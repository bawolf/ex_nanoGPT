defmodule ExNanoGPT.Trainer do
  @moduledoc """
  Training loop for the GPT model.

  Mirrors nanoGPT's train.py:
  - Forward pass -> cross-entropy loss -> backward (Nx.Defn.grad) -> AdamW step
  - Gradient accumulation to simulate larger batch sizes
  - Cosine LR schedule with warmup
  - Gradient clipping by global norm
  - Periodic evaluation on train/val sets
  - Checkpoint saving/loading (only when val loss improves)
  """

  alias ExNanoGPT.{Batch, Data, Model, Optimizer}

  @typedoc """
  Training configuration.
  """
  @type train_config :: %{
          max_iters: pos_integer(),
          eval_interval: pos_integer(),
          eval_iters: pos_integer(),
          log_interval: pos_integer(),
          batch_size: pos_integer(),
          block_size: pos_integer(),
          gradient_accumulation_steps: pos_integer(),
          always_save_checkpoint: boolean(),
          out_dir: String.t()
        }

  @default_train_config %{
    max_iters: 5000,
    eval_interval: 500,
    eval_iters: 200,
    log_interval: 10,
    batch_size: 64,
    block_size: 256,
    gradient_accumulation_steps: 1,
    always_save_checkpoint: true,
    out_dir: "out"
  }

  @spec default_train_config() :: train_config()
  def default_train_config, do: @default_train_config

  @doc """
  Run the full training loop.
  """
  @spec train(Model.config(), train_config(), Optimizer.config()) :: Model.params()
  def train(
        model_config,
        train_config \\ @default_train_config,
        optim_config \\ Optimizer.default_config()
      ) do
    File.mkdir_p!(train_config.out_dir)

    IO.puts("Loading data...")
    train_data = Data.load_bin(Path.join(Data.data_dir(), "train.bin"))
    val_data = Data.load_bin(Path.join(Data.data_dir(), "val.bin"))

    IO.puts("Initializing model...")
    key = Nx.Random.key(1337)
    params = Model.init_params(model_config, key)
    param_count = Model.count_params(params)
    IO.puts("  #{Float.round(param_count / 1.0e6, 2)}M parameters")

    opt_state = Optimizer.init(params)

    IO.puts("Training for #{train_config.max_iters} iterations...")
    IO.puts("  gradient accumulation steps: #{train_config.gradient_accumulation_steps}")

    IO.puts(
      "  effective batch size: #{train_config.batch_size * train_config.gradient_accumulation_steps}"
    )

    best_val_loss = :infinity

    # nanoGPT: while True ... if iter_num > max_iters: break
    # iter_num goes 0..max_iters inclusive = max_iters+1 steps
    {final_params, _final_opt_state, _best_val_loss} =
      Enum.reduce(0..train_config.max_iters, {params, opt_state, best_val_loss}, fn iter,
                                                                                    {params,
                                                                                     opt_state,
                                                                                     best_val_loss} ->
        # Evaluate and checkpoint
        {best_val_loss} =
          if rem(iter, train_config.eval_interval) == 0 do
            train_loss = estimate_loss(params, model_config, train_data, train_config)
            val_loss = estimate_loss(params, model_config, val_data, train_config)

            IO.puts(
              "step #{iter}: train loss #{Float.round(train_loss, 4)}, val loss #{Float.round(val_loss, 4)}"
            )

            if val_loss < best_val_loss or train_config.always_save_checkpoint do
              new_best = min(val_loss, best_val_loss)

              if iter > 0 do
                save_checkpoint(
                  params,
                  opt_state,
                  model_config,
                  iter,
                  best_val_loss,
                  train_config.out_dir
                )

                IO.puts("saving checkpoint to #{train_config.out_dir}")
              end

              {new_best}
            else
              {best_val_loss}
            end
          else
            {best_val_loss}
          end

        # Gradient accumulation
        {grads, last_micro_loss} =
          accumulate_gradients(params, train_data, model_config, train_config, iter)

        # Log using the last micro-step loss, scaled up (matches nanoGPT's approximation)
        if rem(iter, train_config.log_interval) == 0 do
          lr = Optimizer.get_lr(elem(opt_state, 0), optim_config)
          lossf = Nx.to_number(last_micro_loss) * train_config.gradient_accumulation_steps
          IO.puts("  iter #{iter}: loss #{Float.round(lossf, 4)}, lr #{Float.round(lr, 8)}")
        end

        {new_params, new_opt_state} = Optimizer.step(params, grads, opt_state, optim_config)
        {new_params, new_opt_state, best_val_loss}
      end)

    final_params
  end

  @doc """
  Accumulate gradients over multiple micro-batches, then average.

  Each micro-step computes loss / gradient_accumulation_steps (scaling
  before backward, matching nanoGPT). Returns the averaged gradients
  and the last micro-step's (scaled) loss for logging.
  """
  @spec accumulate_gradients(
          Model.params(),
          Nx.Tensor.t(),
          Model.config(),
          train_config(),
          non_neg_integer()
        ) ::
          {map(), Nx.Tensor.t()}
  def accumulate_gradients(params, data, model_config, train_config, iter) do
    accum_steps = train_config.gradient_accumulation_steps

    {accumulated, last_loss} =
      Enum.reduce(0..(accum_steps - 1), {nil, nil}, fn micro_step, {acc_grads, _last_loss} ->
        seed = iter * accum_steps + micro_step
        batch_key = Nx.Random.key(seed)

        {x, y} =
          Batch.get_batch(data, batch_key,
            batch_size: train_config.batch_size,
            block_size: train_config.block_size
          )

        dropout_key = Nx.Random.key(seed + 1_000_000)
        {loss, grads} = compute_loss_and_grads(params, x, y, model_config, dropout_key)

        # Scale loss and grads by 1/accum_steps (matches nanoGPT's loss = loss / grad_accum)
        grads =
          if accum_steps > 1 do
            scale_grads(grads, 1.0 / accum_steps)
          else
            grads
          end

        scaled_loss = if accum_steps > 1, do: Nx.divide(loss, accum_steps), else: loss

        acc_grads =
          if acc_grads == nil do
            grads
          else
            add_grads(acc_grads, grads)
          end

        {acc_grads, scaled_loss}
      end)

    {accumulated, last_loss}
  end

  @doc """
  Compute forward pass loss and gradients w.r.t. parameters.

  All tensors are packed into a single tuple input so they're all traced
  together by `value_and_grad` (avoids EXLA/Expr backend conflicts from
  closure captures). Gradients for non-params tensors are discarded.
  """
  @spec compute_loss_and_grads(
          Model.params(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Model.config(),
          Nx.Tensor.t()
        ) ::
          {Nx.Tensor.t(), map()}
  def compute_loss_and_grads(params, x, y, model_config, key) do
    n_head = model_config.n_head
    dropout_rate = model_config.dropout

    loss_fn = fn {params, x, y, key} ->
      logits = Model.forward_train(x, params, key, n_head: n_head, dropout_rate: dropout_rate)
      Model.cross_entropy_loss(logits, y)
    end

    {loss, {param_grads, _x_grads, _y_grads, _key_grads}} =
      Nx.Defn.value_and_grad(loss_fn).({params, x, y, key})

    {loss, param_grads}
  end

  @doc """
  Estimate loss over multiple batches (for evaluation).
  """
  @spec estimate_loss(Model.params(), Model.config(), Nx.Tensor.t(), train_config()) :: float()
  def estimate_loss(params, model_config, data, train_config) do
    losses =
      for i <- 0..(train_config.eval_iters - 1) do
        batch_key = Nx.Random.key(i + 50_000)

        {x, y} =
          Batch.get_batch(data, batch_key,
            batch_size: train_config.batch_size,
            block_size: train_config.block_size
          )

        logits =
          Model.forward_train(x, params, Nx.Random.key(0),
            n_head: model_config.n_head,
            dropout_rate: 0.0
          )

        loss = Model.cross_entropy_loss(logits, y)
        Nx.to_number(loss)
      end

    Enum.sum(losses) / length(losses)
  end

  @doc """
  Save a checkpoint to disk as Erlang Term Format.
  """
  @spec save_checkpoint(
          Model.params(),
          term(),
          Model.config(),
          non_neg_integer(),
          float(),
          String.t()
        ) :: :ok
  def save_checkpoint(params, opt_state, model_config, iter, best_val_loss, out_dir) do
    checkpoint = %{
      params: params,
      optimizer_state: opt_state,
      model_config: model_config,
      iter: iter,
      best_val_loss: best_val_loss
    }

    path = Path.join(out_dir, "ckpt.etf")
    File.write!(path, :erlang.term_to_binary(checkpoint))
    :ok
  end

  @doc """
  Load a checkpoint from disk.
  """
  @spec load_checkpoint(String.t()) :: map()
  def load_checkpoint(out_dir) do
    path = Path.join(out_dir, "ckpt.etf")
    path |> File.read!() |> :erlang.binary_to_term()
  end

  # --- Gradient tree arithmetic ---

  defp add_grads(%Nx.Tensor{} = a, %Nx.Tensor{} = b), do: Nx.add(a, b)

  defp add_grads(a, b) when is_map(a) and is_map(b) and not is_struct(a) do
    Map.new(a, fn {k, v} -> {k, add_grads(v, b[k])} end)
  end

  defp add_grads(a, b) when is_tuple(a) and is_tuple(b) do
    [Tuple.to_list(a), Tuple.to_list(b)]
    |> Enum.zip()
    |> Enum.map(fn {x, y} -> add_grads(x, y) end)
    |> List.to_tuple()
  end

  defp scale_grads(%Nx.Tensor{} = t, s), do: Nx.multiply(t, s)

  defp scale_grads(map, s) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, scale_grads(v, s)} end)
  end

  defp scale_grads(tuple, s) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.map(&scale_grads(&1, s)) |> List.to_tuple()
  end
end
