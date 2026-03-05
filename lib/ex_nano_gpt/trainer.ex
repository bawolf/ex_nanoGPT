defmodule ExNanoGPT.Trainer do
  @moduledoc """
  Training loop for the GPT model.

  Mirrors nanoGPT's train.py:
  - Forward pass -> cross-entropy loss -> backward (Nx.Defn.grad) -> AdamW step
  - Gradient accumulation to simulate larger batch sizes
  - Cosine LR schedule with warmup
  - Gradient clipping by global norm
  - Periodic evaluation on train/val sets
  - Checkpoint saving/loading
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
    out_dir: "out"
  }

  @spec default_train_config() :: train_config()
  def default_train_config, do: @default_train_config

  @doc """
  Run the full training loop.
  """
  @spec train(Model.config(), train_config(), Optimizer.config()) :: Model.params()
  def train(model_config, train_config \\ @default_train_config, optim_config \\ Optimizer.default_config()) do
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
    IO.puts("  effective batch size: #{train_config.batch_size * train_config.gradient_accumulation_steps}")

    {final_params, _final_opt_state} =
      Enum.reduce(0..(train_config.max_iters - 1), {params, opt_state}, fn iter, {params, opt_state} ->
        if rem(iter, train_config.eval_interval) == 0 do
          train_loss = estimate_loss(params, model_config, train_data, train_config)
          val_loss = estimate_loss(params, model_config, val_data, train_config)
          IO.puts("step #{iter}: train loss #{Float.round(train_loss, 4)}, val loss #{Float.round(val_loss, 4)}")
          save_checkpoint(params, opt_state, model_config, iter, train_config.out_dir)
        end

        # Gradient accumulation: run N micro-steps, average the gradients
        grads = accumulate_gradients(params, train_data, model_config, train_config, iter)

        if rem(iter, train_config.log_interval) == 0 do
          lr = Optimizer.get_lr(elem(opt_state, 0), optim_config)
          loss = compute_loss(params, train_data, model_config, train_config, iter)
          IO.puts("  iter #{iter}: loss #{Float.round(Nx.to_number(loss), 4)}, lr #{Float.round(lr, 8)}")
        end

        {new_params, new_opt_state} = Optimizer.step(params, grads, opt_state, optim_config)
        {new_params, new_opt_state}
      end)

    train_loss = estimate_loss(final_params, model_config, train_data, train_config)
    val_loss = estimate_loss(final_params, model_config, val_data, train_config)
    IO.puts("Final: train loss #{Float.round(train_loss, 4)}, val loss #{Float.round(val_loss, 4)}")

    save_checkpoint(final_params, nil, model_config, train_config.max_iters, train_config.out_dir)

    final_params
  end

  @doc """
  Accumulate gradients over multiple micro-batches, then average.

  Matches nanoGPT's gradient accumulation loop:
  each micro-step computes gradients on a small batch, and they're
  averaged together before the optimizer step. This simulates a larger
  effective batch size without requiring more memory.
  """
  @spec accumulate_gradients(Model.params(), Nx.Tensor.t(), Model.config(), train_config(), non_neg_integer()) :: map()
  def accumulate_gradients(params, data, model_config, train_config, iter) do
    accum_steps = train_config.gradient_accumulation_steps

    # Run N micro-steps, accumulating gradients
    accumulated =
      Enum.reduce(0..(accum_steps - 1), nil, fn micro_step, acc_grads ->
        seed = iter * accum_steps + micro_step
        batch_key = Nx.Random.key(seed)
        {x, y} = Batch.get_batch(data, batch_key,
          batch_size: train_config.batch_size,
          block_size: train_config.block_size
        )

        dropout_key = Nx.Random.key(seed + 1_000_000)
        {_loss, grads} = compute_loss_and_grads(params, x, y, model_config, dropout_key)

        if acc_grads == nil do
          grads
        else
          add_grads(acc_grads, grads)
        end
      end)

    # Average by dividing by accumulation steps
    if accum_steps > 1 do
      scale_grads(accumulated, 1.0 / accum_steps)
    else
      accumulated
    end
  end

  @doc """
  Compute forward pass loss and gradients w.r.t. parameters.

  All tensors are packed into a single tuple input so they're all traced
  together by `value_and_grad` (avoids EXLA/Expr backend conflicts from
  closure captures). Gradients for non-params tensors are discarded.
  """
  @spec compute_loss_and_grads(Model.params(), Nx.Tensor.t(), Nx.Tensor.t(), Model.config(), Nx.Tensor.t()) ::
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
  Compute loss only (no gradients), for logging.
  """
  @spec compute_loss(Model.params(), Nx.Tensor.t(), Model.config(), train_config(), non_neg_integer()) :: Nx.Tensor.t()
  def compute_loss(params, data, model_config, train_config, iter) do
    batch_key = Nx.Random.key(iter)
    {x, y} = Batch.get_batch(data, batch_key,
      batch_size: train_config.batch_size,
      block_size: train_config.block_size
    )

    logits = Model.forward_train(x, params, Nx.Random.key(0),
      n_head: model_config.n_head, dropout_rate: 0.0)
    Model.cross_entropy_loss(logits, y)
  end

  @doc """
  Estimate loss over multiple batches (for evaluation).
  """
  @spec estimate_loss(Model.params(), Model.config(), Nx.Tensor.t(), train_config()) :: float()
  def estimate_loss(params, model_config, data, train_config) do
    losses =
      for i <- 0..(train_config.eval_iters - 1) do
        batch_key = Nx.Random.key(i + 50_000)
        {x, y} = Batch.get_batch(data, batch_key,
          batch_size: train_config.batch_size,
          block_size: train_config.block_size
        )

        logits = Model.forward_train(x, params, Nx.Random.key(0),
          n_head: model_config.n_head, dropout_rate: 0.0)
        loss = Model.cross_entropy_loss(logits, y)
        Nx.to_number(loss)
      end

    Enum.sum(losses) / length(losses)
  end

  @doc """
  Save a checkpoint to disk as Erlang Term Format.
  """
  @spec save_checkpoint(Model.params(), term(), Model.config(), non_neg_integer(), String.t()) :: :ok
  def save_checkpoint(params, opt_state, model_config, iter, out_dir) do
    checkpoint = %{
      params: params,
      optimizer_state: opt_state,
      model_config: model_config,
      iter: iter
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
