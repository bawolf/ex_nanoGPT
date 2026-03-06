# Cloud Training Script for ExNanoGPT v2
#
# This script trains a v2 (nanochat-style) model from scratch.
# Designed to run on a cloud GPU (NVIDIA A100/H100 via EXLA).
#
# Usage:
#   NX_BACKEND=exla elixir scripts/cloud_train.exs \
#     --data data/train.bin \
#     --vocab-size 32768 \
#     --n-layer 12 \
#     --n-embd 768 \
#     --batch-size 32 \
#     --max-steps 50000 \
#     --lr 3.0e-4 \
#     --output checkpoints/v2
#
# Cost estimates (2025 pricing):
# ┌─────────────┬───────────┬──────────┬───────────┬─────────┐
# │ GPU         │ $/hr      │ tok/s    │ 10B tok   │ Cost    │
# ├─────────────┼───────────┼──────────┼───────────┼─────────┤
# │ A100 40GB   │ $1.10     │ ~50K     │ ~55 hrs   │ ~$61    │
# │ A100 80GB   │ $1.60     │ ~60K     │ ~46 hrs   │ ~$74    │
# │ H100 80GB   │ $2.50     │ ~120K    │ ~23 hrs   │ ~$58    │
# │ A10G 24GB   │ $0.75     │ ~20K     │ ~139 hrs  │ ~$104   │
# │ L4 24GB     │ $0.50     │ ~25K     │ ~111 hrs  │ ~$56    │
# └─────────────┴───────────┴──────────┴───────────┴─────────┘
#
# Notes:
# - Estimates based on nanochat's ~1B param model with sequence_len=2048
# - tok/s = tokens per second throughput
# - 10B tokens is roughly what nanochat trains on
# - Actual costs depend on provider (Lambda, RunPod, Vast.ai, etc.)
# - Lambda Labs: A100 ~$1.10/hr, H100 ~$2.50/hr
# - RunPod: A100 ~$0.89/hr (spot), H100 ~$2.19/hr (spot)
# - For a quick test run, 100M tokens on an A100 takes ~30 min (~$0.55)
#
# Providers:
# - Lambda Labs: https://lambdalabs.com/service/gpu-cloud
# - RunPod: https://www.runpod.io/gpu-cloud
# - Vast.ai: https://vast.ai/
# - Modal: https://modal.com/ (serverless GPU, pay per second)
#
# Setup on cloud instance:
#   1. Install Elixir + EXLA: asdf install erlang latest && asdf install elixir latest
#   2. Clone repo: git clone https://github.com/bawolf/ex_nanoGPT
#   3. cd ex_nanoGPT && NX_BACKEND=exla mix deps.get && mix deps.compile
#   4. Prepare data (see below)
#   5. Run this script
#
# Data preparation:
#   Use the BPE tokenizer to convert text -> binary token file:
#     elixir scripts/prepare_data.exs --input data/corpus.txt --output data/train.bin --vocab-size 32768

alias ExNanoGPT.V2.{Model, Tokenizer}

defmodule CloudTrainer do
  import Nx.Defn

  def run(args) do
    opts = parse_args(args)

    IO.puts("=== ExNanoGPT v2 Cloud Training ===")
    IO.puts("Config: #{inspect(opts)}")

    config = %Model{
      sequence_len: opts[:seq_len],
      vocab_size: opts[:vocab_size],
      n_layer: opts[:n_layer],
      n_head: opts[:n_head],
      n_kv_head: opts[:n_kv_head],
      n_embd: opts[:n_embd],
      window_pattern: opts[:window_pattern]
    }

    IO.puts("Initializing model...")
    params = Model.init_params(config, Nx.Random.key(42))
    n_params = Model.count_params(params)
    IO.puts("Parameters: #{n_params} (#{Float.round(n_params / 1_000_000, 1)}M)")

    IO.puts("Loading training data from #{opts[:data]}...")
    data = load_data(opts[:data])
    IO.puts("Tokens: #{byte_size(data) |> div(4)}")

    IO.puts("Training for #{opts[:max_steps]} steps...")
    train_loop(params, config, data, opts)
  end

  defp train_loop(params, config, data, opts) do
    batch_size = opts[:batch_size]
    seq_len = config.sequence_len
    lr = opts[:lr]
    max_steps = opts[:max_steps]
    output_dir = opts[:output]

    File.mkdir_p!(output_dir)

    # AdamW state init
    adam_state = init_adam(params)
    key = Nx.Random.key(0)

    Enum.reduce(0..(max_steps - 1), {params, adam_state, key}, fn step, {params, adam_state, key} ->
      {batch_input, batch_target, key} = sample_batch(data, batch_size, seq_len, key)

      {loss_val, grads} = value_and_grad_fn(params, batch_input, batch_target, config)

      loss_num = Nx.to_number(loss_val)

      cosine_lr = cosine_schedule(step, max_steps, lr)
      {params, adam_state} = adam_update(params, grads, adam_state, cosine_lr)

      if rem(step, 100) == 0 do
        IO.puts("step #{step}/#{max_steps} | loss #{Float.round(loss_num, 4)} | lr #{Float.round(cosine_lr, 6)}")
      end

      if rem(step, 5000) == 0 and step > 0 do
        save_checkpoint(params, config, output_dir, step)
      end

      {params, adam_state, key}
    end)
  end

  defp value_and_grad_fn(params, input, target, config) do
    fun = fn params ->
      logits = Model.forward(input, params, config)
      Model.cross_entropy_loss(logits, target)
    end

    Nx.Defn.value_and_grad(fun).(params)
  end

  defp sample_batch(data, batch_size, seq_len, key) do
    n_tokens = byte_size(data) |> div(4)
    max_start = n_tokens - seq_len - 1

    {starts, key} = Nx.Random.randint(key, 0, max_start, shape: {batch_size})
    starts = Nx.to_flat_list(starts)

    {inputs, targets} =
      starts
      |> Enum.map(fn start ->
        bytes = binary_part(data, start * 4, (seq_len + 1) * 4)
        ids = for <<id::little-signed-32 <- bytes>>, do: id
        {Enum.take(ids, seq_len), Enum.drop(ids, 1) |> Enum.take(seq_len)}
      end)
      |> Enum.unzip()

    input = Nx.tensor(inputs, type: :s64)
    target = Nx.tensor(targets, type: :s64)
    {input, target, key}
  end

  defp load_data(path) do
    File.read!(path)
  end

  defp cosine_schedule(step, max_steps, max_lr) do
    warmup = min(2000, div(max_steps, 10))
    min_lr = max_lr * 0.1

    cond do
      step < warmup -> max_lr * step / warmup
      true ->
        progress = (step - warmup) / max(max_steps - warmup, 1)
        min_lr + 0.5 * (max_lr - min_lr) * (1 + :math.cos(:math.pi() * progress))
    end
  end

  defp init_adam(params), do: init_adam_state(params)
  defp init_adam_state(%Nx.Tensor{} = t), do: %{m: Nx.broadcast(0.0, t), v: Nx.broadcast(0.0, t), t: 0}
  defp init_adam_state(map) when is_map(map), do: Map.new(map, fn {k, v} -> {k, init_adam_state(v)} end)
  defp init_adam_state(tuple) when is_tuple(tuple), do: tuple |> Tuple.to_list() |> Enum.map(&init_adam_state/1) |> List.to_tuple()
  defp init_adam_state(other), do: other

  defp adam_update(params, grads, state, lr) do
    do_adam_update(params, grads, state, lr, 0.9, 0.999, 1.0e-8, 0.01)
  end

  defp do_adam_update(%Nx.Tensor{} = p, %Nx.Tensor{} = g, %{m: m, v: v, t: t}, lr, b1, b2, eps, wd) do
    t = t + 1
    m = Nx.add(Nx.multiply(b1, m), Nx.multiply(1 - b1, g))
    v = Nx.add(Nx.multiply(b2, v), Nx.multiply(1 - b2, Nx.multiply(g, g)))
    m_hat = Nx.divide(m, 1 - :math.pow(b1, t))
    v_hat = Nx.divide(v, 1 - :math.pow(b2, t))
    update = Nx.add(Nx.divide(m_hat, Nx.add(Nx.sqrt(v_hat), eps)), Nx.multiply(wd, p))
    p = Nx.subtract(p, Nx.multiply(lr, update))
    {p, %{m: m, v: v, t: t}}
  end
  defp do_adam_update(map, grads, state, lr, b1, b2, eps, wd) when is_map(map) do
    {new_params, new_state} =
      Enum.reduce(Map.keys(map), {%{}, %{}}, fn k, {p_acc, s_acc} ->
        {p, s} = do_adam_update(Map.fetch!(map, k), Map.fetch!(grads, k), Map.fetch!(state, k), lr, b1, b2, eps, wd)
        {Map.put(p_acc, k, p), Map.put(s_acc, k, s)}
      end)
    {new_params, new_state}
  end
  defp do_adam_update(tuple, grads, state, lr, b1, b2, eps, wd) when is_tuple(tuple) do
    zipped = Enum.zip([Tuple.to_list(tuple), Tuple.to_list(grads), Tuple.to_list(state)])
    {ps, ss} = Enum.map(zipped, fn {p, g, s} -> do_adam_update(p, g, s, lr, b1, b2, eps, wd) end) |> Enum.unzip()
    {List.to_tuple(ps), List.to_tuple(ss)}
  end
  defp do_adam_update(other, _g, state, _lr, _b1, _b2, _eps, _wd), do: {other, state}

  defp save_checkpoint(params, config, dir, step) do
    path = Path.join(dir, "step_#{step}.etf")
    data = :erlang.term_to_binary(%{params: params, config: config, step: step})
    File.write!(path, data)
    IO.puts("  Saved checkpoint: #{path}")
  end

  defp parse_args(args) do
    {parsed, _, _} = OptionParser.parse(args,
      strict: [
        data: :string, vocab_size: :integer, n_layer: :integer,
        n_head: :integer, n_kv_head: :integer, n_embd: :integer,
        batch_size: :integer, max_steps: :integer, lr: :float,
        output: :string, seq_len: :integer, window_pattern: :string
      ]
    )

    Keyword.merge([
      data: "data/train.bin", vocab_size: 32768, n_layer: 12,
      n_head: 6, n_kv_head: 6, n_embd: 768, batch_size: 32,
      max_steps: 50000, lr: 3.0e-4, output: "checkpoints/v2",
      seq_len: 2048, window_pattern: "SSSL"
    ], parsed)
  end
end

CloudTrainer.run(System.argv())
