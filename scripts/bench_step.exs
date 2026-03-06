# Benchmark a single training step with the real Shakespeare config.
# Usage: mix run scripts/bench_step.exs

IO.puts("Preparing data...")
meta = ExNanoGPT.Data.prepare()

model_config = %{
  vocab_size: meta.vocab_size,
  block_size: 256,
  n_layer: 6,
  n_head: 6,
  n_embd: 384,
  dropout: 0.2,
  bias: false
}

IO.puts("Initializing model...")
key = Nx.Random.key(1337)
params = ExNanoGPT.Model.init_params(model_config, key)
param_count = ExNanoGPT.Model.count_params(params)
IO.puts("  #{Float.round(param_count / 1.0e6, 2)}M parameters")

train_data = ExNanoGPT.Data.load_bin(Path.join(ExNanoGPT.Data.data_dir(), "train.bin"))

# Warm up: first step includes JIT compilation
IO.puts("\nStep 1 (includes JIT compilation)...")
t0 = System.monotonic_time(:millisecond)
batch_key = Nx.Random.key(0)
{x, y} = ExNanoGPT.Batch.get_batch(train_data, batch_key, batch_size: 64, block_size: 256)
dropout_key = Nx.Random.key(1_000_000)
{loss, grads} = ExNanoGPT.Trainer.compute_loss_and_grads(params, x, y, model_config, dropout_key)
t1 = System.monotonic_time(:millisecond)
IO.puts("  loss: #{Float.round(Nx.to_number(loss), 4)}")
IO.puts("  time: #{t1 - t0}ms (includes compilation)")

# Second step: steady state
IO.puts("\nStep 2 (steady state)...")
t0 = System.monotonic_time(:millisecond)
batch_key2 = Nx.Random.key(1)
{x2, y2} = ExNanoGPT.Batch.get_batch(train_data, batch_key2, batch_size: 64, block_size: 256)
dropout_key2 = Nx.Random.key(1_000_001)
{loss2, _grads2} = ExNanoGPT.Trainer.compute_loss_and_grads(params, x2, y2, model_config, dropout_key2)
t2 = System.monotonic_time(:millisecond)
IO.puts("  loss: #{Float.round(Nx.to_number(loss2), 4)}")
IO.puts("  time: #{t2 - t0}ms")

step_ms = t2 - t0
total_steps = 5001
IO.puts("\n--- Estimate for full training ---")
IO.puts("  #{step_ms}ms per step")
IO.puts("  #{total_steps} steps")
IO.puts("  ~#{Float.round(step_ms * total_steps / 1000.0 / 60.0, 1)} minutes")
IO.puts("  ~#{Float.round(step_ms * total_steps / 1000.0 / 3600.0, 1)} hours")
