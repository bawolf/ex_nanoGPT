# Training script for Shakespeare character-level GPT.
# Usage: mix run scripts/train.exs [--smoke]
#
# --smoke: tiny model, 10 steps, just to verify the pipeline works

smoke? = "--smoke" in System.argv()

# Step 1: Prepare data
IO.puts("=== Preparing data ===")
meta = ExNanoGPT.Data.prepare()

# Step 2: Configure model and training
{model_config, train_config, optim_config} =
  if smoke? do
    IO.puts("\n=== SMOKE TEST (tiny model, 10 steps) ===")

    model_config = %{
      vocab_size: meta.vocab_size,
      block_size: 32,
      n_layer: 1,
      n_head: 2,
      n_embd: 32,
      dropout: 0.0,
      bias: true
    }

    train_config = %{
      max_iters: 10,
      eval_interval: 5,
      eval_iters: 2,
      log_interval: 1,
      batch_size: 4,
      block_size: 32,
      gradient_accumulation_steps: 1,
      always_save_checkpoint: true,
      out_dir: "out-smoke"
    }

    optim_config = %{ExNanoGPT.Optimizer.default_config() |
      learning_rate: 1.0e-3,
      warmup_iters: 0,
      lr_decay_iters: 10,
      min_lr: 1.0e-4
    }

    {model_config, train_config, optim_config}
  else
    IO.puts("\n=== Shakespeare char training (matches nanoGPT config) ===")

    model_config = %{
      vocab_size: meta.vocab_size,
      block_size: 256,
      n_layer: 6,
      n_head: 6,
      n_embd: 384,
      dropout: 0.2,
      bias: false
    }

    train_config = %{
      max_iters: 5000,
      eval_interval: 250,
      eval_iters: 200,
      log_interval: 10,
      batch_size: 64,
      block_size: 256,
      gradient_accumulation_steps: 1,
      always_save_checkpoint: false,
      out_dir: "out-shakespeare-char"
    }

    optim_config = %{ExNanoGPT.Optimizer.default_config() |
      learning_rate: 1.0e-3,
      warmup_iters: 100,
      lr_decay_iters: 5000,
      min_lr: 1.0e-4,
      beta2: 0.99
    }

    {model_config, train_config, optim_config}
  end

# Step 3: Train
IO.puts("\nModel config: #{inspect(model_config)}")
IO.puts("Train config: #{inspect(train_config)}")

t0 = System.monotonic_time(:millisecond)
params = ExNanoGPT.Trainer.train(model_config, train_config, optim_config)
elapsed = System.monotonic_time(:millisecond) - t0
IO.puts("\nTraining took #{Float.round(elapsed / 1000.0, 1)}s")

# Step 4: Generate sample text
IO.puts("\n=== Generating sample text ===")
key = Nx.Random.key(1337)

max_tokens = if smoke?, do: 50, else: 500
text = ExNanoGPT.Sampler.generate_text(params, model_config, meta, "\n", key,
  max_new_tokens: max_tokens,
  temperature: 0.8,
  top_k: 200
)

IO.puts("---")
IO.puts(text)
IO.puts("---")
