# ExNanoGPT

An Elixir/Nx port of [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) -- the simplest, most readable implementation of a GPT language model, rewritten in Elixir using [Nx](https://github.com/elixir-nx/nx) and [EXLA](https://github.com/elixir-nx/nx/tree/main/exla).

Every matrix multiply, every attention head, every gradient update is explicit Nx code. No magic libraries -- the AdamW optimizer, training loop, and sampler are all built from scratch.

## Learn by Building

This repo includes **11 interactive Livebook notebooks** where you rebuild each component yourself, with exercises, solutions, and verification against the reference implementation.

| # | Notebook | What you build |
|---|----------|---------------|
| 00 | [What Is a Language Model](notebooks/00_what_is_a_language_model.livemd) | Tokenizer, batching, Shakespeare data |
| 01 | [Embeddings](notebooks/01_embeddings.livemd) | Token + position embeddings |
| 02 | [Layer Normalization](notebooks/02_layer_norm.livemd) | Mean/variance normalization |
| 03 | [Self-Attention](notebooks/03_self_attention.livemd) | Q/K/V, causal mask, multi-head |
| 04 | [MLP Block](notebooks/04_mlp.livemd) | GELU, feed-forward, 4x expansion |
| 05 | [Transformer Block](notebooks/05_transformer_block.livemd) | Pre-norm residuals, composing attn + MLP |
| 06 | [Full GPT Model](notebooks/06_full_model.livemd) | Assembly, weight tying, forward pass |
| 07 | [Cross-Entropy Loss](notebooks/07_cross_entropy_loss.livemd) | Log-softmax, NLL, ignore_index |
| 08 | [AdamW Optimizer](notebooks/08_adamw_optimizer.livemd) | Moments, bias correction, LR schedule |
| 09 | [Training Loop](notebooks/09_training_loop.livemd) | value_and_grad, gradient accumulation |
| 10 | [Sampling & Generation](notebooks/10_sampling.livemd) | Temperature, top-k, autoregressive loop |

### Getting Started with Livebook

1. **Install Livebook** (if you haven't already):

   ```bash
   # Option A: Install as an Elixir escript
   mix escript.install hex livebook

   # Option B: Download the desktop app from https://livebook.dev
   ```

2. **Start Livebook**:

   ```bash
   livebook server
   ```

   This opens Livebook in your browser (usually at `http://localhost:8080`).

3. **Open a notebook**: Click "Open" in the Livebook sidebar, then "From file" and navigate to the `notebooks/` directory in this repo. Start with `00_what_is_a_language_model.livemd`.

   Or open any notebook directly from GitHub by clicking the badge below:

   [![Run in Livebook](https://livebook.dev/badge/v1/blue.svg)](https://livebook.dev/run?url=https%3A%2F%2Fgithub.com%2Fbawolf%2Fex_nanoGPT%2Fblob%2Fmain%2Fnotebooks%2F00_what_is_a_language_model.livemd)

4. **Work through the exercises**: Each notebook has `# TODO` sections for you to fill in. Solutions are hidden in collapsible sections -- try to solve them yourself first, then check your work.

> **Note**: The notebooks install their own dependencies (`Mix.install`), so you don't need to run `mix deps.get` first. Just open and run.

## Project Structure

```
lib/ex_nano_gpt/
  data.ex              # Shakespeare download, char tokenizer, binary encoding
  batch.ex             # Random batch generation
  embedding.ex         # Token + position embeddings, dropout
  layer_norm.ex        # Layer normalization
  attention.ex         # Multi-head causal self-attention
  mlp.ex               # Feed-forward network (GELU)
  block.ex             # Transformer block (attention + MLP + residuals)
  model.ex             # Full GPT model (assembly, weight tying, loss)
  optimizer.ex         # AdamW with cosine LR schedule, gradient clipping
  trainer.ex           # Training loop, gradient accumulation, checkpointing
  sampler.ex           # Text generation (temperature, top-k, Gumbel-max)

notebooks/             # 11 interactive Livebook lessons
scripts/
  train.exs            # Training script (--smoke for quick test)
  bench_step.exs       # Benchmark a single training step
test/                  # Unit tests + golden tests vs Python nanoGPT
```

## Quick Start (Code)

```bash
# Install dependencies
mix deps.get

# Run tests (55 tests including golden tests against Python nanoGPT)
mix test

# Smoke test: train a tiny model for 10 steps to verify the pipeline
mix run scripts/train.exs --smoke

# Full Shakespeare training (requires GPU or patience -- ~113s/step on M2 Air CPU)
mix run scripts/train.exs
```

## Known Limitations

### No GPU training on Mac

EXLA does not support Apple Metal (the GPU API on Apple Silicon Macs). All training runs on CPU, which is ~100-1000x slower than GPU. The full Shakespeare config benchmarks at **~113 seconds per step on an M2 Air** -- roughly 6.5 days for the default 5000 iterations.

This is an upstream limitation in Google's XLA compiler, not an Elixir issue. PyTorch added Apple GPU support (MPS) in 2022; XLA has not. There is an [experimental Metal PjRt plugin](https://github.com/elixir-nx/xla/issues/8) in progress, but it's incomplete (missing f64 support, no CI infrastructure for Apple Silicon).

**What to watch:**
- [elixir-nx/xla#8](https://github.com/elixir-nx/xla/issues/8) -- the main tracking issue for Apple Silicon GPU support
- [alisinabh/nx_metal](https://github.com/alisinabh/nx_metal) -- a community Metal backend for Nx (early stage)
- [Apple's Metal PjRt plugin](https://developer.apple.com/forums/thread/770344) -- the upstream work that would enable EXLA on Metal

**Workarounds:**
- **Rent a GPU** -- A T4 on Vast.ai (~$0.10/hr) or a Colab notebook will train the full model in minutes. EXLA + CUDA works out of the box with `XLA_TARGET=cuda12`.
- **Run the smoke test** -- `mix run scripts/train.exs --smoke` trains a tiny model in seconds on CPU, enough to verify the full pipeline works.
- **Use the notebooks** -- The Livebook lessons work fine on CPU. Exercises use small tensors that compute instantly.

### EXLA compilation latency

The first training step is slow (~2.5 minutes) because XLA JIT-compiles the computation graph. Subsequent steps reuse the compiled code. This is normal XLA behavior -- it trades upfront compilation time for faster execution.

### Nx 0.9.2 pinned

This project pins `nx` and `exla` to `~> 0.9.2` because newer versions (0.11+) have build issues on macOS (`llvm-c/DataTypes.h` not found). If you're on Linux, you can try upgrading to the latest versions.

## GPU Support (Linux)

EXLA supports NVIDIA GPUs via CUDA. The code is identical -- no changes needed:

```bash
XLA_TARGET=cuda12 mix deps.compile
mix run scripts/train.exs
```

AMD GPUs are supported via ROCm (`XLA_TARGET=rocm`). Google TPUs work too (`XLA_TARGET=tpu`).

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for nanoGPT
- [Elixir Nx team](https://github.com/elixir-nx) for Nx/EXLA
