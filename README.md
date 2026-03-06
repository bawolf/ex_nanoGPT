# ExNanoGPT

An Elixir/Nx port of [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) -- the simplest, most readable implementation of a GPT language model, rewritten in Elixir using [Nx](https://github.com/elixir-nx/nx).

Every matrix multiply, every attention head, every gradient update is explicit Nx code. No magic libraries -- the AdamW optimizer, training loop, and sampler are all built from scratch.

**Trains on Apple Silicon GPU** via [EMLX](https://github.com/elixir-nx/emlx) (Metal), or on NVIDIA GPU via [EXLA](https://github.com/elixir-nx/nx/tree/main/exla) (CUDA).

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

## Quick Start

```bash
# Install dependencies
mix deps.get

# Run tests (55 tests including golden tests against Python nanoGPT)
mix test

# Smoke test: train a tiny model for 10 steps
mix run scripts/train.exs --smoke

# Full Shakespeare training (~55s/step on M2 Air GPU, ~21s/step on EXLA+CUDA)
mix run scripts/train.exs
```

## Backend Configuration

The default backend is **EMLX (Metal GPU)** -- no setup required on Apple Silicon Macs. Switch backends with the `NX_BACKEND` environment variable:

| Backend | Env var | Device | Use case |
|---------|---------|--------|----------|
| EMLX GPU | `NX_BACKEND=emlx` (default) | Apple Metal GPU | Mac training & inference |
| EMLX CPU | `NX_BACKEND=emlx_cpu` | CPU | Debugging, non-Mac |
| EXLA | `NX_BACKEND=exla` | CPU (Mac) / CUDA GPU (Linux) | Linux GPU training |

```bash
# Default: EMLX on Apple GPU
mix test

# Switch to EXLA (CPU on Mac, CUDA on Linux)
NX_BACKEND=exla mix test

# EXLA with CUDA on Linux
XLA_TARGET=cuda12 NX_BACKEND=exla mix deps.compile
NX_BACKEND=exla mix run scripts/train.exs
```

### Performance (M2 Air, Shakespeare char, 10.65M params)

| Backend | Device | Per-step | 5000 iters | Tests (55) |
|---------|--------|----------|------------|------------|
| EMLX | Metal GPU | **~55s** | ~3.2 days | 4.9s |
| EXLA | CPU | ~113s | ~6.5 days | 21.4s |

EMLX is ~2x faster on Mac. On Linux with CUDA, EXLA will be much faster (seconds per step).

## Known Limitations

### EMLX Metal sort kernel

MLX's Metal GPU backend has a bug where `Nx.sort` crashes with `Unable to load kernel carg_block_sort_bool` -- it dispatches a boolean sort kernel for float tensors, causing a hard crash (SIGABRT, exit code 134).

**Affected code**: `ExNanoGPT.Sampler.apply_top_k/2` in `lib/ex_nano_gpt/sampler.ex`. The natural implementation uses `Nx.sort` (matching nanoGPT's `torch.topk`), but we use a rank-based O(vocab^2) workaround instead.

**Only affects EMLX GPU** -- `NX_BACKEND=exla` works fine with `Nx.sort`.

**To revert**: When the upstream MLX sort kernel is fixed, replace `apply_top_k` with the 4-line `Nx.sort` version documented in the source comment.

**Workaround for your own code**: Transfer to CPU for the sort: `tensor |> Nx.backend_transfer({EMLX.Backend, device: :cpu}) |> Nx.sort() |> Nx.backend_transfer({EMLX.Backend, device: :gpu})`.

Track: [elixir-nx/emlx](https://github.com/elixir-nx/emlx)

### No f64 on Metal

Metal does not support 64-bit floats. This project uses f32 everywhere so this is a non-issue, but if you add code that creates f64 tensors it will fail on EMLX GPU.

### EMLX is still early

[EMLX](https://github.com/elixir-nx/emlx) is not yet on Hex (installed from GitHub). Some Nx operations are missing (interior padding, reduce, window_reduce). None of these are used by this project, but they may affect extensions.

**What to watch:**
- [elixir-nx/emlx](https://github.com/elixir-nx/emlx) -- active development by the Nx core team
- [elixir-nx/nx#1504](https://github.com/elixir-nx/nx/pull/1504) -- Metal PjRt plugin support for Nx

### Training is still slow

Even with EMLX GPU, ~55s/step means ~3 days for the full 5000 iterations. This is an Elixir/Nx overhead issue -- the same model trains in seconds per step in PyTorch on the same hardware. The gap will narrow as EMLX's compiler matures, but for now this project is best used for **learning** (the notebooks work instantly) rather than production training.

**Workarounds:**
- **Rent a GPU** -- A T4 on Vast.ai (~$0.10/hr) trains the full model in minutes with EXLA + CUDA.
- **Run the smoke test** -- `mix run scripts/train.exs --smoke` trains a tiny model in 5 seconds, enough to verify everything works.

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for nanoGPT
- [Elixir Nx team](https://github.com/elixir-nx) for Nx, EXLA, and EMLX
