# ExNanoGPT

An Elixir/Nx port of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat) -- the simplest, most readable implementation of a GPT language model, rewritten in Elixir using [Nx](https://github.com/elixir-nx/nx).

Every matrix multiply, every attention head, every gradient update is explicit Nx code. No magic libraries -- the AdamW optimizer, BPE tokenizer, training loop, and sampler are all built from scratch.

**Two model versions:**
- **v1 (nanoGPT)**: Character-level GPT-2 architecture with learned position embeddings, LayerNorm, GELU
- **v2 (nanochat)**: Modern architecture with RoPE, RMSNorm, GQA, ReLU², KV cache, BPE tokenizer, SFT, and Phoenix LiveView chat UI

**Trains on Apple Silicon GPU** via [EMLX](https://github.com/elixir-nx/emlx) (Metal), or on NVIDIA GPU via [EXLA](https://github.com/elixir-nx/nx/tree/main/exla) (CUDA).

## Learn by Building

This repo includes **17 interactive Livebook notebooks** where you rebuild each component yourself, with exercises, visualizations, "Break It" experiments, and verification against the reference implementation.

**New to Nx?** No problem -- every notebook teaches Nx concepts inline as they come up, translating Nx patterns into familiar Elixir idioms. See the [Elixir → Nx cheat sheet](#elixir--nx-cheat-sheet) below for a quick reference.

**Part 1: nanoGPT (v1) -- Character-level GPT**

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

**Part 2: nanochat (v2) -- Modern Chat Model**

| # | Notebook | What you build |
|---|----------|---------------|
| 11 | [RoPE & RMSNorm](notebooks/11_rope_and_rmsnorm.livemd) | Rotary embeddings, RMS normalization |
| 12 | [Modern Attention](notebooks/12_modern_attention.livemd) | GQA, QK norm, sliding window |
| 13 | [Modern Transformer](notebooks/13_modern_transformer.livemd) | ReLU², softcapping, residual scaling |
| 14 | [BPE Tokenization](notebooks/14_bpe_tokenization.livemd) | Byte-pair encoding from scratch |
| 15 | [KV Cache](notebooks/15_kv_cache.livemd) | Cached inference, O(n) generation |
| 16 | [Conversation & SFT](notebooks/16_conversation_and_sft.livemd) | Chat format, loss masking, fine-tuning |
| 17 | [Serving with LiveView](notebooks/17_serving_with_liveview.livemd) | Phoenix chat UI, streaming generation |

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
  npy.ex               # NumPy .npy file reader (for weight loading)

  v2/                  # nanochat-style modern architecture
    model.ex           # Compact single-file model (~300 lines)
    kv_cache.ex        # KV cache for fast inference
    tokenizer.ex       # BPE tokenizer (train, encode, decode)
    weight_loader.ex   # Load nanochat PyTorch checkpoints
    conversation.ex    # Conversation rendering + SFT loss masking

lib/ex_nano_gpt_web/   # Phoenix LiveView chat UI
  endpoint.ex          # HTTP endpoint
  router.ex            # Routes
  layouts.ex           # HTML root layout
  live/chat_live.ex    # Chat interface with streaming generation

notebooks/             # 17 interactive Livebook lessons (10 v1 + 7 v2)
scripts/
  train.exs            # v1 training script
  cloud_train.exs      # v2 cloud training with cost estimates
  convert_checkpoint.py # PyTorch -> .npy checkpoint converter
  golden_v2/           # Python scripts for golden test generation
test/                  # 91 tests (unit + golden tests vs Python)
```

## Quick Start

```bash
# Install dependencies
mix deps.get

# Run tests (91 tests including golden tests against Python nanoGPT + nanochat)
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

## v2: nanochat Modern Architecture

The `v2/` modules implement the modern transformer architecture from [nanochat](https://github.com/karpathy/nanochat):

- **RoPE** instead of learned position embeddings (no block_size limit)
- **RMSNorm** instead of LayerNorm (no learnable params, faster)
- **GQA** (grouped-query attention) with separate Q/K/V projections
- **QK normalization** for training stability
- **Sliding window attention** (per-layer window sizes)
- **ReLU²** activation (sparser than GELU)
- **Value embeddings** on alternating layers
- **Per-layer residual scaling** (resid_lambda, x0_lambda)
- **Logit softcapping** (prevents extreme confidence)
- **BPE tokenizer** built from scratch
- **KV cache** for O(n) autoregressive generation
- **SFT support** with conversation rendering and loss masking

### Chat UI

A Phoenix LiveView streaming chat interface is included:

```bash
# Start the chat server
iex -S mix phx.server
# Open http://localhost:4000
```

### Loading Pre-trained Weights

To load a nanochat checkpoint:

```bash
# 1. Convert PyTorch checkpoint to .npy files
python scripts/convert_checkpoint.py path/to/checkpoint.pt converted_weights/

# 2. Load in Elixir
{params, config} = ExNanoGPT.V2.WeightLoader.load("converted_weights/")
logits = ExNanoGPT.V2.Model.forward(token_ids, params, config)
```

### Cloud Training

See `scripts/cloud_train.exs` for a complete training script with:
- Cost estimates: ~$56-74 for 10B tokens depending on GPU
- Quick test: ~$0.55 for 100M tokens on A100
- Supported providers: Lambda Labs, RunPod, Vast.ai, Modal

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

## Elixir → Nx Cheat Sheet

| Elixir | Nx | Notes |
|--------|----|-------|
| `[1, 2, 3]` | `Nx.tensor([1, 2, 3])` | Fixed type, GPU-ready |
| `length(list)` | `Nx.shape(tensor)` | Returns a tuple like `{5}` |
| `Enum.at(list, i)` | `tensor[i]` | Returns a tensor, not a value |
| `Enum.slice(list, a..b)` | `tensor[a..b]` | Range indexing |
| `hd(list)` | `Nx.to_number(t[0])` | Convert to Elixir value |
| `Enum.map(list, &f/1)` | `Nx.multiply(tensor, 2)` | Vectorized, no map needed |
| `Enum.sum(list)` | `Nx.sum(tensor)` | Or `Nx.sum(t, axes: [0])` |
| `Enum.zip_reduce(a, b, ...)` | `Nx.dot(a, b)` | Matrix multiply |
| `:rand.uniform()` | `Nx.Random.uniform(key, ...)` | Functional, returns `{tensor, key}` |
| `Enum.map(ids, &table[&1])` | `Nx.take(table, ids, axis: 0)` | Parallel gather |
| `if cond, do: a, else: b` | `Nx.select(cond, a, b)` | Element-wise conditional |
| `List.replace_at(l, i, v)` | `Nx.put_slice(t, starts, v)` | Immutable update |
| `def f(x), do: ...` | `defn f(x), do: ...` | JIT-compiled for GPU |

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat)
- [Elixir Nx team](https://github.com/elixir-nx) for Nx, EXLA, and EMLX
