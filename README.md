# ExNanoGPT

[![CI](https://github.com/bawolf/ex_nanoGPT/actions/workflows/ci.yml/badge.svg)](https://github.com/bawolf/ex_nanoGPT/actions/workflows/ci.yml)

A GPT in **176 lines of Elixir**. A modern chat model in **292 lines**.

```elixir
# The entire v1 GPT forward pass (lib/ex_nano_gpt/v1_compact.ex)
def forward(idx, params, config) do
  %{n_head: n_head, n_layer: n_layer} = config
  seq_len = Nx.axis_size(idx, 1)

  x = Nx.add(Nx.take(params.wte, idx, axis: 0),
             Nx.take(params.wpe, Nx.iota({seq_len}), axis: 0))

  x = Enum.reduce(0..(n_layer - 1), x, fn i, x ->
    block(x, elem(params.blocks, i), n_head: n_head)
  end)

  x = layer_norm(x, params.ln_f_w, params.ln_f_b)
  Nx.dot(x, [-1], params.wte, [-1])
end
```

That's a real, working GPT-2. Token embeddings, position embeddings, N transformer blocks, final layer norm, weight-tied output projection. Every matrix multiply is explicit [Nx](https://github.com/elixir-nx/nx) code -- no magic libraries.

This is an Elixir port of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat). Two model versions, both as single-file implementations:

- **[v1_compact.ex](lib/ex_nano_gpt/v1_compact.ex)** (176 lines): GPT-2 architecture -- LayerNorm, GELU, learned position embeddings
- **[v2_compact.ex](lib/ex_nano_gpt/v2_compact.ex)** (292 lines): nanochat architecture -- RoPE, RMSNorm, GQA, ReLU², sliding window attention, KV cache, softcapping

Trains on Apple Silicon GPU via [EMLX](https://github.com/elixir-nx/emlx) (Metal), or on NVIDIA GPU via [EXLA](https://github.com/elixir-nx/nx/tree/main/exla) (CUDA).

## Understand Every Line

Want to know what those 176 lines actually do? **17 Livebook notebooks** build each model piece by piece -- with exercises, visualizations, "Break It" experiments, and verification against the compact implementations.

**New to Nx?** Every notebook teaches Nx inline, translating patterns into familiar Elixir idioms. See the [cheat sheet](#elixir--nx-cheat-sheet) below.

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
  v1_compact.ex        # ★ Complete GPT-2 in one file (176 lines)
  v2_compact.ex        # ★ Complete nanochat in one file (292 lines)

  # The same code, broken out module-by-module for teaching:
  data.ex              # Shakespeare download, char tokenizer
  batch.ex             # Random batch generation
  embedding.ex         # Token + position embeddings
  layer_norm.ex        # Layer normalization
  attention.ex         # Multi-head causal self-attention
  mlp.ex               # Feed-forward network (GELU)
  block.ex             # Transformer block
  model.ex             # Full GPT model assembly
  optimizer.ex         # AdamW optimizer
  trainer.ex           # Training loop
  sampler.ex           # Text generation (temperature, top-k)

  v2/                  # nanochat modules (same as v2_compact, exploded)
    model.ex kv_cache.ex tokenizer.ex weight_loader.ex conversation.ex

lib/ex_nano_gpt_web/   # Phoenix LiveView chat UI
notebooks/             # 17 Livebook lessons (10 v1 + 7 v2)
test/                  # Unit + golden tests vs Python
```

## Quick Start

```bash
# Install dependencies
mix deps.get

# Run tests (unit + golden tests against Python nanoGPT + nanochat)
mix test

# Smoke test: train a tiny model for 10 steps
mix run scripts/train.exs --smoke

# Full Shakespeare training (~55s/step on M2 Air GPU, ~21s/step on EXLA+CUDA)
mix run scripts/train.exs
```

## Backends

Default is **EMLX** (Apple Metal GPU) -- no setup required on Apple Silicon. Set `NX_BACKEND=exla` for CUDA on Linux. Set `NX_BACKEND=emlx_cpu` for CPU-only. Training is ~55s/step on M2 Air (EMLX GPU), much faster on CUDA. For full training, rent a T4 on Vast.ai (~$0.10/hr) with EXLA + CUDA.

```bash
# EXLA with CUDA on Linux
XLA_TARGET=cuda12 NX_BACKEND=exla mix deps.compile
NX_BACKEND=exla mix run scripts/train.exs
```

## v2: Chat UI & Pre-trained Weights

Start the Phoenix LiveView chat interface:

```bash
iex -S mix phx.server  # http://localhost:4000
```

**Option A: Download Karpathy's pre-trained weights** (recommended)

```bash
./scripts/download_weights.sh            # downloads nanochat-d32 → weights/
./scripts/download_weights.sh nanochat-d34  # or the larger 2.2B model
```

Then enter `weights/` as the path in the chat UI and click "Load Model".

**Option B: Train from scratch** (cloud GPU)

```bash
mix run scripts/cloud_train.exs  # see script for cost estimates (~$0.55 for 100M quick test)
```

**v1 Shakespeare** doesn't use the chat UI -- it generates text directly:

```bash
mix run scripts/train.exs        # trains & generates Shakespeare
mix run scripts/train.exs --smoke  # quick 10-step smoke test
```

**Load weights programmatically:**

```elixir
{params, config} = ExNanoGPT.V2.WeightLoader.load("weights/")
logits = ExNanoGPT.V2.Model.forward(token_ids, params, config)
```

## Known Limitations

- **EMLX sort bug**: `Nx.sort` crashes on Metal GPU. Workaround in `sampler.ex` uses rank-based top-k. Revert instructions in source. Track: [elixir-nx/emlx](https://github.com/elixir-nx/emlx)
- **No f64 on Metal**: This project uses f32 everywhere, but custom f64 code will fail on EMLX GPU.
- **EMLX is early**: Not yet on Hex (installed from GitHub). Some Nx ops missing but none used here. Watch: [elixir-nx/nx#1504](https://github.com/elixir-nx/nx/pull/1504)
- **Training speed**: ~55s/step on Mac vs seconds in PyTorch. Best for learning; rent a GPU for full training.

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
