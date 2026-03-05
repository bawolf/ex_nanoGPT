# ExNanoGPT

An Elixir/Nx port of [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) -- the simplest, most readable implementation of a GPT language model, rewritten in Elixir using [Nx](https://github.com/elixir-nx/nx) for numerical computing and [EXLA](https://github.com/elixir-nx/nx/tree/main/exla) for hardware acceleration.

## Why?

This project exists to learn how transformers work by building one from scratch in a language where nothing is hidden behind library abstractions. Every matrix multiply, every attention head, every gradient update is explicit Nx code.

The Python nanoGPT is ~300 lines of model code and ~300 lines of training code. This port aims for similar simplicity in Elixir.

## What it does

Trains a character-level GPT on Shakespeare, then generates new text in the same style. The default configuration (6 layers, 6 heads, 384 embedding dim, ~65 char vocab) trains in minutes on a MacBook.

## Setup

```bash
mix deps.get
mix deps.compile
```

## Project structure

```
lib/
  ex_nano_gpt.ex          # Top-level module
  ex_nano_gpt/
    data.ex                # Shakespeare download, tokenization, batching
    model.ex               # Full GPT model (forward pass, init)
    attention.ex           # Multi-head causal self-attention
    mlp.ex                 # Feed-forward network
    block.ex               # Transformer block (attention + MLP + residuals)
    layer_norm.ex          # Layer normalization
    train.ex               # Training loop, optimizer, LR schedule
    sample.ex              # Text generation (temperature, top-k)

notebooks/                 # Livebook lessons for learning
test/                      # Tests including golden tests vs Python nanoGPT
```

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for nanoGPT
- [Elixir Nx team](https://github.com/elixir-nx) for Nx/EXLA
- [nickgnd](https://github.com/nickgnd/gpt-from-scratch-with-nx-and-axon) for prior art on GPT in Elixir
