"""
Extract golden test values from nanoGPT's Python implementation.

Run this script to generate .npy files that the Elixir tests
compare against to verify numerical correctness.

Usage:
    python test/golden/extract.py

Requires: torch, numpy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def save(name, tensor):
    """Save a tensor as a .npy file."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    path = os.path.join(OUT_DIR, f"{name}.npy")
    np.save(path, tensor)
    print(f"  Saved {name}.npy with shape {tensor.shape}")


def extract_layer_norm():
    """Extract LayerNorm reference values."""
    print("\n=== LayerNorm ===")
    torch.manual_seed(42)

    ndim = 8
    ln = nn.LayerNorm(ndim, elementwise_affine=True)
    # Match nanoGPT: weight=ones, bias=zeros initially
    # But let's use non-trivial weights to make the test meaningful
    with torch.no_grad():
        ln.weight.copy_(torch.tensor([1.0, 2.0, 0.5, 1.5, 1.0, 0.8, 1.2, 0.9]))
        ln.bias.copy_(torch.tensor([0.1, -0.1, 0.0, 0.2, -0.2, 0.0, 0.1, -0.1]))

    x = torch.randn(2, 4, ndim)
    save("ln_input", x)
    save("ln_weight", ln.weight)
    save("ln_bias", ln.bias)

    out = ln(x)
    save("ln_output", out)


def extract_embeddings():
    """Extract embedding reference values."""
    print("\n=== Embeddings ===")
    torch.manual_seed(42)

    vocab_size = 10
    block_size = 8
    n_embd = 16

    wte = nn.Embedding(vocab_size, n_embd)
    wpe = nn.Embedding(block_size, n_embd)

    # Use specific weights for reproducibility
    torch.manual_seed(42)
    nn.init.normal_(wte.weight, mean=0.0, std=0.02)
    torch.manual_seed(43)
    nn.init.normal_(wpe.weight, mean=0.0, std=0.02)

    save("emb_wte", wte.weight)
    save("emb_wpe", wpe.weight)

    # Test input: batch of 2, seq len 5
    idx = torch.tensor([[1, 3, 5, 7, 2], [0, 9, 4, 6, 8]], dtype=torch.long)
    save("emb_idx", idx)

    tok_emb = wte(idx)  # (2, 5, 16)
    save("emb_tok", tok_emb)

    pos = torch.arange(0, 5, dtype=torch.long)
    pos_emb = wpe(pos)  # (5, 16)
    save("emb_pos", pos_emb)

    combined = tok_emb + pos_emb
    save("emb_combined", combined)


if __name__ == "__main__":
    print("Extracting golden test values...")
    extract_layer_norm()
    extract_embeddings()
    print("\nDone! Golden test files saved to:", OUT_DIR)
