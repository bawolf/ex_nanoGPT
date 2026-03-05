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
import sys

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


def extract_attention():
    """Extract CausalSelfAttention reference values."""
    print("\n=== CausalSelfAttention ===")
    torch.manual_seed(42)

    n_embd = 32
    n_head = 4
    head_dim = n_embd // n_head  # 8
    batch = 2
    seq_len = 6

    # Input
    x = torch.randn(batch, seq_len, n_embd)
    save("attn_input", x)

    # c_attn: projects to Q, K, V in one matmul
    c_attn_weight = torch.randn(n_embd, 3 * n_embd) * 0.02
    c_attn_bias = torch.zeros(3 * n_embd)
    save("attn_c_attn_weight", c_attn_weight)
    save("attn_c_attn_bias", c_attn_bias)

    # c_proj: output projection
    c_proj_weight = torch.randn(n_embd, n_embd) * 0.02
    c_proj_bias = torch.zeros(n_embd)
    save("attn_c_proj_weight", c_proj_weight)
    save("attn_c_proj_bias", c_proj_bias)

    sys.path.insert(0, os.path.join(OUT_DIR, '..', '..', 'nanoGPT_ref'))
    from model import CausalSelfAttention, GPTConfig

    config = GPTConfig(
        block_size=seq_len,
        vocab_size=10,
        n_layer=1,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=True,
    )
    attn = CausalSelfAttention(config)

    with torch.no_grad():
        attn.c_attn.weight.copy_(c_attn_weight.T)  # PyTorch Linear stores transposed
        attn.c_attn.bias.copy_(c_attn_bias)
        attn.c_proj.weight.copy_(c_proj_weight.T)
        attn.c_proj.bias.copy_(c_proj_bias)

    attn.eval()
    with torch.no_grad():
        out = attn(x)
    save("attn_output", out)

    # Also save intermediate values for debugging
    with torch.no_grad():
        qkv = attn.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(n_embd, dim=2)
        save("attn_q_flat", q)
        save("attn_k_flat", k)
        save("attn_v_flat", v)

        # Reshape into heads
        B, T, C = x.size()
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)
        save("attn_q_heads", q)
        save("attn_k_heads", k)
        save("attn_v_heads", v)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))
        # Causal mask
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        save("attn_scores_masked", att)

        att = F.softmax(att, dim=-1)
        save("attn_weights", att)

        y = att @ v
        save("attn_after_v", y)


def extract_mlp():
    """Extract MLP reference values."""
    print("\n=== MLP ===")
    torch.manual_seed(42)

    n_embd = 32
    batch = 2
    seq_len = 6

    x = torch.randn(batch, seq_len, n_embd)
    save("mlp_input", x)

    c_fc_weight = torch.randn(n_embd, 4 * n_embd) * 0.02
    c_fc_bias = torch.zeros(4 * n_embd)
    c_proj_weight = torch.randn(4 * n_embd, n_embd) * 0.02
    c_proj_bias = torch.zeros(n_embd)

    save("mlp_c_fc_weight", c_fc_weight)
    save("mlp_c_fc_bias", c_fc_bias)
    save("mlp_c_proj_weight", c_proj_weight)
    save("mlp_c_proj_bias", c_proj_bias)

    sys.path.insert(0, os.path.join(OUT_DIR, '..', '..', 'nanoGPT_ref'))
    from model import MLP, GPTConfig

    config = GPTConfig(
        block_size=seq_len,
        vocab_size=10,
        n_layer=1,
        n_head=4,
        n_embd=n_embd,
        dropout=0.0,
        bias=True,
    )
    mlp = MLP(config)

    with torch.no_grad():
        mlp.c_fc.weight.copy_(c_fc_weight.T)
        mlp.c_fc.bias.copy_(c_fc_bias)
        mlp.c_proj.weight.copy_(c_proj_weight.T)
        mlp.c_proj.bias.copy_(c_proj_bias)

    mlp.eval()
    with torch.no_grad():
        out = mlp(x)
    save("mlp_output", out)


def extract_block():
    """Extract Block (full transformer block) reference values."""
    print("\n=== Block ===")
    torch.manual_seed(42)

    n_embd = 32
    n_head = 4
    batch = 2
    seq_len = 6

    x = torch.randn(batch, seq_len, n_embd)
    save("block_input", x)

    sys.path.insert(0, os.path.join(OUT_DIR, '..', '..', 'nanoGPT_ref'))
    from model import Block, GPTConfig

    config = GPTConfig(
        block_size=seq_len,
        vocab_size=10,
        n_layer=1,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=True,
    )

    torch.manual_seed(42)
    block = Block(config)

    # Save all block params
    sd = block.state_dict()
    for name, param in sd.items():
        clean_name = name.replace('.', '_')
        save(f"block_param_{clean_name}", param)

    block.eval()
    with torch.no_grad():
        out = block(x)
    save("block_output", out)


if __name__ == "__main__":
    import sys
    print("Extracting golden test values...")
    extract_layer_norm()
    extract_embeddings()
    extract_attention()
    extract_mlp()
    extract_block()
    print("\nDone! Golden test files saved to:", OUT_DIR)
