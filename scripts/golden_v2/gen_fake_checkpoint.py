"""
Generate a tiny fake nanochat checkpoint for weight loader tests.

Creates individual .npy files + metadata.json in the same format
as convert_checkpoint.py would produce.
"""

import os
import json
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test", "support", "golden_v2", "fake_ckpt")
os.makedirs(OUT_DIR, exist_ok=True)

# Tiny config for testing
config = {
    "sequence_len": 16,
    "vocab_size": 64,
    "n_layer": 2,
    "n_head": 4,
    "n_kv_head": 2,
    "n_embd": 32,
    "window_pattern": "SL",
}

HEAD_DIM = config["n_embd"] // config["n_head"]  # 8
KV_DIM = config["n_kv_head"] * HEAD_DIM  # 16
VE_GATE_CH = 32

np.random.seed(42)
metadata = {"config": config, "params": {}}

def save_param(key, shape):
    arr = np.random.randn(*shape).astype(np.float32) * 0.01
    filename = key.replace("/", "_") + ".npy"
    np.save(os.path.join(OUT_DIR, filename), arr)
    metadata["params"][key] = {"shape": list(shape), "dtype": "float32", "file": filename}
    print(f"  {key}: {shape}")
    return arr

# Note: PyTorch Linear weights are (out_features, in_features)
save_param("transformer.wte.weight", (64, 32))
save_param("lm_head.weight", (64, 32))
save_param("transformer.resid_lambdas", (2,))
save_param("transformer.x0_lambdas", (2,))

for i in range(2):
    save_param(f"transformer.h.{i}.attn.c_q.weight", (4 * HEAD_DIM, 32))  # (out=n_head*hd, in=n_embd)
    save_param(f"transformer.h.{i}.attn.c_k.weight", (KV_DIM, 32))
    save_param(f"transformer.h.{i}.attn.c_v.weight", (KV_DIM, 32))
    save_param(f"transformer.h.{i}.attn.c_proj.weight", (32, 4 * HEAD_DIM))
    save_param(f"transformer.h.{i}.mlp.c_fc.weight", (4 * 32, 32))
    save_param(f"transformer.h.{i}.mlp.c_proj.weight", (32, 4 * 32))

# Value embed on layer 1 only (has_ve(1, 2) = True)
save_param("transformer.value_embeds.1.weight", (64, KV_DIM))

with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nFake checkpoint saved to {OUT_DIR}")
