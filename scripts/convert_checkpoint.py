"""
Convert a nanochat PyTorch checkpoint to individual .npy files for Elixir.

Usage:
    python scripts/convert_checkpoint.py <checkpoint.pt> <output_dir> [meta.json]

This reads the state_dict from a nanochat checkpoint and saves each parameter
as a separate .npy file with a flattened key name. Also saves a metadata.json
with the model config and parameter shapes.

The optional meta.json argument provides model config (sequence_len, n_head, etc.)
from nanochat's separate meta file (e.g. meta_000650.json from HuggingFace).

Example output files:
    output_dir/transformer.wte.weight.npy
    output_dir/transformer.h.0.attn.c_q.weight.npy
    output_dir/metadata.json
"""

import os
import sys
import json
import numpy as np
import torch


def extract_config(checkpoint, meta_path=None):
    """Extract model config from checkpoint or external meta JSON."""
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "model_config" in meta:
            return meta["model_config"]
        return meta.get("config", {})

    if not isinstance(checkpoint, dict):
        return {}

    for key in ("model_config", "config"):
        cfg = checkpoint.get(key, None)
        if cfg is None:
            continue
        if hasattr(cfg, "__dict__"):
            return {k: v for k, v in cfg.__dict__.items()
                    if isinstance(v, (int, float, str, bool))}
        if isinstance(cfg, dict) and cfg:
            return cfg

    return {}


def convert(checkpoint_path, output_dir, meta_path=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = extract_config(checkpoint, meta_path)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and any(k.startswith("transformer") for k in checkpoint):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    metadata = {"config": config, "params": {}}

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        arr = tensor.detach().float().numpy()
        filename = key.replace("/", "_") + ".npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, arr)
        metadata["params"][key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "file": filename,
        }
        print(f"  {key}: {arr.shape} -> {filename}")

    meta_path_out = os.path.join(output_dir, "metadata.json")
    with open(meta_path_out, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nConfig: {json.dumps(config, indent=2)}")
    print(f"Metadata saved to {meta_path_out}")
    print(f"Total params saved: {len(metadata['params'])}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(f"Usage: {sys.argv[0]} <checkpoint.pt> <output_dir> [meta.json]")
        sys.exit(1)
    meta = sys.argv[3] if len(sys.argv) == 4 else None
    convert(sys.argv[1], sys.argv[2], meta)
