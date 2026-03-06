"""
Convert a nanochat PyTorch checkpoint to individual .npy files for Elixir.

Usage:
    python scripts/convert_checkpoint.py <checkpoint.pt> <output_dir>

This reads the state_dict from a nanochat checkpoint and saves each parameter
as a separate .npy file with a flattened key name. Also saves a metadata.json
with the model config and parameter shapes.

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


def convert(checkpoint_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        config = checkpoint.get("config", {})
    elif isinstance(checkpoint, dict) and any(k.startswith("transformer") for k in checkpoint):
        state_dict = checkpoint
        config = {}
    else:
        state_dict = checkpoint
        config = {}

    metadata = {"config": {}, "params": {}}

    if hasattr(config, "__dict__"):
        metadata["config"] = {
            k: v for k, v in config.__dict__.items()
            if isinstance(v, (int, float, str, bool))
        }
    elif isinstance(config, dict):
        metadata["config"] = config

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

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")
    print(f"Total params saved: {len(metadata['params'])}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <checkpoint.pt> <output_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
