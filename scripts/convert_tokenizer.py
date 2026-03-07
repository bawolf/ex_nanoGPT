"""
Convert nanochat's tiktoken tokenizer to JSON for Elixir.

Usage:
    python scripts/convert_tokenizer.py <tokenizer.pkl> <output.json>

Reads a pickled tiktoken Encoding and writes a JSON file mapping each token ID
to its byte sequence. Our Elixir Tokenizer.load_vocab_json/1 can load this.

Requires: pip install tiktoken
"""

import sys
import json
import pickle


SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


def convert(tokenizer_path, output_path):
    print(f"Loading tokenizer: {tokenizer_path}")
    with open(tokenizer_path, "rb") as f:
        tok = pickle.load(f)

    n_vocab = tok.n_vocab
    print(f"Vocab size: {n_vocab}")

    vocab = {}
    for token_id in range(n_vocab):
        try:
            raw = tok.decode_bytes([token_id])
            vocab[str(token_id)] = list(raw)
        except Exception:
            vocab[str(token_id)] = [token_id % 256]

    special_start = n_vocab
    special_map = {}
    for i, name in enumerate(SPECIAL_TOKENS):
        special_map[name] = special_start + i

    data = {
        "vocab": vocab,
        "special_tokens": special_map,
        "vocab_size": special_start + len(SPECIAL_TOKENS),
    }

    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"Saved: {output_path}")
    print(f"  Base vocab: {n_vocab}")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"  Total vocab: {data['vocab_size']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <tokenizer.pkl> <output.json>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
