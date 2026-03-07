#!/usr/bin/env bash
#
# Download and convert a nanochat checkpoint for use with ExNanoGPT.
#
# Usage:
#   ./scripts/download_weights.sh              # downloads nanochat-d32 to weights/
#   ./scripts/download_weights.sh nanochat-d34  # downloads the larger d34 model
#   ./scripts/download_weights.sh nanochat-d34 my_weights/
#
# Requirements: python3, pip install torch numpy (CPU-only is fine)

set -euo pipefail

MODEL="${1:-nanochat-d32}"
OUT_DIR="${2:-weights}"
HF_BASE="https://huggingface.co/karpathy"

case "$MODEL" in
  nanochat-d32)
    CHECKPOINT="model_000650.pt"
    META="meta_000650.json"
    ;;
  nanochat-d34)
    CHECKPOINT="model_000650.pt"
    META="meta_000650.json"
    ;;
  *)
    echo "Unknown model: $MODEL (expected nanochat-d32 or nanochat-d34)"
    exit 1
    ;;
esac

DOWNLOAD_DIR="${OUT_DIR}/raw"
CHECKPOINT_PATH="${DOWNLOAD_DIR}/${CHECKPOINT}"
META_PATH="${DOWNLOAD_DIR}/${META}"

mkdir -p "$DOWNLOAD_DIR"

download() {
  local url="$1" dest="$2"
  echo "    ${url}"
  if command -v curl &> /dev/null; then
    curl -L --progress-bar -o "$dest" "$url"
  elif command -v wget &> /dev/null; then
    wget --show-progress -O "$dest" "$url"
  else
    echo "Error: neither curl nor wget found. Install one and retry."
    exit 1
  fi
}

echo "==> Downloading ${MODEL} checkpoint + meta..."
download "${HF_BASE}/${MODEL}/resolve/main/${CHECKPOINT}" "$CHECKPOINT_PATH"
download "${HF_BASE}/${MODEL}/resolve/main/${META}" "$META_PATH"

TOKENIZER_URL="${HF_BASE}/${MODEL}/resolve/main/tokenizer.pkl"
TOKENIZER_PATH="${DOWNLOAD_DIR}/tokenizer.pkl"

download "${TOKENIZER_URL}" "$TOKENIZER_PATH"

echo ""
echo "==> Converting weights to .npy format..."
python3 scripts/convert_checkpoint.py "$CHECKPOINT_PATH" "$OUT_DIR" "$META_PATH"

echo ""
echo "==> Converting tokenizer (requires: pip install tiktoken)..."
python3 scripts/convert_tokenizer.py "$TOKENIZER_PATH" "${OUT_DIR}/tokenizer.json"

echo ""
echo "==> Done! Converted weights are in: ${OUT_DIR}/"
echo ""
echo "To use in the chat UI:"
echo "  1. Start the server:  iex -S mix phx.server"
echo "  2. Open http://localhost:4000"
echo "  3. Enter '${OUT_DIR}' as the weights path and click 'Load Model'"
echo ""
echo "Or load in IEx:"
echo "  {params, config} = ExNanoGPT.V2.WeightLoader.load(\"${OUT_DIR}\")"
