#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-models}"
mkdir -p "$MODEL_DIR"

echo "==> Downloading LTX-2.3 base model (required)..."
hf download Lightricks/LTX-2 \
  ltx-2.3-22b-dev.safetensors --local-dir "$MODEL_DIR"

echo "==> Downloading text encoder (required)..."
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir "$MODEL_DIR/gemma-3-12b-it-qat-q4_0-unquantized"

echo "==> Downloading spatial upscaler (for two-stage pipeline)..."
hf download Lightricks/LTX-2 \
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors --local-dir "$MODEL_DIR"

echo "==> Downloading distilled LoRA (for two-stage pipeline)..."
hf download Lightricks/LTX-2 \
  ltx-2.3-22b-distilled-lora-384.safetensors --local-dir "$MODEL_DIR"

echo "==> Downloading ID-LoRA checkpoints..."
hf download AviadDahan/ID-LoRA-CelebVHQ \
  lora_weights.safetensors --local-dir "$MODEL_DIR/id-lora-celebvhq"

hf download AviadDahan/ID-LoRA-TalkVid \
  lora_weights.safetensors --local-dir "$MODEL_DIR/id-lora-talkvid"

echo "==> All models downloaded to $MODEL_DIR/"
