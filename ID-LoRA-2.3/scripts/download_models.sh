#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-models}"
mkdir -p "$MODEL_DIR"

echo "==> Downloading LTX-2.3 base model (required)..."
hf download Lightricks/LTX-2.3 \
  ltx-2.3-22b-dev.safetensors --local-dir "$MODEL_DIR"

echo "==> Downloading text encoder (required)..."
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir "$MODEL_DIR/gemma-3-12b-it-qat-q4_0-unquantized"

echo "==> Downloading spatial upscaler (for two-stage pipeline)..."
hf download Lightricks/LTX-2.3 \
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors --local-dir "$MODEL_DIR"

echo "==> Downloading distilled LoRA (for two-stage pipeline)..."
hf download Lightricks/LTX-2.3 \
  ltx-2.3-22b-distilled-lora-384.safetensors --local-dir "$MODEL_DIR"

echo "==> Downloading ID-LoRA checkpoints (LTX-2.3)..."
hf download AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K \
  lora_weights.safetensors --local-dir "$MODEL_DIR/id-lora-celebvhq-ltx2.3"

hf download AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K \
  lora_weights.safetensors --local-dir "$MODEL_DIR/id-lora-talkvid-ltx2.3"

echo "==> All models downloaded to $MODEL_DIR/"
