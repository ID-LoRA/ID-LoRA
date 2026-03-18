#!/usr/bin/env bash
# Run all three ID-LoRA 2.3 inference pipelines in parallel
set -euo pipefail
cd "$(dirname "$0")"

PROMPT='[VISUAL]: A medium shot features a young man with medium-length, curly brown hair and light blue eyes, sitting on a beige or light brown couch. He is wearing a light blue button-up shirt and a red and white patterned tie. His mouth is slightly open as he speaks or reacts. In the background, there is a blurry room setting with warm lighting.
[SPEECH]: We are proud to introduce ID-LoRA.
[SOUNDS]: The speaker has a moderate volume and a conversational tone, sounding engaged and natural. They are close to the microphone. Light, instrumental background music plays softly, creating a calm atmosphere.'

NEG='blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.'

COMMON=(
  --lora-path models/id-lora-celebvhq-ltx2.3/lora_weights.safetensors
  --reference-audio examples/reference.wav
  --first-frame examples/first_frame.png
  --checkpoint models/ltx-2.3-22b-dev.safetensors
  --text-encoder-path models/gemma-3-12b-it-qat-q4_0-unquantized
  --seed 42
  --height 512
  --width 512
  --num-frames 121
  --num-inference-steps 30
  --quantize
  --video-guidance-scale 3.0
  --audio-guidance-scale 7.0
  --identity-guidance-scale 3.0
  --stg-scale 1.0
  --negative-prompt "$NEG"
)

mkdir -p logs

# Each pipeline needs ~24-48GB VRAM. Use separate GPUs for parallel execution.
echo "Starting one_stage on GPU 5..."
CUDA_VISIBLE_DEVICES=5 uv run python ID-LoRA-2.3/scripts/inference_one_stage.py \
  "${COMMON[@]}" --prompt "$PROMPT" --output-dir ID-LoRA-2.3/examples/one_stage \
  > logs/one_stage.log 2>&1 &
PID1=$!

echo "Starting two_stage on GPU 6..."
CUDA_VISIBLE_DEVICES=6 uv run python ID-LoRA-2.3/scripts/inference_two_stage.py \
  "${COMMON[@]}" --prompt "$PROMPT" --output-dir ID-LoRA-2.3/examples/two_stage \
  --upsampler-path models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --distilled-lora-path models/ltx-2.3-22b-distilled-lora-384.safetensors \
  > logs/two_stage.log 2>&1 &
PID2=$!

echo "Starting two_stage_hq on GPU 7 (15 steps, Res2s default)..."
CUDA_VISIBLE_DEVICES=7 uv run python ID-LoRA-2.3/scripts/inference_two_stage_hq.py \
  "${COMMON[@]}" --num-inference-steps 15 --prompt "$PROMPT" --output-dir ID-LoRA-2.3/examples/two_stage_hq \
  --upsampler-path models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --distilled-lora-path models/ltx-2.3-22b-distilled-lora-384.safetensors \
  > logs/two_stage_hq.log 2>&1 &
PID3=$!

echo "PIDs: one_stage=$PID1 two_stage=$PID2 two_stage_hq=$PID3"
echo "Waiting for all pipelines..."
wait $PID1 && echo "one_stage DONE" || echo "one_stage FAILED"
wait $PID2 && echo "two_stage DONE" || echo "two_stage FAILED"
wait $PID3 && echo "two_stage_hq DONE" || echo "two_stage_hq FAILED"
