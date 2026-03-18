# ID-LoRA 2.3 — LTX-2.3 Support

This directory contains the LTX-2.3 (22B) variant of ID-LoRA. It mirrors the base ID-LoRA
structure but targets the newer, larger LTX-2.3 model with improved text conditioning and
audio quality.

## What's Different from the Base Model

| Aspect | Base (LTX-2, 19B) | LTX-2.3 (22B) |
|--------|-------------------|----------------|
| Model size | 19B parameters | 22B parameters |
| Text conditioning | Single feature extractor | Separate video + audio feature extractors with per-token RMSNorm |
| Prompt AdaLN | Not used | Active (modulates cross-attention to text) |
| Vocoder | HiFi-GAN | BigVGAN v2 + bandwidth extension |
| Inference scripts | one-stage, two-stage | one-stage, two-stage, **two-stage HQ** (new) |

Version detection is automatic — the packages read the checkpoint config and select the
correct architecture.

## Requirements

- Python 3.11+
- CUDA 12.x
- 24+ GB VRAM (one-stage with quantization)
- 48+ GB VRAM (two-stage)

## Installation

Since the LTX-2.3 packages share the same Python module names as the base packages
(`ltx_core`, `ltx_pipelines`, `ltx_trainer`), they **cannot** be installed side-by-side.
To use LTX-2.3, point the uv workspace at the `ID-LoRA-2.3/packages` directory.

From the repository root, edit `pyproject.toml` and change the workspace members:

```toml
[tool.uv.workspace]
members = ["ID-LoRA-2.3/packages/*"]
```

Then sync:

```bash
uv sync --frozen
```

## Download Models

```bash
bash ID-LoRA-2.3/scripts/download_models.sh
```

This downloads:

| File | Size | Purpose |
|------|------|---------|
| `ltx-2.3-22b-dev.safetensors` | ~44 GB | Base LTX-2.3 model |
| `gemma-3-12b-it-qat-q4_0-unquantized` | ~6 GB | Text encoder |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ~700 MB | Spatial upscaler (two-stage) |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | ~900 MB | Distilled LoRA (two-stage) |
| `id-lora-celebvhq/lora_weights.safetensors` | ~1.1 GB | ID-LoRA CelebVHQ checkpoint (LTX-2.3) |
| `id-lora-talkvid/lora_weights.safetensors` | ~1.1 GB | ID-LoRA TalkVid checkpoint (LTX-2.3) |

## Inference

All scripts are in `ID-LoRA-2.3/scripts/`. They share the same CLI interface as the base
scripts.

### Two-Stage (Recommended)

Generates at 512x512 then upscales to 1024x1024 with a distilled LoRA refinement pass.

```bash
python ID-LoRA-2.3/scripts/inference_two_stage.py \
  --lora-path models/id-lora-celebvhq/lora_weights.safetensors \
  --reference-audio examples/reference.wav \
  --first-frame examples/first_frame.png \
  --prompt "[VISUAL]: A close-up of a person speaking. [SPEECH]: Hello world. [SOUNDS]: Clear speech." \
  --output-dir outputs/two_stage \
  --quantize
```

### Two-Stage HQ (New in v2.3)

Higher-quality variant using the Res2s sampler and rescaling guidance. Uses fewer steps (15
vs 30) but produces higher fidelity results.

```bash
python ID-LoRA-2.3/scripts/inference_two_stage_hq.py \
  --lora-path models/id-lora-celebvhq/lora_weights.safetensors \
  --reference-audio examples/reference.wav \
  --first-frame examples/first_frame.png \
  --prompt "[VISUAL]: A close-up of a person speaking. [SPEECH]: Hello world. [SOUNDS]: Clear speech." \
  --output-dir outputs/two_stage_hq \
  --quantize
```

### One-Stage (Faster, Lower VRAM)

Generates at a single resolution without upscaling.

```bash
python ID-LoRA-2.3/scripts/inference_one_stage.py \
  --lora-path models/id-lora-celebvhq/lora_weights.safetensors \
  --reference-audio examples/reference.wav \
  --first-frame examples/first_frame.png \
  --prompt "[VISUAL]: A close-up of a person speaking. [SPEECH]: Hello world. [SOUNDS]: Clear speech." \
  --output-dir outputs/one_stage \
  --quantize
```

### Batch Inference

All scripts support a `--prompts-file` argument for batch processing. Pass a JSON file
with a list of prompt objects instead of individual `--prompt` / `--reference-audio` /
`--first-frame` arguments.

### Using Example Args

```bash
python ID-LoRA-2.3/scripts/inference_one_stage.py @ID-LoRA-2.3/examples/one_stage/args.json
python ID-LoRA-2.3/scripts/inference_two_stage.py @ID-LoRA-2.3/examples/two_stage/args.json
```

## Training

Training uses the same `audio_ref_only_ic` strategy as the base model. The only difference
is the model checkpoint path.

### Dataset Preparation

Follow the same dataset preparation workflow as the base model (see the main README),
using the LTX-2.3 packages:

```bash
python ID-LoRA-2.3/packages/ltx-trainer/scripts/process_dataset.py \
  --model-path models/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path models/gemma-3-12b-it-qat-q4_0-unquantized \
  --data-root datasets/your_dataset \
  --output-root datasets/your_dataset_preprocessed
```

### Run Training

```bash
python ID-LoRA-2.3/packages/ltx-trainer/scripts/train.py \
  ID-LoRA-2.3/configs/training_celebvhq.yaml
```

Or for TalkVid:

```bash
python ID-LoRA-2.3/packages/ltx-trainer/scripts/train.py \
  ID-LoRA-2.3/configs/training_talkvid.yaml
```

### Multi-GPU Training

```bash
accelerate launch \
  --config_file ID-LoRA-2.3/packages/ltx-trainer/configs/accelerate/ddp.yaml \
  ID-LoRA-2.3/packages/ltx-trainer/scripts/train.py \
  ID-LoRA-2.3/configs/training_celebvhq.yaml
```

## Directory Structure

```
ID-LoRA-2.3/
├── README.md                   # This file
├── packages/                   # LTX-2.3 packages
│   ├── ltx-core/               # Core model (transformer, VAEs, text encoder)
│   ├── ltx-pipelines/          # Inference pipelines
│   └── ltx-trainer/            # Training toolkit
├── scripts/                    # Inference scripts
│   ├── download_models.sh      # Model download helper
│   ├── inference_one_stage.py  # Single-stage pipeline
│   ├── inference_two_stage.py  # Two-stage with upsampling
│   └── inference_two_stage_hq.py  # Two-stage HQ (Res2s)
├── configs/                    # ID-LoRA training configs
│   ├── training_celebvhq.yaml
│   └── training_talkvid.yaml
└── examples/                   # Example args
    ├── one_stage/args.json
    └── two_stage/args.json
```
