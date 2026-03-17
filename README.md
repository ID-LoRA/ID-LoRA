<h2 align="center">ID-LoRA: Identity-Driven Audio-Video Personalization with In-Context LoRA</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2603.10256"><img src="https://img.shields.io/badge/arXiv-2603.10256-b31b1b.svg" alt="arXiv"></a>
  <a href="https://id-lora.github.io"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/AviadDahan"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow.svg" alt="HuggingFace Models"></a>
</p>

<p align="center">
  <img src="assets/teaser.png" alt="ID-LoRA teaser" width="720">
</p>

## ⚡ TL;DR

**ID-LoRA** enables identity-preserving **audio–video generation in a single model**.  
Given a **text prompt, reference image, and short audio clip**, it generates a talking video where the **voice sounds like the reference speaker and the face matches the subject**.

- 🎤 **Voice identity transfer** from a short reference audio
- 🧑 **Visual identity control** via first-frame conditioning
- 🎬 **Unified audio–video diffusion** (not cascaded pipelines)
- ⚡ **Zero-shot inference** — load LoRA weights, no per-speaker training
- 🪶 **Lightweight** — trained on ~3K pairs on a single GPU

Built on **LTX-2** with **In-Context LoRA** for identity conditioning.


## 🔍 Overview

**ID-LoRA** (Identity-Driven In-Context LoRA) jointly generates a subject's appearance and voice in a single model, letting a text prompt, a reference image, and a short audio clip govern both modalities together. Built on top of [LTX-2](https://github.com/Lightricks/LTX-Video), it is the first method to personalize visual appearance and voice within a single generative pass.

Unlike cascaded pipelines that treat audio and video separately, ID-LoRA operates in a unified latent space where a single text prompt can simultaneously dictate the scene's visual content, environmental acoustics, and speaking style -- while preserving the subject's vocal identity and visual likeness.

Key features:
- 🎵 **Unified audio-video generation** -- voice and appearance synthesized jointly, not cascaded
- 🗣️ **Audio identity transfer** -- the generated speaker sounds like the reference
- 🌍 **Prompt-driven environment control** -- text prompts govern speaking style, environment sounds, and scene content
- 🖼️ **First-frame conditioning** -- provide an image to control the face and scene
- ⚡ **Zero-shot at inference** -- just load the LoRA weights, no per-speaker fine-tuning needed
- 🔬 **Two-stage pipeline** -- high-quality output with 2x spatial upsampling
- 🪶 **Lightweight** -- trained with only ~3K pairs on a single GPU

## 🗺️ Roadmap

- [x] Pre-trained checkpoints (CelebV-HQ, TalkVid)
- [x] Inference scripts (one-stage, two-stage)
- [x] Training code
- [ ] ComfyUI support
- [ ] LTX-2.3 support
- [ ] Training datasets (CelebV-HQ preprocessed, TalkVid preprocessed) -- HuggingFace Datasets
- [ ] Evaluation datasets and benchmark splits (CelebV-HQ v3.2 eval, TalkVid eval) -- HuggingFace Datasets
- [ ] Evaluation scripts

## 🐛 Recent Fixes

> **If you cloned this repo before these fixes**, pull the latest changes to get them:
> ```bash
> git pull origin main
> ```

- **Missing `ltx_core.model` subpackage**: The core model code (`ltx_core.model`) from [LTX-Video](https://github.com/Lightricks/LTX-Video) was not included in the repository, causing `ModuleNotFoundError: No module named 'ltx_core.model'`. This has been added.
- **Audio loading fix**: Replaced `torchaudio.load()` with `soundfile` in inference scripts to avoid a `torchcodec`/FFmpeg dependency issue that caused `RuntimeError: Could not load libtorchcodec` on systems without a system-wide FFmpeg installation.

## 🛠️ Installation

### 📋 Prerequisites

- Python 3.11+
- CUDA 12.x with 24+ GB VRAM (48 GB recommended for two-stage)
- [uv](https://docs.astral.sh/uv/) package manager

### ⚙️ Setup

```bash
git clone https://github.com/ID-LoRA/ID-LoRA.git
cd ID-LoRA

# Install all dependencies (frozen lockfile for reproducibility)
uv sync --frozen
```

### 📥 Download Models

ID-LoRA requires the base LTX-2 model and supporting components. Download everything with:

```bash
bash scripts/download_models.sh
```

This downloads to `models/` by default. To use a custom directory: `bash scripts/download_models.sh /path/to/models`.

## 🚀 Inference

> **📢 Reference audio should be ~5 seconds long.** The model was trained on 5-second reference clips, so providing a reference audio of approximately this duration yields optimal speaker similarity. Shorter or longer clips may degrade voice identity transfer.

### 📝 Prompt Format

ID-LoRA uses a structured prompt with three tagged sections:

```
[VISUAL]: <scene and appearance description>
[SPEECH]: <exact words the person should say>
[SOUNDS]: <speaker vocal style + ambient/environmental sounds>
```

| Section | What to write | Example |
|---------|--------------|---------|
| `[VISUAL]` | Shot type, subject appearance, clothing, setting, lighting, actions | *A medium shot of a woman with short black hair wearing a white blouse, standing in a modern kitchen with warm lighting.* |
| `[SPEECH]` | The literal words to be spoken | *Hello everyone, welcome to our channel.* |
| `[SOUNDS]` | Speaker vocal qualities (volume, tone, distance from mic) and background sounds | *The speaker has a calm, conversational tone at moderate volume, close to the microphone. Soft birds chirping in the background.* |

**Full example:**

```
[VISUAL]: A medium shot features a young man with curly brown hair and light blue eyes, sitting on a beige couch. He is wearing a light blue shirt and a red patterned tie. His mouth is slightly open as he speaks. In the background, there is a blurry room with warm lighting.
[SPEECH]: We are proud to introduce ID-LoRA.
[SOUNDS]: The speaker has a moderate volume and a conversational tone, sounding engaged and natural. They are close to the microphone. Light, instrumental background music plays softly, creating a calm atmosphere.
```

**Tips:**
- Be descriptive in `[VISUAL]` — shot type, colors, clothing, and background details all help, mention that the person is speaking to avoid voice-overs.
- `[SPEECH]` should contain the exact transcript, not a summary.
- `[SOUNDS]` controls both the speaking style (tone, volume, mic proximity) and ambient sounds (music, nature, room noise). Describing these steers the audio generation.

### 🎯 Two-Stage (Recommended -- Higher Quality)

The two-stage pipeline generates at the target resolution, then spatially upsamples 2x with a distilled LoRA for sharper output.

```bash
python scripts/inference_two_stage.py \
  --lora-path models/id-lora-celebvhq/lora_weights.safetensors \
  --reference-audio reference_speaker.wav \
  --first-frame first_frame.png \
  --prompt "[VISUAL]: A person speaks in a sunlit park... [SPEECH]: Hello world... [SOUNDS]: ..." \
  --output-dir outputs/
```

Stage 1 generates at `512x512` (configurable via `--height`/`--width`), stage 2 refines at `1024x1024`.

### ⚡ One-Stage (Fast / Low VRAM)

Single-resolution generation without upsampling. Faster and uses less VRAM.

```bash
python scripts/inference_one_stage.py \
  --lora-path models/id-lora-celebvhq/lora_weights.safetensors \
  --reference-audio reference_speaker.wav \
  --first-frame first_frame.png \
  --prompt "[VISUAL]: A person speaks in a sunlit park... [SPEECH]: Hello world... [SOUNDS]: ..." \
  --output-dir outputs/
```

### 🔧 Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora-path` | required | Path to ID-LoRA `.safetensors` checkpoint |
| `--reference-audio` | required | Reference audio file (`.wav`, ~5 s) for speaker identity |
| `--first-frame` | required | First-frame image for face/scene conditioning |
| `--prompt` | required | Text prompt (or use `--prompts-file` for batch) |
| `--height` / `--width` | 512 / 512 | Generation resolution (must be divisible by 32) |
| `--num-frames` | 121 | Number of frames (`frames % 8 == 1`) |
| `--quantize` | off | Enable int8 quantization for lower VRAM |
| `--video-guidance-scale` | 3.0 | CFG scale for video |
| `--audio-guidance-scale` | 7.0 | CFG scale for audio |
| `--identity-guidance-scale` | 3.0 | Identity guidance strength |
| `--seed` | 42 | Random seed |

### 📦 Batch Inference

Create a JSON file with multiple samples:

```json
[
  {
    "prompt": "...",
    "reference_path": "ref1.wav",
    "first_frame_path": "frame1.png",
    "output_name": "sample_001"
  },
  {
    "prompt": "...",
    "reference_path": "ref2.wav",
    "first_frame_path": "frame2.png",
    "output_name": "sample_002"
  }
]
```

```bash
python scripts/inference_two_stage.py \
  --lora-path models/id-lora-celebvhq/lora_weights.safetensors \
  --prompts-file prompts.json \
  --output-dir outputs/
```

### 🎬 Pre-generated Examples

The [`examples/`](examples/) directory contains pre-generated outputs you can inspect before running inference yourself:

- [`examples/two_stage/output_0000.mp4`](examples/two_stage/output_0000.mp4) — two-stage pipeline
- [`examples/one_stage/output_0000.mp4`](examples/one_stage/output_0000.mp4) — one-stage pipeline

Both were generated with `--quantize` enabled (int8), seed 42, and the prompt:

> *"We are proud to introduce ID-LoRA."*

Full generation configs are in [`examples/two_stage/args.json`](examples/two_stage/args.json) and [`examples/one_stage/args.json`](examples/one_stage/args.json).

## 🏋️ Training

### 📂 Dataset Preparation

ID-LoRA training requires preprocessed video-audio pairs with:
- Video latents (encoded with LTX-2 VAE)
- Audio latents (encoded with audio VAE)
- Reference audio latents (from a different segment of the same speaker)
- First-frame images
- Text captions (visual description + speech transcription + sound description)

See `packages/ltx-trainer/` for the preprocessing pipeline.

### 🏃 Run Training

```bash
CUDA_VISIBLE_DEVICES=0 python packages/ltx-trainer/scripts/train.py \
  configs/training_celebvhq.yaml
```

Example configs are in `configs/`:
- `training_celebvhq.yaml` -- CelebV-HQ dataset
- `training_talkvid.yaml` -- TalkVid dataset

Both use the `audio_ref_only_ic` strategy with negative temporal positions, LoRA rank 128, and 6000 training steps.

## 📦 Pre-trained Checkpoints

| Checkpoint | Dataset | LoRA Rank | Training Steps | Download |
|-----------|---------|-----------|----------------|----------|
| ID-LoRA-CelebVHQ | CelebV-HQ | 128 | 6,000 | [🤗 HuggingFace](https://huggingface.co/AviadDahan/ID-LoRA-CelebVHQ) |
| ID-LoRA-TalkVid | TalkVid | 128 | 6,000 | [🤗 HuggingFace](https://huggingface.co/AviadDahan/ID-LoRA-TalkVid) |

## 🧪 Method

<p align="center">
  <img src="assets/architecture.png" alt="ID-LoRA architecture" width="800">
</p>

ID-LoRA adapts the LTX-2 joint audio-video diffusion backbone (19B parameter DiT with 48 layers) for identity-preserving generation via In-Context LoRA:

1. **Reference conditioning** -- Reference audio is encoded via the Audio VAE and concatenated with noisy target latents along the sequence dimension. The video stream uses standard text-to-video generation with first-frame conditioning, providing a strong visual anchor for face identity.

2. **Negative temporal positions** -- Reference audio tokens receive negative temporal positions in the RoPE space (t ∈ [-T_ref, 0)), cleanly separating them from target tokens (t ∈ [0, T_tgt]) while preserving internal temporal structure. This addresses the positional entanglement problem in cross-video settings where reference and target share no temporal correspondence.

3. **Identity guidance** -- A classifier-free guidance variant applied to the audio stream that amplifies speaker-specific features (vocal timbre, speaking rhythm, pronunciation) by contrasting predictions with and without the reference signal, while leaving scene content to be governed by the text prompt.

4. **Audio-only in-context** -- Only audio reference tokens are prepended (no video reference tokens), keeping the IC mechanism lightweight. The video stream remains free to generate visuals guided by the text prompt and first frame.

The LoRA adapter (rank 128) targets audio self-attention, audio-video cross-attention, and audio FFN layers, learning identity transfer from ~3K training pairs on a single GPU in 6,000 steps.

## 📊 Results

### 📈 Comparison with Baselines

ID-LoRA outperforms cascaded baselines (CosyVoice 3.0 + WAN2.2, VoiceCraft + WAN2.2, ElevenLabs + WAN2.2) and the state-of-the-art commercial Kling 2.6 Pro on speaker similarity and lip synchronization. The advantage widens on the harder cross-video split where reference and target conditions diverge.

**CelebV-HQ -- cross-video (hard) split:**

| Method | Spk Sim ↑ | Face Sim ↑ | LSE-D ↓ | LSE-C ↑ | CLAP ↑ | WER ↓ |
|--------|-----------|------------|---------|---------|--------|-------|
| CosyVoice 3.0 + WAN2.2 | 0.391 | 0.890 | 11.40 | 1.50 | 0.249 | 0.362 |
| ElevenLabs + WAN2.2 | 0.357 | 0.894 | 11.86 | 1.72 | 0.238 | 0.154 |
| Kling 2.6 Pro | 0.385 | 0.854 | 9.49 | 3.47 | 0.316 | 0.121 |
| **ID-LoRA (Ours)** | **0.477** | 0.874 | **8.49** | **3.90** | **0.363** | **0.113** |

**TalkVid split:**

| Method | Spk Sim ↑ | Face Sim ↑ | LSE-D ↓ | LSE-C ↑ | CLAP ↑ | WER ↓ |
|--------|-----------|------------|---------|---------|--------|-------|
| CosyVoice 3.0 + WAN2.2 | 0.579 | 0.770 | 12.34 | 1.20 | 0.315 | 0.223 |
| ElevenLabs + WAN2.2 | 0.491 | 0.772 | 12.42 | 1.31 | 0.319 | 0.064 |
| Kling 2.6 Pro | 0.506 | 0.754 | 11.59 | 2.40 | 0.326 | 0.040 |
| **ID-LoRA (Ours)** | **0.599** | 0.772 | 10.62 | 3.09 | 0.385 | 0.054 |
| **ID-LoRA (CelebV-HQ → TalkVid)** | 0.595 | 0.767 | **10.32** | **3.12** | **0.412** | 0.065 |

### 👥 Human Evaluation

In A/B preference studies on Amazon Mechanical Turk (9 annotators per item, ~285-290 annotations per question), ID-LoRA is significantly preferred over both Kling 2.6 Pro and ElevenLabs + WAN2.2 across all axes (p < 0.001):

<p align="center">
  <img src="assets/human_eval.png" alt="Human evaluation A/B preference" width="720">
</p>

**vs. Kling 2.6 Pro:** voice similarity 73% vs 20%, environment sounds 55% vs 21%, speech manners 65% vs 31%.

**vs. ElevenLabs + WAN2.2:** voice similarity 81% vs 18%, environment sounds 69% vs 26%, speech manners 55% vs 40%.

### 🔊 Environment Sound Interaction

A MOS study evaluating physically grounded audio-visual correspondence across 10 interaction scenarios (dropping a box, clapping, drumming, etc.) shows ID-LoRA achieves higher overall MOS than Kling 2.6 Pro (3.05 vs 2.90), winning on 8 of 10 scenarios:

<p align="center">
  <img src="assets/mos_study.png" alt="Environment sound MOS study" width="700">
</p>

## 📝 Citation

```bibtex
@misc{dahan2026idloraidentitydrivenaudiovideopersonalization,
  title     = {ID-LoRA: Identity-Driven Audio-Video Personalization
               with In-Context LoRA},
  author    = {Aviad Dahan and Moran Yanuka and Noa Kraicer and Lior Wolf and Raja Giryes},
  year      = {2026},
  eprint    = {2603.10256},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url       = {https://arxiv.org/abs/2603.10256}
}
```

## ⚖️ License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Built on [LTX-2](https://github.com/Lightricks/LTX-Video) by Lightricks
- Text encoder: [Gemma 3](https://ai.google.dev/gemma) by Google
- Speaker verification: [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) + ECAPA-TDNN
