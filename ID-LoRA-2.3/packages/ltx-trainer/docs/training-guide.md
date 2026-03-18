# Training Guide

This guide covers how to run training jobs, from basic single-GPU training to advanced distributed setups and automatic
model uploads.

## ⚡ Basic Training (Single GPU)

After preprocessing your dataset and preparing a configuration file, you can start training using the trainer script:

```bash
uv run python scripts/train.py configs/ltx2_av_lora.yaml
```

The trainer will:

1. **Load your configuration** and validate all parameters
2. **Initialize models** and apply optimizations
3. **Run the training loop** with progress tracking
4. **Generate validation videos** (if configured)
5. **Save the trained weights** in your output directory

### Output Files

**For LoRA training:**

- `lora_weights.safetensors` - Main LoRA weights file
- `training_config.yaml` - Copy of training configuration
- `validation_samples/` - Generated validation videos (if enabled)

**For full model fine-tuning:**

- `model_weights.safetensors` - Full model weights
- `training_config.yaml` - Copy of training configuration
- `validation_samples/` - Generated validation videos (if enabled)

## 🖥️ Distributed / Multi-GPU Training

We use Hugging Face 🤗 [Accelerate](https://huggingface.co/docs/accelerate/index) for multi-GPU DDP and FSDP.

### Configure Accelerate

Run the interactive wizard once to set up your environment (DDP / FSDP, GPU count, etc.):

```bash
uv run accelerate config
```

This stores your preferences in `~/.cache/huggingface/accelerate/default_config.yaml`.

### Use the Provided Accelerate Configs (Recommended)

We include ready-to-use Accelerate config files in `configs/accelerate/`:

- [ddp.yaml](../configs/accelerate/ddp.yaml) — Standard DDP
- [ddp_compile.yaml](../configs/accelerate/ddp_compile.yaml) — DDP with `torch.compile` (Inductor)
- [fsdp.yaml](../configs/accelerate/fsdp.yaml) — Standard FSDP (auto-wraps `BasicAVTransformerBlock`)
- [fsdp_compile.yaml](../configs/accelerate/fsdp_compile.yaml) — FSDP with `torch.compile` (Inductor)

Launch with a specific config using `--config_file`:

```bash
# DDP (2 GPUs shown as example)
CUDA_VISIBLE_DEVICES=0,1 \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml \
  scripts/train.py configs/ltx2_av_lora.yaml

# DDP + torch.compile
CUDA_VISIBLE_DEVICES=0,1 \
uv run accelerate launch --config_file configs/accelerate/ddp_compile.yaml \
  scripts/train.py configs/ltx2_av_lora.yaml

# FSDP (4 GPUs shown as example)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run accelerate launch --config_file configs/accelerate/fsdp.yaml \
  scripts/train.py configs/ltx2_av_lora.yaml

# FSDP + torch.compile
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run accelerate launch --config_file configs/accelerate/fsdp_compile.yaml \
  scripts/train.py configs/ltx2_av_lora.yaml
```

**Notes:**

- The number of processes is taken from the Accelerate config (`num_processes`). Override with `--num_processes X` or
  restrict GPUs with `CUDA_VISIBLE_DEVICES`.
- The compile variants enable `torch.compile` with the Inductor backend via Accelerate's `dynamo_config`.
- FSDP configs auto-wrap the transformer blocks (`fsdp_transformer_layer_cls_to_wrap: BasicAVTransformerBlock`).

### Launch with Your Default Accelerate Config

If you prefer to use your default Accelerate profile:

```bash
# Use settings from your default accelerate config
uv run accelerate launch scripts/train.py configs/ltx2_av_lora.yaml

# Override number of processes on the fly (e.g., 2 GPUs)
uv run accelerate launch --num_processes 2 scripts/train.py configs/ltx2_av_lora.yaml

# Select specific GPUs
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch scripts/train.py configs/ltx2_av_lora.yaml
```

> [!TIP]
> You can disable the in-terminal progress bars with `--disable-progress-bars` flag in the trainer CLI if desired.

### Benefits of Distributed Training

- **Faster training**: Distribute workload across multiple GPUs
- **Larger effective batch sizes**: Combine gradients from multiple GPUs
- **Memory efficiency**: Each GPU handles a portion of the batch

> [!NOTE]
> Distributed training requires that all GPUs have sufficient memory for the model and batch size. The effective batch
> size becomes `batch_size × num_processes`.

## 🤗 Pushing Models to Hugging Face Hub

You can automatically push your trained models to the Hugging Face Hub by adding the following to your configuration:

```yaml
hub:
  push_to_hub: true
  hub_model_id: "your-username/your-model-name"
```

### Prerequisites

Before pushing, make sure you:

1. **Have a Hugging Face account** - Sign up at [huggingface.co](https://huggingface.co)
2. **Are logged in** via `huggingface-cli login` or have set the `HUGGING_FACE_HUB_TOKEN` environment variable
3. **Have write access** to the specified repository (it will be created if it doesn't exist)

### Login Options

**Option 1: Interactive login**

```bash
uv run huggingface-cli login
```

**Option 2: Environment variable**

```bash
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

### What Gets Uploaded

The trainer will automatically:

- **Create a model card** with training details and sample outputs
- **Upload model weights**
- **Push sample videos as GIFs** in the model card
- **Include training configuration and prompts**

## 📊 Weights & Biases Logging

Enable experiment tracking with W&B by adding to your configuration:

```yaml
wandb:
  enabled: true
  project: "ltx-2-trainer"
  entity: null  # Your W&B username or team
  tags: [ "ltx2", "lora" ]
  log_validation_videos: true
```

This will log:

- Training loss and learning rate
- Validation videos
- Model configuration
- Training progress

## 🚀 Next Steps

After training completes:

- **Run inference with your trained LoRA** - The [`ltx-pipelines`](../../ltx-pipelines/) package provides
  production-ready inference
  pipelines that support loading custom LoRAs. Available pipelines include text-to-video, image-to-video,
  IC-LoRA video-to-video, and more. See the [`ltx-pipelines`](../../ltx-pipelines/) package for usage details.
- **Test your model** with validation prompts
- **Iterate and improve** based on validation results
- **Share your results** by pushing to Hugging Face Hub

## 💡 Tips for Successful Training

- **Start small**: Begin with a small dataset and a few hundred steps to verify everything works
- **Monitor validation**: Keep an eye on validation samples to catch overfitting
- **Adjust learning rate**: Lower learning rates often produce better results
- **Use gradient checkpointing**: Essential for training with limited GPU memory
- **Save checkpoints**: Regular checkpoints help recover from interruptions

## Need Help?

If you encounter issues during training, see the [Troubleshooting Guide](troubleshooting.md).

Join our [Discord community](https://discord.gg/ltxplatform) for real-time help!
