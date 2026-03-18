#!/usr/bin/env python3
"""
ID-LoRA one-stage inference pipeline (fast / low-VRAM).

Generates audio-video at a single resolution with ID-LoRA + full guidance suite.
No spatial upsampling -- faster and uses less VRAM than the two-stage pipeline.

Usage
-----
# Single-sample inference
python scripts/inference_one_stage.py \
    --lora-path models/id-lora-celebvhq.safetensors \
    --reference-audio reference.wav \
    --first-frame first_frame.png \
    --prompt "A person speaks warmly in a sunlit park..." \
    --output-dir outputs/

# Batch inference from prompts file
python scripts/inference_one_stage.py \
    --lora-path models/id-lora-celebvhq.safetensors \
    --prompts-file prompts.json \
    --output-dir outputs/
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import av
import torch
import torchaudio
from tqdm import tqdm

if not sys.stdout.isatty():
    from functools import partial
    tqdm = partial(tqdm, mininterval=30, ncols=80)

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider, STGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, SpatioTemporalScaleFactors
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.guidance import BatchedPerturbationConfig, Perturbation, PerturbationConfig, PerturbationType
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.loader.registry import DummyRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    AudioEncoderConfigurator,
    AudioProcessor,
    decode_audio as vae_decode_audio,
)
from ltx_core.model.transformer import (
    LTXModelConfigurator,
    LTXV_MODEL_COMFY_RENAMING_MAP,
    X0Model,
)
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape

from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines.utils.helpers import cleanup_memory, euler_denoising_loop, modality_from_latent_state
from ltx_pipelines.utils.media_io import preprocess
from ltx_pipelines.utils.types import PipelineComponents

from ltx_trainer.video_utils import read_video, save_video


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = "models/ltx-2-19b-dev.safetensors"
DEFAULT_TEXT_ENCODER = "models/gemma-3-12b-it-qat-q4_0-unquantized"

DEFAULT_VIDEO_HEIGHT = 512
DEFAULT_VIDEO_WIDTH = 512
DEFAULT_NUM_FRAMES = 121
DEFAULT_FRAME_RATE = 25.0
RESOLUTION_DIVISOR = 32
MAX_LONG_SIDE = 512
MAX_PIXELS = 576 * 1024

DEFAULT_INFERENCE_STEPS = 30
DEFAULT_VIDEO_GUIDANCE_SCALE = 3.0
DEFAULT_AUDIO_GUIDANCE_SCALE = 7.0
DEFAULT_IDENTITY_GUIDANCE_SCALE = 3.0

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def get_video_dimensions(video_path: str | Path) -> tuple[int, int]:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        return stream.height, stream.width


def snap_to_divisor(value: int, divisor: int = RESOLUTION_DIVISOR) -> int:
    return max(int(round(value / divisor)) * divisor, divisor)


def compute_resolution_match_aspect(
    src_h: int, src_w: int,
    max_long: int = MAX_LONG_SIDE,
    max_pixels: int = MAX_PIXELS,
    divisor: int = RESOLUTION_DIVISOR,
) -> tuple[int, int]:
    scale = max_long / max(src_h, src_w)
    pixel_scale = (max_pixels / (src_h * src_w)) ** 0.5
    scale = min(scale, pixel_scale)
    return snap_to_divisor(int(round(src_h * scale)), divisor), snap_to_divisor(int(round(src_w * scale)), divisor)


# ---------------------------------------------------------------------------
# One-stage ID-LoRA pipeline
# ---------------------------------------------------------------------------

class IDLoraOneStagePipeline:
    """
    One-stage pipeline for audio identity transfer with ID-LoRA.

    Generates at a single resolution (no spatial upsampling).
    Faster and uses less VRAM than the two-stage pipeline.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device,
        quantize: bool = False,
        stg_scale: float = 1.0,
        stg_blocks: list[int] | None = None,
        stg_mode: str = "stg_av",
        identity_guidance: bool = True,
        identity_guidance_scale: float = 3.0,
        av_bimodal_cfg: bool = True,
        av_bimodal_scale: float = 3.0,
    ):
        self.dtype = torch.bfloat16
        self.device = device
        self._checkpoint_path = checkpoint_path
        self._loras = loras
        self._quantize = quantize

        self._stg_scale = stg_scale
        self._stg_blocks = stg_blocks if stg_blocks is not None else [29]
        self._stg_mode = stg_mode
        self._identity_guidance = identity_guidance
        self._identity_guidance_scale = identity_guidance_scale
        self._av_bimodal_cfg = av_bimodal_cfg
        self._av_bimodal_scale = av_bimodal_scale

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,
            fp8transformer=False,
        )

        self.pipeline_components = PipelineComponents(dtype=self.dtype, device=device)
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)
        self._transformer = None

    def load_models(self):
        print("Loading video encoder...")
        self._video_encoder = self.model_ledger.video_encoder()

        print("Loading transformer...")
        transformer_builder = Builder(
            model_path=self._checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
            loras=tuple(self._loras),
            registry=DummyRegistry(),
        )
        transformer = transformer_builder.build(device=self.device, dtype=self.dtype)
        if self._quantize:
            print("Applying int8-quanto quantization...")
            from ltx_trainer.quantization import quantize_model
            transformer = quantize_model(transformer, "int8-quanto")
            cleanup_memory()
        self._transformer = X0Model(transformer).eval()

        print("Loading video decoder...")
        self._video_decoder = self.model_ledger.video_decoder()

        print("Loading audio encoder...")
        self._audio_encoder = Builder(
            model_path=self.model_ledger.checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=DummyRegistry(),
        ).build(device=self.device, dtype=self.dtype).eval()

        print("Loading audio decoder...")
        self._audio_decoder = self.model_ledger.audio_decoder()
        self._vocoder = self.model_ledger.vocoder()

        print("Creating audio processor...")
        self._audio_processor = AudioProcessor(
            sample_rate=16000, mel_bins=64, mel_hop_length=160, n_fft=1024,
        ).to(self.device)

    def text_encoder(self):
        return self.model_ledger.text_encoder()

    def _stg_config(self) -> BatchedPerturbationConfig:
        perturbations: list[Perturbation] = [
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=self._stg_blocks)
        ]
        if self._stg_mode == "stg_av":
            perturbations.append(Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=self._stg_blocks))
        return BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=perturbations)])

    def _av_bimodal_config(self) -> BatchedPerturbationConfig:
        return BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=[
            Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
            Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
        ])])

    @torch.inference_mode()
    def __call__(
        self,
        v_context_p: torch.Tensor,
        a_context_p: torch.Tensor,
        v_context_n: torch.Tensor,
        a_context_n: torch.Tensor,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        reference_audio: torch.Tensor | None = None,
        reference_audio_sample_rate: int = 16000,
        condition_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-stage generation with audio identity transfer.

        Args:
            reference_audio:  [C, samples] waveform from reference speaker
            condition_image:  [C, H, W] in [0, 1] -- first frame for face conditioning
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        video_cfg = CFGGuider(video_guidance_scale)
        audio_cfg = CFGGuider(audio_guidance_scale)
        stg_guider = STGGuider(self._stg_scale)
        av_bimodal_guider = CFGGuider(self._av_bimodal_scale if self._av_bimodal_cfg else 0.0)

        stg_pcfg = self._stg_config() if stg_guider.enabled() else None
        av_pcfg = self._av_bimodal_config() if av_bimodal_guider.enabled() else None

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
            dtype=torch.float32, device=self.device
        )

        output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate
        )

        video_state, video_tools, ref_vid_len = self._create_video_state(
            output_shape=output_shape,
            condition_image=condition_image,
            noiser=noiser,
            frame_rate=frame_rate,
        )
        audio_state, audio_tools, ref_aud_len = self._create_audio_state(
            output_shape=output_shape,
            reference_audio=reference_audio,
            reference_audio_sample_rate=reference_audio_sample_rate,
            noiser=noiser,
        )

        total_steps = len(sigmas) - 1

        def denoising_func(video_state, audio_state, sigmas, step_idx):
            sigma = sigmas[step_idx]
            print(f"  Step {step_idx + 1}/{total_steps} (sigma={sigma.item():.4f})", flush=True)

            pv = modality_from_latent_state(video_state, v_context_p, sigma)
            pa = modality_from_latent_state(audio_state, a_context_p, sigma)
            dv_pos, da_pos = self._transformer(video=pv, audio=pa, perturbations=None)

            delta_v = torch.zeros_like(dv_pos)
            delta_a = torch.zeros_like(da_pos) if da_pos is not None else None

            if video_cfg.enabled() or audio_cfg.enabled():
                nv = modality_from_latent_state(video_state, v_context_n, sigma)
                na = modality_from_latent_state(audio_state, a_context_n, sigma)
                dv_neg, da_neg = self._transformer(video=nv, audio=na, perturbations=None)
                delta_v = delta_v + video_cfg.delta(dv_pos, dv_neg)
                if delta_a is not None:
                    delta_a = delta_a + audio_cfg.delta(da_pos, da_neg)

            if self._identity_guidance and self._identity_guidance_scale > 0 and ref_aud_len > 0:
                tgt_aud = LatentState(
                    latent=audio_state.latent[:, ref_aud_len:],
                    denoise_mask=audio_state.denoise_mask[:, ref_aud_len:],
                    positions=audio_state.positions[:, :, ref_aud_len:],
                    clean_latent=audio_state.clean_latent[:, ref_aud_len:],
                )
                nrv = modality_from_latent_state(video_state, v_context_p, sigma)
                nra = modality_from_latent_state(tgt_aud, a_context_p, sigma)
                _, da_noref = self._transformer(video=nrv, audio=nra, perturbations=None)
                if delta_a is not None and da_noref is not None:
                    id_delta = self._identity_guidance_scale * (da_pos[:, ref_aud_len:] - da_noref)
                    full_id = torch.zeros_like(delta_a)
                    full_id[:, ref_aud_len:] = id_delta
                    delta_a = delta_a + full_id

            if stg_guider.enabled() and stg_pcfg is not None:
                pv_s, pa_s = self._transformer(video=pv, audio=pa, perturbations=stg_pcfg)
                delta_v = delta_v + stg_guider.delta(dv_pos, pv_s)
                if delta_a is not None and pa_s is not None:
                    delta_a = delta_a + stg_guider.delta(da_pos, pa_s)

            if av_bimodal_guider.enabled() and av_pcfg is not None:
                pv_b, pa_b = self._transformer(video=pv, audio=pa, perturbations=av_pcfg)
                delta_v = delta_v + av_bimodal_guider.delta(dv_pos, pv_b)
                if delta_a is not None and pa_b is not None:
                    delta_a = delta_a + av_bimodal_guider.delta(da_pos, pa_b)

            out_v = dv_pos + delta_v
            out_a = (da_pos + delta_a) if (da_pos is not None and delta_a is not None) else da_pos
            return out_v, out_a

        video_state, audio_state = euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=denoising_func,
        )

        if ref_vid_len > 0:
            video_state = LatentState(
                latent=video_state.latent[:, ref_vid_len:],
                denoise_mask=video_state.denoise_mask[:, ref_vid_len:],
                positions=video_state.positions[:, :, ref_vid_len:],
                clean_latent=video_state.clean_latent[:, ref_vid_len:],
            )
        if ref_aud_len > 0:
            audio_state = LatentState(
                latent=audio_state.latent[:, ref_aud_len:],
                denoise_mask=audio_state.denoise_mask[:, ref_aud_len:],
                positions=audio_state.positions[:, :, ref_aud_len:],
                clean_latent=audio_state.clean_latent[:, ref_aud_len:],
            )

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        torch.cuda.synchronize()
        self._transformer.to("cpu")
        torch.cuda.synchronize()
        cleanup_memory()

        video_latent = video_state.latent.to(dtype=torch.bfloat16)
        decoded_video = self._video_decoder(video_latent)
        decoded_video = ((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0)
        video_tensor = decoded_video[0].float().cpu()

        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        audio_waveform = vae_decode_audio(audio_state.latent, self._audio_decoder, self._vocoder)
        audio_output = audio_waveform.squeeze(0).float().cpu()

        del video_latent, decoded_video
        gc.collect()
        torch.cuda.empty_cache()

        self._transformer.to(self.device)
        torch.cuda.synchronize()
        cleanup_memory()

        return video_tensor, audio_output

    # ------------------------------------------------------------------
    # State-creation helpers (audio-only IC with negative positions)
    # ------------------------------------------------------------------

    def _create_video_state(
        self,
        output_shape: VideoPixelShape,
        condition_image: torch.Tensor | None,
        noiser: GaussianNoiser,
        frame_rate: float,
    ) -> tuple[LatentState, VideoLatentTools, int]:
        video_tools = VideoLatentTools(
            patchifier=self._video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(output_shape),
            fps=frame_rate,
        )
        target_state = video_tools.create_initial_state(device=self.device, dtype=self.dtype)

        if condition_image is not None:
            target_state = self._apply_image_conditioning(target_state, condition_image, output_shape)

        video_state = noiser(latent_state=target_state, noise_scale=1.0)
        return video_state, video_tools, 0

    def _create_audio_state(
        self,
        output_shape: VideoPixelShape,
        reference_audio: torch.Tensor | None,
        reference_audio_sample_rate: int,
        noiser: GaussianNoiser,
    ) -> tuple[LatentState, AudioLatentTools, int]:
        duration = output_shape.frames / output_shape.fps
        audio_tools = AudioLatentTools(
            patchifier=self._audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=duration),
        )
        target_state = audio_tools.create_initial_state(device=self.device, dtype=self.dtype)
        ref_seq_len = 0

        if reference_audio is not None:
            ref_latent, ref_pos = self._encode_audio(reference_audio, reference_audio_sample_rate)
            ref_seq_len = ref_latent.shape[1]

            hop_length = 160
            downsample = 4
            sr = 16000
            time_per_latent = hop_length * downsample / sr
            aud_dur = ref_pos[:, :, -1, 1].max().item()
            ref_pos = ref_pos - aud_dur - time_per_latent

            ref_mask = torch.zeros(1, ref_seq_len, 1, device=self.device, dtype=torch.float32)
            combined = LatentState(
                latent=torch.cat([ref_latent, target_state.latent], dim=1),
                denoise_mask=torch.cat([ref_mask, target_state.denoise_mask], dim=1),
                positions=torch.cat([ref_pos, target_state.positions], dim=2),
                clean_latent=torch.cat([ref_latent, target_state.clean_latent], dim=1),
            )
            audio_state = noiser(latent_state=combined, noise_scale=1.0)
        else:
            audio_state = noiser(latent_state=target_state, noise_scale=1.0)

        return audio_state, audio_tools, ref_seq_len

    def _apply_image_conditioning(
        self, video_state: LatentState, image: torch.Tensor, output_shape: VideoPixelShape
    ) -> LatentState:
        import torch.nn.functional as F
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=(output_shape.height, output_shape.width), mode="bilinear")
        image = image * 2.0 - 1.0
        image = image.unsqueeze(2).to(device=self.device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded = self._video_encoder(image)
        patchified = self._video_patchifier.patchify(encoded)
        n = patchified.shape[1]

        new_latent = video_state.latent.clone()
        new_latent[:, :n] = patchified.to(new_latent.dtype)
        new_clean = video_state.clean_latent.clone()
        new_clean[:, :n] = patchified.to(new_clean.dtype)
        new_mask = video_state.denoise_mask.clone()
        new_mask[:, :n] = 0.0
        return LatentState(
            latent=new_latent, denoise_mask=new_mask,
            positions=video_state.positions, clean_latent=new_clean,
        )

    def _encode_audio(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(device=self.device, dtype=torch.float32)
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        mel = self._audio_processor.waveform_to_mel(waveform, waveform_sample_rate=sample_rate)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_raw = self._audio_encoder(mel.to(torch.float32))
        latent_raw = latent_raw.to(self.dtype)
        B, C, T, Fq = latent_raw.shape
        latent = self._audio_patchifier.patchify(latent_raw)
        positions = self._audio_patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(batch=B, channels=C, frames=T, mel_bins=Fq),
            device=self.device,
        ).to(self.dtype)
        return latent, positions


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_first_frame(path: str | Path) -> torch.Tensor:
    video, _ = read_video(str(path), max_frames=1)
    return video[0]


def load_first_frame_image(path: str | Path) -> torch.Tensor:
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ID-LoRA one-stage inference: fast audio-video generation with speaker identity transfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--lora-path", required=True, help="Path to ID-LoRA checkpoint (.safetensors)")
    parser.add_argument("--reference-audio", default=None,
                        help="Path to reference audio/video file (for single-sample mode)")
    parser.add_argument("--first-frame", default=None,
                        help="Path to first-frame image (for single-sample mode)")
    parser.add_argument("--prompt", default=None,
                        help="Text prompt for generation (for single-sample mode)")
    parser.add_argument("--prompts-file", default=None,
                        help="JSON prompts file for batch mode")

    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Base LTX-2 model path")
    parser.add_argument("--text-encoder-path", default=DEFAULT_TEXT_ENCODER, help="Gemma text encoder path")

    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable int8 quantization (lower VRAM, slightly slower)")

    parser.add_argument("--video-guidance-scale", type=float, default=DEFAULT_VIDEO_GUIDANCE_SCALE)
    parser.add_argument("--audio-guidance-scale", type=float, default=DEFAULT_AUDIO_GUIDANCE_SCALE)
    parser.add_argument("--identity-guidance-scale", type=float, default=DEFAULT_IDENTITY_GUIDANCE_SCALE)
    parser.add_argument("--stg-scale", type=float, default=1.0, help="STG scale (0 disables)")

    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--auto-resolution", "--no-auto-resolution", dest="auto_resolution",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-detect resolution from first frame aspect ratio (default: True)")

    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda")

    if not args.prompts_file and not (args.reference_audio and args.first_frame and args.prompt):
        print("Error: provide either --prompts-file or all of --reference-audio, --first-frame, --prompt")
        sys.exit(1)

    if args.auto_resolution and args.first_frame:
        from PIL import Image
        _img = Image.open(args.first_frame)
        _src_w, _src_h = _img.size
        args.height, args.width = compute_resolution_match_aspect(_src_h, _src_w)
        print(f"Auto-resolution: {_src_w}x{_src_h} -> {args.width}x{args.height}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ID-LoRA Inference -- One-Stage Pipeline (fast / low-VRAM)")
    print("=" * 80)
    print(f"ID-LoRA:                {args.lora_path}")
    print(f"Resolution:             {args.height}x{args.width}")
    print(f"Steps:                  {args.num_inference_steps}")
    print(f"CFG:                    video={args.video_guidance_scale}, audio={args.audio_guidance_scale}")
    print(f"Identity guidance:      scale={args.identity_guidance_scale}")
    print(f"Output:                 {output_dir}")
    print("=" * 80)

    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.prompts_file:
        with open(args.prompts_file) as f:
            data = json.load(f)
        samples = data if isinstance(data, list) else data.get("variations", data.get("samples", [data]))
    else:
        samples = [{
            "prompt": args.prompt,
            "reference_path": args.reference_audio,
            "first_frame_path": args.first_frame,
        }]

    ic_loras = [LoraPathStrengthAndSDOps(
        path=args.lora_path, strength=1.0, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )]

    print("\nCreating one-stage pipeline...")
    pipeline = IDLoraOneStagePipeline(
        checkpoint_path=args.checkpoint,
        gemma_root=args.text_encoder_path,
        loras=ic_loras,
        device=device,
        quantize=args.quantize,
        stg_scale=args.stg_scale,
        identity_guidance=True,
        identity_guidance_scale=args.identity_guidance_scale,
        av_bimodal_cfg=True,
        av_bimodal_scale=3.0,
    )

    print("\nPre-computing text embeddings...")
    text_encoder = pipeline.text_encoder()
    all_prompts = list({s["prompt"] for s in samples})
    embeddings_cache: dict = {}
    for prompt in tqdm(all_prompts, desc="Encoding prompts"):
        ctx_p, ctx_n = encode_text(text_encoder, prompts=[prompt, args.negative_prompt])
        v_p, a_p = ctx_p
        v_n, a_n = ctx_n
        embeddings_cache[prompt] = (v_p.cpu(), a_p.cpu(), v_n.cpu(), a_n.cpu())
    del text_encoder
    cleanup_memory()

    print("\nLoading models...")
    pipeline.load_models()

    print(f"\nGenerating {len(samples)} sample(s)...")
    for i, sample in enumerate(tqdm(samples, desc="Generating")):
        prompt = sample["prompt"]
        ref_path = sample.get("reference_path") or args.reference_audio
        ff_path = sample.get("first_frame_path") or args.first_frame

        ref_audio = None
        ref_sr = 16000
        if ref_path:
            waveform, sr = torchaudio.load(str(ref_path))
            ref_audio = waveform
            ref_sr = sr

        condition_image = None
        if ff_path:
            ff = Path(ff_path)
            if ff.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
                condition_image = load_first_frame(ff)
            else:
                condition_image = load_first_frame_image(ff)

        v_p, a_p, v_n, a_n = embeddings_cache[prompt]
        v_p, a_p = v_p.to(device), a_p.to(device)
        v_n, a_n = v_n.to(device), a_n.to(device)

        video_out, audio_out = pipeline(
            v_context_p=v_p, a_context_p=a_p,
            v_context_n=v_n, a_context_n=a_n,
            seed=args.seed + i,
            height=args.height, width=args.width,
            num_frames=args.num_frames,
            frame_rate=DEFAULT_FRAME_RATE,
            num_inference_steps=args.num_inference_steps,
            video_guidance_scale=args.video_guidance_scale,
            audio_guidance_scale=args.audio_guidance_scale,
            reference_audio=ref_audio,
            reference_audio_sample_rate=ref_sr,
            condition_image=condition_image,
        )

        out_name = sample.get("output_name", f"output_{i:04d}")
        out_path = output_dir / f"{out_name}.mp4"
        audio_sr = pipeline._vocoder.output_sample_rate if audio_out is not None else None
        save_video(
            video_tensor=video_out, output_path=out_path,
            fps=DEFAULT_FRAME_RATE, audio=audio_out, audio_sample_rate=audio_sr,
        )
        print(f"  Saved: {out_path}")
        cleanup_memory()

    print(f"\nDone. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
