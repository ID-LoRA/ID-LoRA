#!/usr/bin/env python3
"""
ID-LoRA two-stage inference pipeline.

Stage 1: Generate audio-video at target resolution with ID-LoRA + full guidance suite.
Stage 2: 2x spatial upsample + refinement with distilled LoRA (no ID-LoRA).
         Audio from stage 1 is frozen; first-frame conditioning re-encoded at 2x.

Usage
-----
# Single-sample inference
python scripts/inference_two_stage.py \
    --lora-path models/id-lora-celebvhq.safetensors \
    --reference-audio reference.wav \
    --first-frame first_frame.png \
    --prompt "A person speaks warmly in a sunlit park..." \
    --output-dir outputs/

# Batch inference from prompts file
python scripts/inference_two_stage.py \
    --lora-path models/id-lora-celebvhq.safetensors \
    --prompts-file prompts.json \
    --output-dir outputs/
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import replace as dc_replace
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
from ltx_core.components.patchifiers import (
    AudioPatchifier,
    SpatioTemporalScaleFactors,
    VideoLatentPatchifier,
    get_pixel_coords,
)
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.guidance import BatchedPerturbationConfig, Perturbation, PerturbationConfig, PerturbationType
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import (
    AudioProcessor,
    decode_audio as vae_decode_audio,
)
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape

from ltx_pipelines.utils import ModelLedger, cleanup_memory, euler_denoising_loop, encode_prompts
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import (
    modality_from_latent_state,
    noise_audio_state,
    noise_video_state,
)
from ltx_pipelines.utils.types import PipelineComponents

from ltx_trainer.video_utils import read_video, save_video


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = "models/ltx-2.3-22b-dev.safetensors"
DEFAULT_TEXT_ENCODER = "models/gemma-3-12b-it-qat-q4_0-unquantized"
DEFAULT_UPSAMPLER = "models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DEFAULT_DISTILLED_LORA = "models/ltx-2.3-22b-distilled-lora-384.safetensors"

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
# Two-stage ID-LoRA pipeline
# ---------------------------------------------------------------------------

class IDLoraTwoStagesPipeline:
    """
    Two-stage pipeline for audio identity transfer with ID-LoRA.

    Stage 1: Generate at height x width with ID-LoRA + full guidance.
    Stage 2: 2x spatial upsample, refine with distilled LoRA only.
    Audio from stage 1 is frozen in stage 2.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        upsampler_path: str,
        distilled_lora_path: str | None,
        ic_loras: list[LoraPathStrengthAndSDOps],
        device: torch.device,
        quantize: bool = False,
        fp8: bool = False,
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
        self._ic_loras = ic_loras
        self._quantize = quantize
        self._fp8 = fp8

        self._stg_scale = stg_scale
        self._stg_blocks = stg_blocks if stg_blocks is not None else [29]
        self._stg_mode = stg_mode
        self._identity_guidance = identity_guidance
        self._identity_guidance_scale = identity_guidance_scale
        self._av_bimodal_cfg = av_bimodal_cfg
        self._av_bimodal_scale = av_bimodal_scale

        quantization = QuantizationPolicy.fp8_cast() if fp8 else None
        self._stage_1_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=upsampler_path,
            loras=tuple(ic_loras),
            quantization=quantization,
        )

        distilled_loras: tuple[LoraPathStrengthAndSDOps, ...] = ()
        if distilled_lora_path and Path(distilled_lora_path).exists():
            distilled_loras = (LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=0.8,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ),)
            print(f"Stage-2 distilled LoRA: {distilled_lora_path}")
        else:
            print(f"WARNING: distilled LoRA not found at {distilled_lora_path!r}, stage 2 runs without it")

        self._stage_2_ledger = self._stage_1_ledger.with_loras(distilled_loras)

        self._pipeline_components = PipelineComponents(dtype=self.dtype, device=device)
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)

    def load_stage_1_models(self):
        print("Loading video encoder...")
        self._video_encoder = self._stage_1_ledger.video_encoder()

        print("Loading stage-1 transformer (ID-LoRA)...")
        if self._quantize:
            from ltx_trainer.quantization import quantize_model
            x0_model = self._stage_1_ledger.transformer()
            quantize_model(x0_model.velocity_model, "int8-quanto")
            cleanup_memory()
            self._stage_1_transformer = x0_model
        else:
            self._stage_1_transformer = self._stage_1_ledger.transformer()

        print("Loading audio encoder...")
        self._audio_encoder = self._stage_1_ledger.audio_encoder()

        print("Creating audio processor...")
        self._audio_processor = AudioProcessor(
            target_sample_rate=16000, mel_bins=64, mel_hop_length=160, n_fft=1024,
        ).to(self.device)

    def _free_stage_1_models(self):
        for attr in ("_stage_1_transformer", "_audio_encoder", "_audio_processor"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.cpu()
                except Exception:
                    pass
                del obj
                setattr(self, attr, None)
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_memory()
        print(f"  Stage-1 models freed.  GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def load_stage_2_models(self):
        print("Loading stage-2 transformer (distilled LoRA)...")
        if self._quantize:
            from ltx_trainer.quantization import quantize_model
            x0_model = self._stage_2_ledger.transformer()
            quantize_model(x0_model.velocity_model, "int8-quanto")
            cleanup_memory()
            self._stage_2_transformer = x0_model
        else:
            self._stage_2_transformer = self._stage_2_ledger.transformer()

        print("Loading video/audio decoders + vocoder...")
        self._video_decoder = self._stage_2_ledger.video_decoder()
        self._audio_decoder = self._stage_2_ledger.audio_decoder()
        self._vocoder = self._stage_2_ledger.vocoder()

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
        Two-stage generation with audio identity transfer.

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

        if getattr(self, "_video_encoder", None) is None:
            self._video_encoder = self._stage_1_ledger.video_encoder()
        if getattr(self, "_audio_encoder", None) is None:
            self._audio_encoder = self._stage_1_ledger.audio_encoder()
        if getattr(self, "_audio_processor", None) is None:
            self._audio_processor = AudioProcessor(
                target_sample_rate=16000, mel_bins=64, mel_hop_length=160, n_fft=1024,
            ).to(self.device)
        if getattr(self, "_stage_1_transformer", None) is None:
            print("Reloading stage-1 transformer (ID-LoRA)...")
            if self._quantize:
                from ltx_trainer.quantization import quantize_model
                x0_model = self._stage_1_ledger.transformer()
                quantize_model(x0_model.velocity_model, "int8-quanto")
                cleanup_memory()
                self._stage_1_transformer = x0_model
            else:
                self._stage_1_transformer = self._stage_1_ledger.transformer()

        # Stage 1: generate at target resolution (audio-only IC, negative positions)
        s1_h, s1_w = height, width
        s1_shape = VideoPixelShape(batch=1, frames=num_frames, width=s1_w, height=s1_h, fps=frame_rate)
        sigmas_s1 = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        video_state, _, ref_vid_len = self._create_video_state(
            output_shape=s1_shape,
            condition_image=condition_image,
            noiser=noiser,
            frame_rate=frame_rate,
        )
        audio_state, _, ref_aud_len = self._create_audio_state(
            output_shape=s1_shape,
            reference_audio=reference_audio,
            reference_audio_sample_rate=reference_audio_sample_rate,
            noiser=noiser,
        )

        total_s1 = len(sigmas_s1) - 1

        def stage_1_denoise(vs: LatentState, as_: LatentState, sigmas: torch.Tensor, idx: int):
            sigma = sigmas[idx]
            print(f"  S1 {idx+1}/{total_s1} sigma={sigma.item():.4f}", flush=True)

            pv = modality_from_latent_state(vs, v_context_p, sigma)
            pa = modality_from_latent_state(as_, a_context_p, sigma)
            dv_pos, da_pos = self._stage_1_transformer(video=pv, audio=pa, perturbations=None)

            dv_delta = torch.zeros_like(dv_pos)
            da_delta = torch.zeros_like(da_pos) if da_pos is not None else None

            if video_cfg.enabled() or audio_cfg.enabled():
                nv = modality_from_latent_state(vs, v_context_n, sigma)
                na = modality_from_latent_state(as_, a_context_n, sigma)
                dv_neg, da_neg = self._stage_1_transformer(video=nv, audio=na, perturbations=None)
                dv_delta = dv_delta + video_cfg.delta(dv_pos, dv_neg)
                if da_delta is not None:
                    da_delta = da_delta + audio_cfg.delta(da_pos, da_neg)

            if self._identity_guidance and self._identity_guidance_scale > 0 and ref_aud_len > 0:
                tgt_aud = LatentState(
                    latent=as_.latent[:, ref_aud_len:],
                    denoise_mask=as_.denoise_mask[:, ref_aud_len:],
                    positions=as_.positions[:, :, ref_aud_len:],
                    clean_latent=as_.clean_latent[:, ref_aud_len:],
                )
                nrv = modality_from_latent_state(vs, v_context_p, sigma)
                nra = modality_from_latent_state(tgt_aud, a_context_p, sigma)
                _, da_noref = self._stage_1_transformer(video=nrv, audio=nra, perturbations=None)
                if da_delta is not None and da_noref is not None:
                    id_delta = self._identity_guidance_scale * (da_pos[:, ref_aud_len:] - da_noref)
                    full_id_delta = torch.zeros_like(da_delta)
                    full_id_delta[:, ref_aud_len:] = id_delta
                    da_delta = da_delta + full_id_delta

            if stg_guider.enabled() and stg_pcfg is not None:
                pv_s, pa_s = self._stage_1_transformer(video=pv, audio=pa, perturbations=stg_pcfg)
                dv_delta = dv_delta + stg_guider.delta(dv_pos, pv_s)
                if da_delta is not None and pa_s is not None:
                    da_delta = da_delta + stg_guider.delta(da_pos, pa_s)

            if av_bimodal_guider.enabled() and av_pcfg is not None:
                pv_b, pa_b = self._stage_1_transformer(video=pv, audio=pa, perturbations=av_pcfg)
                dv_delta = dv_delta + av_bimodal_guider.delta(dv_pos, pv_b)
                if da_delta is not None and pa_b is not None:
                    da_delta = da_delta + av_bimodal_guider.delta(da_pos, pa_b)

            dv_out = dv_pos + dv_delta
            da_out = (da_pos + da_delta) if (da_pos is not None and da_delta is not None) else da_pos
            return dv_out, da_out

        print(f"Stage 1: generating at {s1_h}x{s1_w} ({num_inference_steps} steps, ID-LoRA)...")
        video_state, audio_state = euler_denoising_loop(
            sigmas=sigmas_s1,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=stage_1_denoise,
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

        s1_vid_tools = VideoLatentTools(
            patchifier=self._video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(s1_shape),
            fps=frame_rate,
        )
        s1_aud_tools = AudioLatentTools(
            patchifier=self._audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate),
        )
        video_state = s1_vid_tools.clear_conditioning(video_state)
        video_state = s1_vid_tools.unpatchify(video_state)
        audio_state = s1_aud_tools.clear_conditioning(audio_state)
        audio_state = s1_aud_tools.unpatchify(audio_state)

        s1_video_latent = video_state.latent
        s1_audio_latent = audio_state.latent

        # Transition: free stage-1, upsample, prepare stage-2 conditioning
        self._free_stage_1_models()

        s2_h, s2_w = height * 2, width * 2
        s2_shape = VideoPixelShape(batch=1, frames=num_frames, width=s2_w, height=s2_h, fps=frame_rate)

        print("Upsampling video latent 2x...")
        _upsampler = self._stage_2_ledger.spatial_upsampler()
        upscaled_latent = upsample_video(
            latent=s1_video_latent[:1],
            video_encoder=self._video_encoder,
            upsampler=_upsampler,
        )
        print(f"  {s1_video_latent.shape} -> {upscaled_latent.shape}")

        s2_conditionings = []
        if condition_image is not None:
            s2_image = self._center_crop_resize(condition_image, s2_h, s2_w)
            s2_image = s2_image * 2.0 - 1.0
            s2_image = s2_image.unsqueeze(2).to(device=self.device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                s2_encoded = self._video_encoder(s2_image)
            s2_conditionings.append(
                VideoConditionByLatentIndex(latent=s2_encoded, strength=1.0, latent_idx=0)
            )

        self._video_encoder.cpu()
        _upsampler.cpu()
        del self._video_encoder, _upsampler
        self._video_encoder = None
        cleanup_memory()

        # Stage 2: refine at 2x resolution
        sigmas_s2 = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=self.device)

        print("Loading stage-2 transformer (distilled LoRA)...")
        self.load_stage_2_models()

        def stage_2_denoise(vs: LatentState, as_: LatentState, sigmas: torch.Tensor, idx: int):
            sigma = sigmas[idx]
            v_mod = modality_from_latent_state(vs, v_context_p, sigma)
            a_mod = modality_from_latent_state(as_, a_context_p, sigma)
            dv, da = self._stage_2_transformer(video=v_mod, audio=a_mod, perturbations=None)
            return dv, da

        print(f"Stage 2: refining at {s2_h}x{s2_w} ({len(sigmas_s2)-1} steps)...")
        video_state, video_tools = noise_video_state(
            output_shape=s2_shape, noiser=noiser, conditionings=s2_conditionings,
            components=self._pipeline_components, dtype=self.dtype, device=self.device,
            noise_scale=sigmas_s2[0].item(), initial_latent=upscaled_latent,
        )
        audio_state, audio_tools = noise_audio_state(
            output_shape=s2_shape, noiser=noiser, conditionings=[],
            components=self._pipeline_components, dtype=self.dtype, device=self.device,
            noise_scale=0.0, initial_latent=s1_audio_latent,
        )
        audio_state = dc_replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))
        video_state, audio_state = euler_denoising_loop(
            sigmas=sigmas_s2, video_state=video_state, audio_state=audio_state,
            stepper=stepper, denoise_fn=stage_2_denoise,
        )
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        torch.cuda.synchronize()
        del self._stage_2_transformer
        self._stage_2_transformer = None
        cleanup_memory()

        tiling_config = TilingConfig.default()
        generator_decode = torch.Generator(device=self.device).manual_seed(seed)
        video_chunks = list(vae_decode_video(
            video_state.latent, self._stage_2_ledger.video_decoder(), tiling_config, generator_decode,
        ))
        # Each chunk is [F, H, W, C] uint8 [0,255]; concat on frames dim
        decoded_frames = torch.cat(video_chunks, dim=0)
        video_tensor = decoded_frames.permute(0, 3, 1, 2).float().cpu() / 255.0

        audio_decoder = self._stage_2_ledger.audio_decoder()
        vocoder = self._stage_2_ledger.vocoder()
        decoded_audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)
        audio_output = decoded_audio.waveform.cpu()
        self._vocoder_sr = decoded_audio.sampling_rate

        del audio_decoder, vocoder
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

            # Negative positions: shift reference audio to negative time range
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

    @staticmethod
    def _center_crop_resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
        import torch.nn.functional as F
        src_h, src_w = image.shape[1], image.shape[2]
        img = image.unsqueeze(0)
        if src_h != height or src_w != width:
            ar, tar = src_w / src_h, width / height
            rh, rw = (height, int(height * ar)) if ar > tar else (int(width / ar), width)
            img = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
            h0, w0 = (rh - height) // 2, (rw - width) // 2
            img = img[:, :, h0:h0 + height, w0:w0 + width]
        return img

    def _apply_image_conditioning(
        self, video_state: LatentState, image: torch.Tensor, output_shape: VideoPixelShape
    ) -> LatentState:
        image = self._center_crop_resize(image, output_shape.height, output_shape.width)
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
        mel = self._audio_processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=sample_rate))
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
    """Load first frame from a video file as [C, H, W] in [0, 1]."""
    video, _ = read_video(str(path), max_frames=1)
    return video[0]


def load_first_frame_image(path: str | Path) -> torch.Tensor:
    """Load an image file as [C, H, W] in [0, 1]."""
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
        description="ID-LoRA two-stage inference: generate audio-video with speaker identity transfer",
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
    parser.add_argument("--upsampler-path", default=DEFAULT_UPSAMPLER, help="Spatial upscaler path")
    parser.add_argument("--distilled-lora-path", default=DEFAULT_DISTILLED_LORA, help="Distilled LoRA path")

    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=DEFAULT_VIDEO_HEIGHT,
                        help="Stage 1 height (stage 2 = height*2)")
    parser.add_argument("--width", type=int, default=DEFAULT_VIDEO_WIDTH,
                        help="Stage 1 width (stage 2 = width*2)")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable int8 quantization (lower VRAM, slightly slower)")
    parser.add_argument("--fp8", action="store_true", default=False,
                        help="Enable fp8 quantization via QuantizationPolicy (handled by ModelLedger)")

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
        print(f"Auto-resolution: {_src_w}x{_src_h} -> {args.width}x{args.height} (stage 1)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    s2_steps = len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1
    print("=" * 80)
    print("ID-LoRA Inference -- Two-Stage Pipeline")
    print("=" * 80)
    print(f"ID-LoRA:                {args.lora_path}")
    print(f"Stage 1 resolution:     {args.height}x{args.width}  (stage 2: {args.height*2}x{args.width*2})")
    print(f"Steps:                  stage 1 = {args.num_inference_steps},  stage 2 = {s2_steps}")
    print(f"CFG:                    video={args.video_guidance_scale}, audio={args.audio_guidance_scale}")
    print(f"Identity guidance:      scale={args.identity_guidance_scale}")
    print(f"Output:                 {output_dir}")
    print("=" * 80)

    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Build sample list
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

    print("\nCreating two-stage pipeline...")
    pipeline = IDLoraTwoStagesPipeline(
        checkpoint_path=args.checkpoint,
        gemma_root=args.text_encoder_path,
        upsampler_path=args.upsampler_path,
        distilled_lora_path=args.distilled_lora_path,
        ic_loras=ic_loras,
        device=device,
        quantize=args.quantize,
        fp8=args.fp8,
        stg_scale=args.stg_scale,
        identity_guidance=True,
        identity_guidance_scale=args.identity_guidance_scale,
        av_bimodal_cfg=True,
        av_bimodal_scale=3.0,
    )

    print("\nPre-computing text embeddings...")
    all_prompts = list({s["prompt"] for s in samples})
    embeddings_cache: dict = {}
    for prompt in tqdm(all_prompts, desc="Encoding prompts"):
        results = encode_prompts(
            prompts=[prompt, args.negative_prompt],
            model_ledger=pipeline._stage_1_ledger,
        )
        ctx_p, ctx_n = results
        embeddings_cache[prompt] = (
            ctx_p.video_encoding.cpu(), ctx_p.audio_encoding.cpu(),
            ctx_n.video_encoding.cpu(), ctx_n.audio_encoding.cpu(),
        )
    cleanup_memory()

    print("\nLoading stage-1 models...")
    pipeline.load_stage_1_models()

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
        audio_sr = getattr(pipeline, "_vocoder_sr", 24000)
        save_video(
            video_tensor=video_out, output_path=out_path,
            fps=DEFAULT_FRAME_RATE, audio=audio_out, audio_sample_rate=audio_sr,
        )
        print(f"  Saved: {out_path}")
        cleanup_memory()

    print(f"\nDone. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
