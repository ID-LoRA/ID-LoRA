"""Audio reference-only in-context (IC-LoRA) training strategy.

This strategy implements training with reference audio conditioning only (no reference video):
- Reference AUDIO latents (clean) are concatenated with target audio latents (noised)
- Video uses standard T2V with first-frame conditioning (no reference video concatenation)
- Loss is computed on both video (non-first-frame tokens) and audio (target tokens)
- The model learns to transfer speaker identity from reference audio while generating
  video that matches the target first frame

This is useful when you want speaker voice cloning but face identity comes from a target
first frame rather than from a reference video.
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class AudioRefOnlyICConfig(TrainingStrategyConfigBase):
    """Configuration for audio reference-only IC-LoRA training strategy.

    This strategy uses only reference audio for speaker identity transfer,
    while video identity comes from first-frame conditioning.
    """

    name: Literal["audio_ref_only_ic"] = "audio_ref_only_ic"

    first_frame_conditioning_p: float = Field(
        default=0.9,
        description="Probability of conditioning on the first frame during training (for video)",
        ge=0.0,
        le=1.0,
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for target audio latents",
    )

    reference_audio_latents_dir: str = Field(
        default="reference_audio_latents",
        description="Directory name for latents of reference audio",
    )

    mask_cross_attention_to_reference: bool = Field(
        default=False,
        description=(
            "Whether to mask cross-modal attention to reference audio tokens. "
            "When True, video only attends to target audio (not ref audio). "
            "This ensures audio identity is learned from same-modality reference "
            "(self-attention) while cross-modal attention only syncs target content."
        ),
    )

    mask_reference_from_text_attention: bool = Field(
        default=False,
        description=(
            "Whether to block reference audio tokens from attending to text. "
            "When True, reference audio tokens cannot attend to text embeddings "
            "(which describe the TARGET speech content). This prevents the reference "
            "audio from being confused by text describing different speech. "
            "Target audio tokens still attend to text normally for content guidance. "
            "NOTE: This masking is only applied during training. Inference scripts "
            "do not currently implement this masking, which may cause train/inference "
            "mismatch. Since reference tokens have timestep=0, the impact is likely minimal."
        ),
    )

    use_negative_ref_positions: bool = Field(
        default=False,
        description=(
            "Use negative temporal positions for reference audio. "
            "When True, reference audio positions are [-seq_len, ..., -1] (in time units), "
            "placing the reference 'before' the target in temporal space. "
            "This creates clear separation between reference (identity) and target (generation). "
            "Target audio always starts at position 0."
        ),
    )


class AudioRefOnlyICStrategy(TrainingStrategy):
    """Audio reference-only in-context training strategy for IC-LoRA.

    This strategy implements training with reference audio conditioning only:
    - Reference audio latents (clean) are concatenated with target audio latents (noised)
    - Video uses standard T2V with first-frame conditioning (no IC conditioning)
    - Loss is computed on both modalities:
      - Video: loss on non-first-frame tokens
      - Audio: loss on target tokens only
    - Cross-modal attention can optionally be masked to block video attending to ref audio

    Use case: Speaker voice cloning where face identity comes from a provided first frame
    rather than from a reference video. This is simpler than full AV-IC-LoRA when you
    have a target first frame available.
    """

    config: AudioRefOnlyICConfig

    def __init__(self, config: AudioRefOnlyICConfig):
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        return True

    def get_data_sources(self) -> dict[str, str]:
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.config.audio_latents_dir: "audio_latents",
            self.config.reference_audio_latents_dir: "ref_audio_latents",
        }

    def _create_cross_attention_mask(
        self,
        video_seq_len: int,
        ref_audio_seq_len: int,
        target_audio_seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Create mask for cross-modal attention (a2v only).

        Blocks video from attending to reference audio tokens.
        """
        audio_seq_len = ref_audio_seq_len + target_audio_seq_len
        a2v_mask = torch.zeros(batch_size, video_seq_len, audio_seq_len, device=device, dtype=dtype)
        a2v_mask[:, :, :ref_audio_seq_len] = torch.finfo(dtype).min
        return a2v_mask

    def _create_text_attention_mask_for_audio(
        self,
        ref_audio_seq_len: int,
        target_audio_seq_len: int,
        text_seq_len: int,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Create a per-query text attention mask for audio tokens.

        Blocks reference audio tokens from attending to text embeddings
        while allowing target audio tokens to attend to text normally.
        """
        audio_seq_len = ref_audio_seq_len + target_audio_seq_len
        text_mask_float = (1 - prompt_attention_mask.float()) * torch.finfo(dtype).min
        text_mask_expanded = text_mask_float.unsqueeze(1).expand(batch_size, audio_seq_len, text_seq_len)
        text_attention_mask = text_mask_expanded.clone()
        text_attention_mask[:, :ref_audio_seq_len, :] = torch.finfo(dtype).min
        return text_attention_mask

    def _get_negative_audio_positions(
        self,
        num_time_steps: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Generate negative audio position embeddings for reference audio.

        Shifts the entire position block into negative time, placing
        reference 'before' the target in temporal space.
        """
        positions = self._get_audio_positions(
            num_time_steps=num_time_steps,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        time_per_latent = (
            self._audio_patchifier.hop_length
            * self._audio_patchifier.audio_latent_downsample_factor
            / self._audio_patchifier.sample_rate
        )

        audio_duration = positions[:, :, -1, 1].max().item()
        positions = positions - audio_duration - time_per_latent

        return positions

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for Audio ref-only IC-LoRA training.

        Video: Standard T2V preparation with first-frame conditioning (no IC)
        Audio: IC preparation with reference + target concatenation
        Loss: Computed on both modalities
        """
        # === Video Latents (Standard T2V with first-frame conditioning) ===
        latents = batch["latents"]
        video_latents = latents["latents"]

        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        video_latents = self._video_patchifier.patchify(video_latents)

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        # === Audio Latents (IC conditioning) ===
        audio_data = batch["audio_latents"]
        target_audio_latents = audio_data["latents"]
        ref_audio_data = batch["ref_audio_latents"]
        ref_audio_latents = ref_audio_data["latents"]

        target_audio_latents = self._audio_patchifier.patchify(target_audio_latents)
        ref_audio_latents = self._audio_patchifier.patchify(ref_audio_latents)

        ref_audio_seq_len = ref_audio_latents.shape[1]
        target_audio_seq_len = target_audio_latents.shape[1]

        # === Video Conditioning and Noise ===
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)

        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        video_conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(video_conditioning_mask_expanded, video_latents, noisy_video)

        video_targets = video_noise - video_latents

        video_timesteps = self._create_per_token_timesteps(video_conditioning_mask, sigmas.squeeze())

        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        video_loss_mask = ~video_conditioning_mask

        # === Audio Conditioning and Noise (IC conditioning) ===
        ref_audio_conditioning_mask = torch.ones(
            batch_size, ref_audio_seq_len, dtype=torch.bool, device=device
        )
        target_audio_conditioning_mask = torch.zeros(
            batch_size, target_audio_seq_len, dtype=torch.bool, device=device
        )
        audio_conditioning_mask = torch.cat(
            [ref_audio_conditioning_mask, target_audio_conditioning_mask], dim=1
        )

        audio_noise = torch.randn_like(target_audio_latents)
        noisy_target_audio = (
            (1 - sigmas_expanded) * target_audio_latents + sigmas_expanded * audio_noise
        )

        audio_targets = audio_noise - target_audio_latents

        combined_audio_latents = torch.cat([ref_audio_latents, noisy_target_audio], dim=1)

        audio_timesteps = self._create_per_token_timesteps(
            audio_conditioning_mask, sigmas.squeeze()
        )

        if self.config.use_negative_ref_positions:
            ref_audio_positions = self._get_negative_audio_positions(
                num_time_steps=ref_audio_seq_len,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        else:
            ref_audio_positions = self._get_audio_positions(
                num_time_steps=ref_audio_seq_len,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        target_audio_positions = self._get_audio_positions(
            num_time_steps=target_audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        audio_positions = torch.cat([ref_audio_positions, target_audio_positions], dim=2)

        # === Cross-Attention Masks ===
        a2v_cross_attention_mask = None
        if self.config.mask_cross_attention_to_reference:
            a2v_cross_attention_mask = self._create_cross_attention_mask(
                video_seq_len=video_seq_len,
                ref_audio_seq_len=ref_audio_seq_len,
                target_audio_seq_len=target_audio_seq_len,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        audio_context_attn_mask = None
        if self.config.mask_reference_from_text_attention:
            text_seq_len = audio_prompt_embeds.shape[1]
            audio_context_attn_mask = self._create_text_attention_mask_for_audio(
                ref_audio_seq_len=ref_audio_seq_len,
                target_audio_seq_len=target_audio_seq_len,
                text_seq_len=text_seq_len,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        video_modality = Modality(
            enabled=True,
            latent=noisy_video,
            sigma=sigmas,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
            a2v_cross_attention_mask=a2v_cross_attention_mask,
        )

        audio_modality = Modality(
            enabled=True,
            latent=combined_audio_latents,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
            audio_context_attn_mask=audio_context_attn_mask,
        )

        ref_audio_loss_mask = torch.zeros(
            batch_size, ref_audio_seq_len, dtype=torch.bool, device=device
        )
        target_audio_loss_mask = torch.ones(
            batch_size, target_audio_seq_len, dtype=torch.bool, device=device
        )
        audio_loss_mask = torch.cat([ref_audio_loss_mask, target_audio_loss_mask], dim=1)

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            ref_seq_len=ref_audio_seq_len,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute loss on both video and audio.

        Video: Loss on non-first-frame tokens (masked by video_loss_mask)
        Audio: Loss on target tokens only (after reference)
        """
        video_loss = (video_pred - inputs.video_targets).pow(2)
        video_loss_mask_float = inputs.video_loss_mask.unsqueeze(-1).float()
        video_loss = video_loss.mul(video_loss_mask_float).div(video_loss_mask_float.mean())
        video_loss = video_loss.mean()

        if audio_pred is None or inputs.audio_targets is None:
            return video_loss

        ref_audio_seq_len = inputs.ref_seq_len
        target_audio_pred = audio_pred[:, ref_audio_seq_len:, :]
        audio_loss = (target_audio_pred - inputs.audio_targets).pow(2).mean()

        return video_loss + audio_loss
