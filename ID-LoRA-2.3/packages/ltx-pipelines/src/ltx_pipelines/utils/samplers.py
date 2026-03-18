import logging
from dataclasses import replace
from functools import partial
from typing import Callable

import torch
from tqdm import tqdm

from ltx_core.components.diffusion_steps import Res2sDiffusionStep
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.utils import to_denoised, to_velocity
from ltx_pipelines.utils.helpers import post_process_latent, timesteps_from_mask
from ltx_pipelines.utils.res2s import get_res2s_coefficients
from ltx_pipelines.utils.types import DenoisingFunc, LatentState

logger = logging.getLogger(__name__)


def euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
) -> tuple[LatentState, LatentState]:
    """
    Perform the joint audio-video denoising loop over a diffusion schedule.
    This function iterates over all but the final value in ``sigmas`` and, at
    each diffusion step, calls ``denoise_fn`` to obtain denoised video and
    audio latents. The denoised latents are post-processed with their
    respective denoise masks and clean latents, then passed to ``stepper`` to
    advance the noisy latents one step along the diffusion schedule.
    ### Parameters
    sigmas:
        A 1D tensor of noise levels (diffusion sigmas) defining the sampling
        schedule. All steps except the last element are iterated over.
    video_state:
        The current video :class:`LatentState`, containing the noisy latent,
        its clean reference latent, and the denoising mask.
    audio_state:
        The current audio :class:`LatentState`, analogous to ``video_state``
        but for the audio modality.
    stepper:
        An implementation of :class:`DiffusionStepProtocol` that updates a
        latent given the current latent, its denoised estimate, the full
        ``sigmas`` schedule, and the current step index.
    denoise_fn:
        A callable implementing :class:`DenoisingFunc`. It is invoked as
        ``denoise_fn(video_state, audio_state, sigmas, step_index)`` and must
        return a tuple ``(denoised_video, denoised_audio)``, where each element
        is a tensor with the same shape as the corresponding latent.
    ### Returns
    tuple[LatentState, LatentState]
        A pair ``(video_state, audio_state)`` containing the final video and
        audio latent states after completing the denoising loop.
    """
    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)

        denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        video_state = replace(video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx))
        audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx))

    return (video_state, audio_state)


def gradient_estimating_euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    ge_gamma: float = 2.0,
) -> tuple[LatentState, LatentState]:
    """
    Perform the joint audio-video denoising loop using gradient-estimation sampling.
    This function is similar to :func:`euler_denoising_loop`, but applies
    gradient estimation to improve the denoised estimates by tracking velocity
    changes across steps. See the referenced function for detailed parameter
    documentation.
    ### Parameters
    ge_gamma:
        Gradient estimation coefficient controlling the velocity correction term.
        Default is 2.0. Paper: https://openreview.net/pdf?id=o2ND9v0CeK
    sigmas, video_state, audio_state, stepper, denoise_fn:
        See :func:`euler_denoising_loop` for parameter descriptions.
    ### Returns
    tuple[LatentState, LatentState]
        See :func:`euler_denoising_loop` for return value description.
    """

    previous_audio_velocity = None
    previous_video_velocity = None

    def update_velocity_and_sample(
        noisy_sample: torch.Tensor, denoised_sample: torch.Tensor, sigma: float, previous_velocity: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_velocity = to_velocity(noisy_sample, sigma, denoised_sample)
        if previous_velocity is not None:
            delta_v = current_velocity - previous_velocity
            total_velocity = ge_gamma * delta_v + previous_velocity
            denoised_sample = to_denoised(noisy_sample, total_velocity, sigma)
        return current_velocity, denoised_sample

    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)

        denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        if sigmas[step_idx + 1] == 0:
            return replace(video_state, latent=denoised_video), replace(audio_state, latent=denoised_audio)

        previous_video_velocity, denoised_video = update_velocity_and_sample(
            video_state.latent, denoised_video, sigmas[step_idx], previous_video_velocity
        )
        previous_audio_velocity, denoised_audio = update_velocity_and_sample(
            audio_state.latent, denoised_audio, sigmas[step_idx], previous_audio_velocity
        )

        video_state = replace(video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx))
        audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx))

    return (video_state, audio_state)


def _channelwise_normalize(x: torch.Tensor) -> torch.Tensor:
    return x.sub_(x.mean(dim=(-2, -1), keepdim=True)).div_(x.std(dim=(-2, -1), keepdim=True))


def _get_new_noise(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=generator, dtype=torch.float64, device=generator.device)
    noise = (noise - noise.mean()) / noise.std()
    return _channelwise_normalize(noise)


def _inject_sde_noise(
    state: LatentState,
    sample: torch.Tensor,
    denoised_sample: torch.Tensor,
    step_noise_generator: torch.Generator,
    new_noise_fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor],
    stepper: DiffusionStepProtocol,
    sigmas: torch.Tensor,
    step_idx: int,
    legacy_mode: bool = False,
) -> torch.Tensor:
    sigmas_copy = sigmas.clone()
    new_noise = new_noise_fn(state.latent, step_noise_generator)
    if not legacy_mode:
        timesteps = timesteps_from_mask(state.denoise_mask.double(), sigmas_copy[step_idx].double())
        next_timesteps = timesteps_from_mask(state.denoise_mask.double(), sigmas_copy[step_idx + 1].double())
        sigmas = torch.stack([timesteps, next_timesteps])
        step_idx = 0
    x_next = stepper.step(
        sample=sample,
        denoised_sample=denoised_sample,
        sigmas=sigmas,
        step_index=step_idx,
        noise=new_noise,
    )

    if legacy_mode:
        x_next = post_process_latent(x_next, state.denoise_mask, state.clean_latent)

    return x_next


def res2s_audio_video_denoising_loop(  # noqa: PLR0913,PLR0915
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    noise_seed: int = -1,
    noise_seed_substep: int | None = None,
    bongmath: bool = True,
    bongmath_max_iter: int = 100,
    new_noise_fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor] = _get_new_noise,
    model_dtype: torch.dtype = torch.bfloat16,
    legacy_mode: bool = True,
) -> tuple[LatentState, LatentState]:
    """
    Joint audio-video denoising loop using the res_2s second-order sampler.
    Iterates over the diffusion schedule with a two-stage Runge-Kutta step:
    evaluates the denoiser at the current point and at a midpoint (with SDE
    noise), then combines both with RK coefficients. Supports anchor-point
    refinement (bong iteration) and optional SDE noise injection. Requires
    :class:`Res2sDiffusionStep` as ``stepper``.
    ### Parameters
    sigmas:
        A 1D tensor of noise levels defining the sampling schedule.
    video_state:
        Current video :class:`LatentState` (noisy latent, clean reference, mask).
    audio_state:
        Current audio :class:`LatentState`, same structure as ``video_state``.
    stepper:
        Must be an instance of :class:`Res2sDiffusionStep`; performs SDE step
        with noise injection.
    denoise_fn:
        Callable ``(video_state, audio_state, sigmas, step_index)`` returning
        ``(denoised_video, denoised_audio)``.
    noise_seed:
        Seed for step-level SDE noise; substep seed defaults to ``noise_seed + 10000``.
    noise_seed_substep:
        Optional seed for substep SDE noise; if None, derived from ``noise_seed``.
    bongmath:
        Whether to run iterative anchor refinement (bong iteration) when step size is small.
    bongmath_max_iter:
        Max iterations for bong refinement when enabled.
    new_noise_fn:
        Callable ``(latent, generator) -> noise`` for SDE injection; default
        uses normalized channel-wise Gaussian noise.
    model_dtype:
        Dtype for latent state updates (e.g. bfloat16).
    ### Returns
    tuple[LatentState, LatentState]
        Final ``(video_state, audio_state)`` after the denoising loop.
    """
    # Initialize noise generators with different seeds
    if noise_seed_substep is None:
        noise_seed_substep = noise_seed + 10000  # Offset to ensure different seeds
    step_noise_generator = torch.Generator(device=video_state.latent.device).manual_seed(noise_seed)
    substep_noise_generator = torch.Generator(device=video_state.latent.device).manual_seed(noise_seed_substep)
    sde_noise_injecting_fn = partial(
        _inject_sde_noise, stepper=stepper, new_noise_fn=new_noise_fn, legacy_mode=legacy_mode
    )
    step_noise_injecting_fn = partial(sde_noise_injecting_fn, step_noise_generator=step_noise_generator)
    substep_noise_injecting_fn = partial(sde_noise_injecting_fn, step_noise_generator=substep_noise_generator)

    if not isinstance(stepper, Res2sDiffusionStep):
        raise ValueError("stepper must be an instance of Res2sDiffusionStep")

    n_full_steps = len(sigmas) - 1
    # inject minimal sigma value to avoid division by zero
    if sigmas[-1] == 0:
        sigmas = torch.cat([sigmas[:-1], torch.tensor([0.0011, 0.0], device=sigmas.device)], dim=0)
    # Compute step sizes in hyperbolic space
    hs = -torch.log(sigmas[1:].double().cpu() / (sigmas[:-1].double().cpu()))

    # Initialize phi cache for reuse across loop iterations
    # Cache key: (j, neg_h) where j is phi order and neg_h is negative step value
    phi_cache = {}
    c2 = 0.5  # Midpoint for res_2s

    # Progress bar shows only full two-stage steps; final (sigma_next==0) step is done silently

    for step_idx in tqdm(range(n_full_steps)):
        sigma = sigmas[step_idx].double()
        sigma_next = sigmas[step_idx + 1].double()

        # Initialize anchor point
        x_anchor_video = video_state.latent.clone().double()
        x_anchor_audio = audio_state.latent.clone().double()

        # ====================================================================
        # STAGE 1: Evaluate at current point
        # ====================================================================
        denoised_video_1, denoised_audio_1 = denoise_fn(video_state, audio_state, sigmas, step_idx)
        denoised_video_1 = post_process_latent(denoised_video_1, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio_1 = post_process_latent(denoised_audio_1, audio_state.denoise_mask, audio_state.clean_latent)

        h = hs[step_idx].item()

        # Compute RK coefficients (pass phi_cache for caching)
        a21, b1, b2 = get_res2s_coefficients(h, phi_cache, c2)

        # Compute substep sigma, sqrt is a hardcode for c2 = 0.5
        sub_sigma = torch.sqrt(sigma * sigma_next)

        # ====================================================================
        # Compute substep x using RK coefficient a21
        # ====================================================================
        eps_1_video = denoised_video_1.double() - x_anchor_video
        eps_1_audio = denoised_audio_1.double() - x_anchor_audio

        x_mid_video = x_anchor_video.double() + h * a21 * eps_1_video
        x_mid_audio = x_anchor_audio.double() + h * a21 * eps_1_audio

        # ====================================================================
        # SDE noise injection at substep
        # ====================================================================
        x_mid_video = substep_noise_injecting_fn(
            state=video_state,
            sample=x_anchor_video,
            denoised_sample=x_mid_video,
            sigmas=torch.stack([sigma, sub_sigma]),
            step_idx=0,
        )
        x_mid_audio = substep_noise_injecting_fn(
            state=audio_state,
            sample=x_anchor_audio,
            denoised_sample=x_mid_audio,
            sigmas=torch.stack([sigma, sub_sigma]),
            step_idx=0,
        )
        # ====================================================================
        # ITERATIVE REFINEMENT (Bong Iteration) - Stabilize anchor point
        # ====================================================================
        if bongmath and h < 0.5 and sigma > 0.03:
            for _ in range(bongmath_max_iter):
                x_anchor_video = x_mid_video - h * a21 * eps_1_video
                eps_1_video = denoised_video_1.double() - x_anchor_video
                x_anchor_audio = x_mid_audio - h * a21 * eps_1_audio
                eps_1_audio = denoised_audio_1.double() - x_anchor_audio

        # ====================================================================
        # STAGE 2: Evaluate at substep point (WITH NOISE)
        # ====================================================================
        mid_video_state = replace(video_state, latent=x_mid_video.to(model_dtype))
        mid_audio_state = replace(audio_state, latent=x_mid_audio.to(model_dtype))

        denoised_video_2, denoised_audio_2 = denoise_fn(
            video_state=mid_video_state,
            audio_state=mid_audio_state,
            sigmas=torch.stack([sub_sigma]).to(sigmas.device),
            step_index=0,
        )
        denoised_video_2 = post_process_latent(denoised_video_2, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio_2 = post_process_latent(denoised_audio_2, audio_state.denoise_mask, audio_state.clean_latent)

        # ====================================================================
        # FINAL COMBINATION: Compute x_next using RK coefficients
        # ====================================================================
        eps_2_video = denoised_video_2.double() - x_anchor_video
        eps_2_audio = denoised_audio_2.double() - x_anchor_audio

        x_next_video = x_anchor_video + h * (b1 * eps_1_video + b2 * eps_2_video)
        x_next_audio = x_anchor_audio + h * (b1 * eps_1_audio + b2 * eps_2_audio)

        # ====================================================================
        # SDE NOISE INJECTION AT STEP LEVEL
        # ====================================================================
        x_next_video = step_noise_injecting_fn(
            state=video_state,
            sample=x_anchor_video,
            denoised_sample=x_next_video,
            sigmas=sigmas,
            step_idx=step_idx,
        )
        x_next_audio = step_noise_injecting_fn(
            state=audio_state,
            sample=x_anchor_audio,
            denoised_sample=x_next_audio,
            sigmas=sigmas,
            step_idx=step_idx,
        )

        # Update states
        video_state = replace(video_state, latent=x_next_video.to(model_dtype))
        audio_state = replace(audio_state, latent=x_next_audio.to(model_dtype))

    # Final step if we need to fully remove the noise
    if sigmas[-1] == 0:
        denoised_video_1, denoised_audio_1 = denoise_fn(video_state, audio_state, sigmas, n_full_steps)
        denoised_video_1 = post_process_latent(denoised_video_1, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio_1 = post_process_latent(denoised_audio_1, audio_state.denoise_mask, audio_state.clean_latent)
        video_state = replace(video_state, latent=denoised_video_1.to(model_dtype))
        audio_state = replace(audio_state, latent=denoised_audio_1.to(model_dtype))

    return video_state, audio_state
