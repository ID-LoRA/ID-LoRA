"""UniSpeech speaker verification components."""

from ltx_trainer.metrics.unispeech_sv.wavlm_ecapa import (
    WavLMECAPATDNN,
    load_wavlm_ecapa_model,
)

__all__ = ["WavLMECAPATDNN", "load_wavlm_ecapa_model"]
