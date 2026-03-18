"""
Validation metrics for audio-video generation quality.

Includes:
- SpeakerSimilarityMetric: ECAPA-TDNN speaker verification (SpeechBrain, ~0.8% EER)
- WavLMSpeakerSimilarityMetric: WavLM+ECAPA-TDNN speaker verification (SOTA, 0.431% EER)
- FaceSimilarityMetric: ArcFace face recognition (InsightFace)
- CLAPSimilarityMetric: LAION CLAP caption-to-audio similarity
"""

from ltx_trainer.metrics.evaluation_metrics import (
    SpeakerSimilarityMetric,
    WavLMSpeakerSimilarityMetric,
    FaceSimilarityMetric,
    CLAPSimilarityMetric,
)

__all__ = [
    "SpeakerSimilarityMetric",
    "WavLMSpeakerSimilarityMetric",
    "FaceSimilarityMetric",
    "CLAPSimilarityMetric",
]
