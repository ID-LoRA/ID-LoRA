"""
Speaker, Face, CLAP Similarity, and WER Metrics for Audio-Video LoRA Evaluation

This module computes:
1. Speaker similarity using ECAPA-TDNN (SpeechBrain) or WavLM+ECAPA-TDNN (SOTA)
2. Face similarity using ArcFace embeddings (InsightFace)
3. CLAP similarity for caption-to-audio matching (LAION CLAP)
4. WER/CER using Whisper large-v3 ASR + jiwer

Speaker metrics:
- ECAPA-TDNN (SpeechBrain): ~0.8% EER on VoxCeleb1-E, lighter/faster
- WavLM+ECAPA-TDNN (UniSpeech): 0.431% EER on VoxCeleb1-O, SOTA but heavier

CLAP metrics:
- LAION CLAP (laion/clap-htsat-unfused): Caption-to-audio similarity
- Extracts [ENVIRONMENT_SOUNDS] or [SPEAKING_STYLE] from captions
- Compares text descriptions to audio via joint embedding space

WER/CER metrics:
- Whisper large-v3 (openai/whisper-large-v3) for ASR transcription
- jiwer for WER and CER computation against ground truth text
"""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

logger = logging.getLogger(__name__)


# =============================================================================
# Speaker Similarity (ECAPA-TDNN) - Primary, Established Metric
# =============================================================================

class SpeakerSimilarityMetric:
    """Compute speaker similarity using ECAPA-TDNN embeddings (SpeechBrain).
    
    ECAPA-TDNN is the state-of-the-art speaker verification model, widely used
    in voice cloning research. Pre-trained on VoxCeleb1+2, achieves ~0.8% EER
    on VoxCeleb1-E benchmark.
    
    References:
    - ECAPA-TDNN paper: https://arxiv.org/abs/2005.07143
    - SpeechBrain: https://speechbrain.github.io/
    
    Memory Management:
    - Call offload_to_cpu() after metrics computation to free GPU memory
    - Model will be automatically reloaded on next use
    """
    
    def __init__(self, device: str = "cuda"):
        self._target_device = device
        self.device = device
        self._model = None
        self._is_offloaded = False
        
    def _load_model(self):
        """Lazy load the ECAPA-TDNN model."""
        if self._model is not None:
            return
        
        # Workaround for SpeechBrain + newer torchaudio compatibility
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: ['soundfile']
            
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise ImportError(
                "Please install speechbrain: pip install speechbrain"
            )
        
        reload_msg = " (reloading after offload)" if self._is_offloaded else ""
        logger.info(f"Loading ECAPA-TDNN model for speaker verification{reload_msg}...")
        # Pre-trained on VoxCeleb1+2, standard model for speaker verification
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_ecapa",
            run_opts={"device": self._target_device}
        )
        self._is_offloaded = False
        logger.info("ECAPA-TDNN model loaded successfully")
    
    def extract_embedding(self, audio_path: str | Path) -> torch.Tensor:
        """Extract speaker embedding from an audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.) or video file
            
        Returns:
            Speaker embedding tensor of shape [1, 192]
        """
        self._load_model()
        
        audio_path = Path(audio_path)
        
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self._model.encode_batch(waveform.to(self.device))
        
        # Normalize
        embedding = F.normalize(embedding.squeeze(0), p=2, dim=-1)
        
        return embedding
    
    def compute_similarity(
        self,
        reference_path: str | Path,
        generated_path: str | Path,
    ) -> float:
        """Compute cosine similarity between reference and generated audio.
        
        Args:
            reference_path: Path to reference audio/video
            generated_path: Path to generated audio/video
            
        Returns:
            Cosine similarity score in [-1, 1], higher is better
            Typical same-speaker: 0.7-0.95
            Typical different-speaker: 0.0-0.3
        """
        ref_embedding = self.extract_embedding(reference_path)
        gen_embedding = self.extract_embedding(generated_path)
        
        similarity = F.cosine_similarity(ref_embedding, gen_embedding, dim=-1)
        return similarity.item()
    
    def compute_batch_similarity(
        self,
        reference_paths: list[str | Path],
        generated_paths: list[str | Path],
    ) -> dict:
        """Compute similarity metrics for a batch of samples.
        
        Returns:
            Dictionary with mean, std, min, max similarity scores
        """
        similarities = []
        for ref, gen in zip(reference_paths, generated_paths):
            try:
                sim = self.compute_similarity(ref, gen)
                similarities.append(sim)
            except Exception as e:
                logger.warning(f"Error computing similarity for {ref} vs {gen}: {e}")
        
        if not similarities:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "count": len(similarities),
        }
    
    def offload_to_cpu(self):
        """Unload model to free GPU memory.
        
        SpeechBrain models don't support easy device transfer, so we delete
        the model and let it be reloaded on next use. This is useful during
        training when GPU memory is tight.
        """
        if self._model is not None:
            logger.info("Unloading ECAPA-TDNN model to free GPU memory...")
            del self._model
            self._model = None
            self._is_offloaded = True
            torch.cuda.empty_cache()


# =============================================================================
# Speaker Similarity (WavLM+ECAPA-TDNN) - SOTA from Microsoft UniSpeech
# =============================================================================

class WavLMSpeakerSimilarityMetric:
    """Compute speaker similarity using WavLM Large + ECAPA-TDNN (Microsoft UniSpeech).
    
    This is a hybrid model that combines:
    - WavLM Large as the feature extractor (HuggingFace Transformers)
    - ECAPA-TDNN as the speaker embedding head
    
    This achieves state-of-the-art results on VoxCeleb:
    - Vox1-O: 0.431% EER (best)
    - Vox1-E: 0.538% EER
    - Vox1-H: 1.154% EER
    
    Reference: https://github.com/microsoft/UniSpeech
    
    Memory Management:
    - Call offload_to_cpu() after metrics computation to free GPU memory
    - Model will be automatically moved back to GPU on next use
    """
    
    def __init__(self, device: str = "cuda"):
        self._target_device = torch.device(device)
        self.device = self._target_device
        self._model = None
        
    def _load_model(self):
        """Lazy load the WavLM+ECAPA-TDNN model."""
        if self._model is not None:
            # Move back to target device if offloaded
            if self.device != self._target_device:
                logger.info(f"Moving WavLM+ECAPA-TDNN model back to {self._target_device}...")
                self._model = self._model.to(self._target_device)
                self.device = self._target_device
            return
        
        from ltx_trainer.metrics.unispeech_sv import load_wavlm_ecapa_model
        
        # Find the checkpoint - try several locations
        checkpoint_path = None
        possible_paths = [
            Path(__file__).parent.parent.parent.parent.parent.parent / "models" / "unispeech" / "wavlm_large_finetune.pth",
            Path("models/unispeech/wavlm_large_finetune.pth"),
            Path("/scratch/aviad/github/LTX-2/models/unispeech/wavlm_large_finetune.pth"),
        ]
        
        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            logger.warning(
                "WavLM+ECAPA-TDNN checkpoint not found. Using untrained ECAPA head. "
                "Download from: https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP "
                "and place at models/unispeech/wavlm_large_finetune.pth"
            )
        else:
            logger.info(f"Found WavLM+ECAPA-TDNN checkpoint at {checkpoint_path}")
        
        logger.info("Loading WavLM Large + ECAPA-TDNN model (SOTA 0.431% EER)...")
        self._model = load_wavlm_ecapa_model(
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            device=str(self.device)
        )
        logger.info("WavLM Large + ECAPA-TDNN model loaded successfully")
    
    def extract_embedding(self, audio_path: str | Path) -> torch.Tensor:
        """Extract speaker embedding from an audio file.
        
        Args:
            audio_path: Path to audio file or video file
            
        Returns:
            Speaker embedding tensor of shape [1, 256]
        """
        self._load_model()
        
        audio_path = Path(audio_path)
        
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Model expects [batch, time] shape
        waveform = waveform.squeeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self._model(waveform.unsqueeze(0))
        
        # Normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    def compute_similarity(
        self,
        reference_path: str | Path,
        generated_path: str | Path,
    ) -> float:
        """Compute cosine similarity between reference and generated audio.
        
        Args:
            reference_path: Path to reference audio/video
            generated_path: Path to generated audio/video
            
        Returns:
            Cosine similarity score in [-1, 1], higher is better
        """
        ref_embedding = self.extract_embedding(reference_path)
        gen_embedding = self.extract_embedding(generated_path)
        
        similarity = F.cosine_similarity(ref_embedding, gen_embedding, dim=-1)
        return similarity.item()
    
    def compute_batch_similarity(
        self,
        reference_paths: list[str | Path],
        generated_paths: list[str | Path],
    ) -> dict:
        """Compute similarity metrics for a batch of samples.
        
        Returns:
            Dictionary with mean, std, min, max similarity scores
        """
        similarities = []
        for ref, gen in zip(reference_paths, generated_paths):
            try:
                sim = self.compute_similarity(ref, gen)
                similarities.append(sim)
            except Exception as e:
                logger.warning(f"Error computing WavLM similarity for {ref} vs {gen}: {e}")
        
        if not similarities:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "count": len(similarities),
        }
    
    def offload_to_cpu(self):
        """Offload model to CPU to free GPU memory.
        
        Useful during training when GPU memory is tight.
        The model will be automatically moved back to GPU on next use.
        """
        if self._model is not None and self.device != torch.device("cpu"):
            logger.info("Offloading WavLM+ECAPA-TDNN model to CPU to free GPU memory...")
            self._model = self._model.to("cpu")
            self.device = torch.device("cpu")
            torch.cuda.empty_cache()


# =============================================================================
# Face Similarity (ArcFace / InsightFace)
# =============================================================================

class FaceSimilarityMetric:
    """Compute face similarity using ArcFace embeddings via InsightFace.
    
    ArcFace is a state-of-the-art face recognition model that produces
    discriminative face embeddings for identity verification.
    
    Memory Management:
    - Call offload_to_cpu() after metrics computation to free GPU memory
    - Model will be automatically reloaded on next use (ONNX doesn't support device transfer)
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._is_offloaded = False
        
    def _load_model(self):
        """Lazy load the InsightFace model."""
        if self._model is not None:
            return
            
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "Please install insightface: pip install insightface onnxruntime"
            )
        
        reload_msg = " (reloading after offload)" if self._is_offloaded else ""
        logger.info(f"Loading InsightFace model for face recognition{reload_msg}...")
        
        # Use the buffalo_l model (includes ArcFace for recognition)
        # Always use CPU for InsightFace to avoid competing for GPU memory
        # with the main model (~42GB). ONNX on GPU causes OOM and device
        # mismatch errors when GPU memory is tight.
        self._model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        # Use smaller detection size for better face detection on AI-generated videos
        # 320x320 works better for ~512x512 videos than 640x640
        # ctx_id=-1 forces CPU execution
        self._model.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.3)
        self._is_offloaded = False
        logger.info("InsightFace model loaded successfully")
    
    def extract_embedding(
        self,
        video_path: str | Path,
        frame_idx: int = 0,
    ) -> Optional[np.ndarray]:
        """Extract face embedding from a video frame.
        
        Args:
            video_path: Path to video file
            frame_idx: Which frame to extract face from (0 = first frame)
            
        Returns:
            Face embedding array of shape [512] or None if no face detected
        """
        self._load_model()
        
        try:
            import cv2
        except ImportError:
            raise ImportError("Please install opencv: pip install opencv-python")
        
        # Read video frame
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.warning(f"Could not read frame {frame_idx} from {video_path}")
            return None
        
        # Detect faces and extract embeddings
        faces = self._model.get(frame)
        
        if not faces:
            logger.warning(f"No face detected in {video_path} at frame {frame_idx}")
            return None
        
        # Use the largest face (by bounding box area)
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        return largest_face.embedding
    
    def extract_embedding_multi_frame(
        self,
        video_path: str | Path,
        num_frames: int = 5,
    ) -> Optional[np.ndarray]:
        """Extract and average face embeddings from multiple frames.
        
        This is more robust than single-frame extraction.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
            
        Returns:
            Averaged face embedding or None if no faces detected
        """
        self._load_model()
        
        try:
            import cv2
        except ImportError:
            raise ImportError("Please install opencv: pip install opencv-python")
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        embeddings = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            faces = self._model.get(frame)
            if faces:
                largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embeddings.append(largest_face.embedding)
        
        cap.release()
        
        if not embeddings:
            return None
        
        # Average embeddings and normalize
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return avg_embedding
    
    def compute_similarity(
        self,
        reference_path: str | Path,
        generated_path: str | Path,
        multi_frame: bool = True,
    ) -> Optional[float]:
        """Compute cosine similarity between reference and generated video faces.
        
        Args:
            reference_path: Path to reference video
            generated_path: Path to generated video
            multi_frame: Whether to use multi-frame averaging
            
        Returns:
            Cosine similarity score in [-1, 1], higher is better, or None if face detection failed
        """
        if multi_frame:
            ref_embedding = self.extract_embedding_multi_frame(reference_path)
            gen_embedding = self.extract_embedding_multi_frame(generated_path)
        else:
            ref_embedding = self.extract_embedding(reference_path)
            gen_embedding = self.extract_embedding(generated_path)
        
        if ref_embedding is None or gen_embedding is None:
            return None
        
        similarity = np.dot(ref_embedding, gen_embedding)
        return float(similarity)
    
    def compute_batch_similarity(
        self,
        reference_paths: list[str | Path],
        generated_paths: list[str | Path],
        multi_frame: bool = True,
    ) -> dict:
        """Compute similarity metrics for a batch of samples.
        
        Returns:
            Dictionary with mean, std, min, max similarity scores
        """
        similarities = []
        for ref, gen in zip(reference_paths, generated_paths):
            try:
                sim = self.compute_similarity(ref, gen, multi_frame=multi_frame)
                if sim is not None:
                    similarities.append(sim)
            except Exception as e:
                logger.warning(f"Error computing face similarity for {ref} vs {gen}: {e}")
        
        if not similarities:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "count": len(similarities),
        }
    
    def offload_to_cpu(self):
        """Unload model to free GPU memory.
        
        InsightFace uses ONNX which doesn't support device transfer,
        so we delete the model and let it be reloaded on next use.
        This is useful during training when GPU memory is tight.
        """
        if self._model is not None:
            logger.info("Unloading InsightFace model to free GPU memory...")
            del self._model
            self._model = None
            self._is_offloaded = True
            torch.cuda.empty_cache()


# =============================================================================
# CLAP Similarity (Caption-to-Audio) - LAION CLAP
# =============================================================================

class CLAPSimilarityMetric:
    """Compute CLAP caption-to-audio similarity using LAION CLAP model.

    CLAP (Contrastive Language-Audio Pretraining) measures how well the audio
    matches a text description. This is useful for evaluating if generated
    videos produce audio matching the prompt's audio descriptions.

    Extracts [ENVIRONMENT_SOUNDS] or [SPEAKING_STYLE] from captions and
    compares to the video's audio track using CLAP embeddings.

    References:
    - CLAP paper: https://arxiv.org/abs/2211.06687 (LAION, ICASSP 2023)
    - Model: laion/clap-htsat-unfused

    Memory Management:
    - Call offload_to_cpu() after metrics computation to free GPU memory
    - Model will be automatically reloaded on next use
    """

    def __init__(self, device: str = "cuda", model_id: str | None = None):
        """Initialize CLAP similarity metric.

        Args:
            device: Device to use ('cuda' or 'cpu')
            model_id: CLAP model ID (default: laion/clap-htsat-unfused)
        """
        self._target_device = device
        self.device = device
        self.model_id = model_id or os.environ.get("CLAP_MODEL_ID", "laion/clap-htsat-unfused")
        self._model = None
        self._processor = None
        self._is_offloaded = False

    def _load_model(self):
        """Lazy load the CLAP model and processor."""
        if self._model is not None:
            return

        try:
            from transformers import AutoProcessor, ClapModel
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers"
            )

        reload_msg = " (reloading after offload)" if self._is_offloaded else ""
        logger.info(f"Loading CLAP model ({self.model_id}) for caption-audio similarity{reload_msg}...")
        self._model = ClapModel.from_pretrained(self.model_id).to(self._target_device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._is_offloaded = False
        logger.info("CLAP model loaded successfully")

    _ENV_NONE_REPLACEMENT = "clear speech with no ambient or environmental sounds"

    @staticmethod
    def extract_env_sounds_from_caption(caption: str) -> str:
        """Extract [ENVIRONMENT_SOUNDS]: ... from caption text.

        When the caption says "None", returns a descriptive phrase so CLAP
        can meaningfully score silence vs unwanted ambient sounds.
        """
        if not caption:
            return ""
        m = re.search(
            r"\[ENVIRONMENT_SOUNDS\]\s*:\s*(.+?)(?=\n\[|\n\s*$|\Z)",
            caption,
            re.DOTALL | re.IGNORECASE,
        )
        if not m:
            return ""
        text = m.group(1).strip()
        if text.lower() in ("none", "none."):
            return CLAPSimilarityMetric._ENV_NONE_REPLACEMENT
        return text

    @staticmethod
    def extract_speaking_style_from_caption(caption: str) -> str:
        """Extract [SPEAKING_STYLE]: ... from caption text."""
        if not caption:
            return ""
        m = re.search(
            r"\[SPEAKING_STYLE\]\s*:\s*(.+?)(?=\n\[|\n\s*$|\Z)",
            caption,
            re.DOTALL | re.IGNORECASE,
        )
        return m.group(1).strip() if m else ""

    def _extract_audio_from_video(
        self,
        video_path: str | Path,
        sr: int = 16000,
        max_sec: float = 30.0
    ) -> Path | None:
        """Extract audio from video to a temporary WAV file.

        Args:
            video_path: Path to video file
            sr: Sample rate for output audio (default 16kHz, will be resampled to 48kHz for CLAP)
            max_sec: Maximum seconds to extract

        Returns:
            Path to temporary WAV file, or None if extraction failed
        """
        try:
            fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="clap_audio_")
            os.close(fd)
            out = Path(out_path)
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
                "-t", str(max_sec), "-loglevel", "error", str(out),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0 or not out.is_file():
                if out.exists():
                    out.unlink(missing_ok=True)
                return None
            return out
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None

    def _load_audio_array(self, wav_path: Path) -> tuple[np.ndarray, int]:
        """Load WAV file to float32 mono array.

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            import soundfile as sf
            data, sr = sf.read(str(wav_path), dtype="float32")
        except ImportError:
            import scipy.io.wavfile as wavfile
            sr, data = wavfile.read(str(wav_path))
            data = data.astype("float32") / (2 ** 15)

        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr

    def _resample_to_48k(self, audio_array: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Resample audio to 48kHz for CLAP model.

        CLAP models expect 48kHz audio input.
        """
        if sr == 48000:
            return audio_array, 48000
        try:
            import librosa
            out = librosa.resample(audio_array, orig_sr=sr, target_sr=48000)
            return out, 48000
        except ImportError:
            # Fall back to original sample rate if librosa not available
            return audio_array, sr

    def compute_similarity(
        self,
        audio_path: str | Path,
        caption_text: str,
        use_cosine: bool = True,
    ) -> float:
        """Compute CLAP similarity between audio and caption text.

        Args:
            audio_path: Path to audio or video file (will extract audio if video)
            caption_text: Text description to compare against audio
            use_cosine: If True, return cosine similarity [-1, 1] (standard CLAP score).
                       If False, return raw logit (unbounded, scaled by temperature).

        Returns:
            Similarity score. Higher is better.
            - Cosine mode: [-1, 1], typical good match: 0.2-0.5
            - Logit mode: unbounded, typical: 10-30 for good matches
        """
        self._load_model()

        audio_path = Path(audio_path)

        # Check if we need to extract audio from video
        is_video = audio_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        cleanup_audio = False

        if is_video:
            extracted = self._extract_audio_from_video(audio_path)
            if extracted is None:
                logger.warning(f"Failed to extract audio from {audio_path}")
                return 0.0
            audio_path = extracted
            cleanup_audio = True

        try:
            # Load and resample audio
            audio_array, sr = self._load_audio_array(audio_path)
            audio_array, sr = self._resample_to_48k(audio_array, sr)

            # Process text and audio separately to avoid ClapProcessor
            # forwarding **kwargs to both tokenizer and feature extractor
            text_inputs = self._processor.tokenizer(
                [caption_text], return_tensors="pt", padding=True
            )
            audio_inputs = self._processor.feature_extractor(
                [audio_array], sampling_rate=sr, return_tensors="pt"
            )
            inputs = {
                **text_inputs,
                "input_features": audio_inputs.input_features,
            }
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

                if use_cosine:
                    # Standard CLAP score: cosine similarity of embeddings
                    audio_emb = outputs.audio_embeds
                    text_emb = outputs.text_embeds
                    cos = F.cosine_similarity(audio_emb, text_emb, dim=-1)
                    return float(cos.cpu().item())
                else:
                    # Raw logit (includes temperature scaling)
                    logits = outputs.logits_per_audio
                    if logits is None:
                        return 0.0
                    return float(logits.cpu().numpy().flatten()[0])

        finally:
            if cleanup_audio and audio_path.exists():
                audio_path.unlink(missing_ok=True)

    def compute_similarity_from_video(
        self,
        video_path: str | Path,
        caption: str,
        caption_type: str = "env",
        use_cosine: bool = True,
    ) -> float | None:
        """Compute CLAP similarity for a video using the specified caption type.

        Args:
            video_path: Path to video file
            caption: Full caption containing [ENVIRONMENT_SOUNDS] and/or [SPEAKING_STYLE]
            caption_type: 'env' for ENVIRONMENT_SOUNDS, 'speaking' for SPEAKING_STYLE
            use_cosine: If True, return cosine similarity (standard CLAP score)

        Returns:
            Similarity score, or None if caption section not found
        """
        if caption_type == "env":
            caption_text = self.extract_env_sounds_from_caption(caption)
        elif caption_type == "speaking":
            caption_text = self.extract_speaking_style_from_caption(caption)
        else:
            raise ValueError(f"Unknown caption_type: {caption_type}. Use 'env' or 'speaking'.")

        if not caption_text.strip():
            return None

        return self.compute_similarity(video_path, caption_text, use_cosine=use_cosine)

    def compute_similarity_from_audio(
        self,
        audio_path: str | Path,
        caption: str,
        caption_type: str = "env",
        use_cosine: bool = True,
    ) -> float | None:
        """Compute CLAP similarity using a pre-separated audio WAV file.

        Unlike compute_similarity_from_video, this accepts a WAV file directly
        (e.g. from SAM-Audio separation) and skips video audio extraction.

        Args:
            audio_path: Path to WAV audio file (speaker-only or env-only)
            caption: Full caption containing [ENVIRONMENT_SOUNDS] and/or [SPEAKING_STYLE]
            caption_type: 'env' for ENVIRONMENT_SOUNDS, 'speaking' for SPEAKING_STYLE
            use_cosine: If True, return cosine similarity (standard CLAP score)

        Returns:
            Similarity score, or None if caption section not found or audio missing
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.warning(f"Separated audio not found: {audio_path}")
            return None

        if caption_type == "env":
            caption_text = self.extract_env_sounds_from_caption(caption)
        elif caption_type == "speaking":
            caption_text = self.extract_speaking_style_from_caption(caption)
        else:
            raise ValueError(f"Unknown caption_type: {caption_type}. Use 'env' or 'speaking'.")

        if not caption_text.strip():
            return None

        return self.compute_similarity(audio_path, caption_text, use_cosine=use_cosine)

    def compute_batch_similarity(
        self,
        video_paths: list[str | Path],
        captions: list[str],
        caption_type: str = "env",
        use_cosine: bool = True,
    ) -> dict:
        """Compute CLAP similarity metrics for a batch of video-caption pairs.

        Args:
            video_paths: List of video paths
            captions: List of full captions (with [ENVIRONMENT_SOUNDS] or [SPEAKING_STYLE])
            caption_type: 'env' or 'speaking' to select which caption section to use
            use_cosine: If True, return cosine similarity

        Returns:
            Dictionary with mean, std, min, max, count statistics
        """
        similarities = []

        for video_path, caption in zip(video_paths, captions):
            try:
                sim = self.compute_similarity_from_video(
                    video_path, caption, caption_type=caption_type, use_cosine=use_cosine
                )
                if sim is not None:
                    similarities.append(sim)
            except Exception as e:
                logger.warning(f"Error computing CLAP similarity for {video_path}: {e}")

        if not similarities:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "count": len(similarities),
        }

    def offload_to_cpu(self):
        """Unload model to free GPU memory.

        The model will be automatically reloaded on next use.
        """
        if self._model is not None:
            logger.info("Unloading CLAP model to free GPU memory...")
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._is_offloaded = True
            torch.cuda.empty_cache()


# =============================================================================
# WER/CER (Whisper ASR + jiwer)
# =============================================================================

class WhisperWERMetric:
    """Compute WER and CER by transcribing audio with Whisper large-v3 and comparing to ground truth.

    Uses HuggingFace transformers pipeline for ASR and jiwer for error rate
    computation. Text is normalized (lowercase, strip, collapse whitespace)
    before comparison.

    Memory Management:
    - Call offload_to_cpu() after metrics computation to free GPU memory
    - Model will be automatically reloaded on next use
    """

    def __init__(self, device: str = "cuda", model_id: str = "openai/whisper-large-v3"):
        self._target_device = device
        self.device = device
        self.model_id = model_id
        self._pipe = None
        self._is_offloaded = False

    def _load_model(self):
        """Lazy load the Whisper ASR pipeline."""
        if self._pipe is not None:
            return

        from transformers import pipeline as hf_pipeline

        reload_msg = " (reloading after offload)" if self._is_offloaded else ""
        logger.info(f"Loading Whisper ASR pipeline ({self.model_id}){reload_msg}...")
        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self._target_device,
            dtype=torch.float16,
        )
        self._is_offloaded = False
        logger.info("Whisper ASR pipeline loaded successfully")

    LANGUAGE_CODES = {
        "english": "en", "spanish": "es", "italian": "it",
        "french": "fr", "chinese": "zh", "hebrew": "he",
    }

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> str:
        """Transcribe an audio file using Whisper.

        Args:
            audio_path: Path to WAV/MP3/FLAC audio file
            language: Language name or ISO code (e.g. "english", "es"). Defaults to "en".

        Returns:
            Transcribed text string
        """
        self._load_model()
        lang_code = self.LANGUAGE_CODES.get(language, language) if language else "en"
        result = self._pipe(
            str(audio_path),
            return_timestamps=False,
            generate_kwargs={"language": lang_code, "task": "transcribe"},
        )
        return result["text"].strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for WER/CER comparison.

        Strips punctuation so that e.g. ``"raisins."`` matches ``"raisins"``
        (Whisper transcriptions typically omit punctuation).
        """
        import re as _re
        text = text.lower().strip()
        text = _re.sub(r"[^\w\s]", "", text)
        text = _re.sub(r"\s+", " ", text)
        return text.strip()

    def compute_wer(
        self,
        audio_path: str | Path,
        ground_truth_text: str,
        language: str | None = None,
    ) -> dict:
        """Transcribe audio and compute WER/CER against ground truth.

        Args:
            audio_path: Path to audio file
            ground_truth_text: Expected transcript text
            language: Language name or ISO code for Whisper (e.g. "spanish", "zh")

        Returns:
            Dict with keys: wer, cer, transcription
        """
        import jiwer

        hypothesis = self.transcribe(audio_path, language=language)

        ref = self._normalize_text(ground_truth_text)
        hyp = self._normalize_text(hypothesis)

        if not ref:
            logger.warning("Empty ground truth text, returning WER/CER = 1.0")
            return {"wer": 1.0, "cer": 1.0, "transcription": hypothesis}

        wer_score = jiwer.wer(ref, hyp)
        cer_score = jiwer.cer(ref, hyp)

        return {
            "wer": float(wer_score),
            "cer": float(cer_score),
            "transcription": hypothesis,
        }

    def offload_to_cpu(self):
        """Unload model to free GPU memory."""
        if self._pipe is not None:
            logger.info("Unloading Whisper ASR pipeline to free GPU memory...")
            del self._pipe
            self._pipe = None
            self._is_offloaded = True
            torch.cuda.empty_cache()