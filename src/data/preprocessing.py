"""Data preprocessing module."""

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
from scipy import signal

from ..utils import (
    apply_cmvn,
    extract_mel_spectrogram,
    load_audio,
    normalize_features,
)


class AudioPreprocessor:
    """Audio preprocessing pipeline."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 3.0,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        f_min: int = 50,
        f_max: int = 7600,
        normalization: str = "cmvn"
    ):
        """
        Initialize preprocessor.
        
        Args:
            sample_rate: Target sample rate
            duration: Audio duration in seconds
            n_mels: Number of mel bins
            n_fft: FFT window size
            hop_length: Hop length for STFT
            f_min: Minimum frequency
            f_max: Maximum frequency
            normalization: Normalization method ("cmvn", "mean_std", None)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.normalization = normalization
    
    def __call__(self, audio_path: str) -> np.ndarray:
        """
        Process audio file using the exact training-compatible robust pipeline.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Processed features (n_mels, time_steps)
        """
        # Load audio (mono, 16 kHz)
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            # Fallback if librosa fails
            y = np.zeros(int(self.sample_rate * self.duration), dtype=np.float32)
            sr = self.sample_rate

        # 1. Voice Activity Detection (VAD) - Trim silences under 30 dB
        # This completely resolves matching on silences!
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)

        # 2. Force fixed duration (3 seconds = 48000 samples)
        max_samples = int(self.sample_rate * self.duration)
        if len(y_trimmed) > max_samples:
            y_trimmed = y_trimmed[:max_samples]
        elif len(y_trimmed) < max_samples:
            y_trimmed = np.pad(y_trimmed, (0, max_samples - len(y_trimmed)), mode='constant')

        # 3. Compute Mel Spectrogram exactly like PyTorch T.MelSpectrogram
        # We use n_fft=400 and win_length=400 to match PyTorch's defaults used in notebooks!
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y_trimmed,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=400,
            hop_length=160,
            win_length=400,
            fmin=0.0,
            fmax=None
        )

        # 4. Natural logarithm: log(mel + 1e-6)
        log_mel = np.log(mel_spectrogram + 1e-6)

        # 5. Cepstral Mean Subtraction (CMS) exactly like Notebooks: lm - lm.mean()
        # This removes the microphone response and environmental acoustic channel bias.
        log_mel = log_mel - np.mean(log_mel)

        return log_mel
    
    def process_batch(self, audio_paths: list) -> np.ndarray:
        """
        Process batch of audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            Batch of processed features
        """
        features_list = []
        
        for audio_path in audio_paths:
            try:
                features = self(audio_path)
                features_list.append(features)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
        
        return features_list


class VoiceActivityDetector:
    """Voice Activity Detection using energy thresholding."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 512,
        hop_length: int = 160,
        threshold_db: float = 40.0
    ):
        """
        Initialize VAD.
        
        Args:
            sample_rate: Sample rate
            frame_length: Frame length for energy computation
            hop_length: Hop length
            threshold_db: Energy threshold in dB
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.threshold_db = threshold_db
    
    def detect(self, y: np.ndarray) -> np.ndarray:
        """
        Detect voice activity.
        
        Args:
            y: Audio signal
            
        Returns:
            Boolean mask indicating voiced frames
        """
        # Compute energy for each frame
        energy = np.array([
            np.sum(y[i:i+self.frame_length]**2)
            for i in range(0, len(y), self.hop_length)
        ])
        
        # Threshold
        threshold = np.max(energy) - self.threshold_db
        voiced = energy > threshold
        
        return voiced
    
    def extract_voiced_frames(self, y: np.ndarray) -> np.ndarray:
        """
        Extract voiced segments.
        
        Args:
            y: Audio signal
            
        Returns:
            Voiced audio segments concatenated
        """
        voiced = self.detect(y)
        
        # Expand boolean mask to sample level
        mask = np.repeat(voiced, self.hop_length)
        mask = mask[:len(y)]
        
        voiced_frames = y[mask > 0]
        
        return voiced_frames


class TemporalAugmentation:
    """Temporal augmentation techniques."""
    
    @staticmethod
    def pitch_shift(y: np.ndarray, sr: int, n_steps: int) -> np.ndarray:
        """
        Pitch shift using librosa.
        
        Args:
            y: Audio signal
            sr: Sample rate
            n_steps: Number of semitone steps
            
        Returns:
            Pitch-shifted audio
        """
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
        """
        Time stretching.
        
        Args:
            y: Audio signal
            rate: Stretch rate (< 1.0: slower, > 1.0: faster)
            
        Returns:
            Time-stretched audio
        """
        return librosa.effects.time_stretch(y, rate=rate)
    
    @staticmethod
    def drop_frames(
        y: np.ndarray,
        sr: int,
        drop_rate: float = 0.1
    ) -> np.ndarray:
        """
        Randomly drop frames.
        
        Args:
            y: Audio signal
            sr: Sample rate
            drop_rate: Fraction of frames to drop
            
        Returns:
            Audio with dropped frames
        """
        frame_length = int(0.01 * sr)  # 10ms frames
        num_frames = len(y) // frame_length
        
        mask = np.ones(num_frames)
        drop_indices = np.random.choice(
            num_frames,
            size=int(num_frames * drop_rate),
            replace=False
        )
        mask[drop_indices] = 0
        
        expanded_mask = np.repeat(mask, frame_length)
        expanded_mask = expanded_mask[:len(y)]
        
        return y[expanded_mask > 0]
