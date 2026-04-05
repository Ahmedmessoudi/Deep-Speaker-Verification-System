"""Audio utilities for preprocessing and feature extraction."""

from typing import Tuple

import librosa
import numpy as np
import torch


def load_audio(
    audio_path: str,
    sr: int = 16000,
    duration: float = 3.0,
    mono: bool = True
) -> np.ndarray:
    """
    Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        duration: Duration in seconds (None for full audio)
        mono: Convert to mono
        
    Returns:
        Audio waveform as numpy array
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=mono)
    
    if duration is not None:
        max_samples = int(sr * duration)
        if len(y) > max_samples:
            y = y[:max_samples]
        elif len(y) < max_samples:
            y = np.pad(y, (0, max_samples - len(y)), mode='constant')
    
    return y


def extract_mfcc(
    y: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160
) -> np.ndarray:
    """
    Extract MFCC features.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length
        
    Returns:
        MFCC feature matrix (n_mfcc, time_steps)
    """
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return mfcc


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    f_min: int = 50,
    f_max: int = 7600
) -> np.ndarray:
    """
    Extract Mel spectrogram.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop length
        f_min: Minimum frequency
        f_max: Maximum frequency
        
    Returns:
        Mel spectrogram (n_mels, time_steps)
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=f_min,
        fmax=f_max
    )
    
    # Convert to log scale
    S_db = librosa.power_to_db(S, ref=np.max)
    
    return S_db


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features (zero mean, unit variance).
    
    Args:
        features: Feature matrix
        
    Returns:
        Normalized features
    """
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True)
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    normalized = (features - mean) / std
    
    return normalized


def add_noise(
    y: np.ndarray,
    noise: np.ndarray,
    snr_db: float = 10.0
) -> np.ndarray:
    """
    Add noise to audio signal at specified SNR.
    
    Args:
        y: Clean audio signal
        noise: Noise signal
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Noisy audio signal
    """
    # Match length
    if len(noise) > len(y):
        noise = noise[:len(y)]
    elif len(noise) < len(y):
        noise = np.tile(noise, (len(y) // len(noise) + 1))[:len(y)]
    
    # Calculate power
    signal_power = np.mean(y ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate scaling factor
    snr_linear = 10 ** (snr_db / 10.0)
    noise_scale = np.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
    
    # Add scaled noise
    noisy_y = y + noise_scale * noise
    
    return noisy_y


def apply_cmn(features: np.ndarray) -> np.ndarray:
    """
    Apply Cepstral Mean Normalization (CMN).
    
    Args:
        features: Feature matrix (n_features, time_steps)
        
    Returns:
        CMN-normalized features
    """
    mean = np.mean(features, axis=1, keepdims=True)
    
    return features - mean


def apply_cmvn(features: np.ndarray) -> np.ndarray:
    """
    Apply Cepstral Mean and Variance Normalization (CMVN).
    
    Args:
        features: Feature matrix (n_features, time_steps)
        
    Returns:
        CMVN-normalized features
    """
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True)
    
    std[std == 0] = 1.0
    
    normalized = (features - mean) / std
    
    return normalized
