"""Data augmentation module."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..utils import add_noise, load_audio


class DataAugmenter:
    """Data augmentation for speaker verification."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        musan_path: Optional[str] = None,
    ):
        """
        Initialize augmenter.
        
        Args:
            sample_rate: Sample rate
            musan_path: Path to MUSAN dataset
        """
        self.sample_rate = sample_rate
        self.musan_path = musan_path if musan_path else "data/musan"
        self.noise_cache = {}
        self.music_cache = {}
        self.babble_cache = {}
    
    def _load_musan_files(self, category: str) -> list:
        """
        Load MUSAN files for a category.
        
        Args:
            category: Category (noise, music, babble)
            
        Returns:
            List of file paths
        """
        musan_dir = Path(self.musan_path) / category
        
        if not musan_dir.exists():
            print(f"Warning: MUSAN directory not found: {musan_dir}")
            return []
        
        files = sorted(musan_dir.glob("**/*.wav"))
        
        return [str(f) for f in files]
    
    def add_noise_augmentation(
        self,
        y: np.ndarray,
        snr_db: float = 10.0,
        category: str = "noise"
    ) -> np.ndarray:
        """
        Add noise augmentation.
        
        Args:
            y: Clean audio signal
            snr_db: Signal-to-noise ratio in dB
            category: Noise category (noise, music, babble)
            
        Returns:
            Augmented audio signal
        """
        if self.musan_path is None or not Path(self.musan_path).exists():
            print("Warning: MUSAN path not found, skipping noise augmentation")
            return y
        
        # Load random noise file
        noise_files = self._load_musan_files(category)
        
        if not noise_files:
            return y
        
        noise_file = np.random.choice(noise_files)
        
        try:
            noise = load_audio(
                noise_file,
                sr=self.sample_rate,
                duration=None,
                mono=True
            )
        except Exception as e:
            print(f"Error loading noise file {noise_file}: {e}")
            return y
        
        # Add noise at specified SNR
        augmented = add_noise(y, noise, snr_db=snr_db)
        
        return augmented
    
    def speed_perturbation(
        self,
        y: np.ndarray,
        factors: Optional[list] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Speed perturbation (using simple resampling).
        
        Args:
            y: Audio signal
            factors: Speed factors to sample from (e.g., [0.9, 1.0, 1.1])
            
        Returns:
            Augmented audio and applied factor
        """
        if factors is None:
            factors = [0.9, 1.0, 1.1]
        
        factor = np.random.choice(factors)
        
        if factor == 1.0:
            return y, factor
        
        # Simple speed change by resampling
        num_samples = int(len(y) / factor)
        indices = np.linspace(0, len(y) - 1, num_samples)
        augmented = np.interp(indices, np.arange(len(y)), y)
        
        return augmented, factor
    
    def additive_white_gaussian_noise(
        self,
        y: np.ndarray,
        snr_db: float = 10.0
    ) -> np.ndarray:
        """
        Add white Gaussian noise.
        
        Args:
            y: Audio signal
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Noisy audio
        """
        signal_power = np.mean(y ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        
        return y + noise
    
    def time_stretching(
        self,
        y: np.ndarray,
        factor: float = 1.0
    ) -> np.ndarray:
        """
        Simple time stretching via resampling.
        
        Args:
            y: Audio signal
            factor: Stretch factor (< 1.0: slower, > 1.0: faster)
            
        Returns:
            Time-stretched audio
        """
        if factor == 1.0:
            return y
        
        num_samples = int(len(y) / factor)
        indices = np.linspace(0, len(y) - 1, num_samples)
        stretched = np.interp(indices, np.arange(len(y)), y)
        
        return stretched
    
    def augment(
        self,
        y: np.ndarray,
        augmentation_type: str = "noise",
        **kwargs
    ) -> np.ndarray:
        """
        Apply augmentation.
        
        Args:
            y: Audio signal
            augmentation_type: Type of augmentation
            **kwargs: Additional arguments for specific augmentation
            
        Returns:
            Augmented audio
        """
        if augmentation_type == "noise":
            snr_db = kwargs.get("snr_db", 10.0)
            category = kwargs.get("category", "noise")
            return self.add_noise_augmentation(y, snr_db, category)
        
        elif augmentation_type == "speed":
            factors = kwargs.get("factors", [0.9, 1.0, 1.1])
            augmented, _ = self.speed_perturbation(y, factors)
            return augmented
        
        elif augmentation_type == "awgn":
            snr_db = kwargs.get("snr_db", 10.0)
            return self.additive_white_gaussian_noise(y, snr_db)
        
        elif augmentation_type == "time_stretch":
            factor = kwargs.get("factor", np.random.uniform(0.9, 1.1))
            return self.time_stretching(y, factor)
        
        else:
            return y
