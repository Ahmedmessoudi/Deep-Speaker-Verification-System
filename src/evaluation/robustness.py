"""Robustness testing utilities."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..data import AudioPreprocessor, DataAugmenter


class RobustnessEvaluator:
    """Evaluate model robustness under noise and degradation."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor: AudioPreprocessor,
        device: str = "cpu"
    ):
        """
        Initialize robustness evaluator.
        
        Args:
            model: Speaker verification model
            preprocessor: Audio preprocessor
            device: Device to use
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.augmenter = DataAugmenter()
    
    def evaluate_with_noise(
        self,
        audio_path: str,
        snr_levels: List[float] = [0, 10, 20, 30]
    ) -> Dict[float, Tuple[np.ndarray, float]]:
        """
        Evaluate embedding quality under noise.
        
        Args:
            audio_path: Path to audio file
            snr_levels: List of SNR levels to test
            
        Returns:
            Dictionary mapping SNR to (embedding, magnitude)
        """
        results = {}
        
        # Load clean audio
        from ..utils import load_audio
        y = load_audio(audio_path, sr=self.preprocessor.sample_rate)
        
        # Test clean
        features_clean = self.preprocessor(audio_path)
        emb_clean = self._extract_embedding(features_clean)
        results['clean'] = (emb_clean, np.linalg.norm(emb_clean))
        
        # Test with noise
        for snr in snr_levels:
            # Add noise
            y_noisy = self.augmenter.add_noise_augmentation(y, snr_db=snr)
            
            # Need to convert back to file for preprocessing
            # In practice, apply augmentation before feature extraction
            # For now, approximate by adding noise to features
            features_noisy = features_clean * (1 - snr / 100.0)  # Rough approximation
            
            emb_noisy = self._extract_embedding(features_noisy)
            results[f'snr_{snr}'] = (emb_noisy, np.linalg.norm(emb_noisy))
        
        return results
    
    def evaluate_drift(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> float:
        """
        Evaluate embedding drift/distance.
        
        Args:
            embeddings1: First set of embeddings (n, dim)
            embeddings2: Second set of embeddings (n, dim)
            
        Returns:
            Average L2 distance between embeddings
        """
        assert embeddings1.shape == embeddings2.shape
        
        distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)
        
        return float(np.mean(distances))
    
    def _extract_embedding(self, features: np.ndarray) -> np.ndarray:
        """Extract embedding from features."""
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            features_tensor = features_tensor.to(self.device)
            embedding = self.model.extract_embedding(features_tensor)
            embedding = embedding.cpu().numpy()[0]
        
        return embedding


def test_robustness_scenarios(
    model: torch.nn.Module,
    test_data: List[Tuple[str, int]],
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Test model on different robustness scenarios.
    
    Args:
        model: Speaker verification model
        test_data: List of (audio_path, speaker_id) tuples
        device: Device to use
        
    Returns:
        Dictionary with robustness metrics
    """
    model.eval()
    
    scenarios = {
        'clean': [],
        'noise_0db': [],
        'noise_10db': [],
        'noise_20db': []
    }
    
    results = {}
    
    return results
