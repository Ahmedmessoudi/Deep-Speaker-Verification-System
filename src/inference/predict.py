"""Inference utilities for speaker verification."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.embedding_extractor import EmbeddingExtractor, CosineDistance


class SpeakerVerificationInference:
    """Speaker verification inference pipeline."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        threshold: float = 0.5
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: Speaker verification model
            device: Device to use
            threshold: Decision threshold
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        
        self.extractor = EmbeddingExtractor(model, device)
        self.distance_metric = CosineDistance()
    
    def verify(
        self,
        feature1: np.ndarray,
        feature2: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[float, bool]:
        """
        Verify if two audio samples are from the same speaker.
        
        Args:
            feature1: First audio features (n_mels, time_steps)
            feature2: Second audio features (n_mels, time_steps)
            threshold: Decision threshold (uses default if None)
            
        Returns:
            Tuple of (similarity_score, is_same_speaker)
        """
        if threshold is None:
            threshold = self.threshold
        
        # Convert to tensors
        feat1_tensor = torch.FloatTensor(feature1).unsqueeze(0)
        feat2_tensor = torch.FloatTensor(feature2).unsqueeze(0)
        
        # Extract embeddings
        emb1 = self.extractor.extract(feat1_tensor)
        emb2 = self.extractor.extract(feat2_tensor)
        
        # Compute similarity
        similarity = self.distance_metric.compute(emb1[0], emb2[0])
        
        # Verify
        is_same_speaker = similarity > threshold
        
        return similarity, is_same_speaker
    
    def enroll_speaker(
        self,
        audio_features_list: list
    ) -> np.ndarray:
        """
        Enroll speaker by averaging embeddings.
        
        Args:
            audio_features_list: List of audio feature arrays
            
        Returns:
            Speaker embedding (averaged)
        """
        embeddings = []
        
        for features in audio_features_list:
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            emb = self.extractor.extract(features_tensor)
            embeddings.append(emb[0])
        
        # Average embeddings
        speaker_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        speaker_embedding = speaker_embedding / (np.linalg.norm(speaker_embedding) + 1e-10)
        
        return speaker_embedding
    
    def identify_speaker(
        self,
        test_features: np.ndarray,
        speaker_embeddings: dict,
        top_k: int = 1
    ) -> list:
        """
        Identify speaker from test audio.
        
        Args:
            test_features: Test audio features
            speaker_embeddings: Dictionary of {speaker_id: embedding}
            top_k: Number of top matches to return
            
        Returns:
            List of (speaker_id, similarity_score) tuples
        """
        # Extract test embedding
        test_tensor = torch.FloatTensor(test_features).unsqueeze(0)
        test_emb = self.extractor.extract(test_tensor)[0]
        
        # Compute similarities to all speakers
        similarities = {}
        
        for speaker_id, speaker_emb in speaker_embeddings.items():
            similarity = self.distance_metric.compute(test_emb, speaker_emb)
            similarities[speaker_id] = similarity
        
        # Sort and return top-k
        sorted_speakers = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_speakers[:top_k]
    
    def set_threshold(self, threshold: float) -> None:
        """Set decision threshold."""
        self.threshold = threshold
    
    def get_threshold(self) -> float:
        """Get current threshold."""
        return self.threshold


class SpeakerDatabase:
    """Database for storing speaker enrollments."""
    
    def __init__(self):
        """Initialize speaker database."""
        self.speakers = {}
    
    def enroll(self, speaker_id: str, embedding: np.ndarray) -> None:
        """
        Enroll speaker.
        
        Args:
            speaker_id: Speaker ID
            embedding: Speaker embedding
        """
        self.speakers[speaker_id] = embedding
    
    def remove(self, speaker_id: str) -> None:
        """
        Remove speaker enrollment.
        
        Args:
            speaker_id: Speaker ID
        """
        if speaker_id in self.speakers:
            del self.speakers[speaker_id]
    
    def get(self, speaker_id: str) -> Optional[np.ndarray]:
        """
        Get speaker embedding.
        
        Args:
            speaker_id: Speaker ID
            
        Returns:
            Speaker embedding or None
        """
        return self.speakers.get(speaker_id)
    
    def list_speakers(self) -> list:
        """List all enrolled speakers."""
        return list(self.speakers.keys())
    
    def save(self, path: str) -> None:
        """
        Save database to file.
        
        Args:
            path: File path
        """
        data = {sid: emb.tolist() for sid, emb in self.speakers.items()}
        
        import json
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str) -> None:
        """
        Load database from file.
        
        Args:
            path: File path
        """
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.speakers = {sid: np.array(emb) for sid, emb in data.items()}
