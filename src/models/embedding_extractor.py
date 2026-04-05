"""Embedding extraction and speaker verification utilities."""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


class EmbeddingExtractor:
    """Extract speaker embeddings from models."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Initialize extractor.
        
        Args:
            model: Speaker verification model
            device: Device to run model on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract embedding from input.
        
        Args:
            x: Input tensor (batch_size, n_mels, time_steps)
            
        Returns:
            Embeddings (batch_size, embedding_dim)
        """
        with torch.no_grad():
            x = x.to(self.device)
            embedding = self.model.extract_embedding(x)
            embedding = embedding.cpu().numpy()
        
        return embedding
    
    def extract_batch(self, batch: torch.Tensor) -> np.ndarray:
        """
        Extract embeddings from batch.
        
        Args:
            batch: Batch tensor
            
        Returns:
            Embeddings array
        """
        return self.extract(batch)


class CosineDistance:
    """Cosine distance similarity metric."""
    
    @staticmethod
    def compute(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        # Normalize
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(similarity)
    
    @staticmethod
    def compute_batch(
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarities.
        
        Args:
            embeddings1: Embeddings array 1 (n, dim)
            embeddings2: Embeddings array 2 (m, dim)
            
        Returns:
            Similarity matrix (n, m)
        """
        # Normalize
        emb1_norm = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-10)
        emb2_norm = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-10)
        
        # Pairwise cosine similarity
        similarities = np.dot(emb1_norm, emb2_norm.T)
        
        return similarities


class SpeakerVerifier:
    """Speaker verification using embeddings."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        threshold: float = 0.5
    ):
        """
        Initialize verifier.
        
        Args:
            model: Speaker verification model
            device: Device to run model on
            threshold: Decision threshold for verification
        """
        self.extractor = EmbeddingExtractor(model, device)
        self.threshold = threshold
        self.distance_metric = CosineDistance()
    
    def verify(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor
    ) -> Tuple[float, bool]:
        """
        Verify if two audio samples are from the same speaker.
        
        Args:
            audio1: First audio features
            audio2: Second audio features
            
        Returns:
            Tuple of (similarity_score, is_same_speaker)
        """
        # Extract embeddings
        emb1 = self.extractor.extract(audio1.unsqueeze(0))
        emb2 = self.extractor.extract(audio2.unsqueeze(0))
        
        # Compute similarity
        similarity = self.distance_metric.compute(emb1[0], emb2[0])
        
        # Verify
        is_same_speaker = similarity > self.threshold
        
        return similarity, is_same_speaker
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set decision threshold.
        
        Args:
            threshold: New threshold
        """
        self.threshold = threshold
    
    def get_threshold(self) -> float:
        """Get current threshold."""
        return self.threshold
