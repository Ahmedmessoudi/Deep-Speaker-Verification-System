"""Loss functions for speaker verification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss."""
    
    def __init__(self):
        """Initialize cross-entropy loss."""
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Predicted logits (batch_size, num_speakers)
            targets: Target speaker IDs (batch_size,)
            
        Returns:
            Loss value
        """
        return self.loss_fn(logits, targets)


class AAMSoftmaxLoss(nn.Module):
    """
    AAM-Softmax loss (Additive Angular Margin + Softmax).
    Combines large margin and softmax for better speaker discrimination.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_speakers: int,
        margin: float = 0.2,
        scale: float = 30.0
    ):
        """
        Initialize AAM-Softmax loss.
        
        Args:
            embedding_dim: Embedding dimension
            num_speakers: Number of speakers
            margin: Angular margin
            scale: Scale factor
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.margin = margin
        self.scale = scale
        
        # Weight matrix (speaker centers in embedding space)
        self.weight = nn.Parameter(
            torch.randn(num_speakers, embedding_dim)
        )
        
        # Initialize with uniform distribution
        nn.init.xavier_uniform_(self.weight)
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, embeddings: Tensor, targets: Tensor) -> Tensor:
        """
        Compute AAM-Softmax loss.
        
        Args:
            embeddings: Speaker embeddings (batch_size, embedding_dim)
            targets: Target speaker IDs (batch_size,)
            
        Returns:
            Loss value
        """
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        
        # Compute cosine similarity
        logits = torch.matmul(embeddings_norm, weight_norm.t())
        
        # Apply scale
        logits = logits * self.scale
        
        # Get target logits and add margin
        target_logits = logits.gather(1, targets.view(-1, 1))
        target_logits = target_logits - self.margin
        
        # Update logits with margin
        logits_with_margin = logits.clone()
        logits_with_margin[torch.arange(len(targets)), targets] = target_logits.squeeze(1)
        
        # Compute cross-entropy loss
        loss = self.ce_loss(logits_with_margin, targets)
        
        return loss


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss (Angular margin loss).
    Uses arc length in the embedding space for margin.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_speakers: int,
        margin: float = 0.5,
        scale: float = 64.0
    ):
        """
        Initialize ArcFace loss.
        
        Args:
            embedding_dim: Embedding dimension
            num_speakers: Number of speakers
            margin: Angular margin (in radians)
            scale: Scale factor
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.margin = margin
        self.scale = scale
        
        # Weight matrix
        self.weight = nn.Parameter(
            torch.randn(num_speakers, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, embeddings: Tensor, targets: Tensor) -> Tensor:
        """
        Compute ArcFace loss.
        
        Args:
            embeddings: Speaker embeddings (batch_size, embedding_dim)
            targets: Target speaker IDs (batch_size,)
            
        Returns:
            Loss value
        """
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        
        # Compute cosine similarity
        cosine = torch.matmul(embeddings_norm, weight_norm.t())
        
        # Clamp to prevent numerical issues
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)
        
        # Compute angles
        theta = torch.acos(cosine)
        
        # Add margin
        target_theta = theta.gather(1, targets.view(-1, 1))
        target_theta_margin = target_theta + self.margin
        
        # Update theta with margin
        theta_with_margin = theta.clone()
        theta_with_margin[torch.arange(len(targets)), targets] = target_theta_margin.squeeze(1)
        
        # Compute logits
        logits = torch.cos(theta_with_margin) * self.scale
        
        # Compute loss
        loss = self.ce_loss(logits, targets)
        
        return loss


class CosFaceLoss(nn.Module):
    """
    CosFace loss (Large Margin Cosine Loss).
    Uses cosine margin for speaker discrimination.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_speakers: int,
        margin: float = 0.35,
        scale: float = 64.0
    ):
        """
        Initialize CosFace loss.
        
        Args:
            embedding_dim: Embedding dimension
            num_speakers: Number of speakers
            margin: Cosine margin
            scale: Scale factor
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.margin = margin
        self.scale = scale
        
        # Weight matrix
        self.weight = nn.Parameter(
            torch.randn(num_speakers, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, embeddings: Tensor, targets: Tensor) -> Tensor:
        """
        Compute CosFace loss.
        
        Args:
            embeddings: Speaker embeddings (batch_size, embedding_dim)
            targets: Target speaker IDs (batch_size,)
            
        Returns:
            Loss value
        """
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        
        # Compute cosine similarity
        cosine = torch.matmul(embeddings_norm, weight_norm.t())
        
        # Apply margin to target class
        target_cosine = cosine.gather(1, targets.view(-1, 1)) - self.margin
        
        # Update cosines
        cosine_with_margin = cosine.clone()
        cosine_with_margin[torch.arange(len(targets)), targets] = target_cosine.squeeze(1)
        
        # Apply scale and compute loss
        logits = cosine_with_margin * self.scale
        loss = self.ce_loss(logits, targets)
        
        return loss


def get_loss_function(
    loss_type: str,
    embedding_dim: int,
    num_speakers: int,
    **kwargs
) -> nn.Module:
    """
    Get loss function based on type.
    
    Args:
        loss_type: Type of loss function
        embedding_dim: Embedding dimension
        num_speakers: Number of speakers
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function module
    """
    if loss_type == "crossentropy":
        return CrossEntropyLoss()
    
    elif loss_type == "aamsoftmax":
        margin = kwargs.get("margin", 0.2)
        scale = kwargs.get("scale", 30.0)
        return AAMSoftmaxLoss(
            embedding_dim=embedding_dim,
            num_speakers=num_speakers,
            margin=margin,
            scale=scale
        )
    
    elif loss_type == "arcface":
        margin = kwargs.get("margin", 0.5)
        scale = kwargs.get("scale", 64.0)
        return ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_speakers=num_speakers,
            margin=margin,
            scale=scale
        )
    
    elif loss_type == "cosface":
        margin = kwargs.get("margin", 0.35)
        scale = kwargs.get("scale", 64.0)
        return CosFaceLoss(
            embedding_dim=embedding_dim,
            num_speakers=num_speakers,
            margin=margin,
            scale=scale
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
