"""X-Vector model implementation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    """
    Time Delay Neural Network (TDNN) layer.
    Implements convolution with fixed context.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout_rate: float = 0.0
    ):
        """
        Initialize TDNN layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            kernel_size: Kernel size for temporal convolution
            dilation: Dilation factor
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding to maintain size
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim, time_steps)
            
        Returns:
            Output tensor (batch_size, output_dim, time_steps)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x


class StatsPooling(nn.Module):
    """
    Statistics pooling: compute mean and std over time.
    Output: [mean, std] concatenated.
    """
    
    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        """
        Initialize stats pooling.
        
        Args:
            input_dim: Input dimension
            output_dim: Optional output dimension (not used, for compatibility)
        """
        super().__init__()
        self.input_dim = input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim, time_steps)
            
        Returns:
            Output tensor (batch_size, 2 * input_dim)
        """
        # Compute mean and std over time
        mean = torch.mean(x, dim=2)  # (batch_size, input_dim)
        std = torch.std(x, dim=2)    # (batch_size, input_dim)
        
        # Concatenate
        output = torch.cat([mean, std], dim=1)  # (batch_size, 2 * input_dim)
        
        return output


class XVector(nn.Module):
    """
    X-Vector model for speaker verification.
    
    Architecture:
    - Input: Mel-spectrogram (batch_size, n_mels, time_steps)
    - TDNN blocks with increasing dilation
    - Stats pooling
    - Feed-forward layers
    - Speaker embedding output
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        tdnn_dim: int = 1024,
        num_speakers: int = 5994,
        embeddings_dim: int = 512,
        dropout_rate: float = 0.5
    ):
        """
        Initialize X-Vector model.
        
        Args:
            input_dim: Input dimension (n_mels)
            tdnn_dim: TDNN layer dimension
            num_speakers: Number of speakers
            embeddings_dim: Embedding dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.tdnn_dim = tdnn_dim
        self.num_speakers = num_speakers
        self.embeddings_dim = embeddings_dim
        
        # TDNN layers with increasing dilation
        self.tdnn1 = TDNN(input_dim, tdnn_dim, kernel_size=5, dilation=1, dropout_rate=dropout_rate)
        self.tdnn2 = TDNN(tdnn_dim, tdnn_dim, kernel_size=3, dilation=2, dropout_rate=dropout_rate)
        self.tdnn3 = TDNN(tdnn_dim, tdnn_dim, kernel_size=3, dilation=3, dropout_rate=dropout_rate)
        self.tdnn4 = TDNN(tdnn_dim, tdnn_dim, kernel_size=1, dilation=1, dropout_rate=dropout_rate)
        self.tdnn5 = TDNN(tdnn_dim, 1024, kernel_size=1, dilation=1, dropout_rate=dropout_rate)
        
        # Stats pooling
        self.stats_pool = StatsPooling(1024)
        
        # Feed-forward layers
        self.fc1 = nn.Linear(1024 * 2, embeddings_dim)
        self.bn_fc1 = nn.BatchNorm1d(embeddings_dim)
        self.fc2 = nn.Linear(embeddings_dim, embeddings_dim)
        self.bn_fc2 = nn.BatchNorm1d(embeddings_dim)
        
        # Speaker classifier
        self.classifier = nn.Linear(embeddings_dim, num_speakers)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, n_mels, time_steps) or (batch_size, time_steps, n_mels)
            return_embedding: If True, return embedding instead of logits
            
        Returns:
            Logits (batch_size, num_speakers) or embedding (batch_size, embeddings_dim)
        """
        # Handle input shape
        if x.dim() == 3 and x.shape[-1] != self.input_dim:
            # Might be (batch, time, features) instead of (batch, features, time)
            if x.shape[1] == self.input_dim:
                x = x.transpose(1, 2)
        
        # TDNN blocks
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        
        # Stats pooling
        x = self.stats_pool(x)
        
        # Feed-forward layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Return embedding or classification output
        if return_embedding:
            return x
        
        x = self.classifier(x)
        
        return x
    
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding.
        
        Args:
            x: Input tensor
            
        Returns:
            Speaker embedding (batch_size, embeddings_dim)
        """
        return self.forward(x, return_embedding=True)
