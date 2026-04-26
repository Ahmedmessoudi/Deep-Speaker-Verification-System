"""ECAPA-TDNN model implementation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEResBlock(nn.Module):
    """
    Squeeze-and-Excitation Residual Block (SE-Res2Block).
    Implements channel attention mechanism.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
        dropout_rate: float = 0.0
    ):
        """
        Initialize SE-Res2Block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size for conv
            dilation: Dilation factor
            scale: Number of branches for res2block
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        if out_channels % scale != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by scale ({scale})"
            )
        self.width = out_channels // scale
        
        # Pointwise projection before Res2Net-style split processing
        self.pre_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        # Branches for Res2Net-style processing
        self.branches = nn.ModuleList()
        for _ in range(scale - 1):
            pad = (kernel_size - 1) * dilation // 2

            branch = nn.Sequential(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=pad,
                    bias=False
                ),
                nn.BatchNorm1d(self.width),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.branches.append(branch)

        self.post_conv = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # Squeeze-and-Excitation
        self.se_conv1 = nn.Conv1d(out_channels, out_channels // 2, 1)
        self.se_conv2 = nn.Conv1d(out_channels // 2, out_channels, 1)
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, in_channels, time_steps)
            
        Returns:
            Output tensor (batch_size, out_channels, time_steps)
        """
        # Pre-projection and split into scale chunks
        x_proj = self.pre_conv(x)
        splits = torch.split(x_proj, self.width, dim=1)

        # Res2Net-style hierarchical aggregation across splits
        branch_outputs = [splits[0]]
        if self.scale > 1:
            prev = None
            for i, branch in enumerate(self.branches):
                if i == 0:
                    cur = branch(splits[1])
                else:
                    cur = branch(splits[i + 1] + prev)
                branch_outputs.append(cur)
                prev = cur

        output = torch.cat(branch_outputs, dim=1)
        output = self.post_conv(output)
        
        # Squeeze-and-Excitation
        se = torch.mean(output, dim=2, keepdim=True)  # Global average pooling
        se = self.se_conv1(se)
        se = torch.relu(se)
        se = self.se_conv2(se)
        se = torch.sigmoid(se)
        
        # Channel attention
        output = output * se
        
        # Add shortcut
        shortcut = self.shortcut(x)
        output = output + shortcut
        output = self.relu(output)
        
        return output


class ECAPATDNN(nn.Module):
    """
    ECAPA-TDNN model for speaker verification.
    
    Architecture:
    - Input: Mel-spectrogram (batch_size, n_mels, time_steps)
    - 1D Conv front-end
    - Multiple SE-Res2Blocks with different dilations
    - Multi-scale feature aggregation
    - Speaker embedding output
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        num_channels: int = 1024,
        num_speakers: int = 5994,
        embeddings_dim: int = 192,
        scale: int = 8,
        dropout_rate: float = 0.5
    ):
        """
        Initialize ECAPA-TDNN model.
        
        Args:
            input_dim: Input dimension (n_mels)
            num_channels: Number of channels in residual blocks
            num_speakers: Number of speakers
            embeddings_dim: Embedding dimension
            scale: Number of branches in Res2Block
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.num_speakers = num_speakers
        self.embeddings_dim = embeddings_dim
        
        # Front-end
        self.conv1 = nn.Conv1d(input_dim, num_channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.relu = nn.ReLU()
        
        # SE-Res2Blocks with different dilations
        self.block1 = SEResBlock(
            num_channels, num_channels, kernel_size=3, dilation=1,
            scale=scale, dropout_rate=dropout_rate
        )
        self.block2 = SEResBlock(
            num_channels, num_channels, kernel_size=3, dilation=2,
            scale=scale, dropout_rate=dropout_rate
        )
        self.block3 = SEResBlock(
            num_channels, num_channels, kernel_size=3, dilation=3,
            scale=scale, dropout_rate=dropout_rate
        )
        self.block4 = SEResBlock(
            num_channels, num_channels, kernel_size=3, dilation=4,
            scale=scale, dropout_rate=dropout_rate
        )
        
        # Multi-layer feature aggregation
        self.mfa_conv = nn.Conv1d(
            num_channels * 4,  # Concatenate outputs from all blocks
            num_channels * 3,
            kernel_size=1
        )
        self.mfa_bn = nn.BatchNorm1d(num_channels * 3)
        
        # Global statistics pooling + context gating
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_std_pool = nn.AdaptiveStdPool1d(1)
        
        # Context gating
        self.context_gate_conv = nn.Sequential(
            nn.Conv1d(num_channels * 3 * 2, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, num_channels * 3, 1),
            nn.Sigmoid()
        )
        
        # Speaker embedding
        self.embed_fc = nn.Linear(num_channels * 3 * 2, embeddings_dim)
        self.embed_bn = nn.BatchNorm1d(embeddings_dim)
        
        # Speaker classifier
        self.classifier = nn.Linear(embeddings_dim, num_speakers)
        
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
        # Handle input shape: ensure (batch, n_mels, time_steps)
        if x.dim() == 3 and x.shape[1] != self.input_dim:
            if x.shape[2] == self.input_dim:
                x = x.transpose(1, 2)
        
        # Front-end
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # SE-Res2Blocks
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        
        # Multi-layer feature aggregation
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.mfa_conv(x_cat)
        x = self.mfa_bn(x)
        x = self.relu(x)
        
        # Global statistics pooling with context gating
        mean = self.global_pool(x)  # (batch_size, channels, 1)
        std = self.global_std_pool(x)  # (batch_size, channels, 1)
        
        # Flatten
        mean = mean.squeeze(-1)  # (batch_size, channels)
        std = std.squeeze(-1)    # (batch_size, channels)
        
        # Context gating
        context = torch.cat([mean.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
        context = context.view(x.size(0), -1, 1)
        gate = self.context_gate_conv(context)
        
        # Apply gating
        x = x * gate
        
        # Global statistics
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2, unbiased=False)
        x = torch.cat([mean, std], dim=1)
        
        # Embedding
        x = self.embed_fc(x)
        x = self.embed_bn(x)
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


class AdaptiveStdPool1d(nn.Module):
    """Adaptive standard deviation pooling for 1D."""
    
    def __init__(self, output_size: int = 1):
        """Initialize adaptive std pool."""
        super().__init__()
        self.output_size = output_size
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute std over time."""
        mean = self.avg_pool(x)
        return torch.std(x, dim=2, keepdim=True, unbiased=False)


# Add to nn module for convenience
if not hasattr(nn, 'AdaptiveStdPool1d'):
    nn.AdaptiveStdPool1d = AdaptiveStdPool1d
