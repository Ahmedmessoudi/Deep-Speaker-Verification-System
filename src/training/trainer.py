"""Model training module."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import get_loss_function


class Trainer:
    """Speaker verification model trainer."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_norm: float = 5.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to use
            max_norm: Gradient clipping max norm
        """
        self.model = model
        self.device = device
        self.max_norm = max_norm
        self.logger = logging.getLogger(__name__)
        
        self.model.to(device)
    
    def setup_training(
        self,
        loss_type: str,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0001,
        **loss_kwargs
    ) -> Tuple[nn.Module, optim.Optimizer]:
        """
        Set up loss function and optimizer.
        
        Args:
            loss_type: Type of loss function
            learning_rate: Learning rate
            weight_decay: Weight decay
            **loss_kwargs: Additional loss function arguments
            
        Returns:
            Tuple of (loss_function, optimizer)
        """
        # Get loss function
        if loss_type == "crossentropy":
            loss_fn = get_loss_function(loss_type)
        else:
            embedding_dim = self.model.embeddings_dim
            num_speakers = self.model.num_speakers
            loss_fn = get_loss_function(
                loss_type,
                embedding_dim=embedding_dim,
                num_speakers=num_speakers,
                **loss_kwargs
            )
        
        loss_fn.to(self.device)
        
        # Get optimizer
        optimizer = optim.Adam(
            list(self.model.parameters()) + list(loss_fn.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        return loss_fn, optimizer
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (features, speaker_ids) in enumerate(progress_bar):
            features = features.to(self.device)
            speaker_ids = speaker_ids.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if loss_fn.__class__.__name__ in ["AAMSoftmaxLoss", "ArcFaceLoss", "CosFaceLoss"]:
                # These loss functions expect embeddings
                embeddings = self.model.extract_embedding(features)
                logits = self.model.classifier(embeddings)
                loss = loss_fn(embeddings, speaker_ids)
            else:
                # Standard loss (crossentropy)
                logits = self.model(features)
                loss = loss_fn(logits, speaker_ids)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_norm
                )
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            if not loss_fn.__class__.__name__ in ["AAMSoftmaxLoss", "ArcFaceLoss", "CosFaceLoss"]:
                _, predicted = torch.max(logits, 1)
                correct = (predicted == speaker_ids).sum().item()
                total_correct += correct
            else:
                _, predicted = torch.max(logits.data, 1)
                correct = (predicted == speaker_ids).sum().item()
                total_correct += correct
            
            total_samples += speaker_ids.size(0)
            
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': total_correct / total_samples
            })
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for features, speaker_ids in progress_bar:
                features = features.to(self.device)
                speaker_ids = speaker_ids.to(self.device)
                
                # Forward pass
                if loss_fn.__class__.__name__ in ["AAMSoftmaxLoss", "ArcFaceLoss", "CosFaceLoss"]:
                    embeddings = self.model.extract_embedding(features)
                    logits = self.model.classifier(embeddings)
                    loss = loss_fn(embeddings, speaker_ids)
                else:
                    logits = self.model(features)
                    loss = loss_fn(logits, speaker_ids)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                correct = (predicted == speaker_ids).sum().item()
                total_correct += correct
                total_samples += speaker_ids.size(0)
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': total_correct / total_samples
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        loss_type: str = "aamsoftmax",
        learning_rate: float = 0.01,
        weight_decay: float = 0.0001,
        save_dir: Optional[str] = None,
        early_stopping: bool = True,
        patience: int = 10,
        **loss_kwargs
    ) -> Dict:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            loss_type: Type of loss function
            learning_rate: Learning rate
            weight_decay: Weight decay
            save_dir: Directory to save checkpoints
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            **loss_kwargs: Loss function kwargs
            
        Returns:
            Training history
        """
        # Setup
        loss_fn, optimizer = self.setup_training(
            loss_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **loss_kwargs
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Training loop
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(
                train_loader,
                loss_fn,
                optimizer,
                scheduler
            )
            
            # Validate
            val_metrics = self.validate(val_loader, loss_fn)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            self.logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}\n"
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Save checkpoint
            if save_dir:
                checkpoint_path = Path(save_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, optimizer, loss_fn)
            
            # Early stopping
            if early_stopping:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    
                    if save_dir:
                        best_checkpoint_path = Path(save_dir) / "best_model.pt"
                        self.save_checkpoint(best_checkpoint_path, epoch, optimizer, loss_fn)
                else:
                    patience_counter += 1
                    self.logger.info(f"Patience: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        self.logger.info("Early stopping triggered")
                        break
        
        return history
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer
            loss_fn: Loss function
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_fn_state_dict': loss_fn.state_dict() if hasattr(loss_fn, 'state_dict') else None
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Epoch from checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {path}")
        
        return checkpoint['epoch']
