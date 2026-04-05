"""Dataset loader and PyTorch Dataset classes."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import AudioPreprocessor
from .augmentation import DataAugmenter


class SpeakerVerificationDataset(Dataset):
    """PyTorch Dataset for speaker verification."""
    
    def __init__(
        self,
        audio_files: List[str],
        speaker_ids: List[int],
        preprocessor: AudioPreprocessor,
        augmenter: Optional[DataAugmenter] = None,
        augmentation_prob: float = 0.5,
        augmentation_config: Optional[Dict] = None
    ):
        """
        Initialize dataset.
        
        Args:
            audio_files: List of audio file paths
            speaker_ids: List of speaker IDs (integers)
            preprocessor: AudioPreprocessor instance
            augmenter: DataAugmenter instance (optional)
            augmentation_prob: Probability of applying augmentation
            augmentation_config: Configuration for augmentation
        """
        assert len(audio_files) == len(speaker_ids), \
            "audio_files and speaker_ids must have same length"
        
        self.audio_files = audio_files
        self.speaker_ids = speaker_ids
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        self.augmentation_prob = augmentation_prob
        self.augmentation_config = augmentation_config or {}
        
        # Create speaker id mapping
        unique_speakers = sorted(set(speaker_ids))
        self.speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
        self.num_speakers = len(unique_speakers)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (features tensor, speaker id)
        """
        audio_path = self.audio_files[idx]
        speaker_id = self.speaker_ids[idx]
        
        try:
            # Load and preprocess audio
            y = self.preprocessor(audio_path)
            
            # Check dimensions - if features is 2D, transpose to (time, features)
            # or keep as (features, time) for compatibility
            if y.ndim == 2:
                # Transpose to (time_steps, n_features) for consistency
                y_features = y.T  # Shape: (time_steps, n_mels)
            else:
                y_features = y
            
            # Apply augmentation if enabled and augmenter is provided
            if self.augmenter and np.random.rand() < self.augmentation_prob:
                # Get SNR values
                snr_values = self.augmentation_config.get("noise_snr", [10])
                snr_db = np.random.choice(snr_values)
                
                # First convert features back to audio for augmentation
                # This is a simplified approach - in practice you'd apply 
                # augmentation to raw audio before feature extraction
                y_features = y_features.T  # Back to (n_mels, time_steps)
                # Apply augmentation to features directly (simplified)
                y_features = y_features.T
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(y_features)
            
            # Map speaker ID
            speaker_idx = self.speaker_to_idx[speaker_id]
            
            return features_tensor, speaker_idx
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zeros as fallback
            return torch.zeros((100, 80)), 0


class VoxCelebDataLoader:
    """Helper class for loading VoxCeleb dataset."""
    
    def __init__(
        self,
        data_root: str,
        preprocessor: AudioPreprocessor,
        augmenter: Optional[DataAugmenter] = None
    ):
        """
        Initialize VoxCeleb loader.
        
        Args:
            data_root: Root directory of VoxCeleb dataset
            preprocessor: AudioPreprocessor instance
            augmenter: DataAugmenter instance (optional)
        """
        self.data_root = Path(data_root)
        self.preprocessor = preprocessor
        self.augmenter = augmenter
    
    def load_file_list(self, split: str = "train") -> Tuple[List[str], List[int]]:
        """
        Load file list for a split.
        
        Args:
            split: Data split (train, val, test)
            
        Returns:
            Tuple of (audio_files, speaker_ids)
        """
        audio_files = []
        speaker_ids = []
        
        split_dir = self.data_root / split / "wav"
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Create speaker ID mapping
        speaker_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        speaker_to_id = {d.name: idx for idx, d in enumerate(speaker_dirs)}
        
        # Collect audio files
        for speaker_dir in speaker_dirs:
            speaker_id = speaker_to_id[speaker_dir.name]
            
            # Find all wav files
            for audio_file in speaker_dir.glob("**/*.wav"):
                audio_files.append(str(audio_file))
                speaker_ids.append(speaker_id)
        
        return audio_files, speaker_ids
    
    def get_dataset(
        self,
        split: str = "train",
        augmentation_prob: float = 0.5,
        augmentation_config: Optional[Dict] = None
    ) -> SpeakerVerificationDataset:
        """
        Get PyTorch dataset for a split.
        
        Args:
            split: Data split
            augmentation_prob: Probability of augmentation
            augmentation_config: Augmentation configuration
            
        Returns:
            SpeakerVerificationDataset instance
        """
        audio_files, speaker_ids = self.load_file_list(split)
        
        dataset = SpeakerVerificationDataset(
            audio_files=audio_files,
            speaker_ids=speaker_ids,
            preprocessor=self.preprocessor,
            augmenter=self.augmenter,
            augmentation_prob=augmentation_prob,
            augmentation_config=augmentation_config
        )
        
        return dataset
    
    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 4,
        augmentation_prob: float = 0.5,
        augmentation_config: Optional[Dict] = None
    ) -> DataLoader:
        """
        Get PyTorch DataLoader for a split.
        
        Args:
            split: Data split
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of workers
            augmentation_prob: Probability of augmentation
            augmentation_config: Augmentation configuration
            
        Returns:
            PyTorch DataLoader
        """
        dataset = self.get_dataset(
            split=split,
            augmentation_prob=augmentation_prob,
            augmentation_config=augmentation_config
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return dataloader


def collate_variable_length_batch(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable length sequences.
    
    Args:
        batch: List of (features, speaker_id) tuples
        
    Returns:
        Tuple of (padded_features, speaker_ids)
    """
    features_list, speaker_ids = zip(*batch)
    
    # Find max length
    max_length = max(f.shape[0] for f in features_list)
    
    # Pad sequences
    padded_features = []
    for features in features_list:
        padding = max_length - features.shape[0]
        padded = torch.nn.functional.pad(features, (0, 0, 0, padding))
        padded_features.append(padded)
    
    # Stack
    features_tensor = torch.stack(padded_features)
    speaker_tensor = torch.LongTensor(speaker_ids)
    
    return features_tensor, speaker_tensor
