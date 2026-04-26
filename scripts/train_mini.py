"""
Training script for speaker verification models — MINI DATASET MODE.

This is a convenience wrapper around the main training pipeline
configured for the mini dataset created by prepare_mini_dataset.py.

Usage:
    cd speaker_verification
    python scripts/train_mini.py --model ecapa_tdnn --epochs 5

The full-dataset training script (train.py) remains unchanged.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import AudioPreprocessor, DataAugmenter, VoxCelebDataLoader
from src.models import ECAPATDNN, XVector
from src.training import Trainer
from src.utils import setup_logger, load_config


def main():
    """Main training function for mini dataset."""
    parser = argparse.ArgumentParser(
        description="Train speaker verification model on MINI dataset"
    )
    parser.add_argument(
        "--config", type=str, default="config/config_mini.yaml",
        help="Config file path (default: config/config_mini.yaml)"
    )
    parser.add_argument(
        "--model", type=str, choices=["xvector", "ecapa_tdnn"],
        default="ecapa_tdnn", help="Model type"
    )
    parser.add_argument(
        "--data_root", type=str, default="data/raw_mini",
        help="Data root directory (default: data/raw_mini)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints_mini/",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples per split (safety cap, default: use all mini files)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    # Setup logging
    log_file = Path(config['logging']['log_file'])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        __name__,
        log_file=str(log_file),
        level=config['logging']['level']
    )

    logger.info("=" * 60)
    logger.info("MINI DATASET TRAINING MODE")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Max samples: {args.max_samples or 'all'}")

    # Get device
    device = config['training']['device']
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    logger.info(f"Using device: {device}")

    # Check that mini dataset exists
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Mini dataset not found at: {data_root}")
        logger.error("Run 'python scripts/prepare_mini_dataset.py' first!")
        return

    # Initialize preprocessor
    logger.info("Initializing data preprocessor...")
    preprocessor = AudioPreprocessor(
        sample_rate=config['data']['sample_rate'],
        duration=config['data']['duration'],
        n_mels=config['data']['n_mels'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        f_min=config['data']['f_min'],
        f_max=config['data']['f_max']
    )

    # Initialize augmenter
    augmenter = None
    if config['augmentation']['enabled']:
        musan_path = config['augmentation']['musan_path']
        if Path(musan_path).exists():
            logger.info("Initializing data augmenter...")
            augmenter = DataAugmenter(
                sample_rate=config['data']['sample_rate'],
                musan_path=musan_path
            )
        else:
            logger.warning(f"MUSAN path not found: {musan_path}, skipping augmentation")

    # Load datasets
    logger.info("Loading mini datasets...")
    data_loader = VoxCelebDataLoader(
        data_root=args.data_root,
        preprocessor=preprocessor,
        augmenter=augmenter
    )

    try:
        train_dataloader = data_loader.get_dataloader(
            split="train",
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,  # Use 0 workers for mini dataset (avoids multiprocessing overhead)
            augmentation_prob=config['augmentation']['prob'],
            augmentation_config={'noise_snr': config['augmentation']['noise_snr']},
            max_samples=args.max_samples
        )

        val_dataloader = data_loader.get_dataloader(
            split="val",
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            augmentation_prob=0.0,
            max_samples=args.max_samples
        )
    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        logger.error("Run 'python scripts/prepare_mini_dataset.py' first!")
        return

    # Determine actual number of speakers from the dataset
    num_speakers = train_dataloader.dataset.num_speakers
    logger.info(f"Detected {num_speakers} speakers in training data")

    # Update config with actual speaker count
    config['model']['xvector']['num_speakers'] = num_speakers
    config['model']['ecapa_tdnn']['num_speakers'] = num_speakers

    # Initialize model
    logger.info(f"Initializing {args.model} model...")

    if args.model == "xvector":
        model = XVector(
            input_dim=config['model']['xvector']['input_dim'],
            tdnn_dim=config['model']['xvector']['tdnn_dim'],
            num_speakers=num_speakers,
            embeddings_dim=config['model']['xvector']['embeddings_dim']
        )
    else:
        model = ECAPATDNN(
            input_dim=config['model']['ecapa_tdnn']['input_dim'],
            num_channels=config['model']['ecapa_tdnn']['num_channels'],
            num_speakers=num_speakers,
            embeddings_dim=config['model']['ecapa_tdnn']['embeddings_dim']
        )

    # Initialize trainer
    trainer = Trainer(model=model, device=device)

    # Create output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    logger.info("Starting training loop...")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Train samples: {len(train_dataloader.dataset)}")
    logger.info(f"  Val samples: {len(val_dataloader.dataset)}")

    history = trainer.train(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=config['training']['num_epochs'],
        loss_type=config['training']['loss_type'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        save_dir=str(output_dir),
        early_stopping=config['training']['early_stopping'],
        patience=config['training']['patience'],
        margin=config['training']['margin'],
        scale=config['training']['scale']
    )

    logger.info("Training completed!")
    logger.info(f"Checkpoints saved to: {output_dir}")

    # Save final history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
