"""Training script for speaker verification models."""

import argparse
import logging
from pathlib import Path

import torch
import yaml

from src.data import AudioPreprocessor, DataAugmenter, VoxCelebDataLoader
from src.models import ECAPATDNN, XVector
from src.training import Trainer
from src.utils import setup_logger, load_config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train speaker verification model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, choices=["xvector", "ecapa_tdnn"], default="ecapa_tdnn", help="Model type")
    parser.add_argument("--data_root", type=str, default="data/raw", help="Data root directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    
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
    
    logger.info("Starting training...")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data root: {args.data_root}")
    
    # Get device
    device = config['training']['device']
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    logger.info(f"Using device: {device}")
    
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
        logger.info("Initializing data augmenter...")
        augmenter = DataAugmenter(
            sample_rate=config['data']['sample_rate'],
            musan_path=config['augmentation']['musan_path']
        )
    
    # Load datasets
    logger.info("Loading datasets...")
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
            num_workers=4,
            augmentation_prob=config['augmentation']['prob'],
            augmentation_config={'noise_snr': config['augmentation']['noise_snr']}
        )
        
        val_dataloader = data_loader.get_dataloader(
            split="val",
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            augmentation_prob=0.0
        )
    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        logger.error("Please ensure VoxCeleb dataset is available at the specified path")
        return
    
    # Initialize model
    logger.info(f"Initializing {args.model} model...")
    
    if args.model == "xvector":
        model = XVector(
            input_dim=config['model']['xvector']['input_dim'],
            tdnn_dim=config['model']['xvector']['tdnn_dim'],
            num_speakers=config['model']['xvector']['num_speakers'],
            embeddings_dim=config['model']['xvector']['embeddings_dim']
        )
    else:
        model = ECAPATDNN(
            input_dim=config['model']['ecapa_tdnn']['input_dim'],
            num_channels=config['model']['ecapa_tdnn']['num_channels'],
            num_speakers=config['model']['ecapa_tdnn']['num_speakers'],
            embeddings_dim=config['model']['ecapa_tdnn']['embeddings_dim']
        )
    
    # Initialize trainer
    trainer = Trainer(model=model, device=device)
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    logger.info("Starting training loop...")
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
    import json
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
