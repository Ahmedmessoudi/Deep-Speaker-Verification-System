"""Evaluation script for speaker verification models."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data import AudioPreprocessor
from src.evaluation import SpeakerVerificationMetrics
from src.models import ECAPATDNN, XVector
from src.utils import setup_logger, load_config


def evaluate_model(
    model: torch.nn.Module,
    test_audio_pairs: list,
    preprocessor: AudioPreprocessor,
    device: str = "cpu",
    threshold: float = 0.5
) -> dict:
    """
    Evaluate model on test pairs.
    
    Args:
        model: Model to evaluate
        test_audio_pairs: List of (audio1_path, audio2_path, is_same_speaker)
        preprocessor: Audio preprocessor
        device: Device to use
        threshold: Decision threshold
        
    Returns:
        Evaluation metrics dictionary
    """
    model.eval()
    
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for audio1, audio2, same_speaker in test_audio_pairs:
            try:
                # Process audio
                features1 = preprocessor(audio1)
                features2 = preprocessor(audio2)
                
                # Extract embeddings
                feat1_tensor = torch.FloatTensor(features1).unsqueeze(0).to(device)
                feat2_tensor = torch.FloatTensor(features2).unsqueeze(0).to(device)
                
                emb1 = model.extract_embedding(feat1_tensor)
                emb2 = model.extract_embedding(feat2_tensor)
                
                # Compute similarity
                emb1_norm = emb1 / (torch.norm(emb1, dim=1, keepdim=True) + 1e-10)
                emb2_norm = emb2 / (torch.norm(emb2, dim=1, keepdim=True) + 1e-10)
                
                similarity = torch.sum(emb1_norm * emb2_norm).item()
                
                y_true.append(1 if same_speaker else 0)
                y_scores.append(similarity)
            
            except Exception as e:
                print(f"Error processing {audio1}, {audio2}: {e}")
                continue
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Compute metrics
    metrics = SpeakerVerificationMetrics.evaluate(y_true, y_scores, threshold)
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate speaker verification model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, choices=["xvector", "ecapa_tdnn"], default="ecapa_tdnn", help="Model type")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--test_data", type=str, default="data/test_pairs.txt", help="Test pairs file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger(__name__, level=config['logging']['level'])
    
    logger.info("Starting evaluation...")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Threshold: {args.threshold}")
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=config['data']['sample_rate'],
        duration=config['data']['duration'],
        n_mels=config['data']['n_mels']
    )
    
    # Load model
    logger.info("Loading model...")
    
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load test data
    logger.info("Loading test data...")
    test_pairs = []
    
    if Path(args.test_data).exists():
        with open(args.test_data, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    audio1, audio2, same_speaker = parts[0], parts[1], int(parts[2])
                    test_pairs.append((audio1, audio2, same_speaker))
    else:
        logger.warning(f"Test data file not found: {args.test_data}")
    
    if not test_pairs:
        logger.error("No test pairs found")
        return
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(
        model,
        test_pairs,
        preprocessor,
        device=device,
        threshold=args.threshold
    )
    
    # Print results
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
