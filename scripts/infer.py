"""Inference script for speaker verification."""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import AudioPreprocessor
from src.models import ECAPATDNN, XVector
from src.inference import SpeakerVerificationInference, SpeakerDatabase
from src.utils import setup_logger, load_config


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Speaker verification inference")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, choices=["xvector", "ecapa_tdnn"], default="ecapa_tdnn", help="Model type")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--audio1", type=str, help="First audio file")
    parser.add_argument("--audio2", type=str, help="Second audio file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--mode", choices=["verify", "enroll", "identify"], default="verify", help="Inference mode")
    parser.add_argument("--speaker_id", type=str, help="Speaker ID (for enroll/identify)")
    parser.add_argument("--db_path", type=str, default="speaker_db.json", help="Speaker database path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger(__name__, level=config['logging']['level'])
    
    logger.info(f"Starting inference ({args.mode})...")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Model: {args.model}")
    
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
    
    # Initialize verifier
    verifier = SpeakerVerificationInference(
        model=model,
        device=device,
        threshold=args.threshold
    )
    
    # Load speaker database
    speaker_db = SpeakerDatabase()
    if Path(args.db_path).exists():
        speaker_db.load(args.db_path)
    
    # Perform inference
    if args.mode == "verify":
        if not args.audio1 or not args.audio2:
            logger.error("--audio1 and --audio2 are required for verify mode")
            return
        
        logger.info(f"Verifying: {args.audio1} vs {args.audio2}")
        
        features1 = preprocessor(args.audio1)
        features2 = preprocessor(args.audio2)

        similarity, is_same = verifier.verify(features1, features2, args.threshold)
        
        logger.info(f"Similarity score: {similarity:.4f}")
        logger.info(f"Same speaker: {is_same}")
    
    elif args.mode == "enroll":
        if not args.audio1 or not args.speaker_id:
            logger.error("--audio1 and --speaker_id are required for enroll mode")
            return
        
        logger.info(f"Enrolling speaker {args.speaker_id}...")
        
        features = preprocessor(args.audio1)
        
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        embedding = model.extract_embedding(features_tensor)
        embedding = embedding.detach().cpu().numpy()[0]
        
        # Normalize
        import numpy as np
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        speaker_db.enroll(args.speaker_id, embedding)
        speaker_db.save(args.db_path)
        
        logger.info(f"Speaker {args.speaker_id} enrolled successfully")
    
    elif args.mode == "identify":
        if not args.audio1:
            logger.error("--audio1 is required for identify mode")
            return
        
        logger.info(f"Identifying speaker from: {args.audio1}")
        
        if not speaker_db.list_speakers():
            logger.error("No speakers in database. Please enroll speakers first.")
            return
        
        features = preprocessor(args.audio1)
        
        speaker_embeddings = {sid: speaker_db.get(sid) for sid in speaker_db.list_speakers()}
        
        matches = verifier.identify_speaker(features, speaker_embeddings, top_k=3)
        
        logger.info("Top matches:")
        for speaker_id, similarity in matches:
            logger.info(f"  {speaker_id}: {similarity:.4f}")


if __name__ == "__main__":
    main()
