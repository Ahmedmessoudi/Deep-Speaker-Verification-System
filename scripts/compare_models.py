"""Compare X-Vector and ECAPA-TDNN checkpoints on the same test pairs."""

import argparse
import logging
import sys
from pathlib import Path
from subprocess import run

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import AudioPreprocessor
from src.evaluation import SpeakerVerificationMetrics
from src.models import ECAPATDNN, XVector
from src.utils import setup_logger, load_config


def evaluate_model(model, test_audio_pairs, preprocessor, device, threshold):
    model.eval()

    y_true = []
    y_scores = []

    with torch.no_grad():
        for audio1, audio2, same_speaker in test_audio_pairs:
            try:
                features1 = preprocessor(audio1)
                features2 = preprocessor(audio2)

                feat1_tensor = torch.FloatTensor(features1).unsqueeze(0).to(device)
                feat2_tensor = torch.FloatTensor(features2).unsqueeze(0).to(device)

                emb1 = model.extract_embedding(feat1_tensor)
                emb2 = model.extract_embedding(feat2_tensor)

                emb1_norm = emb1 / (torch.norm(emb1, dim=1, keepdim=True) + 1e-10)
                emb2_norm = emb2 / (torch.norm(emb2, dim=1, keepdim=True) + 1e-10)

                similarity = torch.sum(emb1_norm * emb2_norm).item()

                y_true.append(1 if same_speaker else 0)
                y_scores.append(similarity)
            except Exception as exc:
                print(f"Error processing {audio1}, {audio2}: {exc}")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    return SpeakerVerificationMetrics.evaluate(y_true, y_scores, threshold)


def load_test_pairs(path):
    pairs = []
    if not Path(path).exists():
        return pairs

    with open(path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 3:
                audio1, audio2, same_speaker = parts[0], parts[1], int(parts[2])
                pairs.append((audio1, audio2, same_speaker))

    return pairs


def load_model(model_name, config, device):
    if model_name == "xvector":
        model = XVector(
            input_dim=config["model"]["xvector"]["input_dim"],
            tdnn_dim=config["model"]["xvector"]["tdnn_dim"],
            num_speakers=config["model"]["xvector"]["num_speakers"],
            embeddings_dim=config["model"]["xvector"]["embeddings_dim"],
        )
    else:
        model = ECAPATDNN(
            input_dim=config["model"]["ecapa_tdnn"]["input_dim"],
            num_channels=config["model"]["ecapa_tdnn"]["num_channels"],
            num_speakers=config["model"]["ecapa_tdnn"]["num_speakers"],
            embeddings_dim=config["model"]["ecapa_tdnn"]["embeddings_dim"],
        )

    model.to(device)
    return model


def maybe_train(args, model_name):
    if not args.auto_train:
        return

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        args.config,
        "--model",
        model_name,
        "--data_root",
        args.data_root,
        "--output_dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
    ]
    run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Compare X-Vector and ECAPA-TDNN models")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--test_data", type=str, default="data/test_pairs.txt", help="Test pairs file")
    parser.add_argument("--xvector_checkpoint", type=str, default="checkpoints/xvector/best_model.pt")
    parser.add_argument("--ecapa_checkpoint", type=str, default="checkpoints/ecapa_tdnn/best_model.pt")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--auto_train", action="store_true", help="Train both models before comparison")
    parser.add_argument("--data_root", type=str, default="data/raw", help="Data root directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs for auto-train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for auto-train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for auto-train")

    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger(__name__, level=config["logging"]["level"])

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    test_pairs = load_test_pairs(args.test_data)
    if not test_pairs:
        logger.error(f"No test pairs found: {args.test_data}")
        return

    maybe_train(args, "xvector")
    maybe_train(args, "ecapa_tdnn")

    preprocessor = AudioPreprocessor(
        sample_rate=config["data"]["sample_rate"],
        duration=config["data"]["duration"],
        n_mels=config["data"]["n_mels"],
    )

    results = {}

    for model_name, checkpoint_path in [
        ("xvector", args.xvector_checkpoint),
        ("ecapa_tdnn", args.ecapa_checkpoint),
    ]:
        if not Path(checkpoint_path).exists():
            logger.error(f"Checkpoint not found for {model_name}: {checkpoint_path}")
            return

        model = load_model(model_name, config, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        metrics = evaluate_model(model, test_pairs, preprocessor, device, args.threshold)
        results[model_name] = metrics

    logger.info("=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)

    for model_name, metrics in results.items():
        logger.info(f"{model_name.upper()}")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("-")


if __name__ == "__main__":
    main()
