"""
Prepare a mini dataset for local testing.

Extracts a small subset (100 voice elements) from VoxCeleb.zip
and a subset of noise files from Musan.zip, then applies noise
augmentation to create an augmented training set.

Usage:
    cd speaker_verification
    python scripts/prepare_mini_dataset.py

This does NOT modify the original zip files or the full dataset config.
"""

import os
import sys
import random
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------
# Configuration
# ----------------------------------------------
NUM_SPEAKERS = 10           # Number of speakers to select
UTTERANCES_PER_SPEAKER = 10 # Utterances per speaker (total = 100)
TRAIN_RATIO = 0.70          # 70 for train
VAL_RATIO = 0.15            # 15 for val
TEST_RATIO = 0.15           # 15 for test
MAX_NOISE_FILES = 20        # Number of MUSAN noise files to extract
SNR_LEVELS = [0, 10, 20]    # dB levels for augmentation
SAMPLE_RATE = 16000
SEED = 42

# Paths (relative to project root)
VOXCELEB_ZIP = PROJECT_ROOT / "data" / "raw" / "Voxceleb.zip"
MUSAN_ZIP = PROJECT_ROOT / "data" / "musan" / "Musan.zip"

OUTPUT_RAW = PROJECT_ROOT / "data" / "raw_mini"
OUTPUT_MUSAN = PROJECT_ROOT / "data" / "musan_mini"
OUTPUT_AUGMENTED = PROJECT_ROOT / "data" / "augmented_mini"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def extract_voxceleb_mini():
    """
    Extract a mini subset of VoxCeleb wav files from the zip archive.

    The zip contains wav files under:
        vox1_test_wav/wav/id{XXXXX}/session_id/NNNNN.wav

    We select NUM_SPEAKERS speakers, UTTERANCES_PER_SPEAKER each,
    and split them into train/val/test.
    """
    print("=" * 60)
    print("STEP 1: Extracting mini VoxCeleb subset")
    print("=" * 60)

    if not VOXCELEB_ZIP.exists():
        print(f"ERROR: VoxCeleb zip not found at {VOXCELEB_ZIP}")
        print("Please place Voxceleb.zip in data/raw/")
        sys.exit(1)

    with zipfile.ZipFile(VOXCELEB_ZIP, 'r') as zf:
        # Find all wav files
        all_wavs = [n for n in zf.namelist() if n.endswith('.wav')]
        print(f"  Found {len(all_wavs)} total .wav files in archive")

        if not all_wavs:
            print("ERROR: No .wav files found in VoxCeleb archive!")
            sys.exit(1)

        # Group by speaker
        speaker_files = defaultdict(list)
        for wav_path in all_wavs:
            parts = wav_path.split('/')
            # Expected: vox1_test_wav/wav/id{XXXXX}/session/file.wav
            speaker_id = None
            for part in parts:
                if part.startswith('id'):
                    speaker_id = part
                    break
            if speaker_id:
                speaker_files[speaker_id].append(wav_path)

        all_speakers = sorted(speaker_files.keys())
        print(f"  Found {len(all_speakers)} speakers with wav files")

        # Select speakers with enough utterances
        eligible_speakers = [
            s for s in all_speakers
            if len(speaker_files[s]) >= UTTERANCES_PER_SPEAKER
        ]

        if len(eligible_speakers) < NUM_SPEAKERS:
            print(f"  WARNING: Only {len(eligible_speakers)} speakers have "
                  f">= {UTTERANCES_PER_SPEAKER} utterances. Using all of them.")
            selected_speakers = eligible_speakers
        else:
            selected_speakers = random.sample(eligible_speakers, NUM_SPEAKERS)

        selected_speakers.sort()
        print(f"  Selected {len(selected_speakers)} speakers: {selected_speakers}")

        # Select utterances for each speaker
        selected_files = []
        for speaker in selected_speakers:
            files = speaker_files[speaker]
            chosen = random.sample(files, min(UTTERANCES_PER_SPEAKER, len(files)))
            selected_files.extend([(f, speaker) for f in chosen])

        random.shuffle(selected_files)
        total = len(selected_files)
        print(f"  Selected {total} utterances total")

        # Split into train/val/test
        n_train = int(total * TRAIN_RATIO)
        n_val = int(total * VAL_RATIO)
        # Rest goes to test
        splits = {
            'train': selected_files[:n_train],
            'val': selected_files[n_train:n_train + n_val],
            'test': selected_files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            print(f"  {split_name}: {len(split_files)} files")

        # Extract files to output directory
        if OUTPUT_RAW.exists():
            print(f"  Cleaning existing output directory: {OUTPUT_RAW}")
            shutil.rmtree(OUTPUT_RAW)

        for split_name, split_files in splits.items():
            for zip_path, speaker_id in split_files:
                # Determine the original filename
                original_filename = Path(zip_path).name
                # Find the session from the zip path
                parts = zip_path.split('/')
                # Find speaker index, session is the next part
                speaker_idx = None
                for i, part in enumerate(parts):
                    if part == speaker_id:
                        speaker_idx = i
                        break

                if speaker_idx is not None and speaker_idx + 1 < len(parts) - 1:
                    session = parts[speaker_idx + 1]
                else:
                    session = "session0"

                # Output path: data/raw_mini/{split}/wav/{speaker_id}/{session}/{file}.wav
                out_path = OUTPUT_RAW / split_name / "wav" / speaker_id / session / original_filename
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract file
                data = zf.read(zip_path)
                with open(out_path, 'wb') as f:
                    f.write(data)

        print(f"  [OK] VoxCeleb mini dataset extracted to: {OUTPUT_RAW}")
        print(f"     Total size: {_get_dir_size_mb(OUTPUT_RAW):.1f} MB")

    return splits


def extract_musan_mini():
    """
    Extract a small subset of MUSAN noise files from the zip archive.

    The zip contains noise files under:
        musan/noise/free-sound/noise-free-sound-NNNN.wav
    """
    print("\n" + "=" * 60)
    print("STEP 2: Extracting mini MUSAN noise subset")
    print("=" * 60)

    if not MUSAN_ZIP.exists():
        print(f"ERROR: MUSAN zip not found at {MUSAN_ZIP}")
        print("Please place Musan.zip in data/musan/")
        sys.exit(1)

    with zipfile.ZipFile(MUSAN_ZIP, 'r') as zf:
        # Find noise wav files
        noise_wavs = [
            n for n in zf.namelist()
            if n.endswith('.wav') and 'noise' in n
        ]
        print(f"  Found {len(noise_wavs)} noise files in archive")

        # Select a subset
        selected_noise = random.sample(
            noise_wavs,
            min(MAX_NOISE_FILES, len(noise_wavs))
        )
        print(f"  Selected {len(selected_noise)} noise files")

        # Clean output directory
        if OUTPUT_MUSAN.exists():
            shutil.rmtree(OUTPUT_MUSAN)

        # Extract files
        for zip_path in selected_noise:
            # Reconstruct relative path under musan_mini
            # Original: musan/noise/free-sound/file.wav
            # Output:   data/musan_mini/noise/free-sound/file.wav
            rel_path = zip_path
            if rel_path.startswith('musan/'):
                rel_path = rel_path[len('musan/'):]

            out_path = OUTPUT_MUSAN / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            data = zf.read(zip_path)
            with open(out_path, 'wb') as f:
                f.write(data)

        print(f"  [OK] MUSAN noise subset extracted to: {OUTPUT_MUSAN}")
        print(f"     Total size: {_get_dir_size_mb(OUTPUT_MUSAN):.1f} MB")

    return selected_noise


def augment_with_noise(splits: dict):
    """
    Apply MUSAN noise augmentation to the training split.

    For each training file, creates augmented copies at different SNR levels.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Augmenting training data with MUSAN noise")
    print("=" * 60)

    # Load noise files
    noise_dir = OUTPUT_MUSAN / "noise"
    if not noise_dir.exists():
        print("  WARNING: No noise directory found, skipping augmentation")
        return

    noise_files = list(noise_dir.glob("**/*.wav"))
    if not noise_files:
        print("  WARNING: No noise files found, skipping augmentation")
        return

    print(f"  Using {len(noise_files)} noise files")

    # Clean output directory
    if OUTPUT_AUGMENTED.exists():
        shutil.rmtree(OUTPUT_AUGMENTED)
    OUTPUT_AUGMENTED.mkdir(parents=True, exist_ok=True)

    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("  WARNING: librosa or soundfile not installed.")
        print("  Install with: pip install librosa soundfile")
        print("  Skipping augmentation step.")
        return

    # Process training files
    train_dir = OUTPUT_RAW / "train" / "wav"
    if not train_dir.exists():
        print(f"  ERROR: Training directory not found: {train_dir}")
        return

    train_wavs = list(train_dir.glob("**/*.wav"))
    print(f"  Processing {len(train_wavs)} training files...")

    augmented_count = 0
    for wav_path in train_wavs:
        try:
            # Load clean audio
            y_clean, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)

            if len(y_clean) == 0:
                continue

            for snr_db in SNR_LEVELS:
                # Pick a random noise file
                noise_path = random.choice(noise_files)
                y_noise, _ = librosa.load(str(noise_path), sr=SAMPLE_RATE, mono=True)

                if len(y_noise) == 0:
                    continue

                # Match noise length to signal length
                if len(y_noise) < len(y_clean):
                    # Repeat noise to match
                    repeats = (len(y_clean) // len(y_noise)) + 1
                    y_noise = np.tile(y_noise, repeats)
                y_noise = y_noise[:len(y_clean)]

                # Add noise at specified SNR
                signal_power = np.mean(y_clean ** 2)
                noise_power = np.mean(y_noise ** 2)

                if noise_power == 0:
                    continue

                snr_linear = 10 ** (snr_db / 10.0)
                scale = np.sqrt(signal_power / (snr_linear * noise_power))
                y_augmented = y_clean + scale * y_noise

                # Normalize to prevent clipping
                max_val = np.max(np.abs(y_augmented))
                if max_val > 1.0:
                    y_augmented = y_augmented / max_val * 0.99

                # Save augmented file
                # Preserve speaker structure: augmented_mini/{speaker}/{session}/{file}_snr{X}.wav
                rel_path = wav_path.relative_to(train_dir)
                stem = wav_path.stem
                out_name = f"{stem}_snr{snr_db}dB.wav"
                out_path = OUTPUT_AUGMENTED / rel_path.parent / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)

                sf.write(str(out_path), y_augmented, SAMPLE_RATE)
                augmented_count += 1

        except Exception as e:
            print(f"  WARNING: Error processing {wav_path.name}: {e}")
            continue

    print(f"  [OK] Created {augmented_count} augmented files")
    print(f"     Output directory: {OUTPUT_AUGMENTED}")
    print(f"     Total size: {_get_dir_size_mb(OUTPUT_AUGMENTED):.1f} MB")


def _get_dir_size_mb(path: Path) -> float:
    """Get directory size in MB."""
    total = 0
    for f in path.rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def print_summary():
    """Print final summary of the mini dataset."""
    print("\n" + "=" * 60)
    print("MINI DATASET SUMMARY")
    print("=" * 60)

    for name, path in [
        ("Raw mini (VoxCeleb)", OUTPUT_RAW),
        ("MUSAN mini (noise)", OUTPUT_MUSAN),
        ("Augmented mini", OUTPUT_AUGMENTED),
    ]:
        if path.exists():
            wavs = list(path.rglob("*.wav"))
            size = _get_dir_size_mb(path)
            print(f"  {name}:")
            print(f"    Path: {path}")
            print(f"    Files: {len(wavs)} .wav files")
            print(f"    Size: {size:.1f} MB")
        else:
            print(f"  {name}: NOT CREATED")

    # Show splits
    for split in ['train', 'val', 'test']:
        split_dir = OUTPUT_RAW / split / "wav"
        if split_dir.exists():
            wavs = list(split_dir.rglob("*.wav"))
            speakers = set()
            for w in wavs:
                # Get speaker dir name
                rel = w.relative_to(split_dir)
                speakers.add(rel.parts[0])
            print(f"\n  {split} split: {len(wavs)} files, {len(speakers)} speakers")

    print("\n" + "=" * 60)
    print("HOW TO USE:")
    print("=" * 60)
    print("  Train with mini dataset:")
    print("    python scripts/train_mini.py --model ecapa_tdnn --epochs 5")
    print("")
    print("  Or use the main script with mini config:")
    print("    python scripts/train.py --config config/config_mini.yaml \\")
    print("        --data_root data/raw_mini --model ecapa_tdnn --epochs 5")
    print("=" * 60)


def main():
    """Main entry point."""
    print("[*] Speaker Verification - Mini Dataset Preparation")
    print(f"   Selecting {NUM_SPEAKERS} speakers x {UTTERANCES_PER_SPEAKER} "
          f"utterances = {NUM_SPEAKERS * UTTERANCES_PER_SPEAKER} voice elements")
    print(f"   Seed: {SEED}")
    print()

    set_seed(SEED)

    # Step 1: Extract VoxCeleb subset
    splits = extract_voxceleb_mini()

    # Step 2: Extract MUSAN noise subset
    extract_musan_mini()

    # Step 3: Augment training data
    augment_with_noise(splits)

    # Summary
    print_summary()


if __name__ == "__main__":
    main()
