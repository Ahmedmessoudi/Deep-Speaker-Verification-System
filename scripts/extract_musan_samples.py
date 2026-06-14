#!/usr/bin/env python3
"""Extract a small subset of MUSAN files from a ZIP or a directory.

Usage:
  python scripts/extract_musan_samples.py --zip path/to/musan.zip --dest data/musan_mini --categories noise music speech --n 10
  or
  python scripts/extract_musan_samples.py --dir /path/to/musan --dest data/musan_mini --categories noise music speech --n 10
"""
import argparse
import random
from pathlib import Path
import zipfile
import shutil
import sys


def extract_from_zip(zip_path: Path, dest: Path, categories, n):
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Normalize names inside zip
        all_files = [f for f in z.namelist() if f.lower().endswith('.wav')]

        for cat in categories:
            cat_prefix = f"musan/{cat}/" if any(f.startswith(f"musan/") for f in all_files) else f"{cat}/"
            cat_files = [f for f in all_files if f.startswith(cat_prefix)]
            if not cat_files:
                print(f"No files found in zip for category '{cat}' (tried prefix '{cat_prefix}')")
                continue

            chosen = random.sample(cat_files, min(n, len(cat_files)))
            out_cat = dest / cat
            out_cat.mkdir(parents=True, exist_ok=True)

            for arcname in chosen:
                # Extract to temporary location in memory then write to out_cat preserving filename
                filename = Path(arcname).name
                target = out_cat / filename
                with z.open(arcname) as src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

            print(f"Extracted {len(chosen)} files for category '{cat}' to {out_cat}")


def copy_from_dir(musan_dir: Path, dest: Path, categories, n):
    for cat in categories:
        cat_dir = musan_dir / cat
        if not cat_dir.exists():
            print(f"Category directory not found: {cat_dir}")
            continue

        wavs = sorted([p for p in cat_dir.rglob('*.wav')])
        if not wavs:
            print(f"No WAV files found under {cat_dir}")
            continue

        chosen = random.sample(wavs, min(n, len(wavs)))
        out_cat = dest / cat
        out_cat.mkdir(parents=True, exist_ok=True)
        for src in chosen:
            dst = out_cat / src.name
            shutil.copy2(src, dst)

        print(f"Copied {len(chosen)} files for category '{cat}' to {out_cat}")


def main():
    parser = argparse.ArgumentParser(description="Extract MUSAN samples into a small directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--zip', type=Path, help='Path to musan.zip')
    group.add_argument('--dir', type=Path, help='Path to musan directory')
    parser.add_argument('--dest', type=Path, default=Path('data/musan_mini'), help='Destination directory')
    parser.add_argument('--categories', nargs='+', default=['noise', 'music', 'speech'], help='Categories to extract')
    parser.add_argument('-n', type=int, default=10, help='Number of files per category')

    args = parser.parse_args()

    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    if args.zip:
        zip_path = args.zip
        if not zip_path.exists():
            print(f"ZIP file not found: {zip_path}")
            sys.exit(1)
        extract_from_zip(zip_path, dest, args.categories, args.n)
    else:
        musan_dir = args.dir
        if not musan_dir.exists():
            print(f"MUSAN directory not found: {musan_dir}")
            sys.exit(1)
        copy_from_dir(musan_dir, dest, args.categories, args.n)

    print("Done.")


if __name__ == '__main__':
    main()
