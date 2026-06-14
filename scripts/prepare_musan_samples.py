"""Prepare a small MUSAN samples directory with a few files per category.

Usage:
    python scripts/prepare_musan_samples.py --musan-path data/musan --out data/musan_samples --num 3
"""
from pathlib import Path
import shutil
import random
import argparse


def collect_samples(musan_path: Path, out_path: Path, num_per_subcat: int = 3):
    if not musan_path.exists():
        print(f"MUSAN path not found: {musan_path}")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    # iterate top-level categories
    for category_dir in sorted([p for p in musan_path.iterdir() if p.is_dir()]):
        cat_name = category_dir.name
        target_cat = out_path / cat_name
        target_cat.mkdir(parents=True, exist_ok=True)

        # gather wav files recursively
        files = list(category_dir.rglob('*.wav'))
        if not files:
            print(f"No wav files found in {category_dir}")
            continue

        # flatten selection by shuffling and selecting up to num_per_subcat
        random.shuffle(files)
        selected = files[:num_per_subcat]

        for src in selected:
            dst = target_cat / src.name
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"Failed to copy {src} -> {dst}: {e}")

        print(f"Collected {len(selected)} samples for category {cat_name} -> {target_cat}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--musan-path', type=str, default='data/musan', help='Path to full MUSAN dataset')
    parser.add_argument('--out', type=str, default='data/musan_samples', help='Output small musan samples folder')
    parser.add_argument('--num', type=int, default=3, help='Number of samples per top-level category')

    args = parser.parse_args()

    musan_path = Path(args.musan_path)
    out_path = Path(args.out)

    collect_samples(musan_path, out_path, num_per_subcat=args.num)


if __name__ == '__main__':
    main()
