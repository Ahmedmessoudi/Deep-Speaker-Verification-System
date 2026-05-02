"""Extract full VoxCeleb and MUSAN datasets from zip files."""

import argparse
import shutil
import zipfile
from pathlib import Path


def _has_non_zip_contents(path: Path, allowed_zip: str) -> bool:
    for entry in path.iterdir():
        if entry.name == allowed_zip:
            continue
        return True
    return False


def _clear_non_zip_contents(path: Path, allowed_zip: str) -> None:
    for entry in path.iterdir():
        if entry.name == allowed_zip:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _extract(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract full VoxCeleb and MUSAN datasets")
    parser.add_argument("--voxceleb_zip", type=str, default="data/raw/Voxceleb.zip")
    parser.add_argument("--musan_zip", type=str, default="data/musan/Musan.zip")
    parser.add_argument("--raw_out", type=str, default="data/raw")
    parser.add_argument("--musan_out", type=str, default="data/musan")
    parser.add_argument("--force", action="store_true", help="Delete existing extracted data before extracting")

    args = parser.parse_args()

    voxceleb_zip = Path(args.voxceleb_zip)
    musan_zip = Path(args.musan_zip)
    raw_out = Path(args.raw_out)
    musan_out = Path(args.musan_out)

    if not voxceleb_zip.exists():
        raise FileNotFoundError(f"VoxCeleb zip not found: {voxceleb_zip}")
    if not musan_zip.exists():
        raise FileNotFoundError(f"MUSAN zip not found: {musan_zip}")

    raw_out.mkdir(parents=True, exist_ok=True)
    musan_out.mkdir(parents=True, exist_ok=True)

    if _has_non_zip_contents(raw_out, voxceleb_zip.name):
        if not args.force:
            raise RuntimeError(f"{raw_out} is not empty. Use --force to overwrite.")
        _clear_non_zip_contents(raw_out, voxceleb_zip.name)

    if _has_non_zip_contents(musan_out, musan_zip.name):
        if not args.force:
            raise RuntimeError(f"{musan_out} is not empty. Use --force to overwrite.")
        _clear_non_zip_contents(musan_out, musan_zip.name)

    print("Extracting VoxCeleb... this can take a long time.")
    _extract(voxceleb_zip, raw_out)
    print(f"Done: {raw_out}")

    print("Extracting MUSAN...")
    _extract(musan_zip, musan_out)
    print(f"Done: {musan_out}")


if __name__ == "__main__":
    main()
