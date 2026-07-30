"""
Dataset preparation and inspection.

This script does three things:
  1. Verifies that the dataset directories exist and are structured correctly.
  2. Reports the class balance (how many Standard vs Non-standard images).
  3. Optionally carves a held-out test set out of the training directory.

Run it once before training:

    python src/prepare_dataset.py
    python src/prepare_dataset.py --make-test-split 0.15
"""

import argparse
import random
import shutil
from pathlib import Path

import config


def count_images(directory: Path) -> dict:
    """Count .png images per class folder."""
    counts = {}
    if not directory.exists():
        return counts
    for class_dir in sorted(directory.iterdir()):
        if class_dir.is_dir():
            counts[class_dir.name] = len(list(class_dir.glob("*.png")))
    return counts


def report(directory: Path, label: str) -> None:
    counts = count_images(directory)
    print(f"\n{label}: {directory}")
    if not counts:
        print("  (missing or empty)")
        return
    total = sum(counts.values())
    for class_name, n in counts.items():
        share = (n / total * 100) if total else 0
        print(f"  {class_name:<16} {n:>5} images  ({share:.1f}%)")
    print(f"  {'TOTAL':<16} {total:>5} images")


def make_test_split(source: Path, dest: Path, fraction: float) -> None:
    """Move a random fraction of each class folder into a test directory."""
    random.seed(config.RANDOM_SEED)
    dest.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue

        images = sorted(class_dir.glob("*.png"))
        n_test = int(len(images) * fraction)
        if n_test == 0:
            print(f"  {class_dir.name}: too few images to split, skipping")
            continue

        selected = random.sample(images, n_test)
        target_dir = dest / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image in selected:
            shutil.move(str(image), str(target_dir / image.name))

        print(f"  {class_dir.name}: moved {n_test} images to {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--make-test-split",
        type=float,
        metavar="FRACTION",
        help="Move this fraction of the training images into the test directory "
             "(e.g. 0.15). This MOVES files - back up your data first.",
    )
    args = parser.parse_args()

    print(config.describe())

    report(config.TRAIN_DIR, "Training / validation set")
    report(config.TEST_DIR, "Test set")

    if args.make_test_split:
        if not 0 < args.make_test_split < 1:
            parser.error("--make-test-split must be between 0 and 1")
        print(f"\nCarving out a {args.make_test_split:.0%} test split...")
        make_test_split(config.TRAIN_DIR, config.TEST_DIR, args.make_test_split)
        report(config.TRAIN_DIR, "Training set after split")
        report(config.TEST_DIR, "Test set after split")

    if not config.ANNOTATION_FILE.exists():
        print(f"\nWarning: annotation file not found at {config.ANNOTATION_FILE}")
        print("NT measurement (extract_nt_measurement.py) needs this file.")


if __name__ == "__main__":
    main()
