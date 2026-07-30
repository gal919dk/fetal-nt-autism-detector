"""
Extract NT (Nuchal Translucency) measurements from the annotation file.

Reads the bounding-box annotations, computes NT height in pixels for every
annotated image, applies the follow-up threshold, and writes a CSV. Annotated
images can optionally be saved for inspection.

    python src/extract_nt_measurement.py
    python src/extract_nt_measurement.py --save-images --limit 20
"""

import argparse

import cv2
import pandas as pd

import config

REQUIRED_COLUMNS = ["structure", "fname", "w_min", "h_min", "w_max", "h_max"]


def load_nt_annotations(annotation_file, image_dir) -> pd.DataFrame:
    """Load the annotation sheet and keep only NT rows with an image on disk."""
    annotations = pd.read_excel(annotation_file)

    missing = [c for c in REQUIRED_COLUMNS if c not in annotations.columns]
    if missing:
        raise SystemExit(
            f"Annotation file is missing columns: {missing}\n"
            f"Found: {list(annotations.columns)}"
        )

    nt_rows = annotations[annotations["structure"] == "NT"].copy()
    print(f"NT annotations in file        : {len(nt_rows)}")

    available = {p.name for p in image_dir.glob("*.png")}
    nt_rows = nt_rows[nt_rows["fname"].isin(available)]
    print(f"NT annotations with an image  : {len(nt_rows)}")

    return nt_rows


def measure(nt_rows: pd.DataFrame, image_dir, save_images: bool, limit=None):
    """Compute NT height per annotation, optionally saving annotated images."""
    results = []
    output_dir = config.OUTPUTS_DIR / "nt_annotated"
    if save_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    for count, (_, row) in enumerate(nt_rows.iterrows()):
        if limit and count >= limit:
            break

        image_path = image_dir / row["fname"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        x_min, y_min = int(row["w_min"]), int(row["h_min"])
        x_max, y_max = int(row["w_max"]), int(row["h_max"])

        nt_height = y_max - y_min
        nt_width = x_max - x_min
        elevated = nt_height > config.NT_PIXEL_THRESHOLD

        results.append(
            {
                "filename": row["fname"],
                "x_min": x_min, "y_min": y_min,
                "x_max": x_max, "y_max": y_max,
                "nt_height_px": nt_height,
                "nt_width_px": nt_width,
                "risk_flag": "Elevated" if elevated else "Normal",
            }
        )

        if save_images:
            colour = (0, 0, 255) if elevated else (0, 200, 0)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colour, 2)
            cv2.putText(
                image,
                f"NT: {nt_height}px",
                (x_min, max(y_min - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(output_dir / row["fname"]), image)

    if save_images:
        print(f"Annotated images saved to {output_dir}")

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-images", action="store_true",
                        help="Write annotated copies of each image")
    parser.add_argument("--limit", type=int,
                        help="Only process the first N annotations")
    args = parser.parse_args()

    image_dir = config.TRAIN_DIR / "Standard"

    if not config.ANNOTATION_FILE.exists():
        raise SystemExit(f"Annotation file not found: {config.ANNOTATION_FILE}")
    if not image_dir.exists():
        raise SystemExit(f"Image directory not found: {image_dir}")

    nt_rows = load_nt_annotations(config.ANNOTATION_FILE, image_dir)
    if nt_rows.empty:
        raise SystemExit("No usable NT annotations found.")

    df = measure(nt_rows, image_dir, args.save_images, args.limit)

    print("\nNT height (pixels)")
    print(df["nt_height_px"].describe().to_string())

    elevated = (df["risk_flag"] == "Elevated").sum()
    print(
        f"\nFlagged for follow-up (> {config.NT_PIXEL_THRESHOLD} px): "
        f"{elevated} / {len(df)} ({elevated / len(df) * 100:.1f}%)"
    )

    csv_path = config.OUTPUTS_DIR / "nt_measurements.csv"
    df.to_csv(csv_path, index=False)
    print(f"Measurements saved to {csv_path}")


if __name__ == "__main__":
    main()
