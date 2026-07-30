"""
Evaluate the trained image-quality classifier on the held-out test set.

Produces:
  - a per-image CSV of predictions
  - accuracy, sensitivity, specificity, precision, F1
  - a confusion-matrix figure

    python src/evaluate_model.py
"""

import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import load_model

import config


def preprocess(image_path, img_size: int):
    """Load an image and prepare it as a single-item batch."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.expand_dims(image / 255.0, axis=0)


def predict_directory(model, test_dir, img_size: int) -> pd.DataFrame:
    """Run inference over every class folder in the test directory."""
    rows = []
    skipped = 0

    for class_name in config.CLASS_NAMES:
        folder = test_dir / class_name
        if not folder.exists():
            print(f"Warning: {folder} not found, skipping")
            continue

        true_label = 1 if class_name == "Standard" else 0

        for image_path in sorted(folder.glob("*.png")):
            batch = preprocess(image_path, img_size)
            if batch is None:
                skipped += 1
                continue

            probability = float(model.predict(batch, verbose=0)[0][0])
            rows.append(
                {
                    "filename": image_path.name,
                    "true_class": class_name,
                    "true_label": true_label,
                    "pred_prob": probability,
                    "pred_label": int(probability >= config.CLASSIFICATION_THRESHOLD),
                }
            )

    if skipped:
        print(f"Skipped {skipped} unreadable file(s)")
    return pd.DataFrame(rows)


def report_metrics(df: pd.DataFrame) -> dict:
    """Print and return the standard classification metrics."""
    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )

    print("\n" + "=" * 52)
    print(f"Test images        : {len(df)}")
    print(f"Accuracy           : {accuracy * 100:.1f}%  ({tp + tn}/{len(df)})")
    print(f"Sensitivity (recall): {sensitivity * 100:.1f}%  (Standard correctly kept)")
    print(f"Specificity        : {specificity * 100:.1f}%  (Non-standard correctly rejected)")
    print(f"Precision          : {precision * 100:.1f}%")
    print(f"F1 score           : {f1:.3f}")
    print("=" * 52)
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print("\n" + classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0
    ))

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, output_path) -> None:
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    parser.add_argument("--test-dir", default=str(config.TEST_DIR))
    args = parser.parse_args()

    model_path = config.Path(args.model)
    test_dir = config.Path(args.test_dir)

    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}\nRun train_model.py first."
        )
    if not test_dir.exists():
        raise SystemExit(
            f"Test directory not found: {test_dir}\n"
            "Set FETUS_TEST_DIR, or create a split with prepare_dataset.py."
        )

    model = load_model(model_path)
    df = predict_directory(model, test_dir, config.IMG_SIZE)

    if df.empty:
        raise SystemExit("No test images found.")

    metrics = report_metrics(df)

    csv_path = config.OUTPUTS_DIR / "predictions_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPer-image predictions saved to {csv_path}")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        config.OUTPUTS_DIR / "confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
