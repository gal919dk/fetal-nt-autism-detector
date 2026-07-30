"""
Train the image-quality classifier.

A small CNN learns to separate Standard fetal ultrasound frames (suitable for
NT measurement) from Non-standard ones.

    python src/train_model.py
    python src/train_model.py --epochs 25
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")  # write plots to file instead of opening a window
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config


def build_model(img_size: int) -> Sequential:
    """A two-block CNN for binary image-quality classification."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu",
                   input_shape=(img_size, img_size, 3)),
            MaxPooling2D(2, 2),

            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),

            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_generators(img_size: int, batch_size: int):
    """Training and validation generators with light augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=config.VALIDATION_SPLIT,
        horizontal_flip=True,
        rotation_range=15,
        brightness_range=(0.85, 1.15),
    )
    # No augmentation on the validation stream - only rescaling.
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=config.VALIDATION_SPLIT,
    )

    common = dict(
        directory=str(config.TRAIN_DIR),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        seed=config.RANDOM_SEED,
    )

    train_gen = train_datagen.flow_from_directory(subset="training", **common)
    val_gen = val_datagen.flow_from_directory(subset="validation", **common)
    return train_gen, val_gen


def plot_history(history, output_path) -> None:
    """Save accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    if not config.TRAIN_DIR.exists():
        raise SystemExit(
            f"Training directory not found: {config.TRAIN_DIR}\n"
            "Set FETUS_TRAIN_DIR or FETUS_DATA_ROOT - see README."
        )

    train_gen, val_gen = build_generators(config.IMG_SIZE, args.batch_size)
    print(f"Class indices: {train_gen.class_indices}")

    model = build_model(config.IMG_SIZE)
    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
    )

    model.save(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

    history_path = config.OUTPUTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2,
        )
    print(f"History saved to {history_path}")

    plot_history(history, config.OUTPUTS_DIR / "training_curves.png")


if __name__ == "__main__":
    main()
