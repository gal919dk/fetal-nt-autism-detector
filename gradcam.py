"""
Grad-CAM explainability for the image-quality classifier.

Produces a heatmap showing which regions of an ultrasound frame drove the
model's Standard / Non-standard decision, overlaid on the original image.

    python src/gradcam.py --image path/to/frame.png
    python src/gradcam.py --directory path/to/folder --limit 10
"""

import argparse

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

import config


def find_last_conv_layer(model) -> str:
    """Return the name of the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found in the model.")


def compute_heatmap(model, image_batch, layer_name: str) -> np.ndarray:
    """Compute a normalised Grad-CAM heatmap for a single-item batch."""
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_batch)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_output[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0).numpy()

    peak = heatmap.max()
    if peak > 0:
        heatmap = heatmap / peak
    return heatmap


def overlay(original_bgr, heatmap: np.ndarray, alpha: float = 0.45):
    """Blend a JET-coloured heatmap over the original image."""
    h, w = original_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    coloured = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(original_bgr, 1 - alpha, coloured, alpha, 0)


def process_image(model, layer_name, image_path, output_dir):
    """Run Grad-CAM on one image and save the overlay."""
    original = cv2.imread(str(image_path))
    if original is None:
        print(f"  Could not read {image_path}")
        return None

    resized = cv2.resize(original, (config.IMG_SIZE, config.IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(rgb / 255.0, axis=0)

    probability = float(model.predict(batch, verbose=0)[0][0])
    predicted = (
        "Standard"
        if probability >= config.CLASSIFICATION_THRESHOLD
        else "Non-standard"
    )

    heatmap = compute_heatmap(model, batch, layer_name)
    blended = overlay(original, heatmap)

    output_path = output_dir / f"gradcam_{image_path.stem}.png"
    cv2.imwrite(str(output_path), blended)

    print(f"  {image_path.name}: {predicted} ({probability:.3f}) -> {output_path.name}")
    return {"filename": image_path.name, "prediction": predicted,
            "probability": probability}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Single image to explain")
    group.add_argument("--directory", help="Folder of .png images to explain")
    parser.add_argument("--limit", type=int, help="Max images when using --directory")
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    args = parser.parse_args()

    model_path = config.Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}\nRun train_model.py first.")

    model = load_model(model_path)
    layer_name = find_last_conv_layer(model)
    print(f"Using conv layer: {layer_name}")

    output_dir = config.OUTPUTS_DIR / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        process_image(model, layer_name, config.Path(args.image), output_dir)
    else:
        directory = config.Path(args.directory)
        if not directory.exists():
            raise SystemExit(f"Directory not found: {directory}")
        images = sorted(directory.glob("*.png"))
        if args.limit:
            images = images[: args.limit]
        print(f"Processing {len(images)} image(s)...")
        for image_path in images:
            process_image(model, layer_name, image_path, output_dir)

    print(f"\nOverlays written to {output_dir}")


if __name__ == "__main__":
    main()
