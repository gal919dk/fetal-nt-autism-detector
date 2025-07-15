# Fetal NT Autism Detector 🧠📈

This project detects whether fetal ultrasound images are **Standard** or **Non-Standard**, based on the visibility and clarity of the **Nuchal Translucency (NT)** region, using a Convolutional Neural Network (CNN).

## 💻 Project Description

This model was trained on labeled ultrasound images using a CNN built with Keras and TensorFlow. It learns to distinguish high-quality ("standard") images — which are suitable for NT measurement — from low-quality ("non-standard") ones.

The NT region is automatically annotated based on coordinates from the dataset, and the model outputs whether follow-up is recommended based on NT thickness and image quality.

## 📂 Files in the Repo

- `train_model.py` – Trains the CNN on fetal ultrasound images.
- `evaluate_model_test.py` – Evaluates predictions and displays NT region and decision (follow-up or not).
- `prepare_dataset.py` – Prepares the training and validation sets.
- `extract_nt_measurement.py` – Loads annotations and draws the NT region on the image.
- `model_standard_vs_nonstandard.py` – The trained model architecture and weights (code-based, not `.h5` file).

## 🧪 Medical Background

The **NT (Nuchal Translucency)** is a fluid-filled space at the back of a fetus's neck, typically measured in the first trimester. Abnormal NT thickness may indicate chromosomal or developmental conditions.

This project explores whether deep learning can help pre-screen images that require clinical follow-up — and serves as a proof of concept for potential future prediction of conditions like **autism** based on fetal imaging patterns.

⚠️ **Disclaimer**: This project is educational only. It is not approved for clinical use.

## 🔧 How to Run

```bash
# Train the model
python train_model.py

# Evaluate and visualize predictions
python evaluate_model_test.py
