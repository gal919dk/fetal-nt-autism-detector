# Fetal NT Analysis — Ultrasound Quality Classification & NT Measurement

Final project — Afeka Academic College of Engineering
**Gal Shemesh** · **Ron Roitman**

A two-stage deep-learning pipeline for first-trimester fetal ultrasound:
a CNN first decides whether a frame is of sufficient quality for Nuchal
Translucency (NT) measurement, and NT thickness is then extracted from the
accepted frames. Grad-CAM overlays show which image regions drove each decision.

> **Disclaimer.** This is an academic proof of concept. It is not a medical
> device, has not been clinically validated, and must not be used for diagnosis
> or patient care.

---

## What the system does

| Stage | Component | Output |
|-------|-----------|--------|
| 1 | **Quality classifier** — CNN (Keras/TensorFlow) | `Standard` / `Non-standard` + confidence |
| 2 | **NT measurement** — bounding-box annotations | NT height in pixels + risk flag |
| 3 | **Explainability** — Grad-CAM | Heatmap overlay on the source frame |

Frames classified `Non-standard` are rejected before measurement, since NT
cannot be measured reliably outside the correct sagittal plane.

---

## Repository layout

```
.
├── src/
│   ├── config.py                  # All paths and hyper-parameters
│   ├── prepare_dataset.py         # Verify dataset, report class balance, make test split
│   ├── train_model.py             # Train the quality classifier
│   ├── evaluate_model.py          # Metrics + confusion matrix on held-out test set
│   ├── extract_nt_measurement.py  # NT height from annotations -> CSV
│   └── gradcam.py                 # Grad-CAM heatmap overlays
├── requirements.txt
├── .env.example                   # Template for local dataset paths
├── LICENSE
└── README.md
```

Generated at runtime (git-ignored): `models/`, `outputs/`.

---

## Dataset

Trained on the public **Dataset for Fetus Framework**, which provides
`Standard` / `Non-standard` ultrasound frames plus `ObjectDetection.xlsx`
containing NT bounding-box coordinates (`structure`, `fname`, `w_min`,
`h_min`, `w_max`, `h_max`).

The dataset is **not** included in this repository. Download it separately and
point the project at it.

---

## Setup

```bash
git clone https://github.com/gal919dk/fetal-nt-autism-detector.git
cd fetal-nt-autism-detector

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Tell the code where your dataset lives:

```bash
export FETUS_DATA_ROOT="/path/to/Dataset for Fetus Framework"
```

Check that everything resolves correctly:

```bash
python src/config.py
```

---

## Usage

```bash
# 1. Inspect the dataset and class balance
python src/prepare_dataset.py

#    Optionally carve out a 15% held-out test set (this MOVES files)
python src/prepare_dataset.py --make-test-split 0.15

# 2. Train the classifier
python src/train_model.py --epochs 25

# 3. Evaluate on the test set
python src/evaluate_model.py

# 4. Extract NT measurements
python src/extract_nt_measurement.py --save-images

# 5. Generate Grad-CAM explanations
python src/gradcam.py --directory "$FETUS_DATA_ROOT/test/Standard" --limit 10
```

### Outputs

| File | Produced by |
|------|-------------|
| `models/model_standard_vs_nonstandard.h5` | `train_model.py` |
| `outputs/training_curves.png` | `train_model.py` |
| `outputs/predictions_report.csv` | `evaluate_model.py` |
| `outputs/confusion_matrix.png` | `evaluate_model.py` |
| `outputs/nt_measurements.csv` | `extract_nt_measurement.py` |
| `outputs/gradcam/` | `gradcam.py` |

---

## Model

**Quality classifier** — sequential CNN, 128×128×3 input:

```
Conv2D(32, 3×3) → MaxPool(2×2)
Conv2D(64, 3×3) → MaxPool(2×2)
Flatten → Dense(128) → Dropout(0.3) → Dense(1, sigmoid)
```

Adam optimiser, binary cross-entropy, batch size 32. Training uses horizontal
flip, ±15° rotation and brightness jitter; validation uses rescaling only.

---

## Medical background

Nuchal Translucency is the fluid-filled space at the back of the fetal neck,
measured routinely in the first trimester (≈11–14 weeks) as part of screening
for chromosomal abnormalities. Measurement is highly plane-dependent and
operator-dependent — an incorrect imaging plane yields an unreliable value,
which is the problem the quality classifier addresses.

Some literature reports statistical associations between increased NT and
later neurodevelopmental outcomes, but these associations are weak and
non-specific. **This project does not predict autism.** It automates image
quality control and NT measurement, and maps measurements onto risk zones
described in the literature.

---

## Known limitations

- NT is reported in **pixels**, not millimetres — converting requires DICOM
  pixel-spacing metadata that the public dataset does not include.
- The NT stage reads existing bounding-box annotations; it does not yet
  segment the NT region from raw pixels.
- Single public dataset, no external validation.
- Trained and evaluated on CPU-scale data volumes.

---

## License

MIT — see [LICENSE](LICENSE).
