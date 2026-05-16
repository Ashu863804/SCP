# Skin Cancer Detection — Phase 2 (EfficientNetB0)

Multiclass skin lesion classification on **HAM10000** using **EfficientNetB0** transfer learning (frozen backbone + custom head). This repository mirrors the original Kaggle notebook pipeline, organized for **Cursor IDE**, **GitHub**, and **Kaggle GPU** training.

## Project structure

```text
skin-cancer-detection/
├── notebooks/
│   └── phase2_training.ipynb    # Clean training notebook (calls src/)
├── src/
│   ├── config.py                  # Paths & hyperparameters
│   ├── dataset.py                 # HAM10000 splits & generators
│   ├── model.py                   # EfficientNetB0 builder
│   ├── train.py                   # Training & fine-tuning
│   ├── evaluate.py                # Metrics & reports
│   └── utils.py                   # Plots & helpers
├── outputs/
│   ├── models/
│   ├── plots/
│   └── reports/
├── requirements.txt
└── README.md
```

## Model (Phase 2)

| Component | Setting |
|-----------|---------|
| Backbone | EfficientNetB0 (ImageNet, `include_top=False`, **frozen**) |
| Head | GAP → BatchNorm → Dense(256, relu) → Dropout(0.5) → Softmax(7) |
| Loss | `CategoricalCrossentropy(label_smoothing=0.1)` |
| Optimizer | Adam `lr=3e-5` (fine-tune: `1e-5`, last 40 layers) |
| Augmentation | rotation 20°, zoom 0.2, horizontal flip + EfficientNet preprocess |
| Class weights | Balanced, smoothed with `power=0.5` |

## Workflow overview

```text
Cursor IDE  →  GitHub (Ashu863804/SCP)  →  Kaggle notebook SCP01  →  clone + src/
```

**Repository:** https://github.com/Ashu863804/SCP

---

## Kaggle notebook **SCP01** (step-by-step)

1. Open your Kaggle notebook **SCP01**.
2. **Settings** → **Accelerator** → **GPU** (T4).
3. **Add Data** → add **HAM10000** (e.g. *Skin Cancer MNIST: HAM10000* by kmader).
4. **Internet** → **On** (required for `git clone`).
5. Copy cells from `notebooks/phase2_training.ipynb` into SCP01, **or** run that file after clone.
6. Run **cell 1** — it always runs:
   ```bash
   rm -rf /kaggle/working/SCP
   git clone https://github.com/Ashu863804/SCP.git /kaggle/working/SCP
   ```
7. Confirm output: `src exists: True`.
8. Run remaining cells (config → dataset → train → evaluate).

**Before each Kaggle session:** push latest code from your PC:

```bash
cd skin-cancer-detection
git add .
git commit -m "your message"
git push origin main
```

Then re-run SCP01 cell 1 on Kaggle to pull the fresh clone.

---

## 1. Clone and develop locally (Cursor)

```bash
git clone https://github.com/Ashu863804/SCP.git
cd SCP
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Place HAM10000 under `data/ham10000-dataset/` **or** set:

```bash
set HAM10000_INPUT_DIR=C:\path\to\ham10000
set HAM10000_ORGANIZED_DIR=C:\path\to\ham10000_organized
```

Open `notebooks/phase2_training.ipynb` in Cursor and run cells (CPU is fine for smoke tests; full training needs GPU).

---

## 2. Push to GitHub

```bash
cd skin-cancer-detection
git add .
git commit -m "Phase 2: modular EfficientNetB0 HAM10000 pipeline"
git push origin main
```

Remote should be: `https://github.com/Ashu863804/SCP.git`

Do **not** commit raw images or large `.h5` weights (see `.gitignore`).

---

## 3. Train on Kaggle (GPU)

### A. Create a Kaggle notebook

1. Go to [kaggle.com](https://www.kaggle.com) → **Code** → **New Notebook**.
2. **Settings** → turn **GPU** on (e.g. T4).
3. **Add Data** → search **HAM10000** (e.g. *Skin Cancer MNIST: HAM10000* by kmader) and add it to the notebook.

### B. Clone repo on Kaggle (SCP01 cell 1)

```python
!rm -rf /kaggle/working/SCP
!git clone https://github.com/Ashu863804/SCP.git /kaggle/working/SCP
```

Turn **Internet ON** in notebook settings.

### C. Run training cells

- Cell 1 clones to `/kaggle/working/SCP` and adds it to `sys.path` before any `import src`.
- If you see `ModuleNotFoundError: No module named 'src'`, run cell 1 again and check `src exists: True`.
- Open the notebook from the cloned repo, or copy its cells into your Kaggle notebook.
- The notebook calls `src/` modules; no large training logic lives in the notebook itself.
- Organized images are written to `/kaggle/working/ham10000_organized/`.
- Artifacts are saved under `/kaggle/working/outputs/` (download from the **Output** tab).

### D. Kaggle dataset path

The code auto-detects common HAM10000 input paths:

- `/kaggle/input/ham10000-dataset/`
- `/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/`

If your dataset slug differs, set in the notebook:

```python
config.kaggle_dataset_slug = "your-dataset-slug"
config.__post_init__()  # refresh paths
```

---

## 4. Outputs

After training:

| Artifact | Location |
|----------|----------|
| Best checkpoint | `outputs/models/best_efficientnet.h5` |
| Final model | `outputs/models/efficientnet_phase2.h5` |
| Training curves | `outputs/plots/training_history.png` |
| Confusion matrix | `outputs/plots/confusion_matrix.png` |
| Classification report | `outputs/reports/classification_report.txt` |

---

## Future phases (not implemented yet)

- Grad-CAM explainability  
- Segmentation  
- Ensemble models  
- Flask / smartphone deployment  

The `src/` layout is ready for these extensions without rewriting the notebook.

---

## License

Academic / FYP use. HAM10000 dataset terms apply on Kaggle.
