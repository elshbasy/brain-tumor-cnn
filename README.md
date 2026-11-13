# Brain Tumor Classification from MRI Scans

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/brain-tumor-cnn-pytorch.ipynb)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![MRI scans](https://img.shields.io/badge/Dataset-7%2C200%20MRI%20scans-0F9D9D)

Classifying brain MRI scans into four classes — **glioma**, **meningioma**, **no tumor**, **pituitary** — by training three convolutional neural networks of increasing sophistication in PyTorch, comparing them on a held-out test set, and explaining their predictions with Grad-CAM heatmaps.

The project is built as a learning progression rather than a single "best" model: each model exposes a classic deep-learning failure mode, and the next one fixes it.

| Stage | Model | Strategy | What it demonstrates |
|---|---|---|---|
| 1 | SimpleCNN | Trained from scratch, no regularization, no augmentation | Textbook overfitting — 99.6% of its 25.7M parameters sit in one fully-connected layer |
| 2 | ImprovedCNN | Augmentation + BatchNorm + Dropout + global average pooling + weight decay | The same task solved with **66x fewer parameters** and much better generalization |
| 3 | ResNet18 (transfer learning) | Two-phase fine-tuning of an ImageNet-pretrained network | Transfer learning outperforms both from-scratch models |

The whole project lives in a single notebook: `brain-tumor-cnn-pytorch.ipynb`.

---

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle, Nickparvar) — 7,200 MRI scans in 4 balanced classes (1,800 per class).

- The dataset is downloaded automatically with `kagglehub` — **no Kaggle account or API key needed**.
- The original `Training` / `Testing` folders are merged and re-split randomly: **70% train / 15% validation / 15% test** (5,040 / 1,080 / 1,080 images), fixed seed 42, so the split is reproducible.
- Images are resized to 224x224 and normalized with ImageNet statistics (required by the pretrained ResNet18).

## Models

### Model 1 — SimpleCNN (from scratch)

```text
Input (3x224x224)
 -> Conv(3->32)   + ReLU + MaxPool      # 224 -> 112
 -> Conv(32->64)  + ReLU + MaxPool      # 112 -> 56
 -> Conv(64->128) + ReLU + MaxPool      # 56 -> 28
 -> Flatten -> Linear(100,352->256) -> ReLU -> Linear(256->4)
```

**25,784,644 parameters** — the huge `Flatten` layer (`128 x 28 x 28 = 100,352` features) makes the first fully-connected layer alone hold 99.6% of all parameters. Trained for 15 epochs (lr 0.001) on non-augmented data. Expected to reach high training accuracy with a large train/validation gap: the classic overfitting case, kept in the project on purpose.

### Model 2 — ImprovedCNN (from scratch)

```text
Input (3x224x224)
 -> Conv(3->32)   + BatchNorm + ReLU + MaxPool + Dropout(0.1)
 -> Conv(32->64)  + BatchNorm + ReLU + MaxPool + Dropout(0.1)
 -> Conv(64->128) + BatchNorm + ReLU + MaxPool + Dropout(0.2)
 -> Conv(128->256)+ BatchNorm + ReLU + AdaptiveAvgPool(1)   # global average pooling
 -> Flatten -> Dropout(0.4) -> Linear(256->4)
```

**390,404 parameters** — global average pooling removes the giant fully-connected layer (66x fewer parameters than Model 1). Trained for 25 epochs (lr 0.001, weight decay 1e-4) on augmented data (random horizontal/vertical flips, rotation up to 30 degrees, color jitter).

### Model 3 — ResNet18 (transfer learning)

ImageNet-pretrained ResNet18 with the final layer replaced by `Linear(512->4)`, fine-tuned in two phases:

| Phase | What trains | Epochs | Learning rate | Weight decay |
|---|---|---|---|---|
| 1 — head training | new fc layer only (backbone frozen) | 5 | 0.001 | — |
| 2 — full fine-tuning | the whole network (unfrozen) | 8 | 0.0001 | 1e-4 |

**11,178,564 parameters.**

## Training setup

| Setting | Value |
|---|---|
| Image size | 224 x 224 (ImageNet normalization) |
| Batch size | 64 |
| Data split | 70 / 15 / 15 (5,040 / 1,080 / 1,080), seed 42 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Model 1 | 15 epochs, lr 0.001, plain data |
| Model 2 | 25 epochs, lr 0.001, weight decay 1e-4, augmented data |
| Model 3 | 5 + 8 epochs (see table above), augmented data |

Four reusable helpers drive the whole notebook: `train_model` (train/validate loop with history), `test_model` (test accuracy), `plot_history` (loss/accuracy curves), and `check_fit` (prints whether the model underfits, overfits, or fits well, based on the train/validation gap).

## Results

Test accuracy on the held-out 1,080-image test set (printed by the notebook at the end of the run):

| Model | Parameters | Test accuracy |
|---|---:|---|
| Model 1 — SimpleCNN (scratch) | 25,784,644 | _fill in after running the notebook_ |
| Model 2 — ImprovedCNN (scratch) | 390,404 | _fill in after running the notebook_ |
| Model 3 — ResNet18 (fine-tuned) | 11,178,564 | _fill in after running the notebook_ |

For context, published accuracies on this dataset / comparable 4-class MRI tasks:

| Study | Model | Reported accuracy |
|---|---|---|
| OkanNet study (2026) | OkanNet (lightweight CNN) | 88.10% |
| OkanNet study (2026) | ResNet-50 baseline | 96.49% |
| Gorenshtein et al. (2025) | custom CNN from scratch | 87.03% |
| Gorenshtein et al. (2025) | ResNet-18 from scratch | 91.99% |
| Bukaita et al. (2025) | fine-tuned ResNet-18 | 99.7% |
| Mehrdad et al. (2026) | optimized ResNet-50 ensemble | 98.97% |

## Grad-CAM

Beyond raw accuracy, the notebook implements Grad-CAM (Selvaraju et al., 2017) from scratch in PyTorch: a forward hook with `retain_grad()` captures the activations and gradients of the last convolutional layer, the gradients are averaged over channels to get one importance weight per feature-map channel, and the weighted sum of the channels becomes a heatmap that is overlaid on the MRI scan (red = the region that pushed the prediction, blue = ignored).

Two visualizations are produced:

1. Six test images with the best model's Grad-CAM overlay (prediction vs. true label, green/red title).
2. The same scan pushed through Model 2 and Model 3 side by side — showing how the from-scratch CNN and the fine-tuned ResNet18 "look" at different regions of the brain.

## Notebook map

| Cell | Content |
|---|---|
| 1 | Project intro (markdown) |
| 2 | Imports, seed (42), device selection |
| 3 | Dataset download via `kagglehub` + class counts |
| 4 | One sample image per class |
| 5 | Transforms, 70/15/15 split, data loaders |
| 6 | Helpers: `train_model`, `test_model`, `plot_history`, `check_fit` |
| 7 | Model 1: SimpleCNN, 15 epochs |
| 8 | Model 1: learning curves + fit check |
| 9 | Model 2: ImprovedCNN, 25 epochs |
| 10 | Model 2: learning curves + fit check |
| 11 | Model 3: load pretrained ResNet18, freeze, phase 1 (5 epochs) |
| 12 | Model 3: phase 2, unfreeze + fine-tune (8 epochs), combined history |
| 13 | Model 3: learning curves + fit check |
| 14 | Test-set comparison of all three models, best model selected |
| 15 | Bar chart of test accuracies |
| 16 | Validation-accuracy curves of the three models |
| 17 | Confusion matrix + classification report (best model) |
| 18 | Grad-CAM implementation (hooks, heatmap, overlay) |
| 19 | Grad-CAM on six test images |
| 20 | Grad-CAM: Model 2 vs. Model 3 on the same scan |

## Getting started

The easiest way to run the project is Google Colab:

1. Click the **Open in Colab** badge at the top (or upload the notebook to your own Colab).
2. `Runtime -> Change runtime type -> T4 GPU`.
3. `Runtime -> Run all`.

The dataset downloads automatically (~160 MB, anonymous, no Kaggle account). A full run takes about **20-30 minutes** on a T4 GPU (53 epochs of training across the three models).

To run locally instead:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
pip install -r requirements.txt
jupyter lab
```

A CUDA GPU is strongly recommended — training on CPU takes several hours.

## Repository structure

```text
.
|-- brain-tumor-cnn-pytorch.ipynb   # the entire project (data -> 3 models -> evaluation -> Grad-CAM)
|-- requirements.txt                # dependencies for local runs (Colab has them preinstalled)
|-- .gitignore
`-- README.md
```

## References

1. Nickparvar, M. *Brain Tumor MRI Dataset*. Kaggle. <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>
2. *OkanNet: A Lightweight Deep Learning Architecture for Brain Tumor Classification.* arXiv:2604.01264 (2026). <https://arxiv.org/html/2604.01264v>
3. He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition.* CVPR (2016) 770-778.
4. Selvaraju, R. R., et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV (2017) 618-626.
5. Paszke, A., et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 32 (2019).
6. Gorenshtein, A., et al. *Lightweight Transfer Learning Models for Multi-Class Brain Tumor Classification* (2025).
7. Bukaita, W., et al. *Comparative Evaluation of CNN and ResNet18 for Brain Tumor Classification.* Medical Research Archives (2025).
8. Mehrdad, V., et al. *Brain Tumor Classification Using Optimized ResNet50 with an Ensemble Approach.* Scientific Reports (2026).

## Acknowledgments

- Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by M. Nickparvar (Kaggle).
- The three-model progression (simple CNN -> improved CNN -> fine-tuned ResNet) follows a classic computer-vision coursework structure.
