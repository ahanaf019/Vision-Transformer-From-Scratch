# Vision Transformer (ViT) from Scratch

This repository provides a clean and minimal implementation of the **Vision Transformer (ViT)** architecture from scratch using  **PyTorch** . It is designed for learning and experimentation with the Transformer-based approach to image classification, without relying on pre-built ViT modules.

---

## 📌 Features

* ✅ Vision Transformer (ViT) architecture implemented from first principles
* ✅ Patch embedding using a simple linear projection
* ✅ Multi-head Self-Attention mechanism
* ✅ Position embeddings
* ✅ Transformer encoder blocks with LayerNorm and residual connections
* ✅ Classification head for image classification
* ✅ End-to-end training pipeline using PyTorch
* ✅ DataLoader with augmentation for classification

---

## 📁 Directory Structure

```
Vision-Transformer-From-Scratch/
├── datasets/                             # Dataset handling utilities
│   ├── classification_dataset.py         # Custom Dataset class for image classification tasks
│   └── __init__.py                       # Package initializer
│
├── models/                               # Model architecture components
│   ├── vit.py                            # Vision Transformer (ViT) wrapper combining all modules
│   └── modules/                          # Core building blocks of the ViT model
│       ├── attention.py                  # Implements scaled dot-product attention
│       ├── multihead_attention.py        # Implements multi-head self-attention
│       ├── patch_embedding.py            # Splits image into patches and embeds them
│       ├── positional_embedding.py       # Adds positional information to patch embeddings
│       └── transformer_encoder.py        # Transformer encoder block with attention and MLP
│
├── trainers/                             # Training logic
│   ├── classification_trainer.py         # Training loop and evaluation logic for classification
│   └── __init__.py                       # Package initializer
│
├── utils/                                # Helper utilities
│   └── utils.py                          # General-purpose utility functions (e.g., metric calculation, logging)
│
├── config.py                             # Centralized configuration for hyperparameters and paths
├── main.py                               # Entry point for training the Vision Transformer
├── ReadMe.md                             # Project documentation (you're here!)
└── requirements.txt			  # Required Packages
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ahanaf019/Vision-Transformer-From-Scratch.git
cd Vision-Transformer-From-Scratch
```

### 2. Install dependencies

We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python main.py
```

---

## 🧠 Model Overview

The Vision Transformer (ViT) follows this architecture:

1. **Image to Patches** : Splits image into fixed-size patches (e.g., 16x16).
2. **Patch Embedding** : Each patch is flattened and projected into a vector.
3. **Position Embedding** : Positional information is added to retain spatial order.
4. **Transformer Encoder** : A stack of transformer blocks (Multi-Head Attention + MLP).
5. **Classification Head** : A linear layer for final prediction using the `[CLS]` token.

---

## ⚙️ Configuration

You can customize training parameters via `config.py`:

```python
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_CLASSES = 10
DIM = 256
DEPTH = 6
HEADS = 8
MLP_DIM = 512
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
```

---

## 🧪 Dataset

This implementation uses **STL-10** by default, which should be already downloaded inside `DB_PATH`.

To switch datasets, modify the `datasets/classification_dataset.py` file accordingly.

---

## 🙌 Acknowledgements

Implemented from the original [ViT paper by Dosovitskiy et al.](https://arxiv.org/abs/2010.11929).
