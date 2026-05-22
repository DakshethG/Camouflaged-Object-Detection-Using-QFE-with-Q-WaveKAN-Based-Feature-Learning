# QFE-COD: Quantum Frequency-Enhanced Camouflaged Object Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Quantum ML](https://img.shields.io/badge/Quantum-PennyLane-purple.svg)
![Status](https://img.shields.io/badge/Status-Research%20Project-success.svg)

### Hybrid Quantum-Classical Architecture for Camouflaged Object Detection using Frequency Enhancement and Q-WaveKAN Feature Learning

</div>

---

# 📌 Overview

QFE-COD (Quantum Frequency-Enhanced Camouflaged Object Detection) is a novel hybrid quantum-classical deep learning framework designed for detecting and segmenting camouflaged objects from highly complex natural scenes.

Traditional CNN-based and Transformer-based segmentation models struggle when foreground objects visually blend with their surroundings in terms of:
- Texture
- Color
- Edge patterns
- Spatial continuity
- Frequency similarity

To overcome these limitations, QFE-COD introduces a multi-domain feature learning pipeline that combines:
- Pyramid Vision Transformers (PVTv2)
- Discrete Wavelet Transform (DWT)
- Frequency Spatial Attention (FSA)
- Hybrid Quantum-Classical Modules (HQCM)
- Mamba-inspired State Space Models (SSM)
- Q-WaveKAN segmentation head

---

# 🧠 Key Features

## ✅ Dual-Stage Detection Pipeline

### Stage 1 — CAM / Non-CAM Classification
A lightweight PVTv2-B2 classifier filters images without camouflaged objects.

### Stage 2 — QFE-COD Segmentation
A heavy segmentation network performs precise pixel-level segmentation only on relevant images.

---

## ✅ Frequency Domain Feature Learning

QFE-COD applies Haar-based Discrete Wavelet Transform (DWT) on deep feature maps:
- LL → Low-frequency structure
- LH → Horizontal edges
- HL → Vertical edges
- HH → Diagonal textures

---

## ✅ Hybrid Quantum-Classical Processing

The model introduces an 8-qubit variational quantum circuit using PennyLane.

The HQCM:
- Encodes frequency vectors into qubit rotations
- Applies entanglement operations
- Learns high-dimensional frequency correlations
- Produces quantum-enhanced embeddings

---

## ✅ Mamba-Inspired Spatial Modeling

Instead of expensive Transformer attention:
- Linear-complexity Selective State Space Models (SSM) are used
- Captures long-range dependencies efficiently
- Improves spatial continuity understanding

---

## ✅ Q-WaveKAN Segmentation Head

Inspired by Kolmogorov-Arnold Networks (KAN):
- Uses learnable spline activations
- Dynamically adapts nonlinearities
- Improves boundary refinement

---

# 🏗️ Architecture

```text
Input Image
      │
      ▼
Stage 1: CAM / Non-CAM Classifier
      │
      ├── Non-CAM → Empty Mask
      │
      └── CAM
           │
           ▼
Stage 2: QFE-COD Segmentation
           │
           ▼
PVTv2-B4 Backbone
           │
           ├── Spatial Features
           │
           └── DWT Frequency Decomposition
                     │
                     ▼
          Frequency Spatial Attention
                     │
                     ▼
          Hybrid Quantum Circuit
                     │
                     ▼
         Quantum Frequency Embeddings
                     │
                     ▼
         Mamba Spatial Decoder
                     │
                     ▼
         Dual-Domain Feature Fusion
                     │
                     ▼
             Q-WaveKAN Head
                     │
                     ▼
          Final Segmentation Mask
```

---

# 📂 Repository Structure

```bash
QFE-COD/
│
├── datasets/
├── models/
├── utils/
├── train/
├── inference/
├── checkpoints/
├── notebooks/
├── results/
├── README.md
├── requirements.txt
└── setup.py
```

---

# 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| Dice Coefficient | Segmentation overlap quality |
| IoU | Intersection over Union |
| MAE | Mean Absolute Error |
| Precision | Positive prediction accuracy |
| Recall | Detection completeness |

---

# 📈 Results

| Model | Dice ↑ | IoU ↑ | MAE ↓ |
|---|---|---|---|
| SINetV2 | 0.742 | 0.661 | 0.048 |
| QFE-COD | **0.8145** | **0.7321** | **0.0312** |

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/QFE-COD.git
cd QFE-COD
```

## Create Environment

```bash
conda create -n qfecod python=3.10
conda activate qfecod
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

## Stage 1 — Classifier

```bash
python train/train_classifier.py
```

## Stage 2 — QFE-COD Segmentation

```bash
python train/train_qfecod.py
```

---

# 🔍 Inference

```bash
python inference/predict.py \
    --image sample.jpg \
    --checkpoint checkpoints/qfecod.pth
```

---

# 📚 Research Contributions

✅ Quantum frequency processing for COD  
✅ First KAN-based segmentation head for COD  
✅ Frequency-spatial hybrid learning pipeline  
✅ Quantum-enhanced wavelet feature representation  
✅ Mamba-based efficient decoder  

---

# 👨‍💻 Author

## Daksheth G
B.Tech Computer Science & Engineering  
Vellore Institute of Technology

Areas of Interest:
- Quantum Machine Learning
- Computer Vision
- Deep Learning
- AI Research

---

# 📜 Citation

```bibtex
@article{qfecod2026,
  title={QFE-COD: Quantum Frequency-Enhanced Camouflaged Object Detection},
  author={Daksheth G and Sathya K},
  journal={Research Project},
  year={2026}
}
```

---

# 📄 License

This project is licensed under the MIT License.
