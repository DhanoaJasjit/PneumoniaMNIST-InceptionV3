# PneumoniaMNIST-InceptionV3
Fine-tuned InceptionV3 model for PneumoniaMNIST classification, achieving AUC 0.9190 and 48% Normal recall, tailored for medical imaging tasks like PROSPECT’s ultrasound analysis.

# PneumoniaMNIST Classification with InceptionV3

This repository contains a fine-tuned InceptionV3 model for classifying chest X-rays from the PneumoniaMNIST dataset (Normal vs. Pneumonia), developed as part of the PROSPECT project application for chronic pancreatitis evaluation using convolutional neural networks.

## Project Overview
- **Objective**: Classify chest X-rays to distinguish Normal from Pneumonia cases, addressing class imbalance and optimizing for clinical reliability.
- **Model**: InceptionV3 (pre-trained on ImageNet) with custom dense layers (128, 64 units, ReLU, L2 regularization), batch normalization, and dropout (0.5, 0.3). Fine-tuned last 50 layers.
- **Dataset**: PneumoniaMNIST (3,882 train, 524 val, 624 test images; 1,134 Normal, 2,748 Pneumonia).
- **Key Features**:
  - 3x oversampling of Normal class (~1,152 images, total 5,034) to address imbalance.
  - Augmentations: flip, brightness (±0.1), contrast (0.8–1.2), hue (±0.05), rotation (0–360°), zoom (0.9–1.1), random crop.
  - Training: 11/20 initial epochs (EarlyStopping patience=7), 11/15 fine-tuning epochs (learning rate 5e-7, patience=5).
- **Results**:
  - AUC: 0.9190
  - Test Accuracy: 80.61%
  - Normal F1: 0.62 (48% recall, 113/234 correct)
  - Pneumonia F1: 0.84 (96% recall, 373/390 correct)
  - Confusion Matrix: [[113, 121], [17, 373]]
- **Outputs**: Training/ROC plots, confusion matrix heatmap, before/after images (28x28).

## Reproducible Instructions
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
2. Modify the data_dir variable (line ~30) from /boot/XRAY DATA to your local dataset directory (e.g., /path/to/your/data).
