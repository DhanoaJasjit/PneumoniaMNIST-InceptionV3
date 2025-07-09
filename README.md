
# PneumoniaMNIST-InceptionV3

Fine-tuned InceptionV3 model for PneumoniaMNIST classification, achieving **AUC 0.9190** and **48% Normal recall**, tailored for medical imaging tasks like PROSPECT’s ultrasound analysis.

This repository contains a fine-tuned InceptionV3 model for classifying chest X-rays from the PneumoniaMNIST dataset (Normal vs. Pneumonia), developed as part of the PROSPECT project application for chronic pancreatitis evaluation using convolutional neural networks.

---

## 🧠 Project Overview

- **Objective**: Classify chest X-rays to distinguish Normal from Pneumonia cases, addressing class imbalance and optimizing for clinical reliability.
- **Model**: InceptionV3 (pre-trained on ImageNet) with custom dense layers (128, 64 units, ReLU, L2 regularization), batch normalization, and dropout (0.5, 0.3). Fine-tuned last 50 layers.
- **Dataset**: PneumoniaMNIST (3,882 train, 524 val, 624 test images; 1,134 Normal, 2,748 Pneumonia).
- **Key Features**:
  - 3x oversampling of Normal class (~1,152 images, total 5,034) to address imbalance.
  - Augmentations: flip, brightness (±0.1), contrast (0.8–1.2), hue (±0.05), rotation (0–360°), zoom (0.9–1.1), random crop.
  - Training: 11/20 initial epochs (EarlyStopping patience=7), 11/15 fine-tuning epochs (learning rate 5e-7, patience=5).

---

## 📊 Results

- **AUC**: 0.9190
- **Test Accuracy**: 80.61%
- **Normal F1**: 0.62 (48% recall, 113/234 correct)
- **Pneumonia F1**: 0.84 (96% recall, 373/390 correct)
- **Confusion Matrix**:
  ```
  [[113, 121],
   [ 17, 373]]
  ```

---

## 🚀 Reproducible Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**:  
   [PneumoniaMNIST on MedMNIST](https://medmnist.com/)

3. **Update Dataset Path**:
   Modify `data_dir` in `train_model_pneumoniamnist_final.py` (around line 30):
   ```python
   data_dir = "/path/to/your/data"
   ```

   Optional (use a config file):
   ```python
   # config.py
   DATA_DIR = "/path/to/your/data"
   # in script:
   from config import DATA_DIR
   data_dir = DATA_DIR
   ```

4. **Run Training**:
   ```bash
   python train_model_pneumoniamnist_final.py
   ```

5. **Outputs**: ROC plots, confusion matrix heatmap, augmented 28x28 & 299x299 images, model weights.

> ⚠️ **Note**: Always double-check dataset path to avoid `FileNotFoundError`.

---

## 🖼️ Visual Outputs

All visuals are stored in `Visual Output PneumoniaMNIST/`.

### 📈 Training and ROC Plots
<img src="Visual%20Output%20PneumoniaMNIST/j2.png" alt="Training and ROC Plots">

- **Description**:  
  - Left: Accuracy – Training (~0.32) vs Validation (~0.72)  
  - Right: Loss – Training (~1.10–1.21) vs Validation (~0.66–0.76)

---

### 🧮 Confusion Matrix Heatmap
<img src="Visual%20Output%20PneumoniaMNIST/j1.png" alt="Confusion Matrix Heatmap">

- **Description**:  
  Threshold: 0.15  
  Confusion Matrix:
  ```
  [[113, 121],
   [ 17, 373]]
  ```

---

### 🩻 Augmentation Pair 1
<img src="Visual%20Output%20PneumoniaMNIST/j3.png" alt="Augmentation Pair 1">

- **Description**:  
  Original 28x28 Pneumonia X-ray ➝ Augmented 299x299 (predicted Pneumonia).  
  Shows effects of flipping, brightness, and rotation.

---

### 🩺 Augmentation Pair 2
<img src="Visual%20Output%20PneumoniaMNIST/j5.png" alt="Augmentation Pair 2">

- **Description**:  
  Original 28x28 Normal X-ray ➝ Augmented 299x299 (misclassified as Pneumonia).  
  Highlights model confusion in borderline cases.

---

### 📉 ROC Curve
<img src="Visual%20Output%20PneumoniaMNIST/j4.png" alt="ROC Curve">

- **Description**:  
  ROC Curve with **AUC = 0.9190**  
  Strong class separability, confirming robustness for clinical use in PROSPECT.

---

## 📚 Lessons Learned

### ❌ Mistakes & Learnings
- **Dependency Conflicts**:
  - Faced `ModuleNotFoundError` due to version mismatches.
  - ✅ Fix: Pin `scikit-learn==1.4.2`, `imbalanced-learn==0.10.1`

- **Data Path Issues**:
  - Forgot to update dataset path ➝ `FileNotFoundError`.
  - ✅ Fix: Always test file paths in early stages.

- **Metric Imbalance**:
  - Focused on Pneumonia recall (96%) ➝ Normal recall dropped (48%).
  - ✅ Lesson: Balance recall and precision for clinical realism.

- **Time Pressure**:
  - Rushed final tuning post **July 9, 2025** ➝ missed deeper evaluation.
  - ✅ Lesson: Buffer time for debugging + fine-tuning is essential.

---

## 🔮 Future Improvements

- Use `venv` or `conda` to isolate environments and avoid version issues.
- Add dataset path validation and auto-download prompts.
- Try 2x oversampling (~768 Normal) + **focal loss** to reduce Normal false negatives.
- Tune for longer (e.g., 20 epochs @ 3e-7) with warm restarts.
- Add Grad-CAM and explainability for model transparency.
- Expand this README with troubleshooting and versioning notes.

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) file for details.
