# LungVision AI — Verified Model Inventory

**Author:** Staff Backend Architect & MLOps Lead  
**Governing Skill:** Gravitas Engineering Reliability Runtime  
**Status:** Frozen & Verified Checkpoints  

---

## 1. Verified Model Table

| Model ID | Architecture | Task | Checkpoint Path | Input Dimensions | Preprocessing | Output Labels | Verified Status | Supporting Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `densenet121_nih_6disease` | DenseNet121 (`classifier = nn.Linear(1024, 6)`) | Multi-label chest X-ray classification | `lungvision_disease_classification/checkpoints/densenet121_nih_best.pth` | $224 \times 224 \times 3$ | RGB conversion, bilinear resize to 224x224, ImageNet mean/std norm | 6 diseases: Cardiomegaly, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax | **Verified** | Epoch 4 checkpoint (84.4 MB), Macro AUC 0.8206, `nih_training_history.csv` |
| `original_resnet18` | ResNet18 (`fc = nn.Linear(512, 1)`) | Binary Normal vs. Abnormal classification | `LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth` | $224 \times 224 \times 3$ | RGB conversion, bilinear resize to 224x224, ImageNet mean/std norm | Class 0: Normal, Class 1: Abnormal | **Verified** | Epoch 4 checkpoint (134.2 MB), SHA-256 `a293bd9e...`, Val AUC 0.9332 |
| `blur_augmented_resnet18` | ResNet18 (`fc = nn.Linear(512, 1)`) | Binary Normal vs. Abnormal classification | `LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth` | $224 \times 224 \times 3$ | RGB conversion, bilinear resize to 224x224, ImageNet mean/std norm | Class 0: Normal, Class 1: Abnormal | **Verified** | Epoch 1 checkpoint (134.2 MB), SHA-256 `7d863445...`, Val AUC 0.9380 |

---

## 2. Model 1: DenseNet121 6-Disease Classifier Details

- **Checkpoint File Size:** 84,457,897 bytes
- **Total Keys in Checkpoint:** 727 weight entries
- **Training Epoch:** 4 (out of 5)
- **Validation Macro AUC:** 0.8205820182751643
- **Validation Macro AUPRC:** 0.1669838201752586
- **Training Dataset:** NIH ChestX-ray14 (112,120 frontal radiographs)
- **Output Activation:** Independent Sigmoid per label $\sigma(z_i) = \frac{1}{1 + e^{-z_i}}$
- **Calibrated Operational Thresholds (from `lungVision/nih_validation_thresholds.csv`):**
  - **Cardiomegaly:** `0.9501953125` (Sensitivity: 36.3%, Specificity: 97.7%)
  - **Edema:** `0.876953125` (Sensitivity: 37.8%, Specificity: 97.3%)
  - **Consolidation:** `0.73291015625` (Sensitivity: 43.5%, Specificity: 89.0%)
  - **Pneumonia:** `0.7529296875` (Sensitivity: 13.7%, Specificity: 97.0%)
  - **Atelectasis:** `0.74951171875` (Sensitivity: 49.1%, Specificity: 88.3%)
  - **Pneumothorax:** `0.58251953125` (Sensitivity: 50.1%, Specificity: 92.7%)

---

## 3. Model 2 & 3: Binary ResNet18 Checkpoint Details

### Original ResNet18
- **SHA-256 Checksum:** `a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f`
- **File Size:** 134,260,367 bytes
- **Parameter Count:** 11,177,025
- **Locked Calibration Threshold:** `0.1764627248048782` (derives 90.0% sensitivity, 79.5% specificity on IN-CXR validation)

### Blur-Augmented ResNet18
- **SHA-256 Checksum:** `7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24`
- **File Size:** 134,263,007 bytes
- **Parameter Count:** 11,177,025
- **Locked Calibration Threshold:** `0.187923327088356` (derives 90.0% sensitivity, 82.2% specificity on IN-CXR validation)

---

## 4. Checkpoint Integrity Protection

All three checkpoint binaries are permanently locked.
- No retraining scripts are executed.
- No model weight files are modified or overwritten.
- All models run strictly in evaluation mode (`model.eval()` with `torch.no_grad()`).

