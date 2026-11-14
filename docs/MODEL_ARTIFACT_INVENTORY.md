# LungVision AI — Model Artifact & Checkpoint Inventory

**Document Version:** 1.0.0  
**Audit Date:** November 13, 2026  
**Auditor:** ML Infrastructure Engineer & Model Auditor (Gravitas Runtime)  
**Status:** **AUTHORITATIVE CHECKPOINT AUDIT COMPLETE (All Hashes Verified)**

---

## 1. Executive Summary

This inventory establishes the verified provenance, cryptographic SHA-256 signatures, architectural dimensions, training datasets, and locked decision thresholds for all machine learning models in LungVision AI.

---

## 2. Active Production & Research Inference Checkpoints

### Model 1: DenseNet121 6-Disease Multi-Label Classifier
- **Model ID:** `densenet121_nih_6disease`
- **Primary Checkpoint Path:** `lungvision_disease_classification/checkpoints/densenet121_nih_best.pth`
- **File Size:** $84,457,897\text{ bytes}$ ($80.55\text{ MB}$)
- **SHA-256 Checksum:** `8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af`
- **Base Architecture:** `torchvision.models.densenet121`
- **Classifier Head:** `nn.Linear(in_features=1024, out_features=6)`
- **Total Parameters:** 6,959,942
- **Training Epoch:** 4 (Checkpoint selected from 5-epoch run based on macro validation AUROC)
- **Training Dataset:** NIH ChestX-ray14 ($112,120\text{ radiographs}$)
- **Input Dimension:** $224 \times 224 \times 3$ (RGB normalized to ImageNet mean/std)
- **Verified Label Ordering (Exact & Immutable):**
  1. `Cardiomegaly`
  2. `Edema`
  3. `Consolidation`
  4. `Pneumonia`
  5. `Atelectasis`
  6. `Pneumothorax`
- **Locked Calibration Thresholds (Youden Index on NIH Validation Set):**
  - Cardiomegaly: `0.9502`
  - Edema: `0.8770`
  - Consolidation: `0.7329`
  - Pneumonia: `0.7529`
  - Atelectasis: `0.7495`
  - Pneumothorax: `0.5825`

---

### Model 2: Original ResNet18 Binary Classifier
- **Model ID:** `original_resnet18`
- **Primary Checkpoint Path:** `LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth`
- **Duplicate Checkpoint Path:** `LungVision_AI_Model_Registry/models/original_resnet18/model.pth`
- **File Size:** $134,260,367\text{ bytes}$ ($128.04\text{ MB}$)
- **SHA-256 Checksum:** `a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f`
- **Base Architecture:** `torchvision.models.resnet18`
- **Classifier Head:** `nn.Linear(in_features=512, out_features=1)`
- **Total Parameters:** 11,177,025
- **Task:** Binary classification (`0: Normal`, `1: Abnormal`)
- **Training Epoch:** 4
- **Locked Operating Threshold:** `0.176463` (Calibrated at 90% sensitivity on IN-CXR validation)
- **VinDr-CXR 15k External Validation AUROC:** `0.601515` (Baseline), `0.600478` (CLAHE)
- **Brier Score Calibration:** `0.258521`

---

### Model 3: Blur-Augmented ResNet18 Binary Classifier
- **Model ID:** `blur_augmented_resnet18`
- **Primary Checkpoint Path:** `LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth`
- **Duplicate Checkpoint Path:** `LungVision_AI_Model_Registry/models/blur_augmented_resnet18/model.pth`
- **File Size:** $134,263,007\text{ bytes}$ ($128.04\text{ MB}$)
- **SHA-256 Checksum:** `7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24`
- **Base Architecture:** `torchvision.models.resnet18` (Trained with Gaussian low-pass spatial blur)
- **Classifier Head:** `nn.Linear(in_features=512, out_features=1)`
- **Total Parameters:** 11,177,025
- **Task:** Binary classification (`0: Normal`, `1: Abnormal`)
- **Training Epoch:** 1
- **Locked Operating Threshold:** `0.187923` (Calibrated at 90% sensitivity on IN-CXR validation)
- **VinDr-CXR 15k External Validation AUROC:** `0.604319` (Baseline), `0.613423` (CLAHE)
- **Brier Score Calibration:** `0.254394`

---

## 3. Training Epoch Archive Models (`lungvision_phase2_models/`)

The following checkpoints represent historical training trajectory artifacts from Phase 2 exploration:

| Checkpoint File | Size | SHA-256 Checksum | Epoch / Role |
| :--- | :---: | :--- | :--- |
| `resnet18_best.pth` | $128.04\text{ MB}$ | `e4e525de9107774f00804664804d593b23b6e645f930d3ddf45b5f2d898d7fb6` | Best Phase 2 checkpoint |
| `resnet18_epoch_1.pth` | $128.04\text{ MB}$ | `c34549ebaee5e1c6eee3ac0185df731e6944300b2653189e3358701124ec58f2` | Trajectory Epoch 1 |
| `resnet18_epoch_2.pth` | $128.04\text{ MB}$ | `7e6fac963b483001df441964cc46603cece39f97ffbb46083edb8f3d560ed8b9` | Trajectory Epoch 2 |
| `resnet18_epoch_3.pth` | $128.04\text{ MB}$ | `479bbb4da1fc08d880f10da7aadd938200f6484e89c6fdb0785c241b04d05212` | Trajectory Epoch 3 |
| `resnet18_epoch_4.pth` | $128.04\text{ MB}$ | `69f6c78e9fe5f9a1672d2d16503a4a8f0889b8ee04b2fe392e9c970c2ea0ee6d` | Trajectory Epoch 4 |
| `resnet18_epoch_5.pth` | $128.04\text{ MB}$ | `aea85feff729589385259d53bce1a1182f67d5d74736ea906bc2b0f6efe1ec3f` | Trajectory Epoch 5 |

---

## 4. Checkpoint Integrity Verification Protocol

Automated integrity verification is implemented in `scripts/verification/reconnaissance_inventory.py`.  
Whenever models are backed up, consolidated, or deployed, their SHA-256 checksums must match the values recorded in this table.

