# LungVision AI — Complete Model Checkpoint & Neural Weight Inventory
**Document ID:** `LV-AUDIT-MDL-001`  
**Version:** 1.0.0  
**Status:** Authoritative Forensic Record  
**Auditor:** Principal ML Research Engineer & Checkpoint Integrity Auditor  
**Audit Framework:** Gravitas Engineering Reliability Runtime  
**Machine-Readable Companion:** [models.csv](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/models.csv)  

---

## 1. Executive Summary & Verification Overview

A forensic search across all repository directories identified **11 PyTorch checkpoint files (`.pth`) on disk**, totaling **1,332.61 MB** of storage, plus **6 redundant copies preserved inside `results.zip`** (768.25 MB).

Every checkpoint was independently inspected on CPU using PyTorch `weights_only=True` safe loading, calculating exact parameter counts, layer key structures, byte sizes, and cryptographic SHA-256 hashes.

### Key Inventory Findings:
1. **Three Authoritative Active Models:**
   * `densenet121_nih` (DenseNet121): 6-disease multi-label screening model.
   * `original_resnet18` (ResNet18): Baseline binary normal/abnormal classifier.
   * `blur_augmented_resnet18` (ResNet18): Motion-robust blur-augmented binary classifier.
2. **Duplicate Checkpoint Copies:**
   * `original_resnet18/model.pth` exists in duplicate across `LungVision_AI_Model_Registry/` and `LungVision_AI_Final_Research_Bundle/` (Identical SHA-256: `a293bd9e441b2713...`).
   * `blur_augmented_resnet18/model.pth` exists in duplicate across `LungVision_AI_Model_Registry/` and `LungVision_AI_Final_Research_Bundle/` (Identical SHA-256: `7d8634454d1ac0df...`).
3. **Phase 2 Training Checkpoints:**
   * 6 checkpoints in `lungvision_phase2_models/` represent an independent ResNet18 training trajectory across 5 epochs plus the selected best checkpoint (`resnet18_best.pth`, SHA-256 `e4e525de...`).
   * All 6 Phase 2 models on disk match the copies inside `results.zip` bit-for-bit.

---

## 2. Master Model Checkpoint Catalog (19 Fields)

| Field | Checkpoint 1 (DenseNet121) | Checkpoint 2 (ResNet18 Baseline) | Checkpoint 3 (ResNet18 Blur-Augmented) | Checkpoint 4 (Phase 2 Best) | Checkpoints 5–9 (Phase 2 Epochs 1–5) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model ID** | `densenet121_nih` | `original_resnet18` | `blur_augmented_resnet18` | `phase2_resnet18_best` | `phase2_resnet18_epoch_{1..5}` |
| **Model Name** | NIH Six-Disease Classifier | Baseline Binary Classifier | Blur-Augmented Binary Classifier | Phase 2 Best Binary Model | Phase 2 Training Epoch Checkpoints |
| **Architecture** | DenseNet121 (`torchvision.models`) | ResNet18 (`torchvision.models`) | ResNet18 (`torchvision.models`) | ResNet18 (`torchvision.models`) | ResNet18 (`torchvision.models`) |
| **File Path (Primary)** | `lungvision_disease_classification/checkpoints/densenet121_nih_best.pth` | `LungVision_AI_Model_Registry/models/original_resnet18/model.pth` | `LungVision_AI_Model_Registry/models/blur_augmented_resnet18/model.pth` | `lungvision_phase2_models/resnet18_best.pth` | `lungvision_phase2_models/resnet18_epoch_{1..5}.pth` |
| **File Path (Duplicate)**| *None* | `LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth` | `LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth` | *Inside `results.zip`* | *Inside `results.zip`* |
| **Format** | PyTorch (`.pth`) | PyTorch (`.pth`) | PyTorch (`.pth`) | PyTorch (`.pth`) | PyTorch (`.pth`) |
| **Size (Bytes)** | 84,457,897 bytes | 134,260,367 bytes | 134,263,007 bytes | 134,260,367 bytes | 134,260,367 bytes each |
| **Size (MB)** | 80.55 MB | 128.04 MB | 128.04 MB | 128.04 MB | 128.04 MB each |
| **SHA-256 Hash** | `8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af` | `a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f` | `7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24` | `e4e525de9107774ff0e814df3ba896cf1b0ffcf3a6ae64c9ae45731f2495b6cb` | See Section 3 below |
| **Parameter Count** | 6,960,006 parameters | 11,177,538 parameters | 11,177,538 parameters | 11,177,538 parameters | 11,177,538 parameters |
| **Input Shape** | `(1, 3, 224, 224)` RGB | `(1, 3, 224, 224)` RGB | `(1, 3, 224, 224)` RGB | `(1, 3, 224, 224)` RGB | `(1, 3, 224, 224)` RGB |
| **Output Labels** | 6 multi-label thoracic classes | 2 classes (`[Normal, Abnormal]`) | 2 classes (`[Normal, Abnormal]`) | 2 classes (`[Normal, Abnormal]`) | 2 classes (`[Normal, Abnormal]`) |
| **Output Class Order**| `Cardiomegaly`, `Edema`, `Consolidation`, `Pneumonia`, `Atelectasis`, `Pneumothorax` | `[0: Normal, 1: Abnormal]` | `[0: Normal, 1: Abnormal]` | `[0: Normal, 1: Abnormal]` | `[0: Normal, 1: Abnormal]` |
| **Training Dataset** | NIH ChestX-ray14 (86,524 train images) | IN-CXR + NIH Binary Normal/Abnormal | IN-CXR + Synthetic Defocus/Motion Blur | Phase 2 Curated Clean Split | Phase 2 Curated Clean Split |
| **Training Status** | COMPLETED (Converged) | COMPLETED (Converged) | COMPLETED (Converged) | COMPLETED (Best Val AUROC) | COMPLETED (Epoch Checkpoints) |
| **Best Epoch** | Epoch 14 | Epoch 8 | Epoch 11 | Epoch 3 | Epochs 1, 2, 3, 4, 5 |
| **Validation Metrics**| Mean AUROC = 0.8124 | Val AUROC = 0.8841 | Val AUROC = 0.8792 | Val AUROC = 0.8815 | Monitored per epoch |
| **Test Metrics** | Held-Out Test AUROC = 0.8091 | Test AUROC = 0.8654 (Clean) | Test AUROC = 0.8612 (Clean) / 0.8420 (Blurred) | Test AUROC = 0.8598 | Logged in phase 2 tables |
| **Decision Thresholds**| `Cardiomegaly: 0.9502`<br>`Edema: 0.8770`<br>`Consolidation: 0.7329`<br>`Pneumonia: 0.7529`<br>`Atelectasis: 0.7495`<br>`Pneumothorax: 0.5825` | `Threshold: 0.5000` (Default probability cutoff) | `Threshold: 0.5000` (Default probability cutoff) | `Threshold: 0.5000` | `Threshold: 0.5000` |
| **Loading Status** | **VERIFIED & TESTED LIVE** | **VERIFIED & TESTED LIVE** | **VERIFIED & TESTED LIVE** | Verified on disk | Verified on disk |
| **Backend API Usage** | Loaded in `xray_service.py` via `POST /api/v1/xray/predict` | Loaded in `xray_service.py` via `POST /api/v1/xray/predict-binary` | Loaded in `xray_service.py` via `POST /api/v1/xray/predict-binary` | Standby / Archival | Archival / Historical |
| **Frontend UI Usage** | Exposed in `XrayPage.tsx` & `DualViewXRayPage.tsx` | Exposed in `XrayPage.tsx` (Model Select dropdown) | Exposed in `XrayPage.tsx` (Model Select dropdown) | Unexposed | Unexposed |

---

## 3. Cryptographic Hashes for Phase 2 Epoch Checkpoints

All Phase 2 intermediate training checkpoints in `lungvision_phase2_models/` have been hashed:

| File Name | Epoch | Size (Bytes) | SHA-256 Checksum | In `results.zip`? |
| :--- | :---: | :---: | :--- | :---: |
| `resnet18_epoch_1.pth` | 1 | 134,260,367 | `c34549ebaee5e1c6b12a023023fc6944e0db0dc470914c68270562e84d436cf2` | **Verified Match** |
| `resnet18_epoch_2.pth` | 2 | 134,260,367 | `7e6fac963b483001ad28ea4e4f76269666df753b708cfc1e878564177c8e6dfa` | **Verified Match** |
| `resnet18_epoch_3.pth` | 3 | 134,260,367 | `479bbb4da1fc08d88bdf20ad764722880b91d92634d0263fcf099f65679c6d46` | **Verified Match** |
| `resnet18_epoch_4.pth` | 4 | 134,260,367 | `69f6c78e9fe5f9a150529d332616a9fe78931a742862d22b28c8bfaef388f704` | **Verified Match** |
| `resnet18_epoch_5.pth` | 5 | 134,260,367 | `aea85feff7295893b8602bca7e8c3f7c469b816a3df7819e078ba196a0f7cff1` | **Verified Match** |
| `resnet18_best.pth` | Best (3)| 134,260,367 | `e4e525de9107774ff0e814df3ba896cf1b0ffcf3a6ae64c9ae45731f2495b6cb` | **Verified Match** |

---

## 4. Checkpoint Deduplication & Storage Analysis

1. **Active vs Duplicate Models:**
   * Primary deployment directory: `LungVision_AI_Model_Registry/models/`.
   * Secondary duplicate directory: `LungVision_AI_Final_Research_Bundle/models/`.
   * **Storage Wasted:** $2 \times 128.04\text{ MB} = \mathbf{256.08\text{ MB}}$.
2. **Phase 2 Models Redundancy:**
   * Disk directory `lungvision_phase2_models/` holds $6 \times 128.04\text{ MB} = 768.25\text{ MB}$.
   * Archive `results.zip` also holds the identical 6 files ($768.25\text{ MB}$).
   * **Storage Redundancy:** $\mathbf{768.25\text{ MB}}$.
3. **Audit Recommendation:**
   Do not delete either copy during audit phase. In future cleanup, `results.zip` can serve as the sole long-term cold archive for phase 2 intermediate epochs, saving 768 MB on disk.

