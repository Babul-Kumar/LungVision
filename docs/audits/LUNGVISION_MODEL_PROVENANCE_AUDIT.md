# LungVision AI — Model Provenance & Checkpoint Verification Audit

**Document Version:** 1.0.0  
**Audit Date:** November 14, 2026  
**Auditor:** Principal Medical AI Software Architect & MLOps Research Engineer  
**Runtime:** Gravitas Engineering Reliability Runtime  
**Governance Scope:** Cryptographic SHA-256 Checksum Verification, Parameter Auditing, Zero Weight Modification  

---

## 1. Executive Summary

This forensic audit verifies the physical integrity, parameter count, architecture, and training history for every registered neural network checkpoint in the LungVision AI repository.

All 3 primary model checkpoints were verified directly on disk:
1. **DenseNet121 NIH 6-Disease Multi-Label Classifier:**  
   Path: `lungvision_disease_classification/checkpoints/densenet121_nih_best.pth`  
   File Size: **84,457,897 bytes** (80.55 MB)  
   Verified SHA-256: `8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af`  
   Trainable Parameters: **6,960,006** (6.96M)  
   Peak Validation Epoch: **Epoch 4** (Validation Macro AUROC: 0.8206)  

2. **Original ResNet18 Baseline Classifier:**  
   Path: `LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth`  
   File Size: **44,790,569 bytes** (42.72 MB)  
   Verified SHA-256: `a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f`  
   Trainable Parameters: **11,177,025** (11.18M)  
   Peak Validation Epoch: **Epoch 4** (Validation AUC: 0.9332)  

3. **Blur-Augmented ResNet18 Robustness Classifier:**  
   Path: `LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth`  
   File Size: **44,790,569 bytes** (42.72 MB)  
   Verified SHA-256: `7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24`  
   Trainable Parameters: **11,177,025** (11.18M)  
   Peak Validation Epoch: **Epoch 1** (Validation AUC: 0.9380)  

---

## 2. Checkpoint Provenance & Physical Audit Table

| Checkpoint Name | Physical File Location | File Size (Bytes) | Cryptographic SHA-256 Hash | State Dict Elements | Total Parameters | Trainable Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `densenet121_nih_best.pth` | `lungvision_disease_classification/checkpoints/densenet121_nih_best.pth` | 84,457,897 | `8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af` | 727 tensors (7,043,775 values) | 6,960,006 | 6,960,006 |
| `original_resnet18/model.pth` | `LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth` | 44,790,569 | `a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f` | 122 tensors (11,186,625 values) | 11,177,025 | 11,177,025 |
| `blur_augmented_resnet18/model.pth` | `LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth` | 44,790,569 | `7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24` | 122 tensors (11,186,625 values) | 11,177,025 | 11,177,025 |

---

## 3. Architecture & Parameter Count Reconciliation

### 3.1 DenseNet121 Parameter Decomposition

A standard torchvision `densenet121` has the following layer structure:
- **Feature Extractor (`features`):**
  - Initial 7x7 Convolution + BatchNorm + ReLU + MaxPool.
  - DenseBlock 1 (6 layers, 12 convs).
  - Transition Layer 1 (1 conv, 1 avgpool).
  - DenseBlock 2 (12 layers, 24 convs).
  - Transition Layer 2 (1 conv, 1 avgpool).
  - DenseBlock 3 (24 layers, 48 convs).
  - Transition Layer 3 (1 conv, 1 avgpool).
  - DenseBlock 4 (16 layers, 32 convs).
  - Final BatchNorm2d.
  - Total Feature Extractor Parameters: **6,953,856**.

- **Classifier Head:**
  - Standard ImageNet Head: `nn.Linear(1024, 1000)` $\implies 1024 \times 1000 + 1000 = 1,025,000$ parameters.
    $$\text{Total ImageNet DenseNet121} = 6,953,856 + 1,025,000 = \mathbf{7,978,856\text{ parameters}}$$
  - Fine-Tuned LungVision 6-Disease Head: `nn.Linear(1024, 6)` $\implies 1024 \times 6 + 6 = 6,150$ parameters.
    $$\text{Total LungVision DenseNet121} = 6,953,856 + 6,150 = \mathbf{6,960,006\text{ parameters}}$$

- **State Dict Tensor Buffer Breakdown:**
  The `state_dict` contains 727 tensor keys totaling $7,043,775$ elements:
  - Trainable parameters (`weights` and `biases`): **6,960,006** elements.
  - Non-trainable BatchNorm running buffers (`running_mean` and `running_var`): **83,769** elements.
  $$6,960,006 + 83,769 = 7,043,775\text{ elements}$$

### 3.2 ResNet18 Parameter Decomposition

Both `original_resnet18` and `blur_augmented_resnet18` share an identical architecture:
- Initial Conv1 ($7 \times 7, 64$ filters) + BN.
- Layer 1: 2 BasicBlocks ($64$ filters) $\implies 4$ convs.
- Layer 2: 2 BasicBlocks ($128$ filters) $\implies 4$ convs + 1 downsample conv.
- Layer 3: 2 BasicBlocks ($256$ filters) $\implies 4$ convs + 1 downsample conv.
- Layer 4: 2 BasicBlocks ($512$ filters) $\implies 4$ convs + 1 downsample conv.
- Final Classifier Head: `nn.Linear(512, 1)` for binary sigmoid output $\implies 512 \times 1 + 1 = 513$ parameters.
- Total Trainable Parameters: **11,177,025**.
- BatchNorm Running Buffers: **9,600** elements.
- Total State Dict Elements: $11,177,025 + 9,600 = \mathbf{11,186,625\text{ elements}}$.

---

## 4. PyTorch 2.6+ Checkpoint Serialization Integrity

### Failure Identified & Mitigated:
Starting in PyTorch 2.6+, `torch.load()` enforces `weights_only=True` by default. Historical checkpoints created with older PyTorch versions often contain NumPy scalar metadata or pickle globals embedded in dictionary objects, triggering `_pickle.UnpicklingError`.

### Applied Runtime Pattern:
In `backend/app/services/model_loader.py:65`:
```python
ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
```
Passing `weights_only=False` explicitly for trusted local repository checkpoints guarantees seamless loading across all modern PyTorch versions while maintaining CPU and CUDA fallback compatibility.

---

## 5. Decision Threshold Calibration Provenance

| Model ID | Target Condition | Baseline Threshold | Calibrated Operating Threshold | Calibration Cohort | Selection Strategy | Achieved Validation Sensitivity | Achieved Validation Specificity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `original_resnet18` | Abnormal (Binary) | 0.5000 | **0.176463** | IN-CXR Validation (1,260 images) | 90% Sensitivity Point | 90.00% | 79.52% |
| `blur_augmented_resnet18` | Abnormal (Binary) | 0.5000 | **0.187923** | IN-CXR Validation (Augmented) | 90% Sensitivity Point | 90.00% | 82.22% |
| `densenet121_nih_6disease` | Cardiomegaly | 0.5000 | **0.950195** | NIH Validation (14,999 images) | Youden J / High-Precision | 36.30% | 97.71% |
| `densenet121_nih_6disease` | Edema | 0.5000 | **0.876953** | NIH Validation (14,999 images) | Youden J / Balanced F1 | 37.80% | 97.30% |
| `densenet121_nih_6disease` | Consolidation | 0.5000 | **0.732910** | NIH Validation (14,999 images) | Youden J / Sensitivity Bias | 43.47% | 89.00% |
| `densenet121_nih_6disease` | Pneumonia | 0.5000 | **0.752930** | NIH Validation (14,999 images) | Youden J / Precision Floor | 13.67% | 96.98% |
| `densenet121_nih_6disease` | Atelectasis | 0.5000 | **0.749512** | NIH Validation (14,999 images) | Youden J / Balanced F1 | 49.14% | 88.27% |
| `densenet121_nih_6disease` | Pneumothorax | 0.5000 | **0.582520** | NIH Validation (14,999 images) | High-Sensitivity Triage | 50.11% | 92.70% |

All thresholds are physically documented in `lungVision/nih_validation_thresholds.csv` and `LungVision_AI_Model_Registry/configuration/model_registry.json`.

