# LungVision AI — Centralized Model Registry & Weights Ledger

This directory houses all locked production checkpoints, experimental model architectures, and statistical shape models used across LungVision AI.

## Model Inventory & Provenance

| Model Name | Directory / Path | Architecture | Parameters | Input Resolution | SHA-256 Digest | Status & Performance |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **DenseNet121 NIH Best** | `densenet121/densenet121_nih_best.pth` | DenseNet121 | 6,960,006 | $224 \times 224$ | `8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af` | **Production Locked**<br>Macro AUC: `0.8206`<br>NIH 6-Disease Triage |
| **ResNet18 Baseline** | `resnet18/original_resnet18.pth` | ResNet18 | 11,177,025 | $224 \times 224$ | `a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f` | **Production Locked**<br>AUC: `0.9348`<br>Binary Normal / Abnormal |
| **ResNet18 Blur-Robust** | `resnet18/blur_augmented_resnet18.pth` | ResNet18 | 11,177,025 | $224 \times 224$ | `7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24` | **Production Locked**<br>AUC: `0.9411`<br>Optical Blur Resistant |
| **MultiScaleDenseNet121-FPN** | `multiscale_densenet121/multiscale_densenet121_fpn_512_v1.pth` | DenseNet121 + FPN + GAP/GMP + SE | 7,144,790 | $512 \times 512$ | `f64728d2742752085cd105d34af00795f53586b71abdf4a67acfa63c8d7b7d2c` | **Research Experimental**<br>Phase 6D Authenticated Checkpoint |
| **Dual-View Shape Predictor (v2)** | `ssm_reconstruction/dual_view_shape_predictor_v2_best.pth` | Dual-Branch ConvNet | 2,048,452 | $512 \times 512$ (Bi-Planar) | `dd592d21d3d9f68587a88184b1f9275089fc037dbafff5e9e6c5a0114b5be413` | **Production 3D Engine**<br>ASSD: `0.072 mm`<br>Dice: `0.985` |
| **Dual-View Shape Predictor (v1)** | `ssm_reconstruction/dual_view_shape_predictor_best.pth` | Dual-Branch ConvNet | 1,480,240 | $224 \times 224$ (Bi-Planar) | `44a7ea7f2471131a130288ca95911e7204aaa31e79b264ee06f80423ff92740b` | **Legacy Baseline** |
| **SSM Lung PCA Model** | `ssm_reconstruction/ssm_lung_pca_v1.json` | Statistical Shape Model | 20 Modes | 6,737 Vertices | `92714969943dbdaf081c7b11b90a7b85e0023649fbf9594cfb1ce8e783746df3` | **100% Explained Variance**<br>Watertight 2-Manifold |
| **Canonical Lung Mesh** | `ssm_reconstruction/canonical_lung_mesh.obj` | 3D Wavefront Mesh | 6,737 Vertices | 13,470 Faces | — | Reference Topological Surface |

---

## Calibrated Decision Thresholds

### 1. NIH ChestX-ray14 (DenseNet121)
- **Cardiomegaly**: `0.9502`
- **Edema**: `0.8770`
- **Consolidation**: `0.7329`
- **Pneumonia**: `0.7529`
- **Atelectasis**: `0.7495`
- **Pneumothorax**: `0.5825`

### 2. Binary Screening (ResNet18)
- **Baseline (`original_resnet18`)**: `0.176463` (Calibrated at 90% Sensitivity operating point)
- **Blur-Augmented (`blur_augmented_resnet18`)**: `0.187923` (Calibrated at 90% Sensitivity operating point)
