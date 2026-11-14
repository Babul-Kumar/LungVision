# LungVision AI — Repository Structure & Inventory Baseline

**Document Version:** 1.0.0  
**Audit Date:** November 13, 2026  
**Auditor:** Senior Software Architect & Repository Maintenance Engineer (Gravitas Runtime)  
**Status:** **PHASE 1 RECONNAISSANCE BASELINE COMPLETE (Pre-Cleanup State Captured)**

---

## 1. Executive Summary

This inventory documents the complete directory structure, file distribution, large binary assets, duplicate artifacts, and codebase references of the LungVision AI repository prior to any consolidation or cleanup.

### Baseline Metrics
- **Total Non-Ignored Files:** 364
- **Top-Level Directories:** 16
- **Total Model Checkpoint Files (.pth):** 11 files (spanning 3 primary model families across 5 directory trees)
- **Large Binary Files (> 1 MB):** 24 files, dominated by PyTorch weights (128 MB and 84 MB) and `results.zip` (858 MB)
- **Duplicate File Groups by SHA-256:** 8 distinct content hash collision sets

---

## 2. Directory Tree Breakdown & File Counts

| Top-Level Directory | File Count | Purpose / Contents | Retention Recommendation |
| :--- | :---: | :--- | :--- |
| `backend/` | 50 | FastAPI backend, modular CT pipeline, tests, config, schemas | **Core Source — Preserve & Maintain** |
| `frontend/` | 78 | React 18 + TypeScript + Three.js application, tests, audit reports | **Core Source — Preserve & Refine** |
| `LungVision_AI_Final_Research_Bundle/` | 102 | External validation logs, VinDr-CXR evaluation, ResNet18 weights, governance | **Authoritative Research Evidence — Preserve** |
| `.agents/` | 29 | Antigravity agent skills including Gravitas runtime | **Runtime Infrastructure — Preserve** |
| `LungVision_AI_Model_Registry/` | 22 | Duplicate ResNet18 weights, configuration, registry metadata | **Target for Safe Consolidation** |
| `lungVision/` | 21 | NIH ChestX-ray14 calibration metrics, CheXpert validation, thresholds | **Authoritative Benchmark Data — Preserve** |
| `lungvision_phase2_models/` | 13 | ResNet18 training epoch checkpoints (epochs 1–5 + best) | **Training History Archive — Document & Preserve** |
| `lungvision_disease_classification/` | 4 | DenseNet121 NIH 6-disease checkpoint (`densenet121_nih_best.pth`) | **Authoritative Checkpoint — Preserve** |
| `lungvision_phase2_data_audit/` | 7 | Early stage data balance and split logs | **Research Archive — Preserve** |
| `lungvision_phase2_dataset_profile/` | 6 | IN-CXR dataset profiling reports | **Research Archive — Preserve** |
| `lungvision_phase2_incxr_artifact_audit/` | 5 | Scanning artifact detection analysis | **Research Archive — Preserve** |
| `lungvision_phase2_pixel_audit/` | 5 | Grayscale and dynamic range intensity audits | **Research Archive — Preserve** |
| `lungvision_phase2_preprocessing_comparison/` | 5 | Bilinear vs Bicubic interpolation benchmarking | **Research Archive — Preserve** |
| `lungvision_phase2_padding_audit/` | 4 | Zero vs reflective boundary padding analysis | **Research Archive — Preserve** |
| `scripts/` | 1 | Verification and maintenance automation scripts | **Operational Scripts — Expand** |
| `.gravitas/` | 1 | Gravitas task contract and policy record | **Governance — Preserve** |
| `root` | 11 | Startup scripts, empty download zips, 858MB results archive | **Cleanup Target (Remove 22-byte dummy zips)** |

---

## 3. Duplicate Files & Hash Collision Analysis

Automated hashing via `scripts/verification/reconnaissance_inventory.py` detected 8 identical content collision groups:

### 3.1 Empty Dummy ZIP Archives (Root)
Four 22-byte empty ZIP archives exist in the root directory from interrupted copy/download operations:
- `LungVision_AI_Disease_Classification_result.zip` (22 bytes) — SHA-256: `8739c76e681f900923b900c9df0ef75cf421d39cabb54650c4b9ad19b6a76d85`
- `LungVision_AI_Final_Research_Bundle.zip` (22 bytes) — SHA-256: `8739c76e...`
- `results (1).zip` (22 bytes) — SHA-256: `8739c76e...`
- `results (2).zip` (22 bytes) — SHA-256: `8739c76e...`
- **Action:** Delete these 4 empty 22-byte placeholder files. They provide zero data and create clutter.

### 3.2 Duplicate ResNet18 Weights Across Directories
Identical model weights exist in two separate directory trees:
- `original_resnet18/model.pth` ($134,260,367\text{ bytes}$):
  - Path A: `LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth` (SHA-256: `a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f`)
  - Path B: `LungVision_AI_Model_Registry/models/original_resnet18/model.pth` (SHA-256: `a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f`)
- `blur_augmented_resnet18/model.pth` ($134,263,007\text{ bytes}$):
  - Path A: `LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth` (SHA-256: `7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24`)
  - Path B: `LungVision_AI_Model_Registry/models/blur_augmented_resnet18/model.pth` (SHA-256: `7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24`)
- **Analysis:** `LungVision_AI_Final_Research_Bundle/` is the authoritative research bundle referenced by `backend/app/config.py`.

### 3.3 Duplicate CSV Datasets
- `nih_test_results.csv` in root is 100% identical (SHA-256: `a24092a2...`) to `lungVision/nih_test_results.csv`.
- `model_registry.json` in `LungVision_AI_Model_Registry/configuration/` is identical (SHA-256: `3e841674...`) to `LungVision_AI_Final_Research_Bundle/model_registry_documentation/configuration/model_registry.json`.

---

## 4. Codebase Dependency & Path Reference Map

To ensure no backend or frontend code breaks during repository organization, every code reference in `backend/` and `frontend/` was traced:

| Referenced String / Path | Referring Source Files | Criticality |
| :--- | :--- | :---: |
| `lungvision_disease_classification/.../densenet121_nih_best.pth` | `backend/app/config.py` | **CRITICAL RUNTIME PATH** |
| `LungVision_AI_Final_Research_Bundle/.../original_resnet18/model.pth` | `backend/app/config.py`, `backend/tests/test_xray_inference.py` | **CRITICAL RUNTIME PATH** |
| `LungVision_AI_Final_Research_Bundle/.../blur_augmented_resnet18/model.pth` | `backend/app/config.py`, `backend/tests/test_xray_inference.py` | **CRITICAL RUNTIME PATH** |
| `lungVision/nih_validation_thresholds.csv` | `backend/app/config.py`, `backend/app/services/xray_service.py` | **CRITICAL RUNTIME PATH** |
| `lungVision/nih_test_calibrated_metrics.csv` | `backend/app/config.py`, `backend/app/services/registry_service.py` | **CRITICAL RUNTIME PATH** |
| `lungVision/chexpert_external_metrics.csv` | `backend/app/config.py`, `backend/app/services/registry_service.py` | **CRITICAL RUNTIME PATH** |
| `LungVision_AI_Final_Research_Bundle/.../vindr_external_metrics.csv` | `backend/app/config.py`, `backend/app/services/registry_service.py` | **CRITICAL RUNTIME PATH** |
| `LungVision_AI_Model_Registry/.../model_registry.json` | `backend/app/config.py`, `backend/app/services/registry_service.py` | **CRITICAL RUNTIME PATH** |
| `demo_thoracic_scan` | `backend/app/services/ct_service.py`, `frontend/src/features/ct/CTExplorerPage.tsx` | **CRITICAL RUNTIME IDENTIFIER** |

**Safety Guard:** Any movement of checkpoints or manifests into a consolidated `models/` or `research/` directory MUST include fallback path resolution in `backend/app/config.py` so both the legacy paths and new consolidated paths resolve seamlessly.

---

## 5. Large Binary Artifacts (> 1 MB)

| Path | File Size | Data Type | Git Management Policy |
| :--- | :---: | :--- | :--- |
| `results.zip` | $818.92\text{ MB}$ | Zip archive of phase 2 audits | Exclude from Git / Archive |
| `lungvision_phase2_models/resnet18_best.pth` | $128.04\text{ MB}$ | PyTorch Model Weights | Preserve in archive |
| `lungvision_phase2_models/resnet18_epoch_1..5.pth` (5 files) | $128.04\text{ MB}$ each | PyTorch Weights per epoch | Preserve in archive |
| `LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth` | $128.04\text{ MB}$ | Active Inference Checkpoint | **Authoritative Active Model** |
| `LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth` | $128.04\text{ MB}$ | Active Inference Checkpoint | **Authoritative Active Model** |
| `lungvision_disease_classification/checkpoints/densenet121_nih_best.pth` | $80.55\text{ MB}$ | Active Inference Checkpoint | **Authoritative Active Model** |
| `lungvision_disease_classification/chexpert_manifest.csv` | $4.85\text{ MB}$ | Evaluation CSV Manifest | Preserve |
| `LungVision_AI_Final_Research_Bundle/external_validation_readiness/vindr_external_manifest.csv` | $3.59\text{ MB}$ | Clinical Dataset Manifest | Preserve |
| `LungVision_AI_Final_Research_Bundle/external_validation_readiness/vindr_dicom_preview.png` | $3.50\text{ MB}$ | Real Clinical Radiograph | **Active Test Image** |
| `LungVision_AI_Final_Research_Bundle/vindr_finding_level_analysis/.../*.png` | $1.5 - 3.3\text{ MB}$ | Research Figures & Grad-CAM | Preserve |
| `lungVision/nih_validation_predictions.csv` | $2.43\text{ MB}$ | Patient-level prediction log | Authoritative Research Data |
| `LungVision_AI_Final_Research_Bundle/.../vindr_frozen_model_predictions.csv` | $1.60\text{ MB}$ | 15,000 Patient prediction log | Authoritative Research Data |

