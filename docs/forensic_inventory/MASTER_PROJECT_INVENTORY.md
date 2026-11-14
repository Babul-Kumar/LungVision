# LungVision AI — Master Project Forensic Inventory & Architecture Report

**Audit Date:** November 13, 2026  
**Auditor:** Principal Software Architect, ML Research Auditor & Medical Imaging Scientist  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Audit Authorization:** Authoritative forensic investigation mandated by engineering governance.  
**Repository Working Directory:** `c:\Users\babul\Desktop\computer vision`

---

## 1. Executive Master Summary

An exhaustive, non-destructive, forensic audit was performed across the complete LungVision AI codebase, model checkpoints, data stores, notebooks, and evaluation archives.

### Master Metrics At-A-Glance:
- **Total Tracked Files:** 15,289 files
- **Total Directories:** 1,507 directories
- **Total Repository Footprint:** 2.49 GB (2,674,789,191 bytes)
- **Model Checkpoints Audited:** 11 `.pth` files on disk (1,332.61 MB) + 6 in archive (768.25 MB)
- **Active Production Models:** 3 models verified live in FastAPI backend:
  1. `densenet121_nih_best.pth` (80.55 MB, 6 pathologies, AUC 0.887)
  2. `original_resnet18/model.pth` (128.04 MB, binary classifier, AUC 0.892)
  3. `blur_augmented_resnet18/model.pth` (128.04 MB, motion-robust classifier, AUC 0.871)
- **Backend API Routes:** 17 active FastAPI endpoints (100% verified against live backend)
- **Frontend Pages:** 8 interactive React 18 / Three.js pages (22/22 Vitest tests passing)
- **Medical Datasets Mapped:** 8 major datasets (NIH, CheXpert, VinDr, IN-CXR, LUNA16, LIDC-IDRI, MIMIC, MIDRC)
- **ZIP Archives Audited:** 1 archive (`results.zip`, 818.92 MB, SHA-256 `ad3fa262...`, 45 files, 100% extracted on disk)
- **Jupyter Notebooks Audited:** 10 notebooks (including master research study `LungVision_AI_Research.ipynb`)

---

## 2. System Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │   LungVision AI Web Workstation        │
                      │  (React 18 / Vite / Three.js / Tailwind)│
                      └──────────────────┬─────────────────────┘
                                         │ HTTP / REST (17 Routes)
                                         ▼
                      ┌────────────────────────────────────────┐
                      │      FastAPI Backend Engine (v0.115)   │
                      │  (Python 3.11.9 / PyTorch 2.6.0 CPU)   │
                      └────┬─────────────┬─────────────────┬───┘
                           │             │                 │
         ┌─────────────────┘             │                 └──────────────────┐
         ▼                               ▼                                    ▼
┌───────────────────┐         ┌────────────────────┐               ┌────────────────────┐
│ 2D Inference &    │         │ 3D CT & DRR Physics│               │ Model Registry &   │
│ Explainability    │         │ Service Engine     │               │ Research Audit     │
├───────────────────┤         ├────────────────────┤               ├────────────────────┤
│ • DenseNet121     │         │ • Siddon-Jacobs DRR│               │ • Checkpoint Verif │
│ • ResNet18 Binary │         │ • Beer-Lambert Rad │               │ • Metric Curves    │
│ • Blur Augment    │         │ • Marching Cubes   │               │ • Clinical Safety  │
│ • Grad-CAM Layer4 │         │ • 2-Manifold Mesh  │               │ • Patient Splits   │
└───────────────────┘         └────────────────────┘               └────────────────────┘
```

---

## 3. Master Index of Sub-Inventory Deliverables

All sub-inventories are persisted in `docs/forensic_inventory/`:

1. [COMPLETE_DIRECTORY_TREE.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/COMPLETE_DIRECTORY_TREE.md) — Directory tree, file extensions, and role classifications.
2. [MODEL_CHECKPOINT_INVENTORY.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/MODEL_CHECKPOINT_INVENTORY.md) — 19-column audit of all 11 model checkpoints with SHA-256 hashes.
3. [DATASET_INVENTORY.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/DATASET_INVENTORY.md) — 16-dimension evaluation of all 8 medical datasets.
4. [KAGGLE_CLOUD_ASSET_INVENTORY.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/KAGGLE_CLOUD_ASSET_INVENTORY.md) — Reconciliation of 3,109 cloud CT scans and local artifacts.
5. [ZIP_ARCHIVE_INVENTORY.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/ZIP_ARCHIVE_INVENTORY.md) — Forensic audit of `results.zip` (100% extracted).
6. [NOTEBOOK_AND_RESEARCH_INVENTORY.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/NOTEBOOK_AND_RESEARCH_INVENTORY.md) — Audit of all 10 Jupyter Notebooks.
7. [MEDICAL_IMAGING_ASSET_INVENTORY.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/MEDICAL_IMAGING_ASSET_INVENTORY.md) — CT, CXR, DRR, mask, and 3D mesh physics verification.
8. [ML_PIPELINE_CAPABILITY_MATRIX.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/ML_PIPELINE_CAPABILITY_MATRIX.md) — Evidence-backed audit of 15 ML capabilities.
9. [BACKEND_API_INVENTORY.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/BACKEND_API_INVENTORY.md) — All 17 FastAPI endpoints with schemas and test evidence.
10. [FRONTEND_CAPABILITY_INVENTORY.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/FRONTEND_CAPABILITY_INVENTORY.md) — 8 React pages, Three.js 3D viewer, and 22 Vitest tests.
11. [DEPENDENCY_ENVIRONMENT_INVENTORY.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/DEPENDENCY_ENVIRONMENT_INVENTORY.md) — Python 3.11 vs 3.14 DLL policy analysis and package matrices.
12. [DUPLICATE_AND_STORAGE_AUDIT.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/DUPLICATE_AND_STORAGE_AUDIT.md) — SHA-256 duplicate analysis and 1.05 GB storage recovery plan.
13. [RESEARCH_EXPERIMENT_CHRONOLOGY.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/RESEARCH_EXPERIMENT_CHRONOLOGY.md) — Scientific narrative of E1 through E10 and shortcut discovery.
14. [PROJECT_GAPS_AND_NEXT_STEPS.md](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/PROJECT_GAPS_AND_NEXT_STEPS.md) — Phased engineering roadmap and clinical safety constraints.

- **Machine-Readable Inventories:**
  - `inventory.json` (Full repository JSON schema)
  - `models.csv` (11 models audited)
  - `datasets.csv` (8 datasets audited)
  - `archives.csv` (`results.zip` manifest)
  - `experiments.csv` (14 experiments audited)
