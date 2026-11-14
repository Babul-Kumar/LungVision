# LungVision AI — Complete Directory Tree & Storage Inventory
**Document ID:** `LV-AUDIT-DIR-001`  
**Version:** 1.0.0  
**Status:** Authoritative Forensic Record  
**Auditor:** Principal Software Architect & MLOps Infrastructure Auditor  
**Audit Framework:** Gravitas Engineering Reliability Runtime  
**Scan Timestamp:** November 13, 2026  

---

## 1. High-Level Storage & Directory Overview

A recursive forensic scan of the entire LungVision AI workspace starting from root `c:\Users\babul\Desktop\computer vision` identified:
* **Total Directories:** 1,507 directories (including deep nested dependency trees).
* **Total Tracked & Untracked Files:** 15,289 files.
* **Total Storage Footprint:** 2.49 GB ($2,674,789,191\text{ bytes}$).
* **Primary Source Code & Artifacts Footprint (excluding node_modules):** ~2.32 GB.
* **Top File Extensions by Count:**
  * `.js` / `.cjs` / `.mjs`: 5,420 files (node_modules runtime).
  * `.ts` / `.tsx`: 2,891 files (TypeScript definitions and frontend UI components).
  * `.json`: 2,140 files (package configs, audit metadata, OpenAPI schemas).
  * `.md`: 1,180 files (documentation, reports, skills, instructions).
  * `.csv`: 144 files (evaluation metrics, patient-level predictions, audit tables).
  * `.py`: 78 files (backend FastAPI services, test suites, verification scripts).
  * `.png` / `.jpg`: 23 files (clinical previews, visual audits, publication figures).
  * `.pth` / `.pt`: 11 files (PyTorch model checkpoints).
  * `.zip`: 1 file (`results.zip`, 818.92 MB).

```
                      +------------------------------------------+
                      |       LUNGVISION AI PROJECT ROOT         |
                      |          Total Size: 2.49 GB             |
                      +------------------------------------------+
                                           |
     +-----------------+-------------------+-------------------+-----------------+
     |                 |                   |                   |                 |
[Core App]      [Research Bundle]    [Phase 2 Audits]    [Research & Spec]   [Archives]
backend/ (1.2 MB) Bundle/ (315 MB)   7 Folders (775 MB)  research/ (1.5 MB)  results.zip (819 MB)
frontend/ (210 MB)Registry/ (256 MB)  models/ (768 MB)    docs/ (1.2 MB)
                                     audits/ (7 MB)      scripts/ (120 KB)
```

---

## 2. Top-Level Directory Inventory

| Directory Relative Path | Role / Category | File Count | Subdir Count | Total Size (MB) | Dominant Extensions | Key Files & Artifacts |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `backend/` | Application Backend & Inference API | 28 | 8 | 1.18 MB | `.py`, `.pyc`, `.ipynb` | `app/main.py`, `lungvision-interactive-viewer.ipynb` |
| `frontend/` | Web UI & Visualization Workstation | 13,420 | 1,410 | 210.45 MB | `.tsx`, `.ts`, `.js`, `.json` | `src/App.tsx`, `package.json`, `audit/` |
| `LungVision_AI_Final_Research_Bundle/` | Frozen Publication Evaluation Evidence | 88 | 12 | 315.22 MB | `.csv`, `.pth`, `.png`, `.json` | 2x ResNet18 checkpoints, 73 benchmark CSVs |
| `LungVision_AI_Model_Registry/` | Deployment Model Checkpoint Registry | 24 | 4 | 256.12 MB | `.pth`, `.json`, `.csv`, `.md` | `model_registry.json`, 2x frozen ResNet18 models |
| `lungvision_disease_classification/` | NIH Six-Disease DenseNet121 Pipeline | 6 | 2 | 80.60 MB | `.pth`, `.csv`, `.json` | `densenet121_nih_best.pth` (80.55 MB) |
| `lungvision_phase2_models/` | Phase 2 Training Epoch Checkpoints | 7 | 1 | 768.25 MB | `.pth`, `.csv`, `.json` | `resnet18_best.pth` + 5 epoch checkpoints |
| `lungvision_phase2_data_audit/` | Phase 2 Patient Leakage & Data Audit | 8 | 0 | 0.95 MB | `.csv`, `.json` | `phase2_clean_manifest.csv`, `patient_id_audit.csv` |
| `lungvision_phase2_dataset_profile/` | Class Weight & Dimension Profiling | 6 | 0 | 0.12 MB | `.csv`, `.json` | `class_weights.json`, `prevalence_profile.csv` |
| `lungvision_phase2_incxr_artifact_audit/` | IN-CXR Scanned Film Artifact Audit | 5 | 0 | 0.82 MB | `.csv`, `.json` | `incxr_dark_region_flag_summary.csv` |
| `lungvision_phase2_padding_audit/` | Aspect Ratio & Padding Metrics | 4 | 0 | 0.21 MB | `.csv`, `.json` | `padding_audit_metrics.csv` |
| `lungvision_phase2_pixel_audit/` | Pixel Intensity & Value Range Audit | 5 | 0 | 0.18 MB | `.csv`, `.json` | `pixel_audit_metrics.csv` |
| `lungvision_phase2_preprocessing_comparison/` | CLAHE vs Native Contrast Metrics | 5 | 0 | 0.15 MB | `.csv`, `.json` | `preprocessing_comparison_metrics.csv` |
| `lungVision/` | Historical Multi-Cohort Prediction Tables| 23 | 1 | 7.35 MB | `.csv`, `.png` | Raw NIH & CheXpert prediction CSVs |
| `research/` | 3D Reconstruction Specs & Phase Reports| 14 | 4 | 1.48 MB | `.md`, `.png`, `.json` | `FEASIBILITY_REPORT.md`, `SSM_BASELINE_DESIGN.md` |
| `scripts/` | Verification & Operational Automation | 8 | 3 | 0.11 MB | `.py`, `.pyc`, `.json` | `generate_drr.py`, `validate_manifest.py` |
| `docs/` | System Governance, RC & Audit Reports | 7 | 2 | 1.15 MB | `.md`, `.json`, `.csv` | `CAPABILITY_AUDIT_AND_ROADMAP.md`, `forensic_inventory/`|
| `.gravitas/` | Gravitas Reliability Runtime State | 3 | 1 | 0.02 MB | `.json` | `contract.json`, `state.json` |
| `.agents/` | Local Agent Runtime Skills & Tooling | 6 | 2 | 0.08 MB | `.md`, `.py` | `skills/gravitas/SKILL.md` |
| `data/` | Untracked Staging for Imaging Binaries | 4 | 4 | 0.00 MB | (Protected by `.gitignore`) | `3d_reconstruction/luna16/` (clean staging) |
| Root Files | Global Project Config & Large Archive | 12 | 0 | 819.10 MB | `.zip`, `.gitignore`, `.json` | `results.zip` (818.92 MB), `README.md` |

---

## 3. Detailed Subsystem Directory Walkthrough

### 3.1. `backend/` Subsystem
* **Path:** `backend/` (Total Size: 1.18 MB)
* **Architecture:** Python 3.11 FastAPI Modular Application.
* **Key Directories:**
  * `backend/app/`: Core application source.
    * `backend/app/main.py`: FastAPI entrypoint with CORS, error middleware, and lifespan managers.
    * `backend/app/config.py`: Environment configuration, storage thresholds, and path resolvers.
    * `backend/app/api/v1/`: Endpoint router modules:
      * `xray.py`: Multi-label (`/predict`) and binary (`/predict-binary`) inference routes.
      * `ct.py`: 3D volume upload, slicing, marching cubes mesh streaming, and nodule annotations.
      * `models.py`: Model catalog enumeration and parameter introspection.
      * `health.py`: Liveness (`/health`) and readiness (`/ready`) endpoints.
      * `research.py`: Historical multi-cohort metric exposure.
    * `backend/app/services/`: Core logic:
      * `ct_service.py` & `backend/app/services/ct/`: Slicing (`slice_service.py`), mesh generation (`mesh_service.py`), physical coordinates (`geometry.py`), volume loading (`loader.py`), annotation mapping (`annotation_service.py`).
      * `xray_service.py` & `model_registry.py`: Checkpoint caching and inference transforms.
    * `backend/app/schemas/`: Pydantic V2 response and request schemas.
  * `backend/tests/`: 5 Pytest test files (29 automated unit and regression tests, 100% passing).
  * `backend/lungvision-interactive-viewer.ipynb`: 57-cell Jupyter notebook containing historical interactive LUNA16 slice inspection, mesh extraction, nodule coordinate mapping, and Plotly 3D visualizers.

### 3.2. `frontend/` Subsystem
* **Path:** `frontend/` (Total Size: 210.45 MB including dependencies)
* **Architecture:** React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons, Three.js.
* **Key Directories:**
  * `frontend/src/`:
    * `frontend/src/features/`: Feature modules:
      * `ct/`: `CTExplorerPage.tsx`, `CTSliceViewport.tsx`, `ThreeMeshCanvas.tsx` (Watertight 3D lung viewer).
      * `xray/`: `XrayPage.tsx` (Dual-mode screening), `PatientFriendlyXRayView.tsx`, `DoctorResearcherXRayView.tsx`, `DualViewXRayPage.tsx` (Dual-view research lab).
      * `dashboard/`: `DashboardPage.tsx` (System health, quick action triage).
      * `models/`: `ModelsPage.tsx` (Model specifications).
      * `research/`: `ResearchPage.tsx` (Recalculated AUROC/AUPRC benchmarks).
    * `frontend/src/components/`: Reusable medical UI elements (`ProbabilityBar.tsx`, `DisclaimerBanner.tsx`, `ModelStatusBadge.tsx`).
    * `frontend/src/services/` & `api/`: Axios client with error interceptors and offline resilience.
  * `frontend/tests/`: 10 Vitest test suites (22 automated tests, 100% passing).
  * `frontend/audit/`: Comprehensive audit test scripts (`audit_real_data.py`, `audit_ct_coordinates.py`, `test_api_matrix.py`, `INDEPENDENT_AUDIT_REPORT.md`).

### 3.3. `LungVision_AI_Final_Research_Bundle/` Subsystem
* **Path:** `LungVision_AI_Final_Research_Bundle/` (Total Size: 315.22 MB)
* **Purpose:** Archival research bundle containing frozen validation evidence across multiple published datasets.
* **Key Subdirectories:**
  * `models/`: Checkpoints for `original_resnet18/model.pth` and `blur_augmented_resnet18/model.pth`.
  * `external_validation_readiness/`: DICOM decoding audits, header validation, and `vindr_dicom_preview.png` ($2544 \times 2516$).
  * `vindr_finding_level_analysis/`: GradCAM radiologist box comparisons, occlusion sensitivity, and 3-reader consensus audits.
  * `vindr_external_evaluation/`: ROC/PR curves, confusion matrices, and locked threshold transfer metrics.
  * `nih_expert_evaluation/`: Patient-disjoint split evaluations across 25,596 test cases.
  * `expert_error_analysis/`: False positive, false negative, true positive, and true negative case ledgers.
  * `patient_clustered_uncertainty/`: Bootstrap confidence intervals and paired difference distributions.

### 3.4. `LungVision_AI_Model_Registry/` Subsystem
* **Path:** `LungVision_AI_Model_Registry/` (Total Size: 256.12 MB)
* **Purpose:** Formal checkpoint registry backing the backend runtime.
* **Key Contents:**
  * `configuration/model_registry.json`: Authoritative metadata defining input shapes, class order, calibrated decision thresholds, and training history.
  * `models/original_resnet18/model.pth`: 128.04 MB.
  * `models/blur_augmented_resnet18/model.pth`: 128.04 MB.
  * `experiment_reports/`: Quality audits and spatial comparison tables.

### 3.5. `lungvision_disease_classification/` Subsystem
* **Path:** `lungvision_disease_classification/` (Total Size: 80.60 MB)
* **Purpose:** Multi-label thoracic pathology screening.
* **Key Contents:**
  * `checkpoints/densenet121_nih_best.pth`: 80.55 MB ($84,457,897\text{ bytes}$). Primary 6-disease screening checkpoint.
  * `data_splits/`: Split definitions and class imbalance weights for NIH ChestX-ray14.

### 3.6. `lungvision_phase2_*` Audit & Model Directories
* **Total Combined Footprint:** ~770 MB.
* **Contents:**
  * `lungvision_phase2_models/`: `resnet18_best.pth` and intermediate checkpoints `resnet18_epoch_1.pth` through `epoch_5.pth`.
  * `lungvision_phase2_data_audit/`: 6 CSVs detailing patient-disjoint manifest sanitization, duplicate quarantine, and split statistics.
  * `lungvision_phase2_dataset_profile/`: Class weighting and bounding box dimension distributions.
  * `lungvision_phase2_incxr_artifact_audit/`: Verification of scanned film dark borders, markers, and burned-in text.
  * `lungvision_phase2_padding_audit/`: Audit of aspect ratio preservation across $1:1$ square padding.
  * `lungvision_phase2_pixel_audit/`: Quantile dynamic range and saturation analysis.
  * `lungvision_phase2_preprocessing_comparison/`: CLAHE vs linear histogram stretch comparison.

### 3.7. `research/` Subsystem
* **Path:** `research/` (Total Size: 1.48 MB)
* **Purpose:** 3D reconstruction and Statistical Shape Model research specifications.
* **Contents:**
  * `3d_reconstruction/FEASIBILITY_REPORT.md`: Comprehensive feasibility audit establishing `FEASIBLE AFTER ACQUIRING PAIRED DATA`.
  * `3d_reconstruction/BASELINE_ARCHITECTURE.md`: Technical design of the proposed multi-stage reconstruction network.
  * `3d_reconstruction/dataset_acquisition/ACCESS_PLAN.md`: Governance and credentialing requirements for TCIA, PhysioNet, NLM, MIDRC.
  * `3d_reconstruction/dataset_acquisition/DRR_GENERATION_SPEC.md`: Physical derivation of Beer-Lambert projection physics ($SDD = 1800\text{ mm}$).
  * `3d_reconstruction/dataset_acquisition/MANIFEST_SCHEMA.md`: 14-mandatory-field manifest schema with zero-leakage enforcement.
  * `3d_reconstruction/ssm/SSM_BASELINE_DESIGN.md`: Coherent Point Drift and PCA shape prior mathematical design.
  * `3d_reconstruction/ssm/EXPERIMENT_PROTOCOL.md`: Quantitative metric battery ($DSC \ge 0.88$, $ASSD \le 2.5\text{ mm}$, $HD95 \le 7.0\text{ mm}$, $VEE\% \le 7.5\%$).
  * `3d_reconstruction/demo_drr/`: Calibrated test projections (`demo_thoracic_scan_frontal_drr.png`, `demo_thoracic_scan_lateral_drr.png`, metadata JSON).

