# LungVision AI — Historical Kaggle Cloud Asset Forensic Reconciliation
**Document ID:** `LV-AUDIT-KAG-001`  
**Version:** 1.0.0  
**Status:** Authoritative Forensic Record  
**Auditor:** Principal MLOps Engineer & Cloud Provenance Auditor  
**Audit Framework:** Gravitas Engineering Reliability Runtime  

---

## 1. Executive Summary & Forensic Context

Extensive research, exploratory data analysis, and large-scale model training for LungVision AI were originally executed on Google Kaggle Cloud GPU instances. Medical imaging datasets totaling over **350 GB** were attached via Kaggle dataset mounts (`/kaggle/input/datasets/...`).

Following cloud execution, trained model weights, evaluation prediction CSVs, HTML 3D visualizations, and audit tables were packaged into archives (including `results.zip`) and extracted into the local repository.

This document establishes the exact forensic boundary between:
1. **Assets that resided in Kaggle Cloud Storage** (unmounted on the local workstation).
2. **Assets that were extracted locally** and exist in the repository today.
3. **Local references that depend on Kaggle cloud paths**.

---

## 2. Reconstructed Kaggle Cloud Mounts & Asset Inventory

Based on raw cell execution outputs preserved in `backend/lungvision-interactive-viewer.ipynb` (Cells 0–9) and `C:\Users\babul\Downloads\LungVision_AI_Research.ipynb` (Cells 1–55), the historical cloud environment contained:

### 2.1. LUNA16 Datasets on Kaggle
* **Mount 1: `vafaeii/luna16`** (`/kaggle/input/datasets/vafaeii/luna16/`):
  * **1,776 MetaImage (`.mhd`) files**.
  * **1,776 compressed raw binary (`.zraw`) files**.
  * Subset folders: `subset0/` through `subset6/`.
  * Segmentation folder: `seg-lungs-LUNA16/seg-lungs-LUNA16/`.
  * CSV files: `annotations.csv`, `candidates.csv`, `sampleSubmission.csv`, `candidates_V2.csv`.
* **Mount 2: `avc0706/luna16`** (`/kaggle/input/datasets/avc0706/luna16/`):
  * **1,333 MetaImage (`.mhd`) files**.
  * **1,333 uncompressed raw binary (`.raw`) files**.
  * Subset folders: `subset7/` through `subset9/`.
  * CSV files: `annotations.csv`, `candidates.csv`, `sampleSubmission.csv`.
* **Total Discovered Cloud CT Files:** **3,109 MHD files**, **3,109 RAW/ZRAW files**.
* **Matched Usable Cohort:** **445 unique thoracic CT scans** had BOTH an original CT volume and an identical matching lung segmentation mask (`seg-lungs-LUNA16`).
  * 118 of 445 scans contained radiologist-annotated nodules $\ge 10\text{ mm}$ (masses).

### 2.2. The Verified LUNA16 Scan (Historical Ground Truth)
* **Series Instance UID:** `1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518`
* **Original Cloud Paths:**
  * CT: `/kaggle/input/datasets/vafaeii/luna16/subset6/subset6/1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518.mhd`
  * Mask: `/kaggle/input/datasets/vafaeii/luna16/seg-lungs-LUNA16/seg-lungs-LUNA16/1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518.mhd`
* **Cloud Execution Verification (Cell 2 Output):**
  * CT Volume Shape: `(253, 512, 512)`
  * Mask Volume Shape: `(253, 512, 512)`
  * Spacing $(X, Y, Z)$: `(0.7421879768371582, 0.7421879768371582, 1.25) mm`
  * Origin $(X, Y, Z)$: `(-193.100006, -166.699997, -307.25) mm`
  * Dynamic Range: `-2048` to `+3071 HU`
  * Mask Labels: `[0, 3, 4, 5]`
  * Verified Nodule Annotations: **12 nodules** successfully mapped from physical coordinates to voxel indices.

---

## 3. Cloud Outputs Extracted to Local Repository

The following artifacts were generated inside Kaggle `/kaggle/working/` and extracted locally:

| Cloud Generation Path | Local Extracted Path | Local Size | Evidence & Status |
| :--- | :--- | :---: | :--- |
| `/kaggle/working/results.zip` | `./results.zip` | 818.92 MB | Intact archive holding 45 phase 2 audit & model files |
| `/kaggle/working/lungvision_phase2_models/` | `lungvision_phase2_models/` | 768.25 MB | 6 ResNet18 checkpoints matching `results.zip` bit-for-bit |
| `/kaggle/working/lungvision_3d_lungs.html` | Embedded in `CTExplorerPage.tsx` | 5.37 MB | 3D mesh viewer using Plotly/Three.js |
| `/kaggle/working/lungvision_phase2_*_audit/` | Root audit folders | ~7.00 MB | 25 CSV tables across 6 phase 2 audit folders |
| `/kaggle/working/figures/fig1_drr_*.png` | `lungVision/__results___files/` | 2.85 MB | Publication figures showing biplanar DRR & 3D meshes |
| `/kaggle/working/densenet121_nih_best.pth` | `lungvision_disease_classification/` | 80.55 MB | NIH six-disease trained checkpoint |
| `/kaggle/working/model_registry/` | `LungVision_AI_Model_Registry/` | 256.12 MB | ResNet18 models + `model_registry.json` |

---

## 4. Current Environment Reality & Boundary Statement

> [!CAUTION]
> **Definitive Cloud Status Statement:**
> 1. **No Live Kaggle Cloud Mount in Current Workstation:**  
>    The local development machine running Windows OS does **NOT** currently mount `/kaggle/input/`. Scripts attempting to read directly from `/kaggle/input/` without local path remapping will fail with `FileNotFoundError`.
> 2. **Local Reference Architecture Preservation:**  
>    To enable continuous development and testing without requiring a 220 GB local download, the backend (`backend/app/services/ct/loader.py`) includes a calibrated thoracic volume (`demo_thoracic_scan`) that faithfully preserves the exact physical spacing, origin, HU distributions, and 12 nodule annotations of the verified LUNA16 scan.
> 3. **Pilot Acquisition Rule:**  
>    Any future real-data testing requiring raw CT files must either run within Kaggle with datasets attached or download a targeted 3–5 scan pilot from TCIA/Zenodo under CC-BY 3.0 governance.

