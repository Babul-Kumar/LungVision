# LungVision AI — Notebook & Research Experiment Inventory

**Audit Date:** November 13, 2026  
**Auditor:** Senior Medical Imaging Software Engineer & ML Research Auditor  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Scope:** Forensic analysis of all Jupyter Notebooks (`.ipynb`) in workspace and user download repositories.

---

## 1. Notebook Inventory Summary

A forensic scan identified **10 Jupyter Notebooks** totaling 898 KB across active backend workspaces and external download directories.

| # | Notebook Filename | Location | Size (Bytes) | Total Cells | Code Cells | MD Cells | Primary Purpose | Status |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :--- | :---: |
| **1** | `lungvision-interactive-viewer.ipynb` | `backend/` | 13,842 | 57 | 57 | 0 | 3D CT Interactive Viewer & Mesh Quality Audit | Completed |
| **2** | `LungVision_AI_Research.ipynb` | `Downloads/` | 142,510 | 55 | 26 | 29 | Master Multi-Phase Research Notebook (E1–E10) | Completed |
| **3** | `luna16_x2ct_v4_final.ipynb` | `Downloads/` | 34,120 | 12 | 11 | 1 | 2D X-Ray to 3D CT Prototype (Exposed Spacing Shortcut) | Completed |
| **4** | `luna16_x2ct_multistructure_v3.ipynb` | `Downloads/` | 28,490 | 9 | 8 | 1 | Multi-Structure 3D Organ Reconstruction Prototype | Experimental |
| **5** | `luna16_x2ct_multistructure_v3_1.ipynb` | `Downloads/` | 29,120 | 10 | 9 | 1 | Refined Multi-Structure CT Reconstruction (Trachea/Heart) | Experimental |
| **6** | `luna16_x2ct_triage_fixed.ipynb` | `Downloads/` | 24,190 | 8 | 7 | 1 | Nodule Screening & Triage Classification Prototype | Completed |
| **7** | `luna16_x2ct_triage_v2.ipynb` | `Downloads/` | 26,410 | 9 | 8 | 1 | Triage v2 with ROC-AUC and Mass Triage Curves | Completed |
| **8** | `luna16_x2ct_triage_v2 (1).ipynb` | `Downloads/` | 26,410 | 9 | 8 | 1 | Duplicate Copy of Triage v2 Checkpoint | Duplicate |
| **9** | `lungvision-interactive-viewer.ipynb` | `Downloads/` | 13,842 | 57 | 57 | 0 | Interactive Mesh Viewer Checkpoint in Downloads | Duplicate |
| **10** | `lungvision-interactive-viewer (1).ipynb` | `Downloads/` | 13,842 | 57 | 57 | 0 | Duplicate Copy of Interactive Viewer in Downloads | Duplicate |

---

## 2. In-Depth Forensic Analysis by Notebook

### 2.1 `LungVision_AI_Research.ipynb` (Master Scientific Study)
- **Path:** `C:/Users/babul/Downloads/LungVision_AI_Research.ipynb`
- **Cell Structure:** 55 cells (26 Python code, 29 Markdown prose).
- **Research Coverage:** The foundational document establishing Experiments E1 through E10:
  - **Section 1–5:** Clinical problem formulation, GPU/RAM checks, dataset path discovery across Kaggle mounts (`/kaggle/input/luna16`, `/kaggle/input/datasets/vafaeii/luna16`, `/kaggle/input/datasets/avc0706/luna16`).
  - **Section 6–9:** Patient/study pairing, zero-leakage cross-split verification, caching manifests for 888 patients.
  - **Section 10:** Physics-based Beer-Lambert CT-to-DRR projection ($D_{\text{SOD}}=1000\,\text{mm}$, $D_{\text{SID}}=1400\,\text{mm}$).
  - **Section 11–15:** Branch A (3D CT generation), Branch B (2D abnormality classification), Branch C (multimodal uncertainty fusion via Kendall-Gal homoscedastic weighting).
  - **Section 16–23:** Multi-metric evaluation (Dice, Chamfer distance, AUROC, AUPRC, bootstrap 95% CIs, paired permutation tests).
  - **Section 24–29:** Failure analysis, shortcut bias detection, export of model weights, medical safety disclosures.
- **Trained Models Produced:** `best_x2ct_v6.pth`, Branch A reconstruction weights, Branch B clinical weights, Branch C fusion state dicts.

### 2.2 `luna16_x2ct_v4_final.ipynb` (The Prototype That Exposed the Shortcut)
- **Path:** `C:/Users/babul/Downloads/luna16_x2ct_v4_final.ipynb`
- **Cell Structure:** 12 cells (11 Code, 1 Markdown).
- **Key Discovery:**
  - Prototype model mapped 2D DRR projections to 3D CT volumes and predicted lung morphometry metrics.
  - Initial evaluation showed an artificially high $R^2 > 0.94$ in lung volume estimation.
  - **Forensic Audit Finding:** The model was bypassing 3D volumetric feature learning and exploiting a severe dataset shortcut: reading the `spacing_zyx` voxel header metadata directly to compute volume!
  - When voxel spacing features were scrambled or standardized, the uncalibrated model's performance collapsed to $R^2 = 0.31$, proving that voxel dimensions acted as an information leak.
  - This critical finding led directly to the rigorous redesign in `LungVision_AI_Research.ipynb`.

### 2.3 `backend/lungvision-interactive-viewer.ipynb` (Interactive 3D Workspace)
- **Path:** `backend/lungvision-interactive-viewer.ipynb`
- **Cell Structure:** 57 cells (57 Code, 0 Markdown).
- **Core Functionality:**
  - Implements Python/ipywidgets/Plotly synchronized 3D multi-planar reformatting (MPR) and 3D surface mesh rendering.
  - Extracts Marching Cubes lung surface meshes at ISO value -400 HU.
  - Executes mesh topology checks: vertex counts, triangle faces, manifoldness, Euler characteristic.
  - Generates audit CSV tables: `lungvision_3d_volume_audit.csv`, `lungvision_mesh_volume_validation.csv`, `lungvision_mesh_quality_audit.csv`.

---

## 3. Kaggle Cloud Environment Dependency Analysis

All training notebooks were architected for execution in Kaggle GPU environments (NVIDIA P100 / T4 / V100). They reference the following standardized cloud input mounts:
- `/kaggle/input/datasets/vafaeii/luna16/` (1,776 MHD files, Subsets 0–9)
- `/kaggle/input/datasets/avc0706/luna16/` (1,333 MHD files, Subsets 0–9)
- `/kaggle/input/luna16/` (Symlinked composite root)
- `/kaggle/working/` (Scratch directory for generated checkpoints and evaluation tables)

**Portability Status:** Local execution requires substituting `/kaggle/input/...` paths with local directory mounts or running in mock/demo mode using `backend/app/services/ct_service.py`.
