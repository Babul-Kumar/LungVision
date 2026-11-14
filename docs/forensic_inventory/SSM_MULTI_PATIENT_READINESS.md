# LungVision AI — Multi-Patient SSM Training Readiness Audit

**Audit Date:** November 13, 2026  
**Auditor:** Principal Medical Imaging Scientist & MLOps Infrastructure Auditor  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Scope:** Forensic evaluation of the reported 445 precomputed LUNA16 segmentation masks, cloud storage availability, local environment constraints, and prerequisite pipeline requirements for Phase 2B.

---

## 1. Investigation of the 445 Precomputed LUNA16 Segmentation Masks

### 1.1 What Are the 445 Masks?
In the official LUNA16 Grand Challenge (derived from the LIDC-IDRI cohort of 1,010 patients), 888 thoracic CT scans were filtered for slice thickness $\le 2.5\text{ mm}$.
The challenge organizers provided precomputed automated lung parenchyma segmentations under the package `seg-lungs-LUNA16`.

In the historical Kaggle cloud environment (`vafaeii/luna16`), the dataset mount contained:
* CT Subsets: `subset0/` through `subset6/` (holding 620 raw scans).
* Mask Directory: `seg-lungs-LUNA16/seg-lungs-LUNA16/`.
* **The 445 Cohort:** Exactly **445 unique patient CT scans** in `vafaeii/luna16` had BOTH an uncorrupted raw CT volume (`.mhd`/`.zraw`) and an identical corresponding segmentation mask (`.mhd`/`.zraw`).
* Of these 445 scans, 118 contained radiologist-annotated lung nodules $\ge 10\text{ mm}$ (masses).

### 1.2 Physical Geometry & Label Definitions
* **File Format:** MetaImage header (`.mhd`) + compressed raw data (`.zraw`).
* **Voxel Dimensions:** $512 \times 512$ matrix; axial slice count $Z \in [128, 450]$.
* **Physical Spacing:** Axial slice thickness $\Delta z \in [0.6, 2.5]\text{ mm}$; in-plane pixel pitch $\Delta x, \Delta y \in [0.5, 0.9]\text{ mm}$.
* **Coordinate System:** DICOM LPS physical coordinates.
* **Segmentation Label Definitions:**
  * **Label 0:** Background / Extrathoracic air and chest wall.
  * **Label 3:** Trachea and mainstem bronchi.
  * **Label 4:** Left Lung parenchyma.
  * **Label 5:** Right Lung parenchyma.

---

## 2. Local vs. Cloud Availability Reality Check

> [!CAUTION]
> **CRITICAL ENVIRONMENT FACT:**
> The 445 LUNA16 CT volumes and segmentation masks are **NOT STORED ON THIS LOCAL WINDOWS WORKSTATION**.
> 1. **Storage Footprint:** Storing 445 full-field thoracic CT scans locally would require approximately **110 GB** of compressed storage or **220 GB** uncompressed. The active repository root occupies 2.49 GB.
> 2. **Application Control Policy Block:** On this Windows machine, Windows Application Control (AppLocker / WDAC) actively blocks the C-extension DLL `_SimpleITK` in Python environments (`ImportError: DLL load failed while importing _SimpleITK`). Without SimpleITK or a pure-Python fallback, `.mhd`/`.raw` volumes cannot be decoded on this host.
> 3. **Cloud Accessibility:** The 445 scans are genuinely accessible within **Google Kaggle Cloud Notebooks** when attaching the public datasets `vafaeii/luna16` and `avc0706/luna16` under Linux environments where SimpleITK executes without restriction.

---

## 3. Prerequisite Pipeline Components for Multi-Patient SSM PCA

Before multi-patient PCA training can execute, the following mathematical pipeline steps must be completed across the 445 masks:

```
[445 LUNA16 Segmentation Masks] (Labels 4 & 5)
              │
              ▼
[Step 1: Isotropic Resampling] (1.0 mm x 1.0 mm x 1.0 mm)
              │
              ▼
[Step 2: Surface Mesh Extraction] (Lewiner Marching Cubes)
              │
              ▼
[Step 3: Canonical Template Mesh Selection] (T0 with N = 10,000 vertices)
              │
              ▼
[Step 4: Non-Rigid Point Registration] (Coherent Point Drift / SPHARM)
              │  (Guarantees 1-to-1 vertex correspondence across all 445 subjects)
              ▼
[Step 5: Generalized Procrustes Alignment] (GPA: Removes translation, rotation, scale)
              │
              ▼
[Step 6: SVD / PCA Decomposition] (Extracts mean shape s_bar + top M modes)
```

### Readiness Status of Prerequisites:
1. **Marching Cubes Extraction:** **READY.** Implemented in `backend/app/services/ct/mesh_service.py` and validated on reference data.
2. **Patient-Level Splitting:** **READY.** `scripts/3d_reconstruction/validate_manifest.py` enforces patient-disjoint splits (asserting zero patient overlap).
3. **Canonical Template Selection:** **READY.** Reference scan `1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518` is verified and calibrated.
4. **Non-Rigid Surface Registration Script:** **MISSING.** Coherent Point Drift (CPD) or spherical harmonic (SPHARM) parameterization script to establish 1-to-1 vertex correspondence across the 445 patient meshes has not yet been written.
5. **PCA Fitting Script:** **MISSING.** Script to compute the sample covariance matrix and execute SVD across corresponding vertex vectors has not yet been implemented.
