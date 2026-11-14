# LungVision AI — Complete Dataset Forensic Inventory
**Document ID:** `LV-AUDIT-DAT-001`  
**Version:** 1.0.0  
**Status:** Authoritative Forensic Record  
**Auditor:** Principal Medical Imaging Scientist & Data Governance Auditor  
**Audit Framework:** Gravitas Engineering Reliability Runtime  
**Machine-Readable Companion:** [datasets.csv](file:///c:/Users/babul/Desktop/computer%20vision/docs/forensic_inventory/datasets.csv)  

---

## 1. Executive Summary & Inventory Matrix

This audit establishes the precise empirical inventory of all medical imaging datasets referenced across the LungVision AI codebase, research bundles, notebooks, and models.

```
+----------------------------------------------------------------------------------------------------+
|                                    MASTER DATASET SUMMARY MATRIX                                   |
+---+----------------+-----------------------+-------------------------+--------------------+--------+
| # | Dataset Name   | Modality              | Data Presence Locally   | Local Storage Size | Status |
+---+----------------+-----------------------+-------------------------+--------------------+--------+
| 1 | NIH CXR-14     | 2D Planar CXR         | Metadata & Predictions  | 7.17 MB            | VERIF. |
| 2 | CheXpert       | 2D Planar CXR         | Metadata & Predictions  | 416.7 KB           | VERIF. |
| 3 | VinDr-CXR      | 2D Planar CXR (DICOM) | Predictions & Previews  | 22.50 MB           | VERIF. |
| 4 | IN-CXR (Open-i)| 2D Paired CXR         | Audit Metrics & Tables  | 18.20 MB           | VERIF. |
| 5 | LUNA16         | 3D Thoracic CT        | Ref. Volume & Metrics   | 1.05 MB            | VERIF. |
| 6 | LIDC-IDRI      | 3D Thoracic CT        | Parent Source Reference | 0 B (Cloud / TCIA) | ACCESS |
| 7 | MIMIC-CXR-JPG  | 2D Paired CXR         | Access Plan Only        | 0 B (PhysioNet)    | PLANNED|
| 8 | MIDRC Cohort   | Paired CXR-CT (<=48h) | Access Plan Only        | 0 B (Gen3 Portal)  | PLANNED|
+---+----------------+-----------------------+-------------------------+--------------------+--------+
```

---

## 2. Detailed Dataset Profiles (16 Dimensions)

### 2.1. NIH ChestX-ray14
* **Source:** National Institutes of Health (NIH) Clinical Center (Wang et al., CVPR 2017).
* **Local Paths:** `lungVision/nih_test_predictions_calibrated.csv`, `lungVision/nih_validation_predictions.csv`, `lungvision_disease_classification/data_splits/`.
* **Data Presence:** **Metadata & Patient-Level Predictions Only**. Raw images ($1024 \times 1024$ PNGs) are not stored locally.
* **Record Count:** 25,596 held-out test predictions + 14,000 validation predictions.
* **Patient Count:** 30,805 total cohort patients; patient-disjoint test partition strictly verified.
* **Image Count:** 112,120 total cohort images.
* **Modality:** Planar digital radiography (Film scans and CR/DX).
* **Frontal / Lateral:** Posteroanterior (PA) and Anteroposterior (AP) frontal projections only. No lateral views.
* **3D CT Availability:** None.
* **Segmentation Masks:** None.
* **Annotations:** 14 multi-label NLP disease categories extracted from reports. LungVision focuses on the 6 verified pathologies.
* **File Formats:** CSV, JSON.
* **Local Storage Footprint:** **7.17 MB**.
* **License & Access:** Public Domain (U.S. Government Work / CC0).
* **Current Usage:** Evaluated and exposed via `GET /api/v1/research/metrics` and `ResearchMetricsPage.tsx`.

---

### 2.2. CheXpert (Stanford ML Group)
* **Source:** Stanford Hospital PACS (Irvin et al., AAAI 2019).
* **Local Paths:** `lungVision/chexpert_*.csv` (10 CSV files).
* **Data Presence:** **Metadata & Predictions Only**.
* **Record Count:** 202 expert-labeled radiologist evaluation cases.
* **Patient Count:** 65,240 patients in full dataset; 202 patients in evaluated benchmark subset.
* **Image Count:** 224,316 images in full dataset.
* **Modality:** Planar chest radiography.
* **Frontal / Lateral:** Both Frontal (PA/AP) and Lateral views exist in the full cohort.
* **3D CT Availability:** None.
* **Segmentation Masks:** None.
* **Annotations:** 14 radiologist-labeled categories with consensus ground truth.
* **File Formats:** CSV.
* **Local Storage Footprint:** **416.7 KB**.
* **License & Access:** Stanford CheXpert Research Data Use Agreement.
* **Current Usage:** External validation benchmarks in backend and frontend.

---

### 2.3. VinDr-CXR (PhysioNet / VinBigData)
* **Source:** Hospital 108 and Hanoi Medical University Hospital (Nguyen et al., Scientific Data 2022).
* **Local Paths:** `LungVision_AI_Final_Research_Bundle/vindr_*` (12 subdirectories, 73 CSVs, 10 PNGs).
* **Data Presence:** **Predictions, Reader Agreement Metrics & High-Resolution Preview PNGs**.
* **Record Count:** 15,000 evaluation studies (13,118 Adult, 1,882 Pediatric).
* **Patient Count:** ~3,000 unique patients.
* **Image Count:** 18,000 DICOM radiographs in full cohort.
* **Modality:** High-resolution digital radiography ($2544 \times 2516$, 16-bit DICOM).
* **Frontal / Lateral:** Frontal PA/AP only.
* **3D CT Availability:** None.
* **Segmentation Masks:** None.
* **Annotations:** 22 local radiological findings (bounding boxes from 3 radiologists) + 6 global disease labels.
* **File Formats:** CSV, PNG (`vindr_dicom_preview.png`, $3.58\text{ MB}$).
* **Local Storage Footprint:** **22.50 MB**.
* **License & Access:** PhysioNet Open Data Use Agreement.
* **Current Usage:** Used as the live test fixture for real image ingestion in `frontend/audit/audit_real_data.py`.

---

### 2.4. Indiana University Chest X-Ray (IN-CXR / Open-i)
* **Source:** Indiana University Hospitals / NLM Open-i Service (Demner-Fushman et al., JAMIA 2016).
* **Local Paths:** `lungvision_phase2_incxr_artifact_audit/` (4 CSVs, 1 JSON metadata).
* **Data Presence:** **Artifact Audit Metrics Only**.
* **Record Count:** 7,470 radiographs across 3,955 radiology reports.
* **Patient Count:** 3,851 unique patients.
* **Image Count:** 7,470 images.
* **Modality:** Planar chest radiography.
* **Frontal / Lateral:** **Paired Frontal and Lateral** (3,612 studies have simultaneous verified pairs).
* **3D CT Availability:** None.
* **Segmentation Masks:** None.
* **Annotations:** Free-text radiology reports and MeSH medical terms.
* **File Formats:** CSV, JSON.
* **Local Storage Footprint:** **18.20 MB**.
* **License & Access:** Public Domain.
* **Current Usage:** Used for scanned film dark border and artifact audits in Phase 2.

---

### 2.5. LUNA16 / LIDC-IDRI
* **Source:** Grand Challenge (Setio et al., IEEE TMI 2017) & The Cancer Imaging Archive (Armato et al., Med Phys 2011).
* **Local Paths:**
  * `backend/app/services/ct/loader.py`: Embedded calibrated thoracic scan (`demo_thoracic_scan`).
  * `backend/lungvision-interactive-viewer.ipynb`: 57-cell interactive viewer.
  * `research/3d_reconstruction/demo_drr/`: Generated DRRs and projection metadata.
* **Data Presence:** **Calibrated Reference Volume, 12 Nodule Annotations & Historical Cloud Analysis**.
  * Raw full CT volumes ($888\times \approx 250\text{ MB} \approx 220\text{ GB}$) were processed in Kaggle cloud storage (`/kaggle/input/datasets/vafaeii/luna16/`).
  * One verified volume geometry is embedded in code: `(64, 128, 128)`, spacing `(0.742, 0.742, 1.25)\text{ mm}`, matching Series UID `1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518`.
* **Record Count:** 888 standardized thoracic CT scans in LUNA16; 1,018 patients in LIDC-IDRI.
* **Patient Count:** 888 unique subjects.
* **Image Count:** 888 volumes (~250,000 axial slices).
* **Modality:** High-resolution helical thoracic Computed Tomography (CT).
* **Frontal / Lateral:** Synthetic DRRs (Forward Beer-Lambert ray casting).
* **3D CT Availability:** **Yes**.
* **Segmentation Masks:** **Yes** (LUNA16 lung lobe masks: 3 = Left Lung, 4 = Right Lung, 5 = Trachea).
* **Annotations:** 1,186 verified nodule coordinates and diameters $\ge 3\text{ mm}$ (12 on reference scan).
* **File Formats:** `.mhd`, `.raw`, `.py`, `.json`, `.png`.
* **Local Storage Footprint:** **1.05 MB** (Volume logic, test DRRs, metadata).
* **License & Access:** Creative Commons Attribution 3.0 Unported (CC-BY 3.0).
* **Current Usage:** Active in `CTExplorerPage.tsx`, `ThreeMeshCanvas.tsx`, and `scripts/3d_reconstruction/generate_drr.py`.

---

### 2.6. MIMIC-CXR-JPG v2.1.0 & MIDRC (External / Planned)
* **MIMIC-CXR:** 377,110 planar radiographs from 65,379 patients. Requires PhysioNet credentialed access with CITI certification. Access plan documented in [ACCESS_PLAN.md](file:///c:/Users/babul/Desktop/computer%20vision/research/3d_reconstruction/dataset_acquisition/ACCESS_PLAN.md).
* **MIDRC:** Paired CXR and chest CT acquired within $\le 48\text{ hours}$ (~1,200 subjects). Requires Gen3 Data Commons account. Access plan documented in [ACCESS_PLAN.md](file:///c:/Users/babul/Desktop/computer%20vision/research/3d_reconstruction/dataset_acquisition/ACCESS_PLAN.md).

