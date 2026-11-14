# LungVision AI — Pre-Experiment Acquisition & Readiness Checklist
**Document ID:** `LV-RESEARCH-3D-CHK-001`  
**Version:** 1.0.0  
**Status:** Approved for Implementation  
**Auditor/Author:** LungVision AI Research & Governance Directorate  
**Framework:** Gravitas Engineering Reliability Runtime  

---

## 1. Executive Summary

Before the first Statistical Shape Model (SSM) or neural 3D lung reconstruction experiment can execute on real clinical data, specific datasets, credentials, software tooling, and annotations must be assembled and verified. This checklist defines the mandatory acquisition gates and prerequisites.

```
+-----------------------------------------------------------------------------------------+
|                  PRE-EXPERIMENT DATASET & GOVERNANCE READINESS GATES                    |
+----+-----------------------------+---------------------+-------------------+------------+
| #  | Requirement / Dataset       | Access Level        | Estimated Size    | Status     |
+----+-----------------------------+---------------------+-------------------+------------+
| 1  | LUNA16 Subset 00 (88 CTs)   | Open (CC-BY 3.0)    | 12.4 GB           | PENDING DL |
| 2  | LUNA16 Lung Segmentations   | Open (CC-BY 3.0)    | 180 MB            | PENDING DL |
| 3  | IN-CXR Paired PNGs (Open-i) | Public Domain (NLM) | 14.2 GB           | PENDING DL |
| 4  | CITI Training Certification | PhysioNet Credential| User Certification| IN PROGRESS|
| 5  | MIMIC-CXR-JPG Access DUA    | PhysioNet Credential| 560 GB (Phase 2)  | SUBMITTED  |
| 6  | MIDRC Paired CXR-CT Cohort  | Gen3 Research DUA   | 45 GB (Benchmark) | PENDING REQ|
| 7  | 16-Landmark 3D Catalog      | Derived Annotation  | 1.5 MB            | READY      |
| 8  | Canonical Lung Template     | Algorithmic Mesh    | 2.8 MB            | READY      |
| 9  | Manifest Validator Script   | In-Repository Code  | -                 | VERIFIED   |
| 10 | CT-to-DRR Generator Script  | In-Repository Code  | -                 | VERIFIED   |
+----+-----------------------------+---------------------+-------------------+------------+
```

---

## 2. Explicit Acquisition & Credential Checklist

### Gate 1: LUNA16 / LIDC-IDRI Open CT Volume Acquisition (Prerequisite for SSM Training)
- [ ] **Action 1.1:** Download `subset00` of LUNA16 (88 standardized thoracic CT scans in `.mhd`/`.raw` format $\le 2.5\text{ mm}$ slice thickness).
  - Source: TCIA / Grand Challenge LUNA16 mirror (`https://luna16.grand-challenge.org/`).
  - License: Creative Commons Attribution 3.0 Unported (CC-BY 3.0).
  - Storage Destination: `data/3d_reconstruction/ct_volumes/luna16/subset00/` (protected from git).
- [ ] **Action 1.2:** Download verified LUNA16 lung lobe segmentation masks (`lung_masks/`).
  - License: CC-BY 3.0.
  - Storage Destination: `data/3d_reconstruction/segmentations/luna16/`.
- [ ] **Action 1.3:** Verify SHA-256 integrity of all downloaded `.mhd`/`.raw` files against the official Grand Challenge checksum manifest.

### Gate 2: Indiana University Chest X-Ray (IN-CXR) Acquisition (Prerequisite for 2D Real CXR Testing)
- [ ] **Action 2.1:** Download `NLMCXR_png.tgz` from NLM Open-i repository (`https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz`).
  - License: Public Domain.
  - Storage Destination: `data/3d_reconstruction/cxr_real/in_cxr/`.
- [ ] **Action 2.2:** Filter and index the 3,612 studies that have simultaneous, verified Frontal (PA/AP) and Lateral (LL/RL) radiographic projections.
- [ ] **Action 2.3:** Execute burned-in text detection and margin cropping audit to scrub any peripheral patient markings.

### Gate 3: Credentialed Research Cohorts (PhysioNet & MIDRC Governance)
- [ ] **Action 3.1:** Complete CITI Course *"Data or Specimens Only Research"* and retain Completion Report Record ID.
- [ ] **Action 3.2:** Submit Data Use Agreement (DUA) on PhysioNet for `mimic-cxr-jpg/2.1.0`.
- [ ] **Action 3.3:** Configure `gen3-client` with institutional credentials for the MIDRC data commons (`https://data.midrc.org/`) to access the paired CXR-CT $\le 48\text{ h}$ benchmark cohort.
- [ ] **Policy Adherence Check:** Strictly confirm no automated credential spoofing, unauthorized scrapers, or third-party token sharing are employed.

### Gate 4: Algorithmic Templates & Spatial Calibration Assets
- [x] **Action 4.1 (COMPLETED):** Verified CT-to-DRR forward projection physics engine (`scripts/3d_reconstruction/generate_drr.py`).
  - Execution tested on `demo_thoracic_scan` in 145 ms; produces calibrated Frontal (PA) and Lateral (LL) DRR PNGs + JSON metadata with SHA-256 hashes.
- [x] **Action 4.2 (COMPLETED):** Manifest schema definition and patient-disjoint validator (`scripts/3d_reconstruction/validate_manifest.py`).
  - Enforces 14 mandatory fields and strictly verifies zero patient data leakage across TRAIN, VAL, and TEST splits.
- [ ] **Action 4.3:** Generate canonical genus-0 template meshes for Left Lung ($5,000$ vertices) and Right Lung ($5,000$ vertices) from reference case `LIDC-IDRI-0001` using Marching Cubes and isotropic remeshing.
- [ ] **Action 4.4:** Generate 16-landmark ground truth coordinate catalog for training cohort.

---

## 3. Pre-Flight Storage & Compute Verification
1. **Disk Space Budget:** Ensure at least $150\text{ GB}$ of free local storage on an encrypted volume (`BitLocker` / `LUKS`) before initiating LUNA16 `subset00` and IN-CXR downloads.
2. **GPU Quota Invariant:** Zero GPU quota consumed during Phase 1 preparation; all projection generation and SSM baseline fitting routines are verified to execute purely on multi-core CPU.
3. **Repository Cleanliness:** Confirm all raw medical image formats (`.mhd`, `.raw`, `.nii.gz`, `.dcm`, `.tgz`) remain strictly untracked by Git.

