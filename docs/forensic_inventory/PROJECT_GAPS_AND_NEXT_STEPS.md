# LungVision AI — Project Gaps, Risk Analysis & Strategic Roadmap

**Audit Date:** November 13, 2026  
**Auditor:** Principal Software Architect & Medical Imaging Scientist  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Scope:** Rigorous identification of technical gaps, computational limitations, clinical risks, and prioritized next steps.

---

## 1. Technical & Scientific Gap Analysis

```
[Current State: Working 2D AI + Physics DRR Engine + Reference Mesh]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
[Pillar 1: Data Pipeline Gaps]        [Pillar 2: 3D Model Gaps]
• Local raw CT storage absent         • SSM trained on single reference
• Missing multi-patient CT cohort      • PCT model is research prototype
• Cloud datasets require sync         • Lacks multi-patient PCA basis
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
[Pillar 3: Clinical & Deployment Gaps]
• Lacks direct PACS / DICOM C-STORE
• FDA Class II SaMD compliance pending
• Multi-reader radiologist trial unperformed
```

### Detailed Gap Inventory:
1. **SSM Training Cohort:** The current SSM baseline demonstrates mathematical deformation on a reference thoracic volume. To capture human anatomical variance (pectus excavatum, emphysema, scoliosis), PCA eigenmodes must be fitted across at least 100+ multi-patient CT segmentations.
2. **Local CT Data Presence:** Full raw LUNA16 CT scans (220 GB) reside on Kaggle cloud mounts. Local development relies on the verified calibrated reference scan.
3. **Point Cloud Transformer Backbones:** The PCT model (`scripts/reconstruction/pct_reconstruction.py`) is structurally sound but requires end-to-end multi-GPU training on paired DRR-CT projections.
4. **Clinical DICOM Network Integration:** The backend currently accepts standard HTTP file uploads (`multipart/form-data`) but lacks native DICOM DIMSE networking (C-STORE, C-FIND, C-MOVE) for hospital PACS integration.

---

## 2. Explicit Non-Recommendations (What NOT to Do)

> [!CAUTION]
> **Strict Engineering Boundaries:**
> 1. **DO NOT re-download the 220 GB LUNA16 dataset locally** unless dedicated encrypted enterprise storage is provisioned. The data already exists and was processed in Kaggle environments.
> 2. **DO NOT train an unconstrained 3D GAN or Diffusion model.** Unconstrained neural 3D generators hallucinate anatomies and fabricate non-existent pulmonary nodules, posing severe patient safety risks.
> 3. **DO NOT extract `results.zip`.** All 45 archived files are confirmed present on disk. Extracting will merely duplicate files or overwrite active metadata.
> 4. **DO NOT bypass patient-level cross-validation splits.** Doing so re-introduces the patient-overlap shortcut bias discovered during earlier research audits.

---

## 3. Prioritized Strategic Next Steps

### Phase 2B: Multi-Patient SSM PCA Training (Next Logical Milestone)
- **Action:** Execute PCA fitting over the precomputed LUNA16 lung masks available in Kaggle cloud storage.
- **Target:** Extract the top 20 principal deformation modes representing $\ge 95\%$ of anatomical lung variance across gender, age, and inspiratory volumes.
- **Resource Requirement:** Cloud compute instance with 32 GB RAM, execution time $\sim 45$ minutes.

### Phase 3: Biplanar DRR-to-SSM Optimization Pipeline
- **Action:** Wire the 10 detected biplanar landmarks and silhouette boundaries to dynamically optimize SSM deformation coefficients using PyTorch L-BFGS.
- **Target:** Achieve mean Chamfer surface distance $< 2.5\,\text{mm}$ and volumetric Dice score $> 0.88$ on held-out test scans.

### Phase 4: Hospital PACS & DICOM Integration
- **Action:** Implement a lightweight `pynetdicom` C-STORE receiver service in `backend/app/services/dicom_listener.py`.
- **Target:** Allow radiologists to push chest radiographs directly from acquisition consoles to LungVision AI.
