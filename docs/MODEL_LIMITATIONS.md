# LungVision AI — Clinical AI Limitations & Safety Governance

## 1. Intended Use & Regulatory Scope

> [!CAUTION]
> **RESEARCH USE ONLY (RUO)**:
> LungVision AI is an experimental computer-assisted detection (CADe) and decision-support prototype. It is NOT cleared or approved by the US FDA, EMA, or other regulatory authorities for primary diagnostic use, automated triage, or treatment selection. All findings must be interpreted by a board-certified radiologist or licensed medical professional.

---

## 2. Dataset Scope & Cohort Boundaries

### 2.1 Training Distribution
- The Phase 7 model was trained on the **NIH ChestX-ray14** dataset, which consists of posteroanterior (PA) and anteroposterior (AP) chest radiographs from clinical centers in North America.
- **Adult Demographics**: Primarily adult patients ($\ge 18$ years of age). Performance on pediatric or neonate populations is unvalidated and likely unreliable due to differences in thymic shadow, cardiac silhouette proportions, and pulmonary lucency.
- **Standard Projections**: Optimized for erect frontal thoracic projections. Decubitus, lordotic, oblique, or lateral projections are out-of-distribution.

---

## 3. Known Failure Modes & Diagnostic Confounders

### 3.1 Medical Devices & Support Hardware
- Endotracheal tubes, tracheostomies, central venous catheters, pacemakers, ICD leads, and sternotomy wires introduce sharp high-contrast metal artifacts.
- MultiScaleDenseNet121 features multi-scale edge filters that may occasionally misattribute apical catheter lines as subtle apical pneumothoraces.

### 3.2 Overlap of Infiltrative Pathologies
- Pneumonia, consolidation, and localized pulmonary edema exhibit substantial radiological overlap (airspace opacification, air bronchograms).
- While the model outputs independent logits for each pathology, multi-pathology co-occurrence requires clinical correlation (fever, WBC, BNP levels, history).

### 3.3 Patient Positioning & Inspiration Depth
- Poor inspiratory effort (fewer than 9-10 posterior ribs visible above the diaphragm) produces basilar atelectasis and artificial apparent cardiomegaly.
- Significant patient rotation alters the cardiothoracic ratio and asymmetrical lung field attenuation.

---

## 4. Operational Mitigations Implemented

1. **Pre-Inference Quality Audit**:
   - Rejects unreadable bitstreams, corrupt files, blank images, and images with dimensions $< 128 \times 128$.
   - Flags degraded high-frequency detail (blur scores $< 10.0$).
2. **Margin-Based Abstention**:
   - Categorizes findings within $\pm 0.05$ of the operating threshold as `BORDERLINE` and prompts for expert clinician overread.
3. **Multi-Scale Grad-CAM**:
   - Provides visual evidence to allow clinicians to inspect which anatomical lung zones drove model activations.
