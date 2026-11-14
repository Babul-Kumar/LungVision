# LungVision AI — 3D Lung Reconstruction Validation Protocol & Metrics

**Document Version:** 1.0.0  
**Protocol Standard:** Quantitative Volumetric & Surface Validation Framework  
**Lead Author:** Senior Medical Imaging Scientist & Evaluation Auditor  
**Status:** Research Protocol Specification  
**Date:** 2026-11-13  

---

## 1. Executive Summary

This protocol defines the formal mathematical metrics, statistical evaluation standards, and data partitioning rules required to validate **patient-specific 3D lung reconstructions** against ground-truth computed tomography (CT) scans.

Every reconstructed lung model must satisfy rigorous geometric, volumetric, and anatomical criteria before deployment in research or clinical evaluation.

---

## 2. Patient-Disjoint Data Partitioning (Zero Patient Leakage)

A fundamental vulnerability in medical machine learning is **patient leakage**—where radiographs or CT scans from the same individual patient appear in both the training set and the test set.

### Splitting Protocol
1. **Partitioning Key:** Splitting is performed exclusively on `patient_id`, never on `study_id` or `image_id`.
2. **Standard Ratio:** $70\%$ Training / $15\%$ Validation / $15\%$ Held-Out Test.
3. **Stratification Variables:**
   - Patient Sex (balanced $50/50$ male/female).
   - Thoracic Volume Tertiles (Small: $< 3,000\text{ cm}^3$, Medium: $3,000\text{--}4,800\text{ cm}^3$, Large: $> 4,800\text{ cm}^3$).
   - Pathological Deformity Status (Normal vs. Pleural Effusion / Lobar Collapse / Hyperinflation).
4. **Leakage Verification Script:** Automated set intersection check asserting:
   $$\text{Patients}_{\text{train}} \cap \text{Patients}_{\text{val}} = \emptyset, \quad \text{Patients}_{\text{train}} \cap \text{Patients}_{\text{test}} = \emptyset, \quad \text{Patients}_{\text{val}} \cap \text{Patients}_{\text{test}} = \emptyset$$

---

## 3. Spatial Registration Protocol (CXR-to-CT Alignment)

When validating against ground-truth CT scans, differences in patient positioning between 2D radiography (standing, arms forward or overhead) and 3D CT (supine, arms extended along the scanner bed) induce physiological deformations.

### Two-Stage Alignment Procedure
1. **Stage 1: Rigid Translation & Rotation Alignment (Procrustes on Bony Landmarks):**
   - Align reconstructed mesh $\mathbf{M}_{\text{pred}}$ to CT segmentation $\mathbf{M}_{\text{true}}$ using rigid Procrustes registration based on 6 skeletal landmarks (sternal notch, T1 vertebra, bilateral clavicular heads, xiphisternal joint, T12 vertebra).
   - Yields optimal rotation $\mathbf{R} \in SO(3)$ and translation $\mathbf{t} \in \mathbb{R}^3$.
2. **Stage 2: Affine Scaling Normalization:**
   - Accommodates global inspiratory lung volume differences between upright inspiratory CXR and supine CT breathing states:
     $$\mathbf{x}_{\text{aligned}} = \mathbf{S} \mathbf{R} \mathbf{x} + \mathbf{t}$$
   - Both rigid and unconstrained metrics must be reported to isolate algorithm reconstruction error from respiratory phase variance.

---

## 4. Formal Mathematical Metrics

### 4.1 Volumetric Overlap: 3D Dice Similarity Coefficient (DSC)
Measures the spatial overlap between predicted volume $V_{\text{pred}}$ and reference CT volume $V_{\text{true}}$:

$$\text{DSC} = \frac{2 \cdot |V_{\text{pred}} \cap V_{\text{true}}|}{|V_{\text{pred}}| + |V_{\text{true}}|}$$

- **Target Acceptance Threshold:** $\text{DSC} \ge 0.88$ for whole lungs; $\text{DSC} \ge 0.85$ for individual left/right lungs.
- **Critical Failure:** $\text{DSC} < 0.80$.

### 4.2 Surface Distance: Average Symmetric Surface Distance (ASSD)
Computes the average Euclidean distance between predicted mesh surface $\mathcal{S}_{\text{pred}}$ and reference surface $\mathcal{S}_{\text{true}}$:

$$\text{ASSD} = \frac{1}{|\mathcal{S}_{\text{pred}}| + |\mathcal{S}_{\text{true}}|} \left(\sum_{\mathbf{p} \in \mathcal{S}_{\text{pred}}} \min_{\mathbf{q} \in \mathcal{S}_{\text{true}}} \|\mathbf{p} - \mathbf{q}\|_2 + \sum_{\mathbf{q} \in \mathcal{S}_{\text{true}}} \min_{\mathbf{p} \in \mathcal{S}_{\text{pred}}} \|\mathbf{q} - \mathbf{p}\|_2\right)$$

- **Target Acceptance Threshold:** $\text{ASSD} \le 2.50\text{ mm}$.
- **Critical Failure:** $\text{ASSD} > 4.50\text{ mm}$.

### 4.3 Outlier Distance: 95th Percentile Hausdorff Distance (HD95)
Measures the maximum distance between surfaces, excluding the top $5\%$ of extreme outliers (e.g. at sharp costophrenic angles):

$$\text{HD95} = \text{percentile}_{95}\left(\left\{\min_{\mathbf{q} \in \mathcal{S}_{\text{true}}} \|\mathbf{p} - \mathbf{q}\|_2 : \mathbf{p} \in \mathcal{S}_{\text{pred}}\right\} \cup \left\{\min_{\mathbf{p} \in \mathcal{S}_{\text{pred}}} \|\mathbf{q} - \mathbf{p}\|_2 : \mathbf{q} \in \mathcal{S}_{\text{true}}\right\}\right)$$

- **Target Acceptance Threshold:** $\text{HD95} \le 7.00\text{ mm}$.
- **Critical Failure:** $\text{HD95} > 12.00\text{ mm}$.

### 4.4 Volume Estimation Error Percentage (VEE%)
Quantifies the clinical accuracy of automated lung volumetry against CT water-displacement volume:

$$\text{VEE}\% = \frac{|\text{Vol}(V_{\text{pred}}) - \text{Vol}(V_{\text{true}})|}{\text{Vol}(V_{\text{true}})} \times 100\%$$

- **Target Acceptance Threshold:** $\text{VEE}\% \le 7.5\%$.
- **Clinical Relevance:** Vital for longitudinal assessment of chronic obstructive pulmonary disease (COPD) and total lung capacity (TLC) monitoring.

### 4.5 Landmark Localization Error (LLE)
Mean Euclidean error across the 16 standard thoracic landmarks:

$$\text{LLE} = \frac{1}{K} \sum_{k=1}^K \|\mathbf{L}_{\text{pred}}^{(k)} - \mathbf{L}_{\text{true}}^{(k)}\|_2$$

- **Target Acceptance Threshold:** $\text{LLE} \le 4.0\text{ mm}$.

---

## 5. Uncertainty Quantification & Calibration

Reconstructed surfaces must communicate their localized confidence to radiologists:
1. **Uncertainty Calibration:** Pearson correlation $r$ between per-vertex predicted uncertainty $\sigma_v$ and true surface error $d(\mathbf{v}, \mathcal{S}_{\text{true}})$.
   $$\text{Requirement:} \quad r(\sigma_v, d_v) \ge 0.65$$
2. **Failure Zone Flagging:** Any anatomical segment with mean uncertainty $\sigma_{\text{region}} > 0.60$ (e.g. retrocardiac space obscured by cardiac opacification) must be automatically highlighted in amber on the 3D Three.js canvas with an explanatory tooltip.

---

## 6. External Generalization Benchmark

To ensure the 3D reconstruction pipeline generalizes across different hospital scanners and demographics:
- **Internal Benchmark:** LUNA16 / LIDC-IDRI synthetic bi-planar DRRs ($N = 250$ test scans).
- **External Clinical Benchmark:** MIDRC multi-center clinical CXR+CT cohort ($N = 150$ patients).
- **Domain Shift Degradation Bound:** $\text{DSC}_{\text{external}} \ge \text{DSC}_{\text{internal}} - 0.05$ (maximum allowable drop: $5.0$ percentage points).

