# LungVision AI — 3D Reconstruction & SSM Baseline Experiment Protocol
**Document ID:** `LV-RESEARCH-3D-EXP-001`  
**Version:** 1.0.0  
**Status:** Approved for Implementation  
**Auditor/Author:** LungVision AI Research Evaluation & Biostatistics Group  
**Framework:** Gravitas Engineering Reliability Runtime  

---

## 1. Experimental Objective & Protocol Scope

This protocol establishes the standardized, reproducible benchmarking and evaluation procedures for reconstructing patient-specific 3D anatomical lung surface models from dual-view (frontal and lateral) chest radiographs. The protocol governs all testing on held-out CT test sets, establishing mathematical definitions, implementation algorithms, acceptance thresholds, and statistical reporting standards.

To avoid subjective evaluation or selective metric reporting, all experiments must report the full metric battery defined herein across all held-out test subjects.

---

## 2. Quantitative Metric Battery & Mathematical Formulations

Reconstruction quality is evaluated across three complementary physical dimensions: **Volumetric Overlap**, **Boundary Surface Accuracy**, and **Clinical Organ Volumetry**.

```
                +------------------------------------------------+
                |        Metric Battery & Acceptance Gates       |
                +------------------------------------------------+
                |  1. Volumetric Overlap:    3D Dice (DSC) >= 0.88|
                |  2. Surface Proximity:     ASSD <= 2.5 mm      |
                |  3. Boundary Extremes:     HD95 <= 7.0 mm      |
                |  4. Clinical Volumetry:    VEE% <= 7.5%        |
                |  5. Landmark Fidelity:     LLE <= 4.0 mm       |
                |  6. Uncertainty Calib.:    Spearman r >= 0.65  |
                +------------------------------------------------+
```

### 2.1. Metric 1: 3D Dice Similarity Coefficient ($DSC$)
Measures the volumetric spatial overlap between the predicted binary 3D lung mask $V_{\text{pred}} \subset \mathbb{Z}^3$ and the ground truth CT segmentation $V_{\text{gt}} \subset \mathbb{Z}^3$:

$$DSC = \frac{2 \cdot |V_{\text{pred}} \cap V_{\text{gt}}|}{|V_{\text{pred}}| + |V_{\text{gt}}|} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

* **Voxelization Procedure:** The reconstructed continuous 3D surface mesh $\mathcal{M}_{\text{pred}}$ is voxelized onto the reference CT voxel grid using a watertight winding-number ray intersection algorithm.
* **Acceptance Gate:**
  * **Mean $DSC \ge 0.88$** across the held-out test set.
  * **Critical Floor:** No individual test subject may exhibit $DSC < 0.78$.

### 2.2. Metric 2: Average Symmetric Surface Distance ($ASSD$)
Measures the average physical Euclidean distance (in millimeters) between the predicted continuous surface $\mathcal{S}_{\text{pred}}$ and the ground truth continuous surface $\mathcal{S}_{\text{gt}}$:

$$ASSD = \frac{1}{|\mathcal{S}_{\text{pred}}| + |\mathcal{S}_{\text{gt}}|} \left( \int_{x \in \mathcal{S}_{\text{pred}}} \min_{y \in \mathcal{S}_{\text{gt}}} \|x - y\|_2 \, dx + \int_{y \in \mathcal{S}_{\text{gt}}} \min_{x \in \mathcal{S}_{\text{pred}}} \|y - x\|_2 \, dy \right)$$

* **Numerical Computation:** Surfaces are densely sampled with $P = 100,000$ points uniformly distributed by face area, and nearest-neighbor Euclidean distances are computed via KD-Tree.
* **Acceptance Gate:**
  * **Mean $ASSD \le 2.50\text{ mm}$**.
  * **Upper Limit:** 90th percentile $ASSD \le 3.50\text{ mm}$.

### 2.3. Metric 3: 95th Percentile Hausdorff Distance ($HD95$)
Quantifies the worst-case localized boundary discrepancy, robustified against non-anatomical outlier points:

$$d(x, \mathcal{S}) = \min_{y \in \mathcal{S}} \|x - y\|_2$$
$$HD95 = \max\left( P_{95\%}\left(\{d(x, \mathcal{S}_{\text{gt}}) \mid x \in \mathcal{S}_{\text{pred}}\}\right),\, P_{95\%}\left(\{d(y, \mathcal{S}_{\text{pred}}) \mid y \in \mathcal{S}_{\text{gt}}\}\right) \right)$$

where $P_{95\%}(\cdot)$ denotes the 95th percentile value of the distance distribution.
* **Acceptance Gate:**
  * **Mean $HD95 \le 7.00\text{ mm}$**.
  * **Worst-Case Floor:** $HD95 \le 12.00\text{ mm}$.

### 2.4. Metric 4: Absolute Volume Estimation Error ($VEE\%$)
Evaluates the physiological fidelity of the total reconstructed lung volume relative to the ground truth CT lung volume:

$$\text{Vol}(\mathcal{M}) = \frac{1}{6} \left| \sum_{(i, j, k) \in \mathcal{F}} \mathbf{v}_i \cdot (\mathbf{v}_j \times \mathbf{v}_k) \right|$$
$$VEE\% = \frac{\left| \text{Vol}(\mathcal{M}_{\text{pred}}) - \text{Vol}(\mathcal{M}_{\text{gt}}) \right|}{\text{Vol}(\mathcal{M}_{\text{gt}})} \times 100\%$$

* **Clinical Significance:** Total lung volume is a vital pulmonary function biomarker. A volume error $\le 7.5\%$ falls well within normal day-to-day inspiratory variation in spirometry testing.
* **Acceptance Gate:**
  * **Mean $VEE\% \le 7.50\%$**.
  * **Worst-Case Floor:** $VEE\% \le 15.00\%$.

### 2.5. Metric 5: 3D Landmark Localization Error ($LLE$)
Evaluates anatomical correspondence across the 16 standardized thoracic landmarks:

$$LLE = \frac{1}{16} \sum_{i=1}^{16} \left\| \mathbf{L}_{\text{pred}}^{(i)} - \mathbf{L}_{\text{gt}}^{(i)} \right\|_2$$

* **Acceptance Gate:**
  * **Mean $LLE \le 4.00\text{ mm}$**.
  * **Apical & Basal Sulcus Landmarks:** $\le 3.50\text{ mm}$.

### 2.6. Metric 6: Surface Uncertainty Calibration ($Spearman\ r$)
Evaluates whether the model's predicted vertex-wise uncertainty $\sigma_i$ accurately predicts real reconstruction error $e_i = \min_{y \in \mathcal{S}_{\text{gt}}} \|\mathbf{v}_i - y\|_2$:

$$r_{\text{calib}} = \text{SpearmanRankCorr}\left( \{\sigma_i\}_{i=1}^N,\, \{e_i\}_{i=1}^N \right)$$

* **Acceptance Gate:**
  * **Rank Correlation $r_{\text{calib}} \ge 0.65$** ($p < 0.001$).
  * Demonstrates that regions of high uncertainty (e.g., hidden mediastinal surfaces) are reliably flagged for clinical review.

---

## 3. Benchmark Comparison Matrix & Performance Targets

| Evaluation Dimension | Metric | Baseline Target (SSM Dual-View) | State-of-the-Art Deep Model Target | Minimum Acceptance Floor |
| :--- | :--- | :---: | :---: | :---: |
| **Volumetric Overlap** | **3D Dice ($DSC$)** | **$\ge 0.880$** | $\ge 0.925$ | $0.780$ |
| **Average Surface Proximity** | **$ASSD$** | **$\le 2.50\text{ mm}$** | $\le 1.60\text{ mm}$ | $3.50\text{ mm}$ |
| **Outlier Surface Distance** | **$HD95$** | **$\le 7.00\text{ mm}$** | $\le 4.50\text{ mm}$ | $12.00\text{ mm}$ |
| **Organ Volumetry** | **$VEE\%$** | **$\le 7.50\%$** | $\le 4.00\%$ | $15.00\%$ |
| **Landmark Localization** | **$LLE$** | **$\le 4.00\text{ mm}$** | $\le 2.50\text{ mm}$ | $6.00\text{ mm}$ |
| **Uncertainty Calibration** | **$Spearman\ r$** | **$\ge 0.650$** | $\ge 0.750$ | $0.500$ |

---

## 4. Subgroup Stratification & Ablation Protocol

To prove model robustness across diverse patient populations and experimental conditions, results must be stratified across the following cohorts:

1. **Inspiratory Volume Quartiles:**
   * Quartile 1: Low inspiratory capacity ($V_{\text{lung}} < 2.8\text{ L}$).
   * Quartiles 2–3: Standard physiological range ($2.8\text{ L} \le V_{\text{lung}} \le 5.2\text{ L}$).
   * Quartile 4: Hyperinflated / COPD lungs ($V_{\text{lung}} > 5.2\text{ L}$).
2. **Pathology Presence vs Normal Anatomy:**
   * Stratified into: Clear lungs, Unilateral consolidation, Pleural effusion, and Cardiomegaly.
3. **Dual-View vs Single-View Ablation:**
   * Every subject must be evaluated under:
     * Condition A: **Dual-View (Frontal + Lateral)**.
     * Condition B: **Single-View (Frontal Only)**.
   * Hypothesis Test: Dual-view must achieve statistically significant improvements in $HD95$ and $ASSD$ over single-view ($p < 0.01$, Wilcoxon signed-rank test).

---

## 5. Statistical Rigor & Reporting Standards

1. **Confidence Intervals:** All reported summary metrics must provide $95\%$ bootstrap confidence intervals computed over $B = 1,000$ resamples.
2. **No Data Snooping:** Hyperparameters (such as regularization weights $w_{\text{prior}}, w_{\text{smooth}}$ and PCA mode cutoff $M$) must be tuned exclusively on the **VAL** split. The **TEST** split must remain sealed until final evaluation.
3. **Reproducibility Manifest:** Every experimental run must record:
   - Random seed (fixed to `42`).
   - Git commit hash of the codebase.
   - SHA-256 hash of the input dataset manifest.
   - Complete execution runtime in seconds and memory footprint.

