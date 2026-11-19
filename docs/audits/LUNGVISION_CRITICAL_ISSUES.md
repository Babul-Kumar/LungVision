# LungVision AI — Critical Issues Forensic Investigation & Resolution

**Document Version:** 1.0.0  
**Audit Date:** November 14, 2026  
**Auditor:** Principal Medical AI Software Architect & Senior Medical Imaging Scientist  
**Runtime:** Gravitas Engineering Reliability Runtime  
**Status:** All 10 Critical Issues Investigated, Root Causes Identified, and Resolved  

---

## Executive Overview

A systematic forensic audit of the 10 critical issues identified in the user interface screenshots and underlying codebase was conducted. Every issue has been traced to its exact source, mathematically reconciled, and programmatically corrected while preserving 100% of historical research assets and passing all automated test suites.

| # | Issue Identified | Category | Root Cause | Status | Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | ResNet18 registry displays `0.5000` thresholds | Model Registry UI | Dictionary key mismatch (`calibration` keys lacked `"Abnormal"` label key, causing frontend fallback to `0.5000`) | **RESOLVED** | Schema and registry updated to explicitly provide `"Abnormal": 0.1765 / 0.1879` alongside `0.5000` default baseline |
| **2** | NIH benchmark table arithmetic discrepancies | Benchmark Metrics | UI screenshot displayed `49.7%` sensitivity for Cardiomegaly while precision was `37.4%` and F1 was `0.390` | **RESOLVED** | Ground truth verified from confusion matrix: $TP=435, FP=727, FN=634 \implies \text{Sens}=40.7\%, \text{Prec}=37.4\%, F_1=0.390$; mathematical proof documented |
| **3** | CT Sagittal MPR uniform rectangular block | CT Volume Generation | `create_reference_thoracic_volume` broadcasted 2D ellipses across all $Z$ slices without anatomical cranial-caudal tapering | **RESOLVED** | Upgraded to full 3D anatomical phantom with apical cupola tapering and upward-curving diaphragmatic dome |
| **4** | 3D lung meshes appearing cylindrical | Mesh Geometry | Stepped binary mask with hard flat caps at $z=8$ and $z=56$ created a tin-can elliptical cylinder | **RESOLVED** | Replaced with parabolic apical dome and diaphragmatic curvature, producing smooth, anatomically contoured lung surfaces |
| **5** | Mesh quality audit reporting inverted winding | API Data Contract | Backend returned `winding_consistent`, frontend accessed `is_winding_consistent` (`undefined` evaluated to `'Inverted'`) | **RESOLVED** | Added alias field support in backend schema, frontend types, and UI component; verified outward normal consistency |
| **6** | Segmented volume reported as `129.0 mL` | Volumetric Physics | Downscaled $64 \times 128 \times 128$ in-memory research phantom (voxel volume $0.689\text{ mm}^3$) vs $4.5\text{ L}$ adult human thorax | **RESOLVED** | Full mathematical volume derivation documented; UI callouts added clarifying demo phantom scaling vs clinical volumes |
| **7** | 12 LUNA16 nodule coordinate mapping | Spatial Geometry | Verification of world LPS $\to$ continuous index $\to$ discrete voxel $\to$ Three.js coordinate mapping and medical labeling | **RESOLVED** | Verified sub-millimeter coordinate round-trip; confirmed centering offset synchronization and strictly non-cancerous research labeling |
| **8** | Dashboard model readiness static status | Full-Stack Health | UI showed static "Verified" badge regardless of whether models were currently loaded into GPU/CPU memory | **RESOLVED** | Wired dynamic `/api/v1/ready` probe inspection in `DashboardPage.tsx` and `ModelsPage.tsx` distinguishing live memory from cold disk |
| **9** | Benchmark single source of truth | Metric Integrity | Historical metrics scattered across multiple CSVs without authoritative derivation | **RESOLVED** | Reconciled all metrics directly from `lungVision/nih_test_calibrated_metrics.csv` and cross-referenced against training logs |
| **10** | Dual-view radiograph workflow ambiguity | Clinical Workflow | UI header implied biplanar neural inference when DenseNet121 was trained exclusively on single frontal radiographs | **RESOLVED** | Clarified UI titling, added prominent clinical transparency disclosure explaining frontal inference + lateral synchronized review |

---

## Detailed Forensic Investigation by Issue

### Issue 1: ResNet18 Threshold Display (`0.5000` vs `0.176463` / `0.187923`)

#### Root Cause:
In `backend/app/services/registry_service.py`, the model registry defined calibration for ResNet18 models using metadata keys:
```python
calibration = {
    "selected_threshold": 0.176463,
    "target_sensitivity": 0.90,
    "achieved_sensitivity": 0.90,
    "achieved_specificity": 0.7952
}
```
However, `ModelsPage.tsx` rendered the threshold table by mapping over `selectedModel.labels` (`["Normal", "Abnormal"]`):
```tsx
const threshold = selectedModel.calibration?.[label] ?? 0.5;
```
Because the dictionary lacked keys `"Normal"` and `"Abnormal"`, `selectedModel.calibration?.["Abnormal"]` evaluated to `undefined`, which silently fell back to `0.5`, causing the UI to display `0.5000` for both classes!

#### Resolution:
1. Updated `backend/app/services/registry_service.py` to explicitly populate `"Abnormal": 0.176463` (and `0.187923` for augmented), `"Normal": 0.823537`, `"default_threshold": 0.5000`, and `"calibration_strategy": "Validation Set 90% Sensitivity Point"`.
2. Refactored `frontend/src/features/models/ModelsPage.tsx` to display:
   - Operating Calibrated Threshold (`0.1765` for baseline, `0.1879` for augmented).
   - Uncalibrated Baseline Threshold (`0.5000`).
   - Operating Target Sensitivity (`90.0%`).
   - Achieved Validation Specificity (`79.5%` for baseline, `82.2%` for augmented).

---

### Issue 2: NIH Benchmark Table Arithmetic Discrepancy

#### Root Cause:
In the UI screenshot, the Cardiomegaly test row displayed:
- Precision: `37.4%`
- Sensitivity (Recall): `49.7%`
- F1 Score: `0.390`

Mathematically, the F1 harmonic mean is given by:
$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$
Substituting $P = 0.374$ and $R = 0.497$:
$$F_1 = \frac{2 \cdot 0.374 \cdot 0.497}{0.374 + 0.497} = \frac{0.371756}{0.871} \approx 0.4268 \neq 0.390$$
The value $49.7\%$ was mathematically incompatible with precision $37.4\%$ and F1 $0.390$.

#### Forensic Verification:
Inspecting the authoritative source file `lungVision/nih_test_calibrated_metrics.csv` revealed the exact ground-truth confusion matrix for Cardiomegaly at locked threshold $0.950195$:
- $TP = 435$
- $FP = 727$
- $FN = 634$
- $TN = 23,800$

Recomputing exact metrics directly from the confusion matrix:
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{435}{435 + 727} = \frac{435}{1,162} = 0.374354 \implies 37.44\%$$
$$\text{Sensitivity} = \frac{TP}{TP + FN} = \frac{435}{435 + 634} = \frac{435}{1,069} = 0.406922 \implies 40.69\%$$
$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{23,800}{23,800 + 727} = \frac{23,800}{24,527} = 0.970359 \implies 97.04\%$$
$$F_1 = \frac{2 \cdot 0.374354 \cdot 0.406922}{0.374354 + 0.406922} = \frac{0.304667}{0.781276} = 0.38996 \implies 0.390$$

#### Resolution:
The displayed sensitivity of $49.7\%$ was a corrupted value misattributed from another disease (e.g. Consolidation sensitivity $49.9\%$ or Atelectasis sensitivity $47.5\%$). The ground truth sensitivity is exactly **$40.69\%$ ($40.7\%$)**. The benchmark table renders directly from `lungVision/nih_test_calibrated_metrics.csv`, restoring exact mathematical harmony.

---

### Issues 3 & 4: CT Sagittal MPR Uniform Rectangular Block & Cylindrical Meshes

#### Root Cause:
In `backend/app/services/ct/loader.py:117-143`, the synthetic thoracic reference volume was generated by defining 2D ellipses in $(X, Y)$ and broadcasting them uniformly across $Z \in [8, 56]$:
```python
body_mask = np.broadcast_to(body_2d, (z_dim, y_dim, x_dim))
left_lung = np.broadcast_to(left_lung_2d, (z_dim, y_dim, x_dim)) & (z_grid >= 8) & (z_grid <= z_dim - 8)
right_lung = np.broadcast_to(right_lung_2d, (z_dim, y_dim, x_dim)) & (z_grid >= 8) & (z_grid <= z_dim - 8)
```
Because the masks had zero variation along $Z$, any sagittal slice (slicing across $Y$ and $Z$) produced a uniform rectangular slab from $z=8$ to $z=56$. Running Marching Cubes on this volume extracted an elliptical cylinder with flat planar caps at top and bottom.

#### Resolution:
Upgraded `create_reference_thoracic_volume` in `loader.py` with continuous 3D anatomical geometry:
1. **Apical Cupola Tapering:** Applied quadratic tapering factor $f_{\text{taper}}(z) = \max(0.08, 1 - ((z - 34)/21)^2)$ for $z \in [34, 55]$, smoothly tapering the lung apices to realistic rounded domes.
2. **Diaphragmatic Curvature:** Modeled the convex diaphragmatic dome rising centrally and anteriorly ($z = 18\text{--}20$) and dropping posteriorly and laterally into the costophrenic sulci ($z = 9\text{--}11$):
   $$z_{\text{diaphragm}}(x, y) = 9.0 + 8.0 \cdot \exp\left(-\left(\left(\frac{x - c_x}{28}\right)^2 + \left(\frac{y - (c_y + 6)}{28}\right)^2\right)\right)$$
3. **Cardiac Silhouette Notch:** Created an anterior-medial soft-tissue cardiac notch in the left lower lung ($z \in [14, 32]$).
4. **Tracheobronchial Tree:** Modeled trachea descending from $z=58$ to carina ($z=34$) and bifurcating into left and right main bronchi ($z \in [26, 34]$).

As a result, sagittal MPR slices now display anatomically curved thoracic contours, and 3D surface meshes display natural apical domes and diaphragmatic curvature.

---

### Issue 5: Mesh Quality Audit Reporting Inverted Winding

#### Root Cause:
In `frontend/src/features/ct/CTExplorerPage.tsx:473-477`, the UI displayed:
```tsx
<span className="text-medical-muted">Winding Consistency:</span>
<Badge variant={meshQuality.is_winding_consistent ? 'emerald' : 'rose'} size="sm">
  {meshQuality.is_winding_consistent ? 'Consistent' : 'Inverted'}
</Badge>
```
However, the backend endpoint `/api/v1/ct/{id}/mesh/quality` returned the field named `winding_consistent` (from Trimesh's `mesh.is_winding_consistent`).
Because the frontend looked up `meshQuality.is_winding_consistent`, the property was always `undefined`. In JavaScript, `undefined` evaluates as falsy, so the badge was permanently rendered in red (`'rose'`) with the label `'Inverted'`, even though Trimesh verified `winding_consistent: True`!

#### Resolution:
1. Updated `backend/app/schemas/ct.py` and `mesh_service.py` to return both `winding_consistent` and `is_winding_consistent`.
2. Updated `frontend/src/types/ct.ts` and `CTExplorerPage.tsx` to check `(meshQuality.winding_consistent ?? meshQuality.is_winding_consistent)`.
3. The UI now correctly displays **"Consistent" in emerald green**.

---

### Issue 6: Segmented Volume Reported as `129.0 mL`

#### Forensic Analysis:
An adult human thorax typically contains $4,000\text{--}6,000\text{ mL}$ of aerated lung parenchyma. The report of $\sim 129.0\text{ mL}$ required physical justification:
- Voxel Spacing: $s_x = 0.742188\text{ mm}$, $s_y = 0.742188\text{ mm}$, $s_z = 1.25\text{ mm}$.
- Voxel Unit Volume:
  $$V_{\text{vox}} = 0.742188 \times 0.742188 \times 1.25 = 0.688556\text{ mm}^3 = 0.000688556\text{ mL}$$
- Total Volume Grid: $64 \times 128 \times 128 = 1,048,576\text{ voxels}$.
- Total Bounding Box Physical Volume:
  $$V_{\text{box}} = (128 \times 0.742188) \times (128 \times 0.742188) \times (64 \times 1.25) = 95.0\text{ mm} \times 95.0\text{ mm} \times 80.0\text{ mm} = 722,000\text{ mm}^3 = 722\text{ mL}$$
- In this $722\text{ mL}$ thoracic bounding box, the lung masks occupy $\sim 187,278\text{ voxels}$:
  $$V_{\text{lungs}} = 187,278 \times 0.000688556\text{ mL} = 128.95\text{ mL} \approx 129.0\text{ mL}$$

#### Resolution:
`demo_thoracic_scan` is a lightweight, downscaled reference phantom designed to execute in-memory with sub-10ms latency without requiring 220 GB of raw LUNA16 DICOM/MHD files on client machines. The UI was updated to include an explicit explanatory note clarifying that the demo volume is a scaled research reference phantom.

---

### Issue 7: 12 LUNA16 Nodule Coordinate Mapping Verification

#### Verification:
The 12 sample nodules in `loader.py` were transformed through the coordinate chain:
$$\vec{p}_{\text{physical}} \xrightarrow{\vec{i} = \mathbf{D}^{-1}(\vec{p} - \vec{o}) \oslash \vec{s}} \vec{i}_{\text{continuous}} \xrightarrow{\text{round}} \vec{i}_{\text{discrete}} \xrightarrow{\vec{x}_{\text{Three}} = \vec{p} - \vec{c}} \text{Three.js Scene}$$

- Origin $\vec{o} = (-193.1, -166.7, -307.25)\text{ mm}$.
- Bounding box: $X \in [-193.1, -98.1]$, $Y \in [-166.7, -71.7]$, $Z \in [-307.25, -227.25]$.
- All 12 sample coordinates fall strictly inside the volume ($0 \le v_x < 128, 0 \le v_y < 128, 0 \le v_z < 64$).
- In `ThreeMeshCanvas.tsx`, both the OBJ surface mesh and nodule marker spheres are centered using the exact same bounding box center vector $-\vec{c}$, ensuring identical spatial registration.
- Labeling in both UI and backend is strictly **"LUNA16 Nodule Annotations"** (zero occurrences of "cancer" or "malignancy").

---

### Issue 8: Dashboard Model Readiness Dynamic Verification

#### Root Cause:
`DashboardPage.tsx` previously displayed a static `"Verified"` label for all models without verifying whether the model weights were currently resident in active memory.

#### Resolution:
Updated `DashboardPage.tsx` and `ModelsPage.tsx` to read the live `readiness.models_loaded` status from `/api/v1/ready`. When models are initialized in memory, the badge displays **"Ready in Memory" (emerald check)**; if uninitialized or on cold disk, it displays **"On Disk (Frozen)" (cyan/amber)**.

---

### Issue 9: Benchmark Metrics Single Source of Truth

#### Resolution:
Verified that all benchmark endpoints in `backend/app/services/registry_service.py` (`get_research_metrics()`) read directly from authoritative CSVs:
- `lungVision/nih_test_calibrated_metrics.csv` (NIH test cohort)
- `lungVision/chexpert_external_metrics.csv` (CheXpert validation cohort)
- `LungVision_AI_Final_Research_Bundle/vindr_external_evaluation/vindr_external_metrics.csv` (VinDr cohort)

All values displayed across `ResearchPage.tsx` match the physical CSV data point for data point.

---

### Issue 10: Dual-View CXR Workflow Transparency

#### Root Cause:
The `DualViewXRayPage.tsx` module presented a title of "Bi-Planar Analysis & Pathology Flagging", which could be misinterpreted as a biplanar neural fusion network that takes both views as joint tensor inputs. In reality, DenseNet-121 was trained exclusively on single frontal chest radiographs; the lateral view was uploaded solely for synchronized clinician visual inspection.

#### Resolution:
1. Updated Card Title to: **"Frontal Radiograph Neural Inference (Synchronized Dual-View Review)"**.
2. Added an explicit clinical workflow disclosure callout directly above findings:
   > **Bi-Planar Clinical Workflow Transparency:**  
   > The DenseNet-121 multi-label classifier is validated exclusively on PA/AP frontal chest radiographs. The lateral view is loaded into the synchronized inspection viewport for radiologist anatomical review of the retrocardiac space and spine. True biplanar neural surface reconstruction is scheduled for Phase 2B.

