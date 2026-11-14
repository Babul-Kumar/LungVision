# LungVision AI — Digitally Reconstructed Radiograph (DRR) Physics Reconciliation

**Audit Date:** November 13, 2026  
**Auditor:** Senior Medical Imaging Physicist & Computational Geometry Auditor  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Scope:** Forensic analysis of CT-to-DRR projection geometry, physical coordinates, Siddon-Jacobs ray marching, attenuation conversion, and discrepancy between $SOD/SID$ parameter sets.

---

## 1. Executive Summary & Geometry Discrepancy Resolution

### 1.1 The Identified Discrepancy
Two conflicting sets of projection geometry parameters appear across the codebase and research documentation:
* **Configuration A:** $SOD = 1000.0\text{ mm} \quad / \quad SID = 1400.0\text{ mm} \quad (M = 1.40\times)$
* **Configuration B:** $SOD = 1450.0\text{ mm} \quad / \quad SDD = 1800.0\text{ mm} \quad (M = 1.2414\times)$

### 1.2 Ground Truth Inspection & Pipeline Attribution
Forensic inspection of the executable codebase, research specifications, and generated artifacts reveals:

1. **Which Pipeline Uses Configuration B ($SOD = 1450\text{ mm} / SDD = 1800\text{ mm}$)?**
   * **`research/3d_reconstruction/dataset_acquisition/DRR_GENERATION_SPEC.md`** (Lines 48–60) explicitly mandates $SDD = 1800.0\text{ mm}$ and $SOD = 1450.0\text{ mm}$.
   * **`scripts/3d_reconstruction/generate_drr.py`** (Lines 49–50) sets:
     ```python
     def __init__(
         self,
         sdd_mm: float = 1800.0,
         sod_mm: float = 1450.0,
         detector_width_mm: float = 430.0,
         detector_height_mm: float = 430.0,
         detector_pixel_pitch_mm: float = 1.0,
         ...
     ):
     ```
   * **`research/3d_reconstruction/demo_drr/demo_thoracic_scan_drr_metadata.json`** records the actual parameters used to generate the verified DRR images:
     ```json
     "projection_geometry": {
       "sdd_mm": 1800.0,
       "sod_mm": 1450.0,
       "pixel_pitch_mm": 1.0,
       "intrinsic_matrix": [[1800.0, 0.0, 215.0], [0.0, 1800.0, 215.0], [0.0, 0.0, 1.0]]
     }
     ```
   * **`backend/app/services/ct/geometry.py`** and **`scripts/3d_reconstruction/validate_manifest.py`** enforce $SDD = 1800.0\text{ mm}, SOD = 1450.0\text{ mm}$.

2. **Where Did Configuration A ($SOD = 1000\text{ mm} / SID = 1400\text{ mm}$) Come From?**
   * Configuration A was an informal desktop prototyping specification introduced in early chat discussions and repeated in previous inventory summaries.
   * **Zero executable scripts use $SOD = 1000\text{ mm} / SID = 1400\text{ mm}$.** No generated output in the repository has this geometry.

### 1.3 Radiological & Clinical Significance of Configuration B
Standard clinical chest radiography follows strict American College of Radiology (ACR) and European Society of Radiology (ESR) diagnostic protocols:
* Standard upright Posteroanterior (PA) chest radiographs are acquired at **72 inches ($182.88\text{ cm} \approx 1800.0\text{ mm}$)** source-to-detector distance.
* Positioning the detector at $SDD = 1800\text{ mm}$ and patient isocenter at $SOD = 1450\text{ mm}$ ensures a low geometric magnification factor:
  $$M = \frac{SDD}{SOD} = \frac{1800.0}{1450.0} = 1.2414\times$$
  This minimizes cardiac magnification and thoracic distortion, matching clinical radiograph appearance.
* Configuration A ($M = 1.40\times$) produces an excessively magnified, distorted projection that exaggerates cardiac and rib boundaries, deviating from authentic chest radiographs.

**Authoritative Governing Decision:** **Configuration B ($SDD = 1800.0\text{ mm}, SOD = 1450.0\text{ mm}$) is the sole validated, verified, and active DRR projection standard.**

---

## 2. Mathematical Formulation & Ray-Tracing Physics

```
                    [X-ray Focal Spot S] (LPS Coordinates)
                             |
                             |  SOD = 1450.0 mm
                             v
                    +--------------------+
                    |   Patient Volume   |  CT Attenuation Grid
                    |  Isocenter C(x,y,z)|  mu(x,y,z) >= 0
                    +--------------------+
                             |
                             |  SDD - SOD = 350.0 mm
                             v
               +------------------------------+
               |  Flat Panel Detector Plane   |  430 mm x 430 mm (pitch: 1.0 mm)
               |  (PA: Anterior, LAT: Left)   |  SDD = 1800.0 mm
               +------------------------------+
```

### 2.1 Anatomical LPS Coordinate Frame
All spatial computations strictly preserve the standard DICOM Patient LPS (Left, Posterior, Superior) coordinate system:
* $+X$: Patient's anatomical Left.
* $+Y$: Patient's anatomical Posterior.
* $+Z$: Patient's anatomical Superior.

### 2.2 Biplanar Projection Geometry
Let $\mathbf{C} = (C_x, C_y, C_z)$ be the patient thoracic volume centroid (isocenter in world LPS millimeters).

#### 1. Posteroanterior (PA / Frontal) Projection:
* **Focal Spot:** Placed posterior to patient:
  $$\mathbf{S}_{\text{PA}} = (C_x,\, C_y + SOD,\, C_z)$$
* **Principal Ray Vector:** Directs anteriorly along $-\hat{\mathbf{y}}$.
* **Detector Plane:** Positioned anteriorly at $Y_{\text{det}} = C_y - (SDD - SOD)$.
* **Detector Orientation:** Horizontal $+u$ axis along $+X$ (Left); vertical $+v$ axis along $-Z$ (Inferior, inverted so image top is Superior).

#### 2. Left Lateral (LL) Projection:
* **Focal Spot:** Placed to patient's anatomical right:
  $$\mathbf{S}_{\text{LAT}} = (C_x - SOD,\, C_y,\, C_z)$$
* **Principal Ray Vector:** Directs leftward along $+\hat{\mathbf{x}}$.
* **Detector Plane:** Positioned to patient's anatomical left at $X_{\text{det}} = C_x + (SDD - SOD)$.
* **Detector Orientation:** Horizontal $+u$ axis along $+Y$ (Posterior); vertical $+v$ axis along $-Z$ (Inferior).

---

## 3. Attenuation Conversion & Line Integration (Beer-Lambert Law)

### 3.1 Hounsfield Unit (HU) to Linear Attenuation Coefficient
At an effective diagnostic X-ray spectrum of $\sim 70\text{ keV}$ ($120\text{ kVp}$ tube voltage with standard filtration), the linear attenuation coefficient of water is:
$$\mu_{\text{water}} = 0.193\text{ cm}^{-1} = 0.0193\text{ mm}^{-1}$$
The CT voxel intensity in Hounsfield Units is converted to physical linear attenuation coefficient $\mu(x,y,z) \ge 0$ via:
$$\mu(x,y,z) = \mu_{\text{water}} \cdot \max\left(0.0,\, 1.0 + \frac{\text{HU}(x,y,z)}{1000.0}\right)$$
* For Air ($\text{HU} = -1000$): $\mu = 0.0\text{ mm}^{-1}$.
* For Water / Soft Tissue ($\text{HU} = 0$): $\mu = 0.0193\text{ mm}^{-1}$.
* For Dense Cortical Bone ($\text{HU} = +1000$): $\mu = 0.0386\text{ mm}^{-1}$.

### 3.2 Discrete Ray Marching Integration
For detector pixel $(u, v)$, the unit direction vector from source $\mathbf{S}$ to detector point $\mathbf{P}_{\text{det}}(u, v)$ is $\hat{\mathbf{d}} = \frac{\mathbf{P}_{\text{det}} - \mathbf{S}}{\|\mathbf{P}_{\text{det}} - \mathbf{S}\|}$.
The line integral through the volume bounding box between ray entry $s_{\text{entry}}$ and exit $s_{\text{exit}}$ is evaluated with sampling step $\Delta s = 0.5\text{ mm}$:
$$P(u, v) = \int_{s_{\text{entry}}}^{s_{\text{exit}}} \mu(\mathbf{S} + s \hat{\mathbf{d}}) \, ds \approx \sum_{k=0}^{K-1} \mu(\mathbf{S} + k \Delta s \hat{\mathbf{d}}) \cdot \Delta s$$
Trilinear interpolation samples $\mu$ continuously within the voxel grid.

### 3.3 Clinical Radiograph Inversion
To match clinical radiograph perception where radio-dense tissue (bone) appears white and radiolucent air appears black, the total attenuation optical path $P(u, v)$ is inverted:
$$I_{\text{display}}(u, v) = 1.0 - \frac{P(u, v) - P_{\text{min}}}{P_{\text{max}} - P_{\text{min}}}$$

---

## 4. Verification of Generated Artifacts (`research/3d_reconstruction/demo_drr/`)

Verification of the generated demo DRRs produced by `scripts/3d_reconstruction/generate_drr.py`:
* **Frontal DRR Image:** `demo_thoracic_scan_frontal_drr.png` (SHA-256 `36f8f39632c3391c...`, $64 \times 128$ resampled preview, duration $145.24\text{ ms}$).
* **Lateral DRR Image:** `demo_thoracic_scan_lateral_drr.png` (SHA-256 `9df0fab505a87c8c...`, $64 \times 128$ resampled preview).
* **Metadata Integrity:** Validated against `DRR_GENERATION_SPEC.md` camera intrinsic matrix:
  $$\mathbf{K} = \begin{bmatrix} 1800.0 & 0.0 & 215.0 \\ 0.0 & 1800.0 & 215.0 \\ 0.0 & 0.0 & 1.0 \end{bmatrix}$$
