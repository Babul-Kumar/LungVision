# LungVision AI — Digitally Reconstructed Radiograph (DRR) Generation Specification
**Document ID:** `LV-RESEARCH-3D-DRR-001`  
**Version:** 1.0.0  
**Status:** Approved for Implementation  
**Auditor/Author:** LungVision AI Medical Imaging Physics & Computational Geometry Group  
**Framework:** Gravitas Engineering Reliability Runtime  

---

## 1. Physical Principle & Optical Geometry

A Digitally Reconstructed Radiograph (DRR) is a synthetically generated planar projection formed by simulating the transmission of X-ray photons through a three-dimensional volumetric Computed Tomography (CT) dataset. DRRs serve as the core bridging representation in 3D-from-2D reconstruction research, providing mathematically exact geometric correspondence between 2D pixel coordinates and 3D anatomical voxels.

```
                      [X-ray Focal Spot (Source)] S = (Sx, Sy, Sz)
                                  \       |       /
                                   \      |      /
                                    \     |     /  Ray path L(u,v)
                                     \    |    /
                               +---------------------+
                               |   Patient Thorax    |  Volume V(x,y,z)
                               |  with attenuation   |  (LPS coordinates)
                               |     mu(x,y,z)       |
                               +---------------------+
                                     /    |    \
                                    /     |     \
                                   /      |      \
                      +---------------------------------------+
                      |         2D Detector Plane             |  (u, v) in mm
                      |    (Flat Panel Digital Radiography)   |  D = SDD
                      +---------------------------------------+
```

---

## 2. Mathematical Formulation & Ray-Tracing Physics

### 2.1. Coordinate Frames & Patient Calibration
All calculations strictly preserve the standard DICOM LPS (Left, Posterior, Superior) patient anatomical coordinate system:
* $+X$-axis: Points towards the patient's anatomical **Left**.
* $+Y$-axis: Points towards the patient's anatomical **Posterior**.
* $+Z$-axis: Points towards the patient's anatomical **Superior** (head).

The patient volume is defined in physical world space $\mathbf{x}_{\text{world}} = (x, y, z) \in \mathbb{R}^3$ via:
$$\mathbf{x}_{\text{world}} = \mathbf{O} + \mathbf{D} \cdot (\mathbf{S}_{\text{vox}} \odot \mathbf{i}_{\text{grid}})$$
where $\mathbf{O} = (O_x, O_y, O_z)$ is the 3D origin, $\mathbf{D} \in \mathbb{R}^{3 \times 3}$ is the direction cosine matrix, $\mathbf{S}_{\text{vox}} = (s_x, s_y, s_z)$ is voxel spacing in millimeters, and $\mathbf{i}_{\text{grid}} = (i, j, k)$ is the zero-indexed integer voxel index.

### 2.2. Standard Radiographic Geometry
To simulate realistic clinical chest radiographs, the projection parameters follow standard American College of Radiology (ACR) and European Society of Radiology (ESR) diagnostic protocols:

1. **Source-to-Detector Distance ($SDD$):**
   * Standard Posteroanterior (PA): $SDD = 1800.0\text{ mm}$ (minimizes heart magnification).
   * Standard Lateral: $SDD = 1800.0\text{ mm}$.
2. **Source-to-Isocenter Distance ($SOD$):**
   * Default: $SOD = 1450.0\text{ mm}$ (patient thoracic center at isocenter).
3. **Detector Geometry:**
   * Active Detector Dimensions: $430.0\text{ mm} \times 430.0\text{ mm}$ ($17\text{ in} \times 17\text{ in}$ standard digital flat panel).
   * Pixel Pitch ($\Delta_{\text{det}}$):
     * Diagnostic Benchmark Resolution: $0.35\text{ mm} \times 0.35\text{ mm}$ ($1228 \times 1228$ matrix).
     * Computational Research Prototyping: $1.40\text{ mm} \times 1.40\text{ mm}$ ($307 \times 307$ matrix) or $1.0\text{ mm} \times 1.0\text{ mm}$.
   * Principal Point $(u_0, v_0)$: Calibrated at detector center $(\frac{W_{\text{det}}}{2}, \frac{H_{\text{det}}}{2})$.

### 2.3. Projection Angles & View Configurations

#### View 1: Posteroanterior (PA / Frontal)
* **X-ray Source Position:** Placed posterior to the patient along the patient $+Y$ axis:
  $$\mathbf{S}_{\text{PA}} = (C_x,\, C_y + SOD,\, C_z)$$
  where $\mathbf{C} = (C_x, C_y, C_z)$ is the thoracic volume centroid / isocenter.
* **Beam Vector:** Directs anteriorly along the $-Y$ axis.
* **Detector Plane:** Positioned anterior to the patient:
  $$Y_{\text{det}} = C_y - (SDD - SOD)$$
* **Detector Orientation:** Horizontal axis along $+X$ (Left), Vertical axis along $-Z$ (Inferior, inverted so image top is Superior).

#### View 2: Left Lateral (LL)
* **X-ray Source Position:** Placed to the patient's anatomical right along the $-X$ axis:
  $$\mathbf{S}_{\text{LAT}} = (C_x - SOD,\, C_y,\, C_z)$$
* **Beam Vector:** Directs leftward along the $+X$ axis.
* **Detector Plane:** Positioned on the patient's anatomical left:
  $$X_{\text{det}} = C_x + (SDD - SOD)$$
* **Detector Orientation:** Horizontal axis along $-Y$ (Anterior $\to$ Posterior), Vertical axis along $-Z$ (Inferior).

---

## 3. Radiometric Transformation & Beer-Lambert Integration

### 3.1. Hounsfield Unit to Linear Attenuation Coefficient Conversion
CT voxel values are recorded in integer Hounsfield Units ($\text{HU}$). Linear attenuation coefficients $\mu$ depend upon tissue electron density and effective photon energy. For a typical diagnostic tungsten anode spectrum at $120\text{ kVp}$ with standard filtration ($\bar{E} \approx 65\text{--}70\text{ keV}$ effective energy), the attenuation coefficient of water is:
$$\mu_{\text{water}} = 0.193\text{ cm}^{-1} = 0.0193\text{ mm}^{-1}$$
The voxel attenuation coefficient $\mu(x,y,z)$ is computed as:
$$\mu(x,y,z) = \mu_{\text{water}} \cdot \max\left(0.0,\, 1.0 + \frac{\text{HU}(x,y,z)}{1000.0}\right)$$
For air ($\text{HU} = -1000$), $\mu = 0.0\text{ mm}^{-1}$.  
For cortical bone ($\text{HU} \approx +1000$), $\mu \approx 0.0386\text{ mm}^{-1}$.

### 3.2. Forward Line Integral (Beer-Lambert Law)
Along any given ray path $L(u, v)$ originating at focal source $\mathbf{S}$ and intersecting detector pixel $(u, v)$ with unit direction vector $\hat{\mathbf{d}}(u, v)$:
$$\mathbf{r}(s) = \mathbf{S} + s \cdot \hat{\mathbf{d}}(u, v), \quad s \in [s_{\text{entry}}, s_{\text{exit}}]$$
The monochromatic transmitted photon intensity $I(u, v)$ satisfies:
$$I(u, v) = I_0 \exp\left(-\int_{s_{\text{entry}}}^{s_{\text{exit}}} \mu(\mathbf{r}(s)) \, ds\right)$$
The primary radiographic optical path length (radiographic projection value) is the line integral:
$$P(u, v) = -\ln\left(\frac{I(u, v)}{I_0}\right) = \int_{s_{\text{entry}}}^{s_{\text{exit}}} \mu(\mathbf{r}(s)) \, ds \approx \sum_{k=0}^{K-1} \mu(\mathbf{r}(k \Delta s)) \cdot \Delta s$$
where $\Delta s$ is the numerical step size along the ray, chosen to satisfy the Nyquist-Shannon criterion:
$$\Delta s \le \frac{1}{2} \min(s_x, s_y, s_z)$$

### 3.3. Quantum Mottle & Noise Simulation
To accurately reflect physical detector quantum noise, the incident photon count per pixel is modeled via a Poisson distribution:
$$N_{\text{trans}}(u, v) \sim \text{Poisson}\left(N_0 \cdot \exp\left(-P(u, v)\right)\right)$$
where $N_0 = 10^5$ represents the unattenuated photon fluence for standard clinical dose. The noisy projection is then recovered as:
$$\tilde{P}(u, v) = -\ln\left(\frac{\max(1, N_{\text{trans}}(u, v))}{N_0}\right)$$

### 3.4. Dynamic Range Windowing & Normalization
For display, model training, and feature extraction, the attenuation integral $P(u, v)$ is inverted and mapped to $[0, 1]$ or $[0, 255]$:
$$I_{\text{display}}(u, v) = 1.0 - \frac{P(u, v) - P_{\text{min}}}{P_{\text{max}} - P_{\text{min}}}$$
This matches standard clinical radiography where dense structures (bone, mediastinum) appear white / radio-opaque, and air-filled lung parenchyma appears dark / radiolucent.

---

## 4. Geometric Transformation Matrix Formalism

The projection geometry is compactly formalized by a $3 \times 4$ camera projection matrix $\mathbf{M}_{\text{proj}}$ mapping homogeneous 3D world coordinates $\tilde{\mathbf{X}} = (X, Y, Z, 1)^T$ to 2D homogeneous detector coordinates $\tilde{\mathbf{u}} = (w u, w v, w)^T$:

$$\tilde{\mathbf{u}} = \mathbf{K} \cdot [\mathbf{R} \mid \mathbf{t}] \cdot \tilde{\mathbf{X}}$$

where:
* $\mathbf{K} \in \mathbb{R}^{3 \times 3}$ is the intrinsic camera matrix:
  $$\mathbf{K} = \begin{bmatrix} f_u & 0 & u_0 \\ 0 & f_v & v_0 \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} \frac{SDD}{\Delta_{\text{det}}} & 0 & u_0 \\ 0 & \frac{SDD}{\Delta_{\text{det}}} & v_0 \\ 0 & 0 & 1 \end{bmatrix}$$
* $\mathbf{R} \in SO(3)$ is the $3 \times 3$ rotation matrix from world (LPS) to camera frame.
* $\mathbf{t} \in \mathbb{R}^3$ is the translation vector mapping world origin to source frame.

---

## 5. Domain Gap & Scientific Boundaries

> [!CAUTION]
> **Scientific Integrity Notice — DRR vs Real Radiograph Discrepancies:**
> Synthetic DRRs generated from CT scans possess key physical and biological differences from real clinical radiographs:
> 1. **Patient Posture & Gravity:** Diagnostic chest CTs are acquired in **supine** position (lying flat). Clinical chest radiographs (PA/LAT) are almost universally acquired in **upright** (standing or sitting) posture. This causes diaphragmatic elevation, lung volume compression, and mediastinal widening in CT-derived DRRs relative to true CXRs.
> 2. **Inspiratory Depth:** CT acquisition protocols optimize for stationary breath-holds, whereas upright CXR instructions emphasize maximal inspiration ($>9$ posterior ribs visible).
> 3. **Compton Scatter:** Real radiographs suffer from diffuse forward Compton scatter, only partially attenuated by anti-scatter grids. Pure line-integral DRRs lack forward scatter unless explicitly modeled via Monte Carlo simulation.
> 4. **Polychromatic Beam Hardening:** CT scans undergo beam hardening corrections, whereas 2D radiographs exhibit energy-dependent heel effect and differential soft tissue filtration.
>
> **Experimental Rule:** Algorithms trained on DRRs must never be claimed to operate on real clinical radiographs without explicit cross-domain adaptation and empirical testing on real paired cohorts (e.g., MIDRC/IN-CXR).

