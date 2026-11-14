# LungVision AI — Statistical Shape Model (SSM) 3D Lung Reconstruction Baseline
**Document ID:** `LV-RESEARCH-3D-SSM-001`  
**Version:** 1.0.0  
**Status:** Approved for Implementation  
**Auditor/Author:** LungVision AI Medical Imaging Scientist & Computational Anatomy Group  
**Framework:** Gravitas Engineering Reliability Runtime  

---

## 1. Executive Summary & Architectural Motivation

Reconstructing patient-specific 3D anatomical organ structures from sparse planar projections (a frontal and lateral chest radiograph) is a severely ill-posed inverse problem. Without strong anatomical priors, ill-posed 3D inversions produce severe reconstruction artifacts: self-intersecting surface meshes, floating disconnected artifacts, anatomically impossible organ volumes, and high sensitivity to 2D segmentation noise.

The **Statistical Shape Model (SSM)** represents the gold standard classical computational anatomy baseline for 3D organ reconstruction:
1. **Biologically Constrained:** The parameter space is restricted to linear combinations of true anatomical variation modes observed in real human CT cohorts.
2. **Compact & Regularized:** Organ shape is parameterized by $M \approx 15\text{--}30$ orthogonal coefficients rather than millions of unconstrained voxel densities or neural weights.
3. **Completely CPU-Compatible:** Optimization requires zero GPU memory and solves in seconds using gradient-based non-linear optimization (L-BFGS-B / Levenberg-Marquardt).
4. **Interpretable:** Each principal mode corresponds to a physiologically meaningful morphological variation (inspiratory expansion, apex height, lateral chest wall flare, diaphragmatic dome curvature).

```
                      [Frontal CXR / DRR]      [Lateral CXR / DRR]
                              |                         |
                      +---------------+         +---------------+
                      | 2D Landmark   |         | 2D Landmark   |
                      | & Silhouette  |         | & Silhouette  |
                      | Detection     |         | Detection     |
                      +---------------+         +---------------+
                              \                         /
                               \                       /
                     +-------------------------------------------+
                     |  Stage 1: Landmark Procrustes Alignment   |
                     |  (Coarse 7-DoF Pose: R0, t0, s0)          |
                     +-------------------------------------------+
                                           |
                     +-------------------------------------------+
                     |  Stage 2: Statistical Shape Prior         |
                     |  s(b) = s_mean + sum(b_k * p_k)           |
                     +-------------------------------------------+
                                           |
                     +-------------------------------------------+
                     |  Stage 3: Multi-View Silhouette Fitting   |
                     |  min L_sil^PA + L_sil^LAT + w_prior*D_M^2 |
                     +-------------------------------------------+
                                           |
                     +-------------------------------------------+
                     |  3D Patient-Specific Lung Surface Mesh    |
                     |  + Organ Volumes (Left/Right)             |
                     |  + Vertex-wise Uncertainty Intervals      |
                     +-------------------------------------------+
```

---

## 2. Mathematical Formulation & Point Distribution Model (PDM)

### 2.1. Surface Topology & Canonical Template Mesh
The anatomical surface of the human lungs is modeled as two separate closed 2-manifold triangular boundary meshes:
* $\mathcal{M}_{\text{left}} = (\mathcal{V}_{\text{left}}, \mathcal{F}_{\text{left}})$: Left Lung ($N_{\text{left}} = 5,000$ vertices).
* $\mathcal{M}_{\text{right}} = (\mathcal{V}_{\text{right}}, \mathcal{F}_{\text{right}})$: Right Lung ($N_{\text{right}} = 5,000$ vertices).

Each lung mesh satisfies the Euler characteristic for genus-0 closed spheres:
$$\chi = |\mathcal{V}| - |\mathcal{E}| + |\mathcal{F}| = 2$$
The combined thoracic shape vector $\mathbf{s} \in \mathbb{R}^{3N}$ ($N = 10,000$ vertices) is formed by concatenating the 3D coordinates:
$$\mathbf{s} = [x_1, y_1, z_1, x_2, y_2, z_2, \dots, x_N, y_N, z_N]^T$$

### 2.2. Establishing Anatomical Surface Correspondence
To construct an SSM across $K$ training thoracic CT segmentation masks:
1. **Canonical Template Extraction:** A representative high-quality reference scan (e.g., from LUNA16/LIDC-IDRI) is segmented, isosurfaced via Marching Cubes, and remeshed using isotropic remeshing to generate a canonical template $\mathbf{T}_0 = (\mathcal{V}_0, \mathcal{F}_0)$.
2. **Dense Non-Rigid Registration (Coherent Point Drift / SPHARM):**
   For each training subject $k \in \{1, \dots, K\}$, the canonical template vertices $\mathcal{V}_0$ are registered onto the subject's segmented lung boundary points $\mathcal{S}_k$ using Coherent Point Drift (CPD) with motion coherence constraints:
   $$\min_{\mathcal{W}_k} \mathcal{E}_{\text{CPD}}(\mathcal{V}_0 + \mathcal{W}_k, \mathcal{S}_k) + \frac{\beta}{2} \text{Tr}\left(\mathcal{W}_k^T \mathbf{G}^{-1} \mathcal{W}_k\right)$$
   where $\mathbf{G}$ is a Gaussian affinity kernel ensuring topologically smooth deformations.
3. This guarantees exact 1-to-1 vertex correspondence across all training subjects with identical face connectivity $\mathcal{F}_0$.

### 2.3. Generalized Procrustes Analysis (GPA)
Before computing statistical variations, all training shapes $\{\mathbf{s}_k\}_{k=1}^K$ undergo Generalized Procrustes Analysis:
1. Translate center of mass to origin: $\sum_{i=1}^N \mathbf{v}_i = \mathbf{0}$.
2. Normalize to unit centroid size: $S(\mathbf{s}) = \sqrt{\sum_{i=1}^N \|\mathbf{v}_i - \bar{\mathbf{v}}\|^2} = 1.0$.
3. Iteratively align orientations via Singular Value Decomposition (SVD) of the cross-covariance matrix to remove global rotational and scale differences.

### 2.4. Principal Component Analysis (PCA) Decomposition
Let $\bar{\mathbf{s}} \in \mathbb{R}^{3N}$ be the aligned sample mean shape:
$$\bar{\mathbf{s}} = \frac{1}{K} \sum_{k=1}^K \mathbf{s}_k$$
The sample covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{3N \times 3N}$ is:
$$\mathbf{\Sigma} = \frac{1}{K - 1} \sum_{k=1}^K (\mathbf{s}_k - \bar{\mathbf{s}})(\mathbf{s}_k - \bar{\mathbf{s}})^T$$
Singular Value Decomposition yields $K-1$ non-zero orthonormal eigenvectors $\mathbf{p}_m \in \mathbb{R}^{3N}$ and corresponding eigenvalues $\lambda_m$:
$$\mathbf{\Sigma} \mathbf{p}_m = \lambda_m \mathbf{p}_m, \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{K-1} \ge 0$$
We retain $M$ modes such that the cumulative explained variance reaches $\ge 95\%$:
$$\frac{\sum_{m=1}^M \lambda_m}{\sum_{j=1}^{K-1} \lambda_j} \ge 0.95$$

Any plausible patient-specific lung shape $\mathbf{s}(\mathbf{b})$ is parameterized by coefficient vector $\mathbf{b} = [b_1, \dots, b_M]^T$:
$$\mathbf{s}(\mathbf{b}) = \bar{\mathbf{s}} + \sum_{m=1}^M b_m \mathbf{p}_m = \bar{\mathbf{s}} + \mathbf{P} \mathbf{b}$$
Under the multi-dimensional Gaussian assumption, the shape prior is governed by the Mahalanobis distance:
$$D_M^2(\mathbf{b}) = \sum_{m=1}^M \frac{b_m^2}{\lambda_m} \sim \chi^2_M$$
Statistical plausibility is enforced by constraining each mode to $|b_m| \le 3 \sqrt{\lambda_m}$.

---

## 3. Physiological Modes of Variation

Empirical clinical studies of thoracic statistical shape models confirm that the leading principal modes capture recognizable anatomical variations:
* **Mode 1 (~48% variance): Global Lung Volume & Inspiratory State.**  
  Simultaneous expansion/contraction of left and right lungs along both cranial-caudal and transverse dimensions.
* **Mode 2 (~18% variance): Thoracic Aspect Ratio (Tall/Narrow vs Short/Wide).**  
  Cranial-caudal apex height relative to lateral basal expansion.
* **Mode 3 (~11% variance): Diaphragmatic Curvature & Basal Angle.**  
  Elevation and flattening of the hemidiaphragms; acute vs blunted costophrenic sulci.
* **Mode 4 (~7% variance): Mediastinal & Cardiac Impression.**  
  Variation in cardiac notch depth and medial left lung contour curvature.
* **Mode 5 (~5% variance): Anteroposterior Curvature & Chest Depth.**  
  Thoracic kyphosis and anterior rib cage curvature.

---

## 4. Multi-View Radiographic Fitting Algorithm

To reconstruct a patient's 3D lung surface from a paired Frontal and Lateral radiograph, the SSM is fitted through a coarse-to-fine two-stage optimization pipeline.

### 4.1. Stage 1: Landmark-Conditioned Rigid Initialization
A set of 16 standardized anatomical keypoints $\mathcal{K} = \{\mathbf{L}_i\}_{i=1}^{16}$ are identified on the canonical mesh and detected on the 2D radiographs (via deep 2D keypoint detectors or manual annotation):
1. Right Lung Apex
2. Left Lung Apex
3. Right Lateral Costophrenic Sulcus
4. Left Lateral Costophrenic Sulcus
5. Right Cardiophrenic Angle
6. Left Cardiophrenic Angle
7. Right Tracheobronchial Angle / Hilum
8. Left Hilum Center
9. Right Anterior Diaphragm Zenith
10. Left Anterior Diaphragm Zenith
11. Right Posterior Costophrenic Angle (Lateral view)
12. Left Posterior Costophrenic Angle (Lateral view)
13. Carina / Tracheal Bifurcation
14. Sternal Angle (Anterior reference)
15. Mid-Thoracic Vertebra T6 (Posterior reference)
16. Xiphoid Process (Inferior anterior boundary)

We estimate global 7-DoF rigid transformation (rotation $\mathbf{R}_0 \in SO(3)$, translation $\mathbf{t}_0 \in \mathbb{R}^3$, scale $s_0 \in \mathbb{R}^+$) via the Umeyama algorithm minimizing:
$$\min_{\mathbf{R}_0, \mathbf{t}_0, s_0} \sum_{i=1}^{16} \sum_{v \in \{\text{PA}, \text{LAT}\}} \left\| \Pi_v(s_0 \mathbf{R}_0 \bar{\mathbf{L}}_i + \mathbf{t}_0) - \mathbf{l}_{i, v} \right\|^2$$
where $\Pi_v(\cdot)$ denotes the perspective camera projection matrix for view $v$.

### 4.2. Stage 2: Non-Linear Multi-View Silhouette Optimization
With pose initialized, we optimize the shape coefficients $\mathbf{b} \in \mathbb{R}^M$ and fine-tune pose $\mathbf{\theta} = (\mathbf{R}, \mathbf{t}, s)$ by minimizing the total energy:

$$\min_{\mathbf{\theta}, \mathbf{b}} \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sil}}^{\text{PA}} + \mathcal{L}_{\text{sil}}^{\text{LAT}} + w_{\text{lm}} \mathcal{L}_{\text{landmark}} + w_{\text{prior}} D_M^2(\mathbf{b}) + w_{\text{Lap}} \mathcal{L}_{\text{smooth}}(\mathbf{s}(\mathbf{b}))$$

#### 1. Silhouette Distance Energy ($\mathcal{L}_{\text{sil}}$):
For each projection view $v \in \{\text{PA}, \text{LAT}\}$, we identify the apparent silhouette contour edges $\mathcal{E}_{\text{sil}}(v)$ of the current 3D mesh (faces whose surface normal dot product with the viewing ray satisfies $(\hat{\mathbf{n}}_{f_1} \cdot \hat{\mathbf{r}}) \cdot (\hat{\mathbf{n}}_{f_2} \cdot \hat{\mathbf{r}}) < 0$).

The projected silhouette points $\mathbf{q}_e = \Pi_v(\mathbf{v}_e)$ are evaluated against the 2D Euclidean Distance Transform $\text{EDT}_v$ of the detected radiograph lung segmentation contour:
$$\mathcal{L}_{\text{sil}}^v = \frac{1}{|\mathcal{E}_{\text{sil}}(v)|} \sum_{e \in \mathcal{E}_{\text{sil}}(v)} \text{EDT}_v\left( \Pi_v(s \mathbf{R} \mathbf{v}_e(\mathbf{b}) + \mathbf{t}) \right)^2$$

#### 2. Statistical Shape Prior Regularizer ($D_M^2$):
Prevents the model from deforming into non-anatomical or impossible shapes:
$$D_M^2(\mathbf{b}) = \sum_{m=1}^M \frac{b_m^2}{\lambda_m}$$

#### 3. Surface Laplacian Smoothness ($\mathcal{L}_{\text{Lap}}$):
Penalizes local high-frequency mesh creases:
$$\mathcal{L}_{\text{Lap}} = \sum_{i=1}^N \left\| \mathbf{v}_i - \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{v}_j \right\|^2$$

---

## 5. Surface Uncertainty Quantification

Because two orthogonal views leave the oblique degrees of freedom partially unconstrained, the SSM provides closed-form vertex-wise reconstruction uncertainty. Under the linear Gaussian observation model:
$$\mathbf{y}_{\text{2D}} = \mathbf{J} \mathbf{b} + \mathbf{\epsilon}, \quad \mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_{\text{img}}^2 \mathbf{I})$$
where $\mathbf{J} = \frac{\partial \Pi}{\partial \mathbf{b}}$ is the silhouette projection Jacobian. The posterior covariance of the shape coefficients is:
$$\mathbf{\Sigma}_{\text{post}} = \left( \mathbf{\Lambda}^{-1} + \frac{1}{\sigma_{\text{img}}^2} \mathbf{J}^T \mathbf{J} \right)^{-1}$$
The 3D spatial variance at each surface vertex $i \in \{1, \dots, N\}$ is given by:
$$\sigma_i^2 = \text{Tr}\left( \mathbf{P}_i \mathbf{\Sigma}_{\text{post}} \mathbf{P}_i^T \right)$$
where $\mathbf{P}_i \in \mathbb{R}^{3 \times M}$ are the eigenvector rows corresponding to vertex $i$.  
Each vertex is assigned a $95\%$ normal-direction confidence margin: $\pm 1.96 \cdot \sigma_i \hat{\mathbf{n}}_i$.

---

## 6. Critical Scientific Boundary & Scope Delimitation

> [!CAUTION]
> **Scientific Integrity Delimitation — What SSM Does and Does NOT Reconstruct:**
> 1. **Surface Geometry vs Internal CT Density:**  
>    The Statistical Shape Model reconstructs the **outer boundary surface** of the lungs ($\mathcal{M}_{\text{left}}$, $\mathcal{M}_{\text{right}}$), and calculates anatomical organ volumetry ($V_{\text{lung}} = \frac{1}{6} \sum_{(i,j,k) \in \mathcal{F}} \mathbf{v}_i \cdot (\mathbf{v}_j \times \mathbf{v}_k)$).  
>    **The SSM DOES NOT reconstruct internal CT Hounsfield Units (HU), internal pulmonary parenchymal densities, airways, or ground-glass opacities.**
> 2. **Clinical Limitation Statement:**  
>    Reconstructed 3D lung meshes are indicated for organ volumetry, thoracic biomechanics, and surgical surface localization. They must **never** be interpreted as a replacement for high-resolution diagnostic CT in nodule detection, interstitial lung disease diagnosis, or pulmonary embolism screening.

