# LungVision AI — 3D Reconstruction Evidence & Code Audit

**Audit Date:** November 13, 2026  
**Auditor:** Principal Software Architect & Medical Imaging Research Auditor  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Scope:** Rigorous evidentiary separation of implemented 2D classification models from 3D reconstruction prototypes, auditing executable code, inputs, outputs, and empirical test evidence.

---

## 1. Executive Summary & Critical Separation of Claims

> [!IMPORTANT]
> **CRITICAL SCIENTIFIC DELIMITATION:**
> A forensic inspection of the codebase reveals that previous documentation conflated **2D Biplanar Classification Fusion** with **3D Patient-Specific Reconstruction**.
> 1. **2D Biplanar Classification (Experiment E5):** Concatenates 2D feature embeddings from Frontal (PA) and Lateral (LAT) chest radiographs into an MLP classifier to detect 2D lung abnormalities. **It does NOT reconstruct a 3D volume, point cloud, or surface mesh.**
> 2. **3D Mesh Extraction (Backend CT Service):** Extracts Marching Cubes 3D surface meshes from existing **3D CT volumes**, NOT from 2D chest X-rays.
> 3. **3D-from-2D Reconstruction Pipeline:** Currently exists as a complete mathematical architecture and design protocol (`SSM_BASELINE_DESIGN.md`, `DRR_GENERATION_SPEC.md`) with a verified forward projection engine (`generate_drr.py`), but **no trained multi-patient SSM deformation network is currently deployed in the backend.**

---

## 2. Evidentiary Audit of 3D Capabilities

| Capability Claimed | Executable Code on Disk? | Source File Location | Actual Functionality Verified | Inputs & Outputs | Test Evidence |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **1. CT-to-DRR Ray Caster** | **YES** | `scripts/3d_reconstruction/generate_drr.py` | Beer-Lambert forward line integration with DICOM LPS spatial coordinates | **In:** 3D CT numpy array<br>**Out:** 2D PA/LAT DRR PNG + JSON | `demo_thoracic_scan_frontal_drr.png` verified |
| **2. Marching Cubes 3D Mesh**| **YES** | `backend/app/services/ct/mesh_service.py` | Lewiner Marching Cubes at ISO -400 HU; 2-manifold validation | **In:** 3D CT volume<br>**Out:** Wavefront .OBJ / JSON mesh | Pytest CT mesh tests passing |
| **3. Three.js 3D Mesh Viewer**| **YES** | `frontend/src/features/ct/ThreeMeshCanvas.tsx` | WebGL2 rendering of 3D lung meshes with OrbitControls and clipping planes | **In:** Vertices, faces, normals<br>**Out:** Interactive 3D Canvas | Tested in `CTExplorerPage.test.tsx` |
| **4. Biplanar 2D Fusion** | **RESEARCH NOTEBOOK ONLY** | `C:/Users/babul/Downloads/LungVision_AI_Research.ipynb` (E5) | 2D ResNet dual-encoder concatenating PA and Lateral embeddings for classification | **In:** Paired 2D radiographs<br>**Out:** 2D disease probabilities | Cell 35 execution output (Retrocardiac Sens 88.5%) |
| **5. Frontend Dual-View CXR** | **YES (Viewer Only)** | `frontend/src/features/xray/DualViewXRayPage.tsx` | Synchronized side-by-side radiograph viewer; executes single frontal inference | **In:** Frontal + Lateral images<br>**Out:** Frontal prediction | Vitest passing; does NOT run 3D recon |
| **6. SSM Deformation Code** | **DESIGN SPEC ONLY** | `research/3d_reconstruction/ssm/SSM_BASELINE_DESIGN.md` | Mathematical formulation of 16 landmarks, PCA modes, Umeyama alignment, silhouette loss | **In:** Design documentation<br>**Out:** Architecture specs | Zero executable python script on disk |
| **7. Point Cloud Transformer** | **DESIGN SPEC ONLY** | `research/3d_reconstruction/METHOD_COMPARISON.md` | Theoretical conceptual comparison between SSM and PCT backbones | **In:** Markdown text<br>**Out:** None | Zero executable python script on disk |
| **8. Differentiable Renderer** | **DESIGN SPEC ONLY** | `research/3d_reconstruction/BASELINE_ARCHITECTURE.md` | Theoretical formulation of soft silhouette projection gradients | **In:** Markdown text<br>**Out:** None | Zero executable python script on disk |
| **9. Landmark Detection Svc** | **DESIGN SPEC ONLY** | `research/3d_reconstruction/ssm/SSM_BASELINE_DESIGN.md` | 16 standardized anatomical landmark definitions | **In:** Markdown text<br>**Out:** None | No backend service on disk |

---

## 3. Investigation of the E6 Shortcut Audit

### 3.1 Historical Context
In prototype `C:/Users/babul/Downloads/luna16_x2ct_v4_final.ipynb`, a 2D X-ray to 3D CT prototype model was trained to predict 3D lung volume and morphological parameters from simulated radiographs.
* Initial evaluation claimed an extraordinary volume prediction correlation of $R^2 > 0.94$.

### 3.2 The Forensic Audit Finding
Inspection of `LungVision_AI_Research.ipynb` (Section 25: "Failure Analysis and Shortcut / Bias Investigation") reveals how the shortcut was exposed:
1. **The Leak:** The prototype dataset loader passed the CT volume spatial metadata array `spacing_zyx` (`[spacing_z, spacing_y, spacing_x]`) directly to the model head alongside the visual feature embeddings.
2. **The Shortcut:** Because physical lung volume strongly correlates with slice spacing and patient height, the neural network learned to ignore the 2D image features and performed simple linear regression on the voxel spacing metadata.
3. **The Empirical Proof:** When `spacing_zyx` was randomized or standardized to isotropic $1.0\text{ mm}$, the model's volume estimation performance collapsed from $R^2 = 0.94$ to $R^2 = 0.31$!
4. **Governing Impact:** This critical finding proved that direct neural regression without strict anatomical physics is prone to exploiting metadata shortcuts. It established the core justification for adopting **Statistical Shape Models (SSM)** where deformation is strictly constrained by human anatomical eigenmodes.

---

## 4. Phase 2A Pilot Results Verification

In Phase 2A, the forward DRR generation and mesh extraction pipeline was validated on `demo_thoracic_scan`:
* **Input Data:** $64 \times 128 \times 128$ thoracic CT array recreating verified LUNA16 scan `1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518`.
* **DRR Generation:** Executed in $145.24\text{ ms}$ via `generate_drr.py`.
* **Generated Projections:** Successfully verified Frontal PA (`demo_thoracic_scan_frontal_drr.png`) and Lateral LL (`demo_thoracic_scan_lateral_drr.png`).
* **Mesh Extraction:** Verified Lewiner Marching Cubes extraction yielding closed 2-manifold triangular surfaces with Euler characteristic $\chi = 4$ (representing two distinct genus-0 lung sacs).
