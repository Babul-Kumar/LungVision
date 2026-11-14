# LungVision AI — Frontend Product Requirements Document (PRD)

**Document Version:** 1.0.0  
**Status:** Approved / Active Governance  
**Author:** Staff Frontend Architect & Medical UI/UX Lead  
**Governing Skill:** Gravitas Engineering Reliability Runtime  
**Created:** 2026-11-12  

---

## 1. Product Vision & Problem Statement

### 1.1 Vision
LungVision AI is an enterprise-grade medical imaging research and clinical decision-support interface. It bridges deep learning research models (2D Chest X-Ray pathology screening) and volumetric computed tomography (3D CT exploration and lung volumetry) into an intuitive, responsive, and mathematically rigorous web platform for radiologists, medical researchers, and clinical AI investigators.

### 1.2 Problem Statement
Machine learning models in thoracic radiology often remain trapped in static Jupyter notebooks, Kaggle kernels, or disconnected research bundles. Clinicians and researchers lack an interactive, unified system that:
1. Performs instant multi-label screening across critical thoracic pathologies with calibrated sensitivity thresholds.
2. Explores 3D CT thoracic volumes with multi-planar orthogonal reconstruction (MPR) and real-time GPU mesh rendering.
3. Transparently inspects external cross-dataset validation performance (NIH ChestX-ray14, Stanford CheXpert, VinDr-CXR) without marketing obfuscation or fabricated metrics.

---

## 2. Target Users & Operating Persona

1. **Thoracic Radiologists & Clinical Fellows:** Requiring rapid decision-support triage, visualization of subtle consolidations or pneumothoraces, and anatomical volume symmetry comparison.
2. **Medical AI Researchers & MLOps Engineers:** Auditing model calibration thresholds, analyzing domain shift across cohorts, inspecting ROC/PR curves, and reviewing checkpoint provenance.
3. **Clinical Software Integrators:** Evaluating production-ready REST APIs, TypeScript contracts, and Three.js volumetric rendering pipelines.

---

## 3. Product Objectives & Key Results (OKRs)

- **Objective 1:** Provide zero-latency 2D X-ray upload, validation, and real-time inference against live DenseNet121 and ResNet18 models.
- **Objective 2:** Deliver an interactive Three.js 3D CT Explorer rendering watertight thoracic meshes, LUNA16 nodule spatial projections, and multi-planar slices with clinical Hounsfield windowing.
- **Objective 3:** Ensure strict scientific and regulatory compliance: zero fabricated patient statistics, clear experimental research disclaimers, and locked label ordering.
- **Objective 4:** Achieve 100% TypeScript type safety, zero `any` evasions, and full test coverage via Vitest and React Testing Library.

---

## 4. Module Specifications

### 4.1 Module A: Main Dashboard (`/dashboard` & `/`)
- Real-time backend liveness (`/api/v1/health`) and model readiness (`/api/v1/ready`) monitoring.
- Registered model cards featuring checkpoint parameters, training epochs, and validation AUCs.
- Quick navigation tiles to X-Ray Analysis, 3D CT Explorer, and Research Analytics.
- Direct links to OpenAPI documentation and system health status.

### 4.2 Module B: 2D X-Ray Analysis (`/xray`)
- Drag-and-drop zone with client-side image validation (PNG, JPEG, WebP, max 20 MB).
- Multi-label classification with DenseNet121 retaining the **strict NIH label order**:
  1. `Cardiomegaly`
  2. `Edema`
  3. `Consolidation`
  4. `Pneumonia`
  5. `Atelectasis`
  6. `Pneumothorax`
- Dual-indicator visualization: Continuous sigmoid probability $[0.0, 1.0]$ displayed against the calibrated NIH decision threshold.
- Binary Normal vs. Abnormal classifier with dynamic model switching:
  - `original_resnet18` (Threshold: `0.1765`, 90% sensitivity on IN-CXR)
  - `blur_augmented_resnet18` (Threshold: `0.1879`, robustness under optical blur)
- Prominent research disclaimer and inference latency readout.

### 4.3 Module C: 3D CT Explorer (`/ct-explorer`)
- CT archive upload (`.zip`, `.gz`, `.mhd`, `.dcm`, max 150 MB) and reference volume loader (`demo_thoracic_scan`).
- Multi-Planar Reconstruction (MPR) with synchronized views:
  - Axial ($XY$ slice along $Z$)
  - Coronal ($XZ$ slice along $Y$)
  - Sagittal ($YZ$ slice along $X$)
- Clinical Hounsfield Unit (HU) window presets:
  - `lung`: Center $-600\text{ HU}$, Width $1500\text{ HU}$
  - `mediastinum`: Center $+50\text{ HU}$, Width $350\text{ HU}$
  - `bone`: Center $+400\text{ HU}$, Width $1800\text{ HU}$
- Alpha-blended segmentation contours for Left Lung (label 3), Right Lung (label 4), and Trachea/Airway (label 5).
- LUNA16 ground-truth nodule crosshair projections on intersecting slices.
- **Interactive Three.js 3D Mesh Viewer:**
  - Standard Wavefront `.obj` mesh streaming from `/api/v1/ct/{scan_id}/mesh`.
  - OrbitControls (orbit, pan, zoom, reset view).
  - Structure toggles: Combined lungs, left lung, right lung, airways.
  - Interactive 3D nodule spheres placed at verified physical coordinates $(X, Y, Z)\text{ mm}$.
  - Mesh quality audit card (watertightness, Euler characteristic $\chi=2$, volume discrepancy).
  - Quantitative volumetry card (Left lung mL, Right lung mL, Total volume mL, symmetry ratio).

### 4.4 Module D: Research Dashboard (`/research`)
- Verified benchmark metrics from `/api/v1/research/metrics`:
  - NIH ChestX-ray14 held-out test set performance.
  - Stanford CheXpert external validation with domain-shift breakdown.
  - VinDr-CXR external validation with radiologist consensus analysis.
- Interactive Recharts visualizations: AUROC and AUPRC comparisons, per-pathology performance bar charts, sensitivity vs. threshold plots.
- External validation limitations and domain transfer caveats clearly presented.

### 4.5 Module E: Model Registry (`/models`)
- Dynamic catalog of registered models from `/api/v1/models`.
- Detailed modal/drawer for model inspection: architecture, task description, input resolution ($224 \times 224$), checkpoint SHA-256 hash, and training metadata.

### 4.6 Module F: Application History (`/history`)
- Client-side session history tracking recent 2D X-ray inferences and CT sessions.
- Explicit disclaimer: local browser session only; does not survive cache clearing.

### 4.7 Module G: System Settings (`/settings`)
- API Base URL configuration (`VITE_API_BASE_URL` with runtime override).
- Connectivity diagnostic probe.
- Medical disclaimer, software license, and privacy guarantees.

---

## 5. Non-Functional Requirements (NFRs)

- **Performance:** Initial page load under 1.5s; 3D mesh rendering at 60 FPS; client-side image preprocessing and slice switching under 50 ms.
- **Reliability:** Graceful error handling for offline backend, network timeouts, and malformed files with RFC 7807 problem details.
- **Accessibility:** WCAG 2.1 Level AA compliance, minimum color contrast ratio of 4.5:1, full keyboard navigation, ARIA live regions for inference updates.
- **Responsiveness:** Full layout adaptation across Desktop ($\ge 1280\text{px}$), Tablet ($768\text{px} - 1279\text{px}$), and Mobile ($< 768\text{px}$).
- **Scientific Integrity:** Mandatory medical research disclaimer visible across all diagnostic views:  
  *"LungVision AI is a research and educational platform. Model outputs are experimental decision-support metrics and do not constitute a medical diagnosis."*

