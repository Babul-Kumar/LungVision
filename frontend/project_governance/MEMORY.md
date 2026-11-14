# LungVision AI — Frontend Persistent Memory (MEMORY.md)

**Document Version:** 1.1.0  
**Status:** **COMPLETED & FULLY VERIFIED**  
**Author:** Staff Frontend Architect & Medical UI/UX Lead  
**Governing Skill:** Gravitas Engineering Reliability Runtime  
**Updated:** 2026-11-12  

---

## 1. Project Overview & Environment Facts

- **Application Name:** LUNGVISION AI Frontend Research Workstation
- **Domain:** Medical Imaging Decision-Support & 3D Volumetric Exploration Workstation
- **Status:** Experimental Research Decision-Support Platform.
- **Frontend Stack:** React 18.3, TypeScript 5.6, Vite 6, Tailwind CSS 3.4, React Router v6, Zustand 5, Axios 1.7, Three.js 0.170, Recharts 2.15, Lucide React, Vitest 2.1.
- **Backend Port & Host:** `http://localhost:8000` (`VITE_API_BASE_URL`).
- **Frontend Port:** `http://localhost:5173` (Whitelisted in backend `CORS_ORIGINS`).
- **Test Metrics:**
  - Frontend Vitest: **13 / 13 PASSED (100%)**
  - Backend Pytest: **29 / 29 PASSED (100%)**
  - Total Full-Stack Verified Tests: **42 / 42 PASSED (100%)**

---

## 2. Verified Backend Integration Contracts

| Method | Endpoint | Purpose | Request Format | Response Highlights |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Process liveness | None | `{"status": "healthy", "version": "1.0.0"}` |
| `GET` | `/api/v1/ready` | Model memory readiness | None | `{"status": "ready", "models_loaded": {...}}` |
| `GET` | `/api/v1/models` | Model registry | None | `{"models": [...]}` with parameter counts & hashes |
| `GET` | `/api/v1/models/{id}` | Single model details | None | Detailed architecture, threshold, training set |
| `GET` | `/api/v1/research/metrics` | Research benchmarks | None | `nih_test_metrics`, `chexpert_external`, `vindr_external` |
| `POST`| `/api/v1/xray/predict` | 6-disease inference | `multipart/form-data` | `predictions: {Cardiomegaly: {...}, ...}` (strict 6 order) |
| `POST`| `/api/v1/xray/predict-binary`| Binary Normal/Abnormal | `multipart/form-data` + `model_id` | `prediction: {abnormal_probability, classification, ...}` |
| `POST`| `/api/v1/ct/upload` | CT volume registration | `multipart/form-data` | `{"scan_id": "ct_...", "filename": "...", ...}` |
| `GET` | `/api/v1/ct/{id}/metadata` | Spatial metadata | None | `dimensions: {x, y, z}`, `spacing_mm`, `origin_mm` |
| `GET` | `/api/v1/ct/{id}/slice` | MPR 2D slice | Query params (`view`, `slice_index`, `window`) | `{"image_base64": "data:image/png;base64,...", "nodules_on_slice": [...]}` |
| `GET` | `/api/v1/ct/{id}/mesh` | 3D Wavefront OBJ stream| Query params (`structure`, `step`, `pad_boundary`) | `text/plain` OBJ data (`v ...`, `f ...`) |
| `GET` | `/api/v1/ct/{id}/mesh/quality`| Trimesh topology audit | Query params (`structure`) | `{"watertight": true, "euler_number": 2, "voxel_volume_ml": ...}` |
| `GET` | `/api/v1/ct/{id}/annotations` | LUNA16 ground truth | None | `{"annotations": [{"coordX_mm", "voxel_x", ...}]}` |
| `GET` | `/api/v1/ct/{id}/measurements`| Quantitative volumetry | None | `{"lungs": {"left_lung_ml", "right_lung_ml", ...}}` |
| `GET` | `/api/v1/ct/{id}/segmentation`| Exploratory status | None | `{"status": "exploratory_structural_only"}` |

---

## 3. Strict Label Ordering & Calibration Thresholds

### Multi-Label DenseNet121 Order
1. `Cardiomegaly`: Calibrated Threshold = `0.950195`
2. `Edema`: Calibrated Threshold = `0.876953`
3. `Consolidation`: Calibrated Threshold = `0.732910`
4. `Pneumonia`: Calibrated Threshold = `0.752930`
5. `Atelectasis`: Calibrated Threshold = `0.749512`
6. `Pneumothorax`: Calibrated Threshold = `0.582520`

### Binary ResNet18 Models
- `original_resnet18`: Locked Threshold = `0.176463` (90% Target Sensitivity)
- `blur_augmented_resnet18`: Locked Threshold = `0.187923` (90% Target Sensitivity, Blur Robust)

---

## 4. Key Architectural Decisions Recorded

1. **Scaffolded standalone React 18 + TypeScript + Vite 6 application in `frontend/`**:
   Decouples client bundle cleanly from Python backend and research artifacts.
2. **Direct Three.js WebGL Engine with OrbitControls**:
   Ensures deterministic memory lifecycle management, clean geometry disposal, and exact SimpleITK coordinate alignment ($Z$-up craniocaudal orientation, camera positioned at $(0, -450, 200)$).
3. **12 LUNA16 3D Nodule Spheres**:
   Rendered in physical millimeters $(X, Y, Z)$ with bi-directional selection between 3D viewport raycaster and annotation table.
4. **MPR Orthogonal Slicer**:
   Multi-planar reconstruction (Axial, Coronal, Sagittal) with clinical HU window presets (`lung`, `mediastinum`, `bone`), segmentation mask overlays, and nodule crosshairs.
5. **Centralized Axios Client with RFC 7807 Error Interceptor**:
   Parses `{ type, title, status, detail, instance }` without ad-hoc error logic.
6. **Zero Mock Policy**:
   Never fallback to synthetic predictions or fake medical findings; connection failures are communicated transparently.
7. **Ephemeral Client-Side History**:
   HTML5 LocalStorage session history with clear privacy and retention notices.
8. **100% Strict TypeScript**:
   `strict: true`, `noUnusedLocals: true`, `noUnusedParameters: true`, zero `any` usage.
