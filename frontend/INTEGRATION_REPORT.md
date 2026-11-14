# LUNGVISION AI — FULL-STACK INTEGRATION REPORT

**Audit Date:** November 2026  
**Auditor:** Senior Frontend Architect & Full-Stack Integration Engineer  
**Scope:** Frontend (React 18 + Vite) ↔ Backend (FastAPI Gateway on port 8000)

---

## 1. Full-Stack End-to-End Contract Matrix

| Endpoint | Method | Request Format | Response Format | Integration Status | Frontend Consumer |
|---|---|---|---|---|---|
| `/api/v1/health` | GET | None | JSON (`HealthResponse`) | **VERIFIED** | `useAppStore`, `Navbar`, `SettingsPage` |
| `/api/v1/ready` | GET | None | JSON (`ReadinessResponse`) | **VERIFIED** | `useAppStore`, `DashboardPage`, `SettingsPage` |
| `/api/v1/models` | GET | None | JSON (`ModelListResponse`) | **VERIFIED** | `DashboardPage`, `ModelsPage` |
| `/api/v1/models/{id}` | GET | Path Param | JSON (`ModelRegistryEntry`) | **VERIFIED** | `ModelsPage` |
| `/api/v1/xray/predict` | POST | `multipart/form-data` (`file`) | JSON (`XRayPredictionResponse`) | **VERIFIED** | `XrayPage` (DenseNet121) |
| `/api/v1/xray/predict-binary` | POST | `multipart/form-data` (`file`, `model_name`) | JSON (`XRayBinaryResponse`) | **VERIFIED** | `XrayPage` (ResNet18) |
| `/api/v1/ct/upload` | POST | `multipart/form-data` (`file`) | JSON (`CTUploadResponse`) | **VERIFIED** | `CTExplorerPage` |
| `/api/v1/ct/{id}/metadata` | GET | Path Param | JSON (`CTMetadataResponse`) | **VERIFIED** | `CTExplorerPage` |
| `/api/v1/ct/{id}/slice` | GET | Query (`plane`, `index`, `window_center`, `window_width`) | JSON (`CTSliceResponse`) | **VERIFIED** | `CTExplorerPage` (MPR Engine) |
| `/api/v1/ct/{id}/mesh` | GET | Query (`structure`, `step_size`, `smooth`) | `text/plain` (Wavefront OBJ) | **VERIFIED** | `ThreeMeshCanvas` (OBJLoader) |
| `/api/v1/ct/{id}/mesh/quality` | GET | Query (`structure`) | JSON (`CTMeshQualityResponse`) | **VERIFIED** | `CTExplorerPage` |
| `/api/v1/ct/{id}/annotations` | GET | Path Param | JSON (`CTAnnotationsResponse`) | **VERIFIED** | `CTExplorerPage`, `ThreeMeshCanvas` |
| `/api/v1/ct/{id}/measurements` | GET | Path Param | JSON (`CTMeasurementResponse`) | **VERIFIED** | `CTExplorerPage` |
| `/api/v1/ct/{id}/segmentation` | GET | Path Param | JSON (`CTMetadataResponse`) | **VERIFIED** | `CTExplorerPage` |
| `/api/v1/research/metrics` | GET | None | JSON (`ResearchMetricsResponse`) | **VERIFIED** | `ResearchPage` |

---

## 2. Cross-Origin Resource Sharing (CORS) Audit

- **FastAPI Backend Settings**:
  `settings.CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"]`
- **Frontend Origin**:
  Runs on `http://localhost:5173`.
- **Verdict**:
  All `OPTIONS` preflight and actual `GET`/`POST` requests are authorized without cross-origin rejections.

---

## 3. Data Format & Numerical Contract Verifications

### 3.1 2D X-Ray Multi-Label Classification
- **Backend Schema**: `predictions: Dict[str, float]`, `calibrated_thresholds: Dict[str, float]`, `experimental_flags: Dict[str, bool]`.
- **Label Order Verification**:
  Frontend locks order: `['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax']`.
  Backend returns probabilities keyed by these exact 6 names.
- **Calibrated Thresholds**:
  Cardiomegaly (0.950195), Edema (0.876953), Consolidation (0.732910), Pneumonia (0.752930), Atelectasis (0.749512), Pneumothorax (0.582520).
- **Frontend Behavior**:
  Matches backend output exactly. When probability exceeds calibrated threshold, `experimental_flags` evaluates to `true` and turns probability indicator amber/rose.

### 3.2 3D Wavefront OBJ Mesh & Three.js Pipeline
- **Backend Route**: `GET /api/v1/ct/{scan_id}/mesh?structure=lung_surface&step_size=2&smooth=true`.
- **Response Format**: `text/plain` formatted as standard Wavefront OBJ (`v x y z`, `vn x y z`, `f v1 v2 v3`).
- **Parsing**: `OBJLoader.parse(objText)` generates a Three.js `BufferGeometry`.
- **Transformation**: Centroid computed via `THREE.Box3` and subtracted from `meshGroup.position` so rotation centers on the lungs.
- **Coordinate System**: Three.js $Z$-up configured via `camera.up.set(0, 0, 1)`, ensuring anatomical congruence with medical DICOM/NIfTI coordinates.

### 3.3 12 LUNA16 Nodule Ground Truth Spheres
- **Backend Route**: `GET /api/v1/ct/{scan_id}/annotations`.
- **Data Attributes**: `nodule_id`, `coordX_mm`, `coordY_mm`, `coordZ_mm`, `diameter_mm`, `inside_volume`.
- **Rendering**: Represented in 3D WebGL space as golden/red glowing spheres at physical coordinates $(X, Y, Z)$.
- **Bi-directional Selection**: Clicking a sphere in the 3D viewport selects the row in the annotation table; clicking a row highlights the sphere in 3D.

---

## 4. Error Handling & Network Resilience

- **RFC 7807 Compliance**: The Axios interceptor parses `{ type, title, status, detail, instance }` from HTTP 400, 404, 422, and 500 responses.
- **Graceful Offline Degradation**:
  If the backend gateway is offline, the navbar renders `Backend Offline`, the station hero reports `Gateway Disconnected`, and interactive pages display actionable connection instructions without crashing.
- **Zero Mock Policy**: Production code does not fallback to fabricated inference outputs or fake diagnoses when the backend fails.

---

## 5. Stated Limitations & Research Boundaries

1. **Synthetic/Reference Scan Scope**: As documented in the independent backend audit report, full multi-scan CT validation on all 888 LUNA16 raw scans remains incomplete on this environment. The frontend operates against the verified reference volume `demo_thoracic_scan`.
2. **Investigational Status**: All model inferences are decision-support outputs and require radiologist validation.

