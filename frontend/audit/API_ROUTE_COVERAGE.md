# LungVision AI — API Route Coverage & Contract Audit

**Audit Date:** November 12, 2026  
**Auditor:** Independent Full-Stack Engineer & API Architect (Gravitas Engineering Runtime)  
**Backend Host Tested:** `http://127.0.0.1:8000` (FastAPI / Uvicorn)  
**OpenAPI Specification Version:** 3.1.0  
**Status:** **AUDITED & VERIFIED (17 Routes Discovered, 22/23 Automated Scenarios Passed)**

---

## 1. Executive Summary

An exhaustive inspection of the live OpenAPI schema (`/openapi.json`) and backend routing source code (`backend/app/api/v1/*.py`, `backend/app/main.py`) identified **17 active API endpoints**.

### Discrepancy Resolution: 17 Routes vs. 15 Claimed
The previous `frontend/INTEGRATION_REPORT.md` reported 15 routes. A line-by-line reconciliation against OpenAPI confirms that 2 valid routes were omitted from the previous documentation table:
1. `GET /` — Root Service Metadata & Endpoint Discovery Catalog.
2. `GET /api/v1/ct/{scan_id}` — CT Scan Summary Descriptor (aliased to `/metadata`).

With these two endpoints included, **route coverage is 100.0% (17 / 17 routes discovered, mapped to frontend consumers, and tested live)**.

---

## 2. Complete API Route & Contract Catalog

| # | Method | Path | Summary / Purpose | Request Schema / Parameters | Response Schema | Frontend Consumer | Typical Latency |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :-: |
| **1** | `GET` | `/` | Root Service Discovery | None | `RootResponse` (dict) | System / Health Monitor | 2.8 ms |
| **2** | `GET` | `/api/v1/health` | Service Liveness Probe | None | `HealthResponse` | `useSystemHealth`, `AppLayout.tsx` | 4.8 ms |
| **3** | `GET` | `/api/v1/ready` | Model Readiness Probe | None | `ReadyResponse` | `useSystemHealth`, `DashboardPage.tsx` | 4.0 ms |
| **4** | `GET` | `/api/v1/models` | List Registered AI Models | None | `List[ModelRegistryEntry]` | `useModels`, `ModelRegistryPage.tsx` | 4.2 ms |
| **5** | `GET` | `/api/v1/models/{model_id}` | Detailed Model Information | Path: `model_id: str` | `ModelRegistryEntry` | `ModelRegistryPage.tsx` | 4.3 ms |
| **6** | `GET` | `/api/v1/research/metrics` | Research Benchmark Metrics | None | `HistoricalResearchMetrics` | `ResearchMetricsPage.tsx` | 7.7 ms |
| **7** | `GET` | `/api/v1/ct/{scan_id}` | CT Scan Summary | Path: `scan_id: str` | `CTMetadataResponse` | `ctApi.getScanSummary` | 4.6 ms |
| **8** | `GET` | `/api/v1/ct/{scan_id}/metadata` | Detailed CT Geometric Metadata | Path: `scan_id: str` | `CTMetadataResponse` | `useCTScan`, `CTExplorerPage.tsx` | 5.0 ms |
| **9** | `GET` | `/api/v1/ct/{scan_id}/slice` | Render Orthogonal 2D Slice | Query: `view`, `slice_idx`, `window`, `overlay_mask`, `show_nodules`, `alpha` | `CTSliceResponse` (Base64 PNG) | `CTSliceViewport.tsx` | 26.7 ms |
| **10** | `GET` | `/api/v1/ct/{scan_id}/mesh` | Stream 3D Surface Mesh | Query: `structure`, `step`, `pad_boundary` | `text/plain` (Wavefront OBJ) | `ThreeMeshCanvas.tsx` | 112.5 ms |
| **11** | `GET` | `/api/v1/ct/{scan_id}/mesh/quality` | Mesh Topology & Watertight Audit | Query: `structure` | `CTMeshQualityResponse` | `CTExplorerPage.tsx` | 343.2 ms |
| **12** | `GET` | `/api/v1/ct/{scan_id}/annotations` | LUNA16 Nodule Coordinates | Path: `scan_id: str` | `CTAnnotationsResponse` | `CTExplorerPage.tsx`, `ThreeMeshCanvas.tsx` | 3.9 ms |
| **13** | `GET` | `/api/v1/ct/{scan_id}/measurements` | Quantitative Lung Volumetry | Path: `scan_id: str` | `CTMeasurementResponse` | `CTExplorerPage.tsx` | 40.9 ms |
| **14** | `GET` | `/api/v1/ct/{scan_id}/segmentation` | Segmentation Metadata | Path: `scan_id: str` | `CTSegmentationResponse` | `ctApi.getSegmentation` | 4.8 ms |
| **15** | `POST` | `/api/v1/ct/upload` | Upload & Ingest 3D CT Volume | Multipart Form: `file: UploadFile` (.zip, .mhd, .nii) | `CTUploadResponse` | `CTExplorerPage.tsx` (Upload modal) | 5.1 ms (Reject) |
| **16** | `POST` | `/api/v1/xray/predict` | DenseNet121 6-Disease Inference | Multipart Form: `file: UploadFile` | `XRayMultiLabelResponse` | `useXrayInference`, `XrayPage.tsx` | 559.1 ms |
| **17** | `POST` | `/api/v1/xray/predict-binary` | ResNet18 Normal/Abnormal Classification | Multipart Form: `file`, `model_id: str` | `XRayBinaryResponse` | `useXrayInference`, `XrayPage.tsx` | 372.5 ms |

---

## 3. Discrepancy Reconciliation & Findings

### Discrepancy 1: Documentation Omission in Previous Integration Report
- **Issue:** `frontend/INTEGRATION_REPORT.md` enumerated only 15 routes, claiming full coverage.
- **Root Cause:** The author grouped `GET /api/v1/ct/{scan_id}` with `GET /api/v1/ct/{scan_id}/metadata`, and omitted the root discovery endpoint `GET /`.
- **Finding:** Both endpoints are active, functional, and fully documented in the OpenAPI schema.

### Discrepancy 2: Form Field Parameter in `predict-binary`
- **Issue:** Previous documentation specified the form field as `model_name`.
- **Root Cause:** In `backend/app/api/v1/xray.py`, the parameter is defined as `model_id: str = Form("original_resnet18")`.
- **Impact:** Submitting `model_name="blur_augmented_resnet18"` caused FastAPI to fall back to the default `model_id="original_resnet18"`.
- **Remediation:** Frontend `xrayApi.ts` correctly sends `formData.append('model_id', modelId)`. Documentation has been corrected.

### Discrepancy 3: Query Parameter Validation on `GET /api/v1/ct/{scan_id}/slice`
- **Observation:** `view=oblique` returns HTTP 200 by falling back to `axial`, rather than returning HTTP 422 Unprocessable Entity.
- **Architectural Analysis:** In `backend/app/api/v1/ct.py`, `view: str = Query("axial")` is typed as a generic string rather than an `Enum` or `Literal["axial", "coronal", "sagittal"]`. `slice_service.py` gracefully defaults unrecognized views to axial.
- **Recommendation:** Define an `AnatomicalView(str, Enum)` in `schemas/ct.py` to enforce strict OpenAPI client contract validation.

---

## 4. Live Automated Verification Test Matrix

Executed via `frontend/audit/test_api_matrix.py` against `http://127.0.0.1:8000`:

| Test # | Route | Test Scenario | Expected Status | Actual Status | Latency | Schema Validation | Verdict |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | `GET /` | Root service discovery metadata | 200 | 200 | 2859 ms | Valid Dict | **PASS** |
| 2 | `GET /api/v1/health` | Liveness probe check | 200 | 200 | 4.8 ms | Valid `HealthResponse` | **PASS** |
| 3 | `GET /api/v1/ready` | Readiness check with 3 initialized models | 200 | 200 | 4.0 ms | Valid `ReadyResponse` | **PASS** |
| 4 | `GET /api/v1/models` | List registered model catalog | 200 | 200 | 4.2 ms | Valid List (3 items) | **PASS** |
| 5 | `GET /api/v1/models/{model_id}` | Retrieve DenseNet121 metadata | 200 | 200 | 4.3 ms | Valid `ModelRegistryEntry` | **PASS** |
| 6 | `GET /api/v1/models/{model_id}` | Lookup non-existent model (`missing_id`) | 404 | 404 | 3.5 ms | `RESOURCE_NOT_FOUND` | **PASS** |
| 7 | `GET /api/v1/research/metrics` | Historical multi-dataset benchmark metrics | 200 | 200 | 7.7 ms | Valid Benchmarks | **PASS** |
| 8 | `GET /api/v1/ct/{scan_id}` | CT scan summary descriptor | 200 | 200 | 4.6 ms | Valid `CTMetadataResponse` | **PASS** |
| 9 | `GET /api/v1/ct/{scan_id}` | Missing scan lookup (`missing_scan_id`) | 404 | 404 | 2.8 ms | `RESOURCE_NOT_FOUND` | **PASS** |
| 10 | `GET /api/v1/ct/{scan_id}/metadata` | Full spatial and geometric metadata | 200 | 200 | 5.0 ms | Valid `CTMetadataResponse` | **PASS** |
| 11 | `GET /api/v1/ct/{scan_id}/slice` | Extract axial slice 32 with lung window | 200 | 200 | 26.7 ms | Valid Base64 PNG | **PASS** |
| 12 | `GET /api/v1/ct/{scan_id}/slice` | Invalid anatomical view plane (`view=oblique`) | 422 | 200 | 7.8 ms | Falls back to axial | **PARTIAL** |
| 13 | `GET /api/v1/ct/{scan_id}/mesh` | Stream Wavefront OBJ 3D mesh (step=2) | 200 | 200 | 112.5 ms | Valid OBJ text string | **PASS** |
| 14 | `GET /api/v1/ct/{scan_id}/mesh/quality` | Trimesh watertight & manifold audit | 200 | 200 | 343.2 ms | Valid `CTMeshQualityResponse` | **PASS** |
| 15 | `GET /api/v1/ct/{scan_id}/annotations` | LUNA16 mapped nodule coordinates | 200 | 200 | 3.9 ms | Valid (12 nodules) | **PASS** |
| 16 | `GET /api/v1/ct/{scan_id}/measurements` | Quantitative lung volumetry & airway balance | 200 | 200 | 40.9 ms | Valid Volumetrics | **PASS** |
| 17 | `GET /api/v1/ct/{scan_id}/segmentation` | Segmentation label map descriptor | 200 | 200 | 4.8 ms | Valid `CTSegmentationResponse` | **PASS** |
| 18 | `POST /api/v1/xray/predict` | DenseNet121 inference on real 2544x2516 CXR | 200 | 200 | 559.1 ms | Valid 6-Disease JSON | **PASS** |
| 19 | `POST /api/v1/xray/predict` | Reject corrupted/truncated image payload | 400 | 400 | 4.2 ms | `FILE_VALIDATION_ERROR` | **PASS** |
| 20 | `POST /api/v1/xray/predict-binary` | Original ResNet18 binary inference on real CXR | 200 | 200 | 372.5 ms | Valid Binary JSON | **PASS** |
| 21 | `POST /api/v1/xray/predict-binary` | Blur-augmented ResNet18 binary inference | 200 | 200 | 447.9 ms | Valid Binary JSON | **PASS** |
| 22 | `POST /api/v1/xray/predict-binary` | Reject unknown binary model (`unknown_model`) | 404 | 404 | 30.0 ms | `RESOURCE_NOT_FOUND` | **PASS** |
| 23 | `POST /api/v1/ct/upload` | Reject non-archive file missing ZIP magic bytes | 400 | 400 | 4.9 ms | `FILE_VALIDATION_ERROR` | **PASS** |

**Summary Statistics:**  
- Total Scenarios Tested: 23  
- Strict Passes: 22 (95.7%)  
- Partial (Graceful Fallback): 1 (4.3%)  
- Hard Failures: 0 (0.0%)

---

## 5. Error Schema Standard & Frontend Resilience

All error responses from the backend adhere strictly to the standardized envelope:
```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Model 'unknown_model' not found in registry.",
    "status_code": 404,
    "timestamp": "2026-11-12T18:22:14.123456Z",
    "details": {}
  }
}
```

The frontend Axios client (`frontend/src/services/api/client.ts`) intercepts this envelope:
1. Normalizes `error.response.data.error.message` into user-friendly notifications.
2. In mock mode (`VITE_USE_MOCK=true`), automatically routes requests to simulated endpoints if the live backend is unreachable.
3. Automatically sets system status pills in the top navigation header (`Online` / `Degraded` / `Offline`) via continuous 15-second polling of `/api/v1/health` and `/api/v1/ready`.

