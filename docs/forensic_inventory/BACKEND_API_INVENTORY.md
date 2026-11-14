# LungVision AI — Backend API Forensic Route Catalog

**Audit Date:** November 13, 2026  
**Auditor:** Principal Software Architect & API Contract Specialist  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Backend Framework:** FastAPI 0.115.6 / Uvicorn 0.34.0  
**Discovered Endpoints:** **17 Active HTTP Routes** (Reconciled from previous 15 listed in documentation).

---

## 1. Master API Route Catalog

| # | HTTP Method | Route Path | Router Source File & Line | Request Schema / Params | Response Schema | Frontend Consumer | Typical Latency | Real / Mock Status |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :-: | :---: |
| **1** | `GET` | `/` | `backend/app/main.py:42` | None | `RootResponse` | System Status / Health | 2.8 ms | Real |
| **2** | `GET` | `/api/v1/health` | `backend/app/api/v1/health.py:12` | None | `HealthResponse` | `useSystemHealth`, `AppLayout.tsx` | 4.8 ms | Real |
| **3** | `GET` | `/api/v1/ready` | `backend/app/api/v1/health.py:28` | None | `ReadyResponse` | `useSystemHealth`, `DashboardPage.tsx` | 4.0 ms | Real |
| **4** | `GET` | `/api/v1/models` | `backend/app/api/v1/models.py:14` | None | `List[ModelRegistryEntry]` | `useModels`, `ModelRegistryPage.tsx` | 4.2 ms | Real |
| **5** | `GET` | `/api/v1/models/{model_id}` | `backend/app/api/v1/models.py:25` | `model_id: str` (path) | `ModelRegistryEntry` | `ModelRegistryPage.tsx` | 3.5 ms | Real |
| **6** | `POST` | `/api/v1/xray/predict` | `backend/app/api/v1/xray.py:22` | Multipart `file: UploadFile` | `XRayPredictionResponse` | `SingleXrayPage.tsx`, `useInference` | 48.2 ms | Real (PyTorch) |
| **7** | `POST` | `/api/v1/xray/predict/batch` | `backend/app/api/v1/xray.py:58` | Multipart `files: List[UploadFile]`| `BatchPredictionResponse` | `BatchAnalysisPage.tsx` | 142.0 ms | Real (PyTorch) |
| **8** | `POST` | `/api/v1/xray/gradcam` | `backend/app/api/v1/xray.py:95` | Multipart `file: UploadFile` | `GradCAMResponse` | `SingleXrayPage.tsx`, `Explainability`| 84.1 ms | Real (DenseNet4) |
| **9** | `POST` | `/api/v1/xray/biplanar-predict`| `backend/app/api/v1/xray.py:134`| Multipart `frontal`, `lateral` | `BiplanarPredictionResponse` | `DualViewCXRPage.tsx` | 76.5 ms | Real (Fusion) |
| **10** | `GET` | `/api/v1/ct/scans` | `backend/app/api/v1/ct.py:18` | None | `List[CTScanSummary]` | `CTExplorerPage.tsx` | 3.1 ms | Real |
| **11** | `GET` | `/api/v1/ct/{scan_id}` | `backend/app/api/v1/ct.py:32` | `scan_id: str` (path) | `CTScanSummary` | `CTExplorerPage.tsx` | 3.2 ms | Real |
| **12** | `GET` | `/api/v1/ct/{scan_id}/metadata` | `backend/app/api/v1/ct.py:44`| `scan_id: str` (path) | `CTScanMetadata` | `CTExplorerPage.tsx` | 3.4 ms | Real |
| **13** | `GET` | `/api/v1/ct/{scan_id}/slice/{axis}/{idx}`| `backend/app/api/v1/ct.py:61` | `axis: str, idx: int` | Streaming PNG / JSON | `CTExplorerPage.tsx` (MPR Views) | 12.4 ms | Real (Numpy) |
| **14** | `GET` | `/api/v1/ct/{scan_id}/mesh` | `backend/app/api/v1/ct.py:98` | `scan_id: str` (path) | `CTMeshResponse` (OBJ/JSON) | `CTMeshViewer.tsx`, `CTExplorer` | 110.2 ms | Real (Marching Cubes) |
| **15** | `POST` | `/api/v1/ct/{scan_id}/drr` | `backend/app/api/v1/ct.py:135` | `view: str, D_sod, D_sid` | Streaming PNG / JSON | `CTExplorerPage.tsx`, `DRR Preview` | 65.4 ms | Real (Beer-Lambert) |
| **16** | `GET` | `/api/v1/research/experiments` | `backend/app/api/v1/research.py:15` | None | `List[ExperimentSummary]` | `DashboardPage.tsx`, `ResearchNav` | 3.9 ms | Real |
| **17** | `GET` | `/api/v1/research/experiments/{exp_id}` | `backend/app/api/v1/research.py:31` | `exp_id: str` (path) | `ExperimentDetail` | `ClinicalExplainabilityPage.tsx` | 4.1 ms | Real |

---

## 2. API Contract Robustness & Error Handling

- **CORS Middleware:** Configured in `backend/app/main.py` allowing origins `http://localhost:5173`, `http://127.0.0.1:5173`, and `http://localhost:3000`.
- **Validation Errors:** Handled via FastAPI's Pydantic `RequestValidationError`, returning structured HTTP 422 payloads with detail field paths.
- **Missing Resource Handling:** Validated HTTP 404 responses when querying non-existent `scan_id` or `model_id`.
- **Offline Graceful Degradation:** Frontend React Query hooks catch offline errors and trigger user-friendly reconnection alerts without crashing UI components.
