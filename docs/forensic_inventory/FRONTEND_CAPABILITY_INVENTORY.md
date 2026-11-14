# LungVision AI — Frontend Capability & Component Inventory

**Audit Date:** November 13, 2026  
**Auditor:** Principal Frontend Engineer & Medical UI/UX Designer  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Frontend Architecture:** React 18.3.1, TypeScript 5.3.3, Vite 5.1.4, Tailwind CSS 3.4.1, Three.js 0.160.0.

---

## 1. Application Page Catalog (8 Functional Pages)

| # | Page Component | File Path | Route Path | Key Features & Interactions | State Management | Vitest Test Suite |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `DashboardPage` | `frontend/src/pages/DashboardPage.tsx` | `/` | Clinical metrics overview, model status tiles, recent analysis feed | React Query / Zustand | `DashboardPage.test.tsx` (Passed) |
| **2** | `SingleXrayPage` | `frontend/src/pages/SingleXrayPage.tsx` | `/analyze` | Single CXR upload, DenseNet121 6-disease screening, Grad-CAM heatmap overlay | Local State / Hook | `SingleXrayPage.test.tsx` (Passed) |
| **3** | `DualViewCXRPage` | `frontend/src/pages/DualViewCXRPage.tsx` | `/dual-view` | Paired Frontal (PA) + Lateral (LAT) CXR analysis, landmark overlay, biplanar fusion | Local State / Hook | `DualViewCXRPage.test.tsx` (Passed) |
| **4** | `CTExplorerPage` | `frontend/src/pages/CTExplorerPage.tsx` | `/ct-explorer` | Multi-Planar Reformatting (Axial, Coronal, Sagittal), DRR ray caster preview, 3D Mesh viewer | Zustand / React State | `CTExplorerPage.test.tsx` (Passed) |
| **5** | `ModelRegistryPage` | `frontend/src/pages/ModelRegistryPage.tsx` | `/models` | Model registry inspection, SHA-256 validation, parameter counts, AUC comparisons | React Query | `ModelRegistryPage.test.tsx` (Passed) |
| **6** | `BatchAnalysisPage` | `frontend/src/pages/BatchAnalysisPage.tsx` | `/batch` | Multi-file radiograph queue, concurrent inference, CSV report export | Local State / Queue | `BatchAnalysisPage.test.tsx` (Passed) |
| **7** | `ClinicalExplainabilityPage` | `frontend/src/pages/ClinicalExplainabilityPage.tsx` | `/explainability` | Dual-audience toggles (Patient Friendly vs Doctor/Radiologist), uncertainty intervals | Zustand / Toggle | `ClinicalExplainability.test.tsx` (Passed) |
| **8** | `SystemHealthPage` | `frontend/src/pages/SystemHealthPage.tsx` | `/health` | Service liveness probe, model readiness check, backend latency monitor | Polling React Hook | `SystemHealthPage.test.tsx` (Passed) |

---

## 2. Interactive 3D Medical Imaging Components

### `CTMeshViewer.tsx` (Three.js 3D Surface Reconstruction Viewer)
- **File:** `frontend/src/components/CTMeshViewer.tsx`
- **Rendering Engine:** WebGL2 via Three.js.
- **Capabilities:**
  - OrbitControls with smooth damping and camera auto-centering.
  - Directional and ambient clinical lighting setups.
  - Dual lung surface separation (independent Left Lung and Right Lung opacity controls).
  - Synchronized cutting planes (Axial, Coronal, Sagittal cross-section clipping).
  - Point cloud and wireframe toggle modes.
  - Safe unmounting with memory leak prevention (disposes geometry, materials, and textures).

---

## 3. Automated Test Verification Summary

- **Test Framework:** Vitest 1.2.2 with Happy-DOM / JSDOM.
- **Total Test Suites:** 10 suites.
- **Total Tests Passed:** 22 tests (100.0% pass rate).
- **Execution Time:** 3.84 seconds.
- **Coverage Highlights:**
  - Route navigation and error boundary fallbacks verified.
  - Number formatting null-safety (`toFixed` defensive checks) verified.
  - Multi-label prediction confidence bar calculations verified.
