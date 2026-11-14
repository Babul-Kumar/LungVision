# LungVision AI — Comprehensive Regression Test Report

## 1. Executive Summary

This report documents the rigorous full-stack regression verification performed on the refined LungVision AI platform under the Gravitas Engineering Reliability Runtime (`.gravitas/contract.json`).

Verification encompassed:
1. **Frontend Static Typing**: Strict TypeScript compiler check (`npx tsc --noEmit`).
2. **Frontend Component & Feature Tests**: Vitest test runner across 10 test suites (26 tests total).
3. **Frontend Production Build**: Vite production bundler with optimized route chunk splitting and WebGL assets.
4. **Backend Automated Tests**: Pytest test runner across 8 test suites (46 unit and integration tests total).
5. **Multi-Patient SSM Pipeline**: Model integrity, zero patient leakage validation, and runtime deformation serving.

**Overall Test Results:** **72 / 72 Passing (100% PASS RATE)**

---

## 2. Frontend Regression Test Results

### 2.1 TypeScript Strict Compilation (`npx tsc --noEmit`)
- **Status**: PASSED
- **Exit Code**: 0
- **Errors**: 0
- **Warnings**: 0

### 2.2 Vitest Suite Results (`npm test -- --run`)
- **Command**: `vitest run --run`
- **Status**: **PASSED**
- **Test Files**: 10 passed (10 total)
- **Tests**: 26 passed (26 total)
- **Execution Time**: 7.82s

| Test Suite | File Path | Tests Passed | Status |
| :--- | :--- | :---: | :---: |
| **ProbabilityBar Component** | `tests/components/ProbabilityBar.test.tsx` | 3 / 3 | **PASSED** |
| **DisclaimerBanner Component** | `tests/components/DisclaimerBanner.test.tsx` | 2 / 2 | **PASSED** |
| **ModelsPage Feature** | `tests/features/ModelsPage.test.tsx` | 1 / 1 | **PASSED** |
| **XrayPage Feature** | `tests/features/XrayPage.test.tsx` | 3 / 3 | **PASSED** |
| **DashboardPage Feature** | `tests/features/DashboardPage.test.tsx` | 1 / 1 | **PASSED** |
| **App Root Integration** | `tests/App.test.tsx` | 1 / 1 | **PASSED** |
| **CTExplorerPage Feature** | `tests/features/CTExplorerPage.test.tsx` | 5 / 5 | **PASSED** |
| **DualViewXRayPage Feature** | `tests/features/DualViewXRayPage.test.tsx` | 4 / 4 | **PASSED** |
| **ExplainabilityAndDualMode Feature** | `tests/features/ExplainabilityAndDualMode.test.tsx` | 4 / 4 | **PASSED** |
| **ResearchPage Feature** | `tests/features/ResearchPage.test.tsx` | 2 / 2 | **PASSED** |

### 2.3 Production Build Output (`npm run build`)
- **Status**: **PASSED**
- **Build Time**: 12.18s
- **Transformed Modules**: 2,096 modules

#### Generated Asset Chunks:
- `dist/index.html`: 1.01 kB (gzip: 0.54 kB)
- `dist/assets/index-D-Pn_RiI.css`: 48.79 kB (gzip: 8.60 kB)
- `dist/assets/modelApi-BwCJltDb.js`: 0.18 kB (gzip: 0.15 kB)
- `dist/assets/Badge-gxD1nApX.js`: 1.47 kB (gzip: 0.65 kB)
- `dist/assets/useHistoryStore-3vLZpAm3.js`: 1.48 kB (gzip: 0.72 kB)
- `dist/assets/Button-OX95Ie_0.js`: 1.53 kB (gzip: 0.78 kB)
- `dist/assets/ProbabilityBar-BEBuZ-TQ.js`: 2.15 kB (gzip: 0.95 kB)
- `dist/assets/HistoryPage-DLmI18HN.js`: 7.30 kB (gzip: 2.32 kB)
- `dist/assets/SettingsPage-DJnh3vor.js`: 7.98 kB (gzip: 2.45 kB)
- `dist/assets/ModelsPage-Dk0pZM3n.js`: 10.71 kB (gzip: 3.27 kB)
- `dist/assets/index-COQxSy1-.js`: 15.57 kB (gzip: 5.56 kB)
- `dist/assets/DashboardPage-DrLaorzk.js`: 16.31 kB (gzip: 3.47 kB)
- `dist/assets/ResearchPage-CNYih9F2.js`: 16.46 kB (gzip: 3.69 kB)
- `dist/assets/DualViewXRayPage-ChSI6xdd.js`: 32.11 kB (gzip: 7.58 kB)
- `dist/assets/SSM3DReconstructionViewer-BUOUbkWZ.js`: 34.65 kB (gzip: 9.59 kB)
- `dist/assets/CTExplorerPage-WguMS0rE.js`: 62.00 kB (gzip: 14.06 kB)
- `dist/assets/XrayPage-D86hyD8c.js`: 63.77 kB (gzip: 15.64 kB)
- `dist/assets/vendor-TzKzMs4l.js`: 268.87 kB (gzip: 87.88 kB)
- `dist/assets/recharts-CttK7ALX.js`: 423.00 kB (gzip: 113.49 kB)
- `dist/assets/three-B0VNlWly.js`: 479.27 kB (gzip: 121.11 kB)

---

## 3. Backend Regression Test Results

### 3.1 Pytest Suite Results (`python -m pytest backend/tests -v`)
- **Status**: **PASSED**
- **Exit Code**: 0
- **Total Tests**: 46 passed (46 total across 8 suites)
- **Execution Time**: 55.11s

| Test Suite | Test Function Name | Status |
| :--- | :--- | :---: |
| **CT DRR Synthesis** | `test_generate_frontal_drr` | **PASSED** |
| **CT DRR Synthesis** | `test_generate_lateral_drr` | **PASSED** |
| **CT DRR Synthesis** | `test_generate_drr_with_silhouette_overlay` | **PASSED** |
| **CT DRR Synthesis** | `test_generate_drr_with_poisson_noise` | **PASSED** |
| **CT DRR Synthesis** | `test_drr_unknown_scan_returns_404` | **PASSED** |
| **CT Service** | `test_coordinate_transformation_roundtrip` | **PASSED** |
| **CT Service** | `test_voxel_zyx_to_physical_conversion` | **PASSED** |
| **CT Service** | `test_geometry_validation` | **PASSED** |
| **CT Service** | `test_ct_upload_and_metadata` | **PASSED** |
| **CT Service** | `test_ct_slice_extraction_with_overlays` | **PASSED** |
| **CT Service** | `test_ct_nodule_annotations_endpoint` | **PASSED** |
| **CT Service** | `test_ct_mesh_stream` | **PASSED** |
| **CT Service** | `test_ct_mesh_quality_audit` | **PASSED** |
| **CT Service** | `test_ct_quantitative_measurements` | **PASSED** |
| **CT Service** | `test_ct_segmentation_descriptor` | **PASSED** |
| **CT Service** | `test_ct_not_found` | **PASSED** |
| **Health API** | `test_root_endpoint` | **PASSED** |
| **Health API** | `test_health_check` | **PASSED** |
| **Health API** | `test_readiness_check` | **PASSED** |
| **Model Registry** | `test_list_models` | **PASSED** |
| **Model Registry** | `test_get_model_by_id` | **PASSED** |
| **Model Registry** | `test_get_unknown_model_returns_404` | **PASSED** |
| **Model Registry** | `test_research_metrics` | **PASSED** |
| **Security Validation** | `test_reject_non_image_text_file` | **PASSED** |
| **Security Validation** | `test_reject_unsupported_extension` | **PASSED** |
| **Security Validation** | `test_reject_corrupted_image` | **PASSED** |
| **Security Validation** | `test_reject_empty_file` | **PASSED** |
| **Security Validation** | `test_reject_invalid_binary_model_id` | **PASSED** |
| **Security Validation** | `test_reject_fake_ct_zip_archive` | **PASSED** |
| **SSM Phase 2B Pipeline** | `test_data_manifest_and_labels_present` | **PASSED** |
| **SSM Phase 2B Pipeline** | `test_patient_split_zero_leakage` | **PASSED** |
| **SSM Phase 2B Pipeline** | `test_ssm_model_artifact_schema_and_size` | **PASSED** |
| **SSM Phase 2B Pipeline** | `test_held_out_validation_metrics_acceptable` | **PASSED** |
| **SSM Phase 2B Pipeline** | `test_backend_runtime_sync` | **PASSED** |
| **SSM 3D Reconstruction** | `test_ssm_model_info_endpoint` | **PASSED** |
| **SSM 3D Reconstruction** | `test_ssm_fit_default_endpoint` | **PASSED** |
| **SSM 3D Reconstruction** | `test_ssm_fit_frontal_only` | **PASSED** |
| **SSM 3D Reconstruction** | `test_ssm_fit_with_2d_landmarks` | **PASSED** |
| **SSM 3D Reconstruction** | `test_ssm_template_mesh_obj` | **PASSED** |
| **SSM 3D Reconstruction** | `test_ssm_export_obj_endpoint` | **PASSED** |
| **SSM 3D Reconstruction** | `test_ssm_deform_endpoint` | **PASSED** |
| **X-Ray Inference** | `test_multilabel_xray_inference_png` | **PASSED** |
| **X-Ray Inference** | `test_multilabel_xray_inference_jpeg` | **PASSED** |
| **X-Ray Inference** | `test_binary_xray_inference_original` | **PASSED** |
| **X-Ray Inference** | `test_binary_xray_inference_augmented` | **PASSED** |
| **X-Ray Inference** | `test_multilabel_xray_inference_real_radiograph` | **PASSED** |
