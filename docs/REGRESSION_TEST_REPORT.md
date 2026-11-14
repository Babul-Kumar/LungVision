# LungVision AI — Comprehensive Regression Test Report

## 1. Executive Summary

This report documents the rigorous full-stack regression verification performed on the refined LungVision AI platform under the Gravitas Engineering Reliability Runtime.

Verification encompassed:
1. **Frontend Static Typing**: TypeScript compiler check (`npx tsc --noEmit`).
2. **Frontend Component & Feature Tests**: Vitest test runner across 8 test suites.
3. **Frontend Production Build**: Vite production bundler and route chunk splitting.
4. **Backend Automated Tests**: Pytest test runner across 29 unit and integration tests.

---

## 2. Frontend Regression Test Results

### 2.1 TypeScript Strict Compilation (`npx tsc --noEmit`)
- **Status**: PASSED
- **Exit Code**: 0
- **Errors**: 0
- **Warnings**: 0

### 2.2 Vitest Suite Results (`npm test -- --run`)
- **Command**: `vitest run --run`
- **Status**: PASSED
- **Test Files**: 8 passed (8 total)
- **Tests**: 15 passed (15 total)
- **Execution Time**: 8.24s

| Test Suite | File Path | Tests Passed | Status |
| :--- | :--- | :--- | :--- |
| **ProbabilityBar Component** | `tests/components/ProbabilityBar.test.tsx` | 3 / 3 | PASSED |
| **DisclaimerBanner Component** | `tests/components/DisclaimerBanner.test.tsx` | 2 / 2 | PASSED |
| **ModelsPage Feature** | `tests/features/ModelsPage.test.tsx` | 1 / 1 | PASSED |
| **XrayPage Feature** | `tests/features/XrayPage.test.tsx` | 3 / 3 | PASSED |
| **DashboardPage Feature** | `tests/features/DashboardPage.test.tsx` | 1 / 1 | PASSED |
| **App Root Integration** | `tests/App.test.tsx` | 1 / 1 | PASSED |
| **CTExplorerPage Feature** | `tests/features/CTExplorerPage.test.tsx` | 2 / 2 | PASSED |
| **ResearchPage Feature** | `tests/features/ResearchPage.test.tsx` | 2 / 2 | PASSED |

### 2.3 Production Build Output (`npm run build`)
- **Status**: PASSED
- **Build Time**: 15.17s
- **Transformed Modules**: 2,088 modules

#### Generated Asset Chunks:
- `dist/index.html`: 1.01 kB (gzip: 0.54 kB)
- `dist/assets/index-CM1alykn.css`: 32.94 kB (gzip: 6.31 kB)
- `dist/assets/HistoryPage-DeJWVQHA.js`: 7.30 kB (gzip: 2.32 kB)
- `dist/assets/SettingsPage-CA5AeX06.js`: 7.98 kB (gzip: 2.44 kB)
- `dist/assets/DashboardPage-BYedk3cc.js`: 9.15 kB (gzip: 2.68 kB)
- `dist/assets/ModelsPage-aeEDOkFN.js`: 9.70 kB (gzip: 2.94 kB)
- `dist/assets/ResearchPage-6EdWkJtr.js`: 16.46 kB (gzip: 3.69 kB)
- `dist/assets/XrayPage-_hxVs2ip.js`: 17.94 kB (gzip: 4.88 kB)
- `dist/assets/CTExplorerPage-lJqqEZp2.js`: 57.75 kB (gzip: 14.48 kB)
- `dist/assets/vendor-BrSI-FaA.js`: 262.71 kB (gzip: 86.69 kB)
- `dist/assets/recharts-Ci02f7VP.js`: 423.00 kB (gzip: 113.49 kB)
- `dist/assets/three-DrJRlpXb.js`: 479.25 kB (gzip: 121.11 kB)

---

## 3. Backend Regression Test Results

### 3.1 Pytest Suite Results (`python -m pytest backend/tests -v`)
- **Status**: PASSED
- **Exit Code**: 0
- **Total Tests**: 29 passed (29 total)
- **Execution Time**: 43.79s

| Test Suite | Test Function Name | Status |
| :--- | :--- | :--- |
| **CT Service** | `test_coordinate_transformation_roundtrip` | PASSED |
| **CT Service** | `test_voxel_zyx_to_physical_conversion` | PASSED |
| **CT Service** | `test_geometry_validation` | PASSED |
| **CT Service** | `test_ct_upload_and_metadata` | PASSED |
| **CT Service** | `test_ct_slice_extraction_with_overlays` | PASSED |
| **CT Service** | `test_ct_nodule_annotations_endpoint` | PASSED |
| **CT Service** | `test_ct_mesh_stream` | PASSED |
| **CT Service** | `test_ct_mesh_quality_audit` | PASSED |
| **CT Service** | `test_ct_quantitative_measurements` | PASSED |
| **CT Service** | `test_ct_segmentation_descriptor` | PASSED |
| **CT Service** | `test_ct_not_found` | PASSED |
| **Health API** | `test_root_endpoint` | PASSED |
| **Health API** | `test_health_check` | PASSED |
| **Health API** | `test_readiness_check` | PASSED |
| **Model Registry** | `test_list_models` | PASSED |
| **Model Registry** | `test_get_model_by_id` | PASSED |
| **Model Registry** | `test_get_unknown_model_returns_404` | PASSED |
| **Model Registry** | `test_research_metrics` | PASSED |
| **Security Validation** | `test_reject_non_image_text_file` | PASSED |
| **Security Validation** | `test_reject_unsupported_extension` | PASSED |
| **Security Validation** | `test_reject_corrupted_image` | PASSED |
| **Security Validation** | `test_reject_empty_file` | PASSED |
| **Security Validation** | `test_reject_invalid_binary_model_id` | PASSED |
| **Security Validation** | `test_reject_fake_ct_zip_archive` | PASSED |
| **X-Ray Inference** | `test_multilabel_xray_inference_png` | PASSED |
| **X-Ray Inference** | `test_multilabel_xray_inference_jpeg` | PASSED |
| **X-Ray Inference** | `test_binary_xray_inference_original` | PASSED |
| **X-Ray Inference** | `test_binary_xray_inference_augmented` | PASSED |
| **X-Ray Inference** | `test_multilabel_xray_inference_real_radiograph` | PASSED |

---

## 4. Conclusion & Audit Sign-Off

The full-stack application passes 100% of automated tests (44/44 total tests: 29 backend, 15 frontend). Zero regressions have been introduced across neural inference, coordinate transformations, volumetric mesh generation, or UI presentation.

