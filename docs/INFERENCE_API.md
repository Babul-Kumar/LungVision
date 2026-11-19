# LungVision AI — Clinical Inference API Reference

## 1. Overview

The LungVision AI Backend Gateway (FastAPI) exposes high-throughput endpoints for thoracic radiograph screening, model governance, and telemetry.

Base URL: `http://127.0.0.1:8000/api/v1`

---

## 2. API Endpoints

### 2.1 Clinical Prediction: `POST /api/v1/predictions`

Executes end-to-end multi-label inference using the Phase 7 MultiScaleDenseNet121 engine.

#### Request Parameters (Multipart Form Data)
| Parameter | Type | Required | Default | Description |
|---|---|:---:|:---:|---|
| `file` | Binary | Yes | - | Frontal chest X-ray image (DICOM, PNG, JPEG, WebP). Max 20 MB. |
| `operating_point` | String | No | `operating_point_A_f1_optimal` | Calibration profile: `operating_point_A_f1_optimal`, `operating_point_B_balanced_youden`, or `operating_point_C_high_sensitivity`. |
| `generate_cam` | Boolean | No | `false` | Whether to compute high-resolution Grad-CAM explainability heatmap overlay. |
| `cam_target_disease` | String | No | `null` | Target pathology for Grad-CAM (e.g. `Pneumothorax`). Defaults to highest positive finding. |

#### Successful Response (`200 OK`)
```json
{
  "status": "COMPLETED",
  "findings": {
    "Cardiomegaly": {
      "disease": "Cardiomegaly",
      "probability": 0.0858,
      "threshold": 0.87,
      "is_positive": false,
      "margin": 0.7842,
      "confidence_level": "HIGH",
      "clinical_note": "Definitive prediction with high boundary separation."
    },
    "Pneumothorax": {
      "disease": "Pneumothorax",
      "probability": 0.8124,
      "threshold": 0.77,
      "is_positive": true,
      "margin": 0.0424,
      "confidence_level": "BORDERLINE",
      "clinical_note": "Borderline prediction (|p - τ| = 0.042); near decision boundary."
    }
  },
  "uncertainty": {
    "overall_confidence": "LOW",
    "mean_margin": 0.4521,
    "borderline_diseases": ["Pneumothorax"],
    "positive_findings": ["Pneumothorax"],
    "recommendation": "HIGH_ACUITY_ALERT",
    "recommendation_reason": "Critical finding(s) detected: Pneumothorax. Immediate radiologist review advised."
  },
  "quality_report": {
    "status": "PASSED",
    "can_proceed": true,
    "metrics": {
      "width": 1024,
      "height": 1024,
      "aspect_ratio": 1.0,
      "mean_intensity": 112.45,
      "std_intensity": 58.21,
      "blur_variance": 184.32,
      "is_blurry": false,
      "is_blank": false,
      "is_overexposed": false,
      "is_underexposed": false
    },
    "warnings": []
  },
  "gradcam": {
    "disease": "Pneumothorax",
    "target_idx": 5,
    "overlay_base64": "data:image/jpeg;base64,...",
    "heatmap_base64": "data:image/png;base64,..."
  },
  "duration_ms": 142.5,
  "model_metadata": {
    "model_id": "lungvision_phase7_msd121_224_direct",
    "architecture": "MultiScaleDenseNet121",
    "total_parameters": 8071942,
    "validation_macro_auroc": 0.819726
  },
  "timestamp": "2026-11-19T12:00:00Z"
}
```

#### Rejection Response (`200 OK` with `status: REJECTED`)
```json
{
  "status": "REJECTED",
  "error": "Resolution (64x64) is below clinical safety minimum (128x128).",
  "quality_report": {
    "status": "IMAGE_QUALITY_INSUFFICIENT",
    "can_proceed": false,
    "rejection_reason": "Resolution (64x64) is below clinical safety minimum (128x128)."
  },
  "duration_ms": 2.4
}
```

---

### 2.2 Operational Health: `GET /api/v1/model/health`

Returns engine status, memory utilization, and checkpoint byte-integrity SHA-256 hash.

```json
{
  "status": "healthy",
  "model_id": "lungvision_phase7_msd121_224_direct",
  "architecture": "MultiScaleDenseNet121",
  "version": "7.0.0",
  "device": "cpu",
  "cuda_info": {},
  "checkpoint_sha256": "7dc735ee00d9e57ece14cd33f2bdbd1e7fd226adca41cdb692ba0af2a33a9b23",
  "checkpoint_path": "models/multiscale_densenet121/msd121_224_weighted_bce_direct.pth",
  "model_loaded_at": "2026-11-19T12:00:00Z",
  "total_inferences": 42
}
```

---

### 2.3 Scientific Metadata: `GET /api/v1/model/info`

Returns comprehensive architecture specifications, dataset provenance, and Brier calibration scores.

---

### 2.4 Backwards-Compatible Screening: `POST /api/v1/xray/predict`

Accepts standard multipart form uploads. Allows selecting between `lungvision_phase7_msd121_224_direct` and legacy `densenet121_nih_6disease` via `model_id` form parameter.
