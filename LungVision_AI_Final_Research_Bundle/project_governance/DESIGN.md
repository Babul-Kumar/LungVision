# LungVision AI — Technical Design Specification (DESIGN.md)

**Document Version:** 1.0.0  
**Status:** Approved / Active Governance  
**Author:** Staff Backend Architect & MLOps Lead  
**Governing Skill:** Gravitas Engineering Reliability Runtime  
**Created:** 2026-11-12  

---

## 1. System Module Design

The backend is built as an asynchronous service powered by FastAPI and PyTorch. Modules are structured according to clean architecture principles:

```text
HTTP Request
     │
     ▼
[Core Middleware] ────► CORS, Exception Handling, Request Logging
     │
     ▼
[API Routers] ────────► Param parsing, multipart unpacking, schema validation
     │
     ▼
[Service Layer] ──────► Business logic, image preprocessing, threshold application
     │
     ▼
[ML Engine] ──────────► In-memory PyTorch models (DenseNet121 / ResNet18)
     │
     ▼
Structured Pydantic Model ──► HTTP JSON Response
```

---

## 2. API Endpoint Specifications

Base URL: `/api/v1`

### 2.1 Health & Readiness Endpoints

#### `GET /api/v1/health`
Checks whether the FastAPI application process is alive.
- **Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-11-12T03:30:00Z"
}
```

#### `GET /api/v1/ready`
Checks whether required PyTorch models are loaded in memory and ready for inference.
- **Response (200 OK):**
```json
{
  "status": "ready",
  "device": "cpu",
  "models_loaded": {
    "densenet121_6disease": true,
    "resnet18_original": true,
    "resnet18_augmented": true
  },
  "timestamp": "2026-11-12T03:30:00Z"
}
```
- **Response (503 Service Unavailable):** Returned if the primary model failed to load.

---

### 2.2 2D X-Ray Classification Endpoints

#### `POST /api/v1/xray/predict`
Executes multi-label classification using the verified 6-disease DenseNet121 model.
- **Content-Type:** `multipart/form-data`
- **Request Parameters:**
  - `file`: Binary image file (`image/png`, `image/jpeg`, `image/webp`). Max size: $20\text{ MB}$.
- **Response Schema (`XRayMultiLabelResponse`):**
```json
{
  "success": true,
  "model": {
    "id": "densenet121_nih_6disease",
    "architecture": "DenseNet121",
    "task": "Multi-label chest X-ray disease classification",
    "training_dataset": "NIH ChestX-ray14",
    "checkpoint_epoch": 4,
    "input_dimensions": [224, 224, 3]
  },
  "predictions": {
    "Cardiomegaly": {
      "probability": 0.0319,
      "threshold": 0.9502,
      "experimental_flag": false
    },
    "Edema": {
      "probability": 0.0054,
      "threshold": 0.8770,
      "experimental_flag": false
    },
    "Consolidation": {
      "probability": 0.0554,
      "threshold": 0.7329,
      "experimental_flag": false
    },
    "Pneumonia": {
      "probability": 0.0009,
      "threshold": 0.7529,
      "experimental_flag": false
    },
    "Atelectasis": {
      "probability": 0.0848,
      "threshold": 0.7495,
      "experimental_flag": false
    },
    "Pneumothorax": {
      "probability": 0.0068,
      "threshold": 0.5825,
      "experimental_flag": false
    }
  },
  "summary": {
    "flagged_conditions_count": 0,
    "highest_probability_disease": "Atelectasis",
    "highest_probability_value": 0.0848
  },
  "inference_time_ms": 42.5,
  "disclaimer": "For research and educational use only. Not a medical diagnosis. Independent clinical verification required."
}
```

#### `POST /api/v1/xray/predict-binary`
Executes binary Normal vs. Abnormal classification using the frozen ResNet18 models.
- **Content-Type:** `multipart/form-data`
- **Request Parameters:**
  - `file`: Binary image file.
  - `model_id`: Optional query/form string (`"original_resnet18"` or `"blur_augmented_resnet18"`, default: `"original_resnet18"`).
- **Response Schema (`XRayBinaryResponse`):**
```json
{
  "success": true,
  "model": {
    "id": "original_resnet18",
    "architecture": "ResNet18",
    "task": "Binary Normal vs. Abnormal classification",
    "training_dataset": "IN-CXR",
    "checkpoint_epoch": 4
  },
  "prediction": {
    "abnormal_probability": 0.1245,
    "calibrated_threshold": 0.1765,
    "classification": "Normal",
    "experimental_flag": false
  },
  "inference_time_ms": 28.1,
  "disclaimer": "For research and educational use only. Not a medical diagnosis."
}
```

---

### 2.3 3D CT Exploration Endpoints

#### `POST /api/v1/ct/upload`
Uploads a CT scan volume or DICOM archive.
- **Content-Type:** `multipart/form-data`
- **Parameters:** `file` (`.zip`, `.nii`, `.nii.gz`, `.mhd`).
- **Response (201 Created):**
```json
{
  "success": true,
  "scan_id": "ct_9b1deb4d3b7d4bad",
  "filename": "chest_scan_sample.zip",
  "size_bytes": 1420584,
  "status": "ready",
  "message": "CT volume registered successfully"
}
```

#### `GET /api/v1/ct/{scan_id}/metadata`
Retrieves volumetric dimensions, spacing, origin, direction matrix, and window presets.
- **Response (200 OK):**
```json
{
  "scan_id": "ct_9b1deb4d3b7d4bad",
  "dimensions": { "x": 512, "y": 512, "z": 128 },
  "spacing_mm": [0.742188, 0.742188, 1.25],
  "origin_mm": [-193.1, -166.7, -307.25],
  "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
  "supported_views": ["axial", "coronal", "sagittal"],
  "has_segmentation": true,
  "window_presets": {
    "lung": { "center": -600, "width": 1500 },
    "mediastinum": { "center": 50, "width": 350 },
    "bone": { "center": 400, "width": 1800 }
  }
}
```

#### `GET /api/v1/ct/{scan_id}/slice`
Extracts an orthogonal 2D slice as a base64-encoded PNG image with clinical windowing and optional segmentation/nodule overlays.
- **Query Params:**
  - `plane`: `"axial"`, `"coronal"`, or `"sagittal"` (default: `"axial"`).
  - `index`: Slice index integer along the chosen orthogonal axis.
  - `window`: Window preset (`"lung"`, `"mediastinum"`, `"bone"`).
  - `overlay_segmentation`: Boolean (`true`/`false`), alpha-blended contours for Left Lung (label 3), Right Lung (label 4), and Trachea (label 5).
  - `overlay_nodules`: Boolean (`true`/`false`), projects nodule crosshairs and circles when intersecting slice.
- **Response (200 OK):**
```json
{
  "scan_id": "ct_9b1deb4d3b7d4bad",
  "plane": "axial",
  "index": 64,
  "max_index": 127,
  "physical_coordinate_mm": -227.25,
  "window_applied": "lung",
  "has_segmentation_overlay": true,
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "intersecting_nodules": [
    {
      "annotation_id": "ann_0",
      "coord_x": -128.5,
      "coord_y": -96.2,
      "coord_z": -227.0,
      "diameter_mm": 5.4,
      "distance_to_plane_mm": 0.25
    }
  ]
}
```

#### `GET /api/v1/ct/{scan_id}/mesh`
Extracts and streams a 3D surface mesh (standard Wavefront OBJ) for anatomical thoracic structures.
- **Query Params:**
  - `structure`: `"lungs"`, `"airways"`, `"left_lung"`, or `"right_lung"` (default: `"lungs"`).
  - `step`: Downsampling step size (default: 2).
  - `pad_boundary`: Boolean (`true`/`false`, default `true`) ensuring watertight closed boundary surfaces.
- **Response:** `200 OK` with `Content-Type: text/plain` containing standard Wavefront `.obj` `v` and `f` directives.

#### `GET /api/v1/ct/{scan_id}/mesh/quality`
Audits the topological quality of the extracted marching cubes mesh via Trimesh.
- **Response (200 OK):**
```json
{
  "scan_id": "ct_9b1deb4d3b7d4bad",
  "structure": "lungs",
  "is_watertight": true,
  "euler_number": 2,
  "is_winding_consistent": true,
  "vertex_count": 14250,
  "face_count": 28496,
  "degenerate_face_count": 0,
  "surface_area_mm2": 84520.4,
  "mesh_volume_ml": 4210.5,
  "voxel_volume_ml": 4235.1,
  "volume_discrepancy_pct": 0.58,
  "quality_tier": "watertight_manifold"
}
```

#### `GET /api/v1/ct/{scan_id}/annotations`
Retrieves all mapped LUNA16 nodule annotations in physical and continuous voxel coordinates.
- **Response (200 OK):**
```json
{
  "scan_id": "ct_9b1deb4d3b7d4bad",
  "series_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518",
  "annotation_count": 12,
  "inside_volume_count": 12,
  "annotations": [
    {
      "annotation_id": "ann_0",
      "coord_x": -128.5,
      "coord_y": -96.2,
      "coord_z": -245.0,
      "diameter_mm": 5.4,
      "voxel_x": 87.04,
      "voxel_y": 94.99,
      "voxel_z": 49.80,
      "inside_volume": true,
      "anatomical_region": "right_lung"
    }
  ],
  "disclaimer": "LUNA16 ground-truth annotations are provided for spatial reference only. Not an autonomous malignancy prediction."
}
```

#### `GET /api/v1/ct/{scan_id}/measurements`
Calculates high-precision anatomical lung volumes from verified segmentation masks.
- **Response (200 OK):**
```json
{
  "scan_id": "ct_9b1deb4d3b7d4bad",
  "voxel_volume_mm3": 0.6885,
  "left_lung_volume_ml": 2045.2,
  "right_lung_volume_ml": 2210.8,
  "total_lung_volume_ml": 4256.0,
  "trachea_volume_ml": 42.1,
  "lung_symmetry_ratio": 0.925,
  "label_distribution": {
    "background_0": 94.2,
    "left_lung_3": 2.7,
    "right_lung_4": 3.0,
    "airway_5": 0.1
  },
  "geometry_valid": true,
  "disclaimer": "Quantitative volume calculations are for research analysis only. Do not use as standalone clinical diagnostic values."
}
```

---

### 2.4 Model Registry & Research Metrics Endpoints

#### `GET /api/v1/models`
Returns detailed configuration and hashes for all available models.

#### `GET /api/v1/models/{model_id}`
Returns granular metadata, training history summary, and threshold details for a specific model.

#### `GET /api/v1/research/metrics`
Returns verified historical benchmarks across datasets:
- NIH test set calibrated metrics (`nih_test_calibrated_metrics.csv`).
- CheXpert external metrics (`chexpert_external_metrics.csv`).
- VinDr external metrics (`vindr_external_metrics.csv`).

---

## 3. Error Response Format (RFC 7807 Compliant)

All client and server errors return a standard JSON structure:
```json
{
  "error": {
    "code": "INVALID_FILE_TYPE",
    "message": "Unsupported file format. Supported formats: image/png, image/jpeg, image/webp",
    "timestamp": "2026-11-12T03:30:00Z",
    "details": {
      "provided_content_type": "text/plain"
    }
  }
}
```

---

## 4. Frontend Integration Contract (React + TypeScript)

### 4.1 TypeScript Interfaces for the Frontend
```typescript
export interface DiseasePrediction {
  probability: number;
  threshold: number;
  experimental_flag: boolean;
}

export interface XRayPredictionResponse {
  success: boolean;
  model: {
    id: string;
    architecture: string;
    task: string;
    training_dataset: string;
    checkpoint_epoch: number;
    input_dimensions: number[];
  };
  predictions: Record<string, DiseasePrediction>;
  summary: {
    flagged_conditions_count: number;
    highest_probability_disease: string;
    highest_probability_value: number;
  };
  inference_time_ms: number;
  disclaimer: string;
}

export interface ModelMetadata {
  id: string;
  name: string;
  architecture: string;
  task: string;
  checkpoint_path: string;
  sha256: string;
  input_size: number[];
  labels: string[];
  status: string;
}
```

