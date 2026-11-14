# LungVision AI — API Documentation & Integration Guide

**API Base URL:** `http://localhost:8000/api/v1`  
**OpenAPI Specification:** `http://localhost:8000/openapi.json`  
**Swagger UI:** `http://localhost:8000/docs`  

---

## 1. 2D Chest X-Ray Endpoints

### 1.1 Multi-Label 6-Disease Classification
**Endpoint:** `POST /api/v1/xray/predict`  
**Description:** Accepts a chest radiograph and predicts probabilities for 6 thoracic pathologies using DenseNet121.  
**Content-Type:** `multipart/form-data`  

#### Request Example (cURL):
```bash
curl -X POST "http://localhost:8000/api/v1/xray/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_chest.png;type=image/png"
```

#### Response Example (200 OK):
```json
{
  "success": true,
  "timestamp": "2026-11-12T03:38:00.123456+00:00",
  "disclaimer": "For research and educational use only. Not a medical diagnosis. Independent verification by a qualified physician is strictly required.",
  "model": {
    "id": "densenet121_nih_6disease",
    "architecture": "DenseNet121",
    "task": "Multi-label chest X-ray disease classification",
    "training_dataset": "NIH ChestX-ray14",
    "checkpoint_epoch": 4,
    "input_dimensions": [224, 224, 3]
  },
  "predictions": {
    "Cardiomegaly": { "probability": 0.0319, "threshold": 0.9502, "experimental_flag": false },
    "Edema": { "probability": 0.0054, "threshold": 0.8770, "experimental_flag": false },
    "Consolidation": { "probability": 0.0554, "threshold": 0.7329, "experimental_flag": false },
    "Pneumonia": { "probability": 0.0009, "threshold": 0.7529, "experimental_flag": false },
    "Atelectasis": { "probability": 0.0848, "threshold": 0.7495, "experimental_flag": false },
    "Pneumothorax": { "probability": 0.0068, "threshold": 0.5825, "experimental_flag": false }
  },
  "summary": {
    "flagged_conditions_count": 0,
    "highest_probability_disease": "Atelectasis",
    "highest_probability_value": 0.0848
  },
  "inference_time_ms": 38.45
}
```

---

### 1.2 Binary Normal vs. Abnormal Classification
**Endpoint:** `POST /api/v1/xray/predict-binary`  
**Description:** Evaluates frontal chest radiograph for general abnormality using ResNet18.  
**Content-Type:** `multipart/form-data`  

#### Request Example (cURL):
```bash
curl -X POST "http://localhost:8000/api/v1/xray/predict-binary" \
     -F "file=@sample_chest.png" \
     -F "model_id=original_resnet18"
```

---

## 2. 3D CT Exploration Endpoints (Integrated from Notebook)

### 2.1 Upload CT Scan
**Endpoint:** `POST /api/v1/ct/upload`  
**Description:** Ingests a 3D CT volume or archive (`.zip`, `.nii.gz`, `.mhd`).  
```bash
curl -X POST "http://localhost:8000/api/v1/ct/upload" -F "file=@scan_archive.zip"
```

### 2.2 Get CT Metadata
**Endpoint:** `GET /api/v1/ct/{scan_id}/metadata`  
**Response (200 OK):**
```json
{
  "success": true,
  "scan_id": "demo_thoracic_scan",
  "filename": "1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518.mhd",
  "dimensions": { "x": 128, "y": 128, "z": 64 },
  "spacing_mm": [0.742188, 0.742188, 1.25],
  "origin_mm": [-193.1, -166.7, -307.25],
  "direction_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
  "supported_views": ["axial", "coronal", "sagittal"],
  "window_presets": {
    "lung": { "center": -600.0, "width": 1500.0 },
    "mediastinum": { "center": 50.0, "width": 350.0 },
    "bone": { "center": 400.0, "width": 1800.0 },
    "abdomen": { "center": 40.0, "width": 400.0 }
  },
  "anatomical_structures": ["lungs", "left_lung", "right_lung", "airways"],
  "labels_present": [0, 3, 4, 5],
  "annotation_count": 12,
  "geometry_validation": { "status": "PASSED", "mismatches": [] }
}
```

### 2.3 Extract Orthogonal Slice
**Endpoint:** `GET /api/v1/ct/{scan_id}/slice`  
**Query Parameters:**
- `view`: `"axial"`, `"coronal"`, `"sagittal"` (default: `"axial"`)
- `slice_index`: integer depth (default: `32`)
- `window`: preset (`"lung"`, `"mediastinum"`, `"bone"`) or custom
- `window_center` & `window_width`: optional numerical overrides
- `overlay_mask`: boolean (default: `true`, renders color contours for left lung [3], right lung [4], airway [5])
- `show_nodules`: boolean (default: `true`, projects intersecting LUNA16 nodule markers)

**Response (200 OK):**
```json
{
  "success": true,
  "scan_id": "demo_thoracic_scan",
  "view": "axial",
  "slice_index": 32,
  "total_slices": 64,
  "dimensions_2d": [128, 128],
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "window_center": -600.0,
  "window_width": 1500.0,
  "aspect_ratio": 1.0,
  "nodules_on_slice": [
    {
      "nodule_id": "nodule_01",
      "pos_2d": [64, 46],
      "distance_slices": 1.0,
      "diameter_mm": 5.4,
      "is_exact_center": false
    }
  ]
}
```

### 2.4 Get LUNA16 Nodule Annotations
**Endpoint:** `GET /api/v1/ct/{scan_id}/annotations`  
**Response (200 OK):**
```json
{
  "success": true,
  "scan_id": "demo_thoracic_scan",
  "total_annotations": 12,
  "inside_volume_count": 12,
  "annotations": [
    {
      "nodule_id": "nodule_01",
      "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518",
      "coordX_mm": -145.2,
      "coordY_mm": -132.4,
      "coordZ_mm": -270.5,
      "continuous_x": 64.538,
      "continuous_y": 46.215,
      "continuous_z": 29.400,
      "voxel_x": 65,
      "voxel_y": 46,
      "voxel_z": 29,
      "diameter_mm": 5.4,
      "inside_volume": true,
      "center_label": 3
    }
  ]
}
```

### 2.5 Stream 3D Surface Mesh (Wavefront OBJ)
**Endpoint:** `GET /api/v1/ct/{scan_id}/mesh`  
**Query Parameters:**
- `structure`: `"lungs"`, `"left_lung"`, `"right_lung"`, `"airways"`
- `step`: marching cubes step factor (1=full resolution, 2=lightweight)
- `pad_boundary`: boolean (applies 1-voxel constant zero padding to ensure watertight surfaces)
- `download`: boolean (sets `Content-Disposition: attachment`)

**Response:** `200 OK` (`text/plain`) returning standard Wavefront OBJ vertices (`v x y z`) in anatomical physical millimeters and triangular faces (`f v1 v2 v3`).

### 2.6 3D Mesh Quality Audit (Trimesh)
**Endpoint:** `GET /api/v1/ct/{scan_id}/mesh/quality`  
**Response (200 OK):**
```json
{
  "success": true,
  "scan_id": "demo_thoracic_scan",
  "structure": "left_lung",
  "watertight": true,
  "winding_consistent": true,
  "euler_number": 2,
  "connected_components": 1,
  "degenerate_faces": 0,
  "surface_area_mm2": 32840.15,
  "bounding_box_volume_mm3": 1420500.0,
  "mesh_volume_ml": 580.45,
  "voxel_volume_ml": 585.12,
  "volume_difference_pct": 0.80
}
```

### 2.7 Quantitative Anatomical Volume Measurements
**Endpoint:** `GET /api/v1/ct/{scan_id}/measurements`  
**Response (200 OK):**
```json
{
  "success": true,
  "scan_id": "demo_thoracic_scan",
  "has_segmentation": true,
  "geometry_validation_status": "PASSED",
  "geometry_mismatches": [],
  "voxel_spacing_mm": [0.742188, 0.742188, 1.25],
  "physical_voxel_volume_mm3": 0.688554,
  "lungs": {
    "left_lung_ml": 585.12,
    "left_lung_voxels": 849780,
    "right_lung_ml": 620.45,
    "right_lung_voxels": 901090,
    "total_lung_volume_ml": 1205.57,
    "total_lung_voxels": 1750870,
    "volume_balance": {
      "absolute_difference_ml": 35.33,
      "relative_difference_pct": 6.04
    }
  },
  "airway": {
    "trachea_volume_ml": 18.42,
    "trachea_voxels": 26750
  },
  "label_audit": [
    { "label": 0, "name": "Background", "voxel_count": 8730994, "volume_ml": 6011.75, "percentage_of_scan": 83.29 },
    { "label": 3, "name": "Left Lung (Label 3)", "voxel_count": 849780, "volume_ml": 585.12, "percentage_of_scan": 8.11 },
    { "label": 4, "name": "Right Lung (Label 4)", "voxel_count": 901090, "volume_ml": 620.45, "percentage_of_scan": 8.59 },
    { "label": 5, "name": "Trachea/Airway (Label 5)", "voxel_count": 26750, "volume_ml": 18.42, "percentage_of_scan": 0.26 }
  ]
}
```

---

## 3. Frontend Integration Types (TypeScript & Three.js)

```typescript
export interface CTMetadata {
  scan_id: string;
  filename: string;
  dimensions: { x: number; y: number; z: number };
  spacing_mm: [number, number, number];
  origin_mm: [number, number, number];
  direction_matrix: number[][];
  supported_views: string[];
  window_presets: Record<string, { center: number; width: number }>;
  labels_present: number[];
  annotation_count: number;
}

export interface CTSliceData {
  scan_id: string;
  view: "axial" | "coronal" | "sagittal";
  slice_index: number;
  total_slices: number;
  dimensions_2d: [number, number];
  image_base64: string;
  window_center: number;
  window_width: number;
  aspect_ratio: number;
  nodules_on_slice: Array<{
    nodule_id: string;
    pos_2d: [number, number];
    distance_slices: number;
    diameter_mm: number;
    is_exact_center: boolean;
  }>;
}

export interface CTNoduleAnnotation {
  nodule_id: string;
  seriesuid: string;
  coordX_mm: number;
  coordY_mm: number;
  coordZ_mm: number;
  voxel_x: number;
  voxel_y: number;
  voxel_z: number;
  diameter_mm: number;
  inside_volume: boolean;
  center_label?: number;
}

export interface CTQuantitativeMeasurements {
  scan_id: string;
  has_segmentation: boolean;
  geometry_validation_status: "PASSED" | "FAILED";
  voxel_spacing_mm: number[];
  physical_voxel_volume_mm3: number;
  lungs: {
    left_lung_ml: number;
    left_lung_voxels: number;
    right_lung_ml: number;
    right_lung_voxels: number;
    total_lung_volume_ml: number;
    total_lung_voxels: number;
    volume_balance: {
      absolute_difference_ml: number;
      relative_difference_pct: number;
    };
  };
  label_audit: Array<{
    label: number;
    name: string;
    voxel_count: number;
    volume_ml: number;
    percentage_of_scan: number;
  }>;
}
```
