/**
 * 3D CT Exploration schemas matching backend/app/schemas/ct.py
 */

export interface CTDimensions {
  x: number;
  y: number;
  z: number;
}

export interface WindowPreset {
  center: number;
  width: number;
}

export interface GeometryValidationStatus {
  status: 'PASSED' | 'FAILED';
  mismatches: string[];
}

export interface CTMetadataResponse {
  scan_id: string;
  filename: string;
  dimensions: CTDimensions;
  spacing_mm: [number, number, number];
  origin_mm: [number, number, number];
  direction_matrix: number[][];
  supported_views: string[];
  window_presets: Record<string, WindowPreset>;
  anatomical_structures: string[];
  labels_present: number[];
  annotation_count: number;
  geometry_validation: GeometryValidationStatus;
  disclaimer: string;
}

export interface NoduleOnSlice {
  nodule_id: string;
  coordX_mm: number;
  coordY_mm: number;
  coordZ_mm: number;
  voxel_x: number;
  voxel_y: number;
  voxel_z: number;
  diameter_mm: number;
  distance_to_plane_mm: number;
}

export interface CTSliceResponse {
  scan_id: string;
  view: 'axial' | 'coronal' | 'sagittal';
  slice_index: number;
  max_slice_index: number;
  physical_coordinate_mm: number;
  window_applied: string;
  has_segmentation_overlay: boolean;
  image_base64: string;
  nodules_on_slice: NoduleOnSlice[];
}

export interface CTAnnotationItem {
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
  center_label?: number | null;
}

export interface CTAnnotationsResponse {
  success: boolean;
  scan_id: string;
  seriesuid?: string;
  total_annotations: number;
  inside_volume_count: number;
  annotations: CTAnnotationItem[];
  disclaimer: string;
}

export interface CTMeshQualityResponse {
  scan_id: string;
  structure: string;
  watertight: boolean;
  euler_number: number;
  winding_consistent?: boolean;
  is_winding_consistent?: boolean;
  vertex_count: number;
  face_count: number;
  degenerate_faces?: number;
  degenerate_face_count?: number;
  surface_area_mm2: number;
  mesh_volume_ml?: number | null;
  voxel_volume_ml?: number | null;
  volume_difference_pct?: number | null;
  volume_discrepancy_percent?: number | null;
  disclaimer: string;
}

export interface VolumeBalance {
  absolute_difference_ml: number;
  relative_difference_pct: number;
}

export interface LungVolumes {
  left_lung_ml: number;
  left_lung_voxels?: number;
  right_lung_ml: number;
  right_lung_voxels?: number;
  total_lung_volume_ml: number;
  total_lung_voxels?: number;
  volume_balance?: VolumeBalance;
  trachea_airway_ml?: number;
  lung_symmetry_ratio?: number;
}

export interface AirwayVolumes {
  trachea_volume_ml: number;
  trachea_voxels: number;
}

export interface LabelAuditItem {
  label: number;
  name: string;
  voxel_count: number;
  volume_mm3: number;
  volume_ml: number;
  percentage_of_scan: number;
}

export interface CTMeasurementResponse {
  scan_id: string;
  has_segmentation: boolean;
  geometry_validation_status: string;
  geometry_mismatches?: string[];
  voxel_spacing_mm?: number[];
  physical_voxel_volume_mm3?: number;
  voxel_volume_mm3?: number;
  lungs: LungVolumes;
  airway?: AirwayVolumes;
  label_audit: LabelAuditItem[];
  disclaimer: string;
}

export interface CTUploadResponse {
  success: boolean;
  scan_id: string;
  filename: string;
  size_bytes: number;
  status: string;
  message: string;
}

export interface DRRProjectionResponse {
  success: boolean;
  scan_id: string;
  view: string;
  view_name: string;
  image_base64: string;
  mask_overlay_base64?: string | null;
  dimensions: {
    width: number;
    height: number;
  };
  physical_fov_mm: {
    width_mm: number;
    height_mm: number;
  };
  geometry: {
    sdd_mm: number;
    sod_mm: number;
    magnification_factor: number;
    axis_integrated: string;
    detector_pixel_pitch_mm: number;
  };
  attenuation_stats: {
    min_line_integral: number;
    max_line_integral: number;
    mean_line_integral: number;
  };
  computation_latency_ms: number;
  disclaimer?: string;
}


