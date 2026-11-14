export interface ModeDescriptor {
  mode_index: number;
  explained_variance_pct: number;
  standard_deviation_mm: number;
  description: string;
}

export interface LandmarkDescriptor {
  landmark_index: number;
  name: string;
  vertex_index: number;
}

export interface SSMModelInfo {
  success: boolean;
  loaded: boolean;
  architecture: string;
  num_vertices: number;
  num_faces: number;
  num_modes: number;
  total_explained_variance_pct: number;
  modes: ModeDescriptor[];
  landmarks: LandmarkDescriptor[];
  reference_volume_ml: number;
  disclaimer?: string;
}

export interface SSMReconstructionRequest {
  frontal_aspect_ratio: number;
  lateral_aspect_ratio?: number | null;
  estimated_inspiratory_effort: number;
  silhouette_scale: number;
  n_iterations?: number;
  target_landmarks_2d?: Record<string, [number, number]>;
  include_mesh_geometry?: boolean;
}

export interface ReconstructionMetrics {
  estimated_dice_score: number;
  estimated_assd_mm: number;
  mean_vertex_uncertainty_mm: number;
  max_vertex_uncertainty_mm: number;
  confidence_margin_pct: number;
  landmark_residual_mm?: number;
}

export interface OrganVolumetry {
  total_lung_volume_ml: number;
  left_lung_ml: number;
  right_lung_ml: number;
  symmetry_ratio: number;
}

export interface MeshDescriptor {
  vertex_count: number;
  face_count: number;
  landmarks: Record<string, [number, number, number]>;
  vertices?: number[][];
  faces?: number[][];
}

export interface SSMReconstructionResponse {
  success: boolean;
  status: string;
  optimization_latency_ms: number;
  iterations_computed: number;
  optimal_coefficients: number[];
  metrics: ReconstructionMetrics;
  volumetry: OrganVolumetry;
  surface_mesh: MeshDescriptor;
  disclaimer?: string;
}

export interface ExportOBJRequest {
  shape_coefficients: number[];
}

export interface SSMDeformRequest {
  shape_coefficients: number[];
  structure?: string;
}

export interface SSMDeformMetrics {
  assd_mm: number;
  hd95_mm: number;
  volume_error_pct: number;
  surface_dice: number;
  mahalanobis_distance: number;
  total_modes_used: number;
  explained_variance_pct: number;
}

export interface SSMDeformResponse {
  success: boolean;
  num_vertices: number;
  num_faces: number;
  structure: string;
  coefficients: number[];
  volumetry: OrganVolumetry;
  metrics: SSMDeformMetrics;
  landmarks: Record<string, [number, number, number]>;
  vertices: number[][];
  faces: number[][];
}


