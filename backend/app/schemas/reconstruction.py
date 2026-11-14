"""
LungVision AI — Statistical Shape Model 3D Reconstruction Schemas
Schema: backend/app/schemas/reconstruction.py
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.common import StandardResponse


class ModeDescriptor(BaseModel):
    mode_index: int
    explained_variance_pct: float
    standard_deviation_mm: float
    description: str


class LandmarkDescriptor(BaseModel):
    landmark_index: int
    name: str
    vertex_index: int


class SSMModelInfoResponse(StandardResponse):
    """Metadata specifications describing the calibrated SSM Point Distribution Model."""
    loaded: bool
    architecture: str
    num_vertices: int
    num_faces: int
    num_modes: int
    total_explained_variance_pct: float
    modes: List[ModeDescriptor]
    landmarks: List[LandmarkDescriptor]
    reference_volume_ml: float


class SSMReconstructionRequest(BaseModel):
    """Parameters for biplanar radiographic surface reconstruction."""
    frontal_aspect_ratio: float = Field(default=1.0, ge=0.5, le=2.5, description="Height-to-width aspect ratio of frontal radiograph")
    lateral_aspect_ratio: Optional[float] = Field(default=None, ge=0.5, le=2.5, description="Height-to-width aspect ratio of lateral radiograph")
    estimated_inspiratory_effort: float = Field(default=0.0, ge=-2.5, le=2.5, description="Normalized inspiratory effort [-2.5 to +2.5]")
    silhouette_scale: float = Field(default=1.0, ge=0.5, le=2.0, description="Global anatomical scale adjustment factor")
    n_iterations: int = Field(default=40, ge=5, le=100, description="Maximum L-BFGS-B optimization iterations")
    target_landmarks_2d: Optional[Dict[str, List[float]]] = Field(default=None, description="Optional 2D radiographic landmark constraints")
    include_mesh_geometry: bool = Field(default=False, description="Whether to include full vertex and face coordinate arrays")


class ExportOBJRequest(BaseModel):
    """Parameters for exporting a reconstructed 3D OBJ mesh."""
    shape_coefficients: List[float] = Field(default_factory=list, description="SSM PCA deformation coefficients")


class ReconstructionMetrics(BaseModel):
    estimated_dice_score: float
    estimated_assd_mm: float
    mean_vertex_uncertainty_mm: float
    max_vertex_uncertainty_mm: float
    confidence_margin_pct: float = 95.0
    landmark_residual_mm: Optional[float] = None


class OrganVolumetry(BaseModel):
    total_lung_volume_ml: float
    left_lung_ml: float
    right_lung_ml: float
    symmetry_ratio: float


class MeshDescriptor(BaseModel):
    vertex_count: int
    face_count: int
    landmarks: Dict[str, List[float]] = {}
    vertices: Optional[List[List[float]]] = None
    faces: Optional[List[List[int]]] = None


class SSMReconstructionResponse(StandardResponse):
    """Patient-specific 3D lung surface reconstruction result."""
    status: str
    optimization_latency_ms: float
    iterations_computed: int
    optimal_coefficients: List[float]
    metrics: ReconstructionMetrics
    volumetry: OrganVolumetry
    surface_mesh: MeshDescriptor


class SSMDeformRequest(BaseModel):
    """Parameters for interactive SSM shape deformation and structure filtering."""
    shape_coefficients: List[float] = Field(default_factory=list, description="PCA mode coefficients")
    structure: str = Field(default="bilateral", description="'bilateral', 'left_lung', or 'right_lung'")


class SSMDeformMetrics(BaseModel):
    assd_mm: float
    hd95_mm: float
    volume_error_pct: float
    surface_dice: float
    mahalanobis_distance: float
    total_modes_used: int
    explained_variance_pct: float


class SSMDeformResponse(StandardResponse):
    """Results of interactive SSM shape deformation."""
    num_vertices: int
    num_faces: int
    structure: str
    coefficients: List[float]
    volumetry: OrganVolumetry
    metrics: SSMDeformMetrics
    landmarks: Dict[str, List[float]]
    vertices: List[List[float]]
    faces: List[List[int]]

