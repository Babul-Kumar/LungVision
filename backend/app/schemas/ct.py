from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.common import StandardResponse


class CTUploadResponse(StandardResponse):
    """Returned upon successful CT volume registration."""
    scan_id: str
    filename: str
    size_bytes: int
    status: str = "ready"
    message: str = "CT volume uploaded and registered for exploratory visualization"


class WindowPreset(BaseModel):
    center: float
    width: float


class CTDimensions(BaseModel):
    x: int
    y: int
    z: int


class CTMetadataResponse(StandardResponse):
    """Metadata describing volumetric properties of a registered CT scan."""
    scan_id: str
    filename: str
    dimensions: CTDimensions
    spacing_mm: List[float]
    origin_mm: List[float]
    direction_matrix: List[List[float]]
    supported_views: List[str] = ["axial", "coronal", "sagittal"]
    window_presets: Dict[str, WindowPreset] = {
        "lung": WindowPreset(center=-600, width=1500),
        "mediastinum": WindowPreset(center=50, width=350),
        "bone": WindowPreset(center=400, width=1800),
        "abdomen": WindowPreset(center=40, width=400)
    }
    anatomical_structures: List[str] = ["lungs", "left_lung", "right_lung", "heart", "airways"]
    labels_present: List[int] = []
    annotation_count: int = 0
    geometry_validation: Dict[str, Any] = {}


class CTSliceResponse(StandardResponse):
    """Slice extraction response for orthogonal plane viewing."""
    scan_id: str
    view: str
    slice_index: int
    total_slices: int
    physical_coordinate_mm: Optional[float] = None
    dimensions_2d: List[int]
    image_base64: str
    window_center: float
    window_width: float
    aspect_ratio: float
    nodules_on_slice: List[Dict[str, Any]] = []


class CTMeshResponse(StandardResponse):
    """Descriptor for extracted 3D surface mesh."""
    scan_id: str
    structure: str
    format: str = "obj"
    vertex_count: int
    face_count: int
    mesh_url: str


class CTAnnotationItem(BaseModel):
    nodule_id: str
    seriesuid: str
    coordX_mm: float
    coordY_mm: float
    coordZ_mm: float
    continuous_x: float
    continuous_y: float
    continuous_z: float
    voxel_x: int
    voxel_y: int
    voxel_z: int
    diameter_mm: float
    inside_volume: bool
    center_label: Optional[int] = None


class CTAnnotationsResponse(StandardResponse):
    """List of LUNA16 mapped nodule annotations for a CT scan."""
    scan_id: str
    total_annotations: int
    inside_volume_count: int
    annotations: List[CTAnnotationItem]


class CTMeshQualityResponse(StandardResponse):
    """Topological and volumetric mesh quality audit metrics via Trimesh."""
    scan_id: str
    structure: str
    watertight: bool
    winding_consistent: bool
    is_winding_consistent: Optional[bool] = None
    euler_number: int
    connected_components: int
    degenerate_faces: int
    degenerate_face_count: Optional[int] = None
    surface_area_mm2: float
    bounding_box_volume_mm3: float
    mesh_volume_ml: Optional[float] = None
    voxel_volume_ml: float
    volume_difference_pct: Optional[float] = None
    volume_discrepancy_percent: Optional[float] = None


class CTMeasurementResponse(StandardResponse):
    """Full-resolution quantitative anatomical lung volume analysis."""
    scan_id: str
    has_segmentation: bool
    geometry_validation_status: str
    geometry_mismatches: List[str]
    voxel_spacing_mm: List[float]
    physical_voxel_volume_mm3: float
    voxel_volume_mm3: Optional[float] = None
    lungs: Dict[str, Any]
    airway: Dict[str, Any]
    heart: Optional[Dict[str, Any]] = None
    label_audit: List[Dict[str, Any]]


class DRRProjectionResponse(StandardResponse):
    """Calibrated Digitally Reconstructed Radiograph (DRR) projection."""
    scan_id: str
    view: str
    view_name: str
    image_base64: str
    mask_overlay_base64: Optional[str] = None
    dimensions: Dict[str, int]
    physical_fov_mm: Dict[str, float]
    geometry: Dict[str, Any]
    attenuation_stats: Dict[str, float]
    computation_latency_ms: float

