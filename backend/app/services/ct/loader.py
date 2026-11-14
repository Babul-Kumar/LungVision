import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
try:
    import SimpleITK as sitk
except Exception as _e:
    sitk = None
from backend.app.core.errors import FileValidationError
from backend.app.core.logging import logger
from backend.app.services.ct.geometry import validate_geometry


@dataclass
class CTVolume:
    """Represents an in-memory 3D CT volume with geometry, segmentation mask, and annotations."""
    scan_id: str
    filename: str
    ct_array: np.ndarray             # Shape: (Z, Y, X)
    spacing_xyz: Tuple[float, float, float]
    origin_xyz: Tuple[float, float, float]
    direction_matrix: np.ndarray     # Shape: (3, 3)
    mask_array: Optional[np.ndarray] = None  # Shape: (Z, Y, X)
    labels: List[int] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    geometry_valid: bool = True
    geometry_mismatches: List[str] = field(default_factory=list)

    @property
    def shape_zyx(self) -> Tuple[int, int, int]:
        return self.ct_array.shape

    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.spacing_xyz
        return float(sx * sy * sz)

    @property
    def voxel_volume_ml(self) -> float:
        return self.voxel_volume_mm3 / 1000.0


def load_itk_volume(
    ct_image: Any,
    mask_image: Optional[Any] = None,
    scan_id: str = "ct_scan",
    filename: str = "volume.mhd"
) -> CTVolume:
    """Constructs a CTVolume instance from loaded SimpleITK image objects."""
    if sitk is None:
        raise RuntimeError("SimpleITK is required to load ITK image objects but is unavailable on this host.")
    ct_arr = sitk.GetArrayFromImage(ct_image)
    spacing = tuple(float(s) for s in ct_image.GetSpacing())
    origin = tuple(float(o) for o in ct_image.GetOrigin())
    dir_mat = np.array(ct_image.GetDirection(), dtype=float).reshape(3, 3)

    mask_arr = None
    labels = []
    geometry_valid = True
    mismatches = []

    if mask_image is not None:
        mask_arr = sitk.GetArrayFromImage(mask_image)
        mask_spacing = tuple(float(s) for s in mask_image.GetSpacing())
        mask_origin = tuple(float(o) for o in mask_image.GetOrigin())
        mask_direction = tuple(float(d) for d in mask_image.GetDirection())

        geometry_valid, mismatches = validate_geometry(
            ct_shape=ct_arr.shape,
            mask_shape=mask_arr.shape,
            ct_spacing=spacing,
            mask_spacing=mask_spacing,
            ct_origin=origin,
            mask_origin=mask_origin,
            ct_direction=tuple(dir_mat.flatten()),
            mask_direction=mask_direction
        )

        labels = [int(v) for v in np.unique(mask_arr)]

    return CTVolume(
        scan_id=scan_id,
        filename=filename,
        ct_array=ct_arr,
        spacing_xyz=spacing,
        origin_xyz=origin,
        direction_matrix=dir_mat,
        mask_array=mask_arr,
        labels=labels,
        geometry_valid=geometry_valid,
        geometry_mismatches=mismatches
    )


def create_reference_thoracic_volume(scan_id: str = "demo_thoracic_scan") -> CTVolume:
    """Creates a high-fidelity synthetic thoracic reference CT volume with multi-label mask and 12 LUNA16 annotations.
    
    Reproduces the exact series UID and spatial characteristics explored in notebook Cell 2, 4, 6 & 19:
      Series UID: 1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518
      Spacing: (0.742188, 0.742188, 1.25) mm
      Origin: (-193.1, -166.7, -307.25) mm
      Labels: 0 (Background), 3 (Left Lung), 4 (Right Lung), 5 (Trachea/Airway)
    """
    z_dim, y_dim, x_dim = 64, 128, 128
    spacing = (0.742188, 0.742188, 1.25)
    origin = (-193.1, -166.7, -307.25)
    dir_mat = np.eye(3, dtype=float)

    # 1. Generate realistic CT Hounsfield Units (-1000 to +400)
    ct_arr = np.full((z_dim, y_dim, x_dim), fill_value=-1000, dtype=np.int16)  # Air background
    mask_arr = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8)

    z_grid, y_grid, x_grid = np.ogrid[:z_dim, :y_dim, :x_dim]
    cz, cy, cx = z_dim / 2.0, y_dim / 2.0, x_dim / 2.0

    # Thoracic body envelope with cranial tapering and smooth chest wall (-500 to +100 HU)
    body_taper = np.clip(1.0 - 0.22 * np.maximum(0.0, (z_grid - cz) / (z_dim * 0.45)) ** 2, 0.78, 1.0)
    body_3d = (((x_grid - cx) / (x_dim * 0.44 * body_taper)) ** 2 + ((y_grid - cy) / (y_dim * 0.38 * body_taper)) ** 2) < 1.0
    ct_arr[body_3d] = 40  # Soft tissue (+40 HU)

    # Spine (posterior cortical bone, +450 HU)
    spine_3d = (((x_grid - cx) / (x_dim * 0.08)) ** 2 + ((y_grid - (cy + y_dim * 0.22)) / (y_dim * 0.08)) ** 2) < 1.0
    ct_arr[spine_3d & body_3d] = 450  # Cortical bone

    # Rib cage (anterior, lateral, and posterior cortical bone rings)
    rib_ring = (((x_grid - cx) / (x_dim * 0.41 * body_taper)) ** 2 + ((y_grid - cy) / (y_dim * 0.35 * body_taper)) ** 2)
    rib_mask = (rib_ring >= 0.88) & (rib_ring <= 1.02) & (z_grid % 5 <= 2) & body_3d
    ct_arr[rib_mask] = 400

    # Trachea & Main Bronchi (Label 5) (-950 HU)
    # Trachea descends in midline from z=58 to z=34
    trachea_main = (((x_grid - cx) / (x_dim * 0.04)) ** 2 + ((y_grid - (cy - y_dim * 0.06)) / (y_dim * 0.04)) ** 2) < 1.0
    trachea_mask = trachea_main & (z_grid >= 34) & (z_grid <= 58)
    
    # Bronchial bifurcation (carina around z=34 down to z=26)
    left_bronchus = (((x_grid - (cx - (34 - z_grid) * 0.8)) / (x_dim * 0.03)) ** 2 + ((y_grid - cy) / (y_dim * 0.03)) ** 2 < 1.0) & (z_grid >= 26) & (z_grid < 34)
    right_bronchus = (((x_grid - (cx + (34 - z_grid) * 0.8)) / (x_dim * 0.03)) ** 2 + ((y_grid - cy) / (y_dim * 0.03)) ** 2 < 1.0) & (z_grid >= 26) & (z_grid < 34)
    
    airway_mask = trachea_mask | left_bronchus | right_bronchus
    ct_arr[airway_mask] = -950
    mask_arr[airway_mask] = 5

    # Anatomical Lung Parenchyma with Apical Dome and Diaphragmatic Curvature
    # Superior apex dome tapering between z=36 and z=56
    lung_taper = np.clip(1.0 - np.maximum(0.0, (z_grid - 34.0) / 21.0) ** 2, 0.08, 1.0)
    
    # Diaphragmatic dome: rises higher in the central/anterior abdomen (z=18-20) and drops posteriorly/laterally (z=8-10)
    diaphragm_dome = 9.0 + 8.0 * np.exp(-(((x_grid - cx) / 28.0) ** 2 + ((y_grid - (cy + 6.0)) / 28.0) ** 2))
    lung_z_bounds = (z_grid >= diaphragm_dome) & (z_grid <= 55)

    # Left Lung (Label 3) (-850 HU)
    left_rx = (x_dim * 0.18) * np.sqrt(lung_taper)
    left_ry = (y_dim * 0.27) * np.sqrt(lung_taper)
    left_lung_mask = (((x_grid - (cx - x_dim * 0.21)) / left_rx) ** 2 + ((y_grid - cy) / left_ry) ** 2 < 1.0) & lung_z_bounds

    # Cardiac silhouette indentation (heart notch in left mid-lower anterior lung)
    cardiac_notch = (((x_grid - (cx - x_dim * 0.08)) / (x_dim * 0.12)) ** 2 + ((y_grid - (cy - y_dim * 0.06)) / (y_dim * 0.14)) ** 2 < 1.0) & (z_grid >= 14) & (z_grid <= 32)
    left_lung_mask = left_lung_mask & ~cardiac_notch & ~airway_mask

    # Right Lung (Label 4) (-850 HU)
    right_rx = (x_dim * 0.19) * np.sqrt(lung_taper)
    right_ry = (y_dim * 0.27) * np.sqrt(lung_taper)
    right_lung_mask = (((x_grid - (cx + x_dim * 0.21)) / right_rx) ** 2 + ((y_grid - cy) / right_ry) ** 2 < 1.0) & lung_z_bounds & ~airway_mask

    # Assign HU and segmentation labels
    ct_arr[left_lung_mask] = -850
    mask_arr[left_lung_mask] = 3

    ct_arr[right_lung_mask] = -850
    mask_arr[right_lung_mask] = 4

    # Mediastinal soft tissue (+40 HU) for heart and great vessels
    ct_arr[cardiac_notch & body_3d] = 40

    # 2. Populate 12 realistic LUNA16 nodule annotations matching the notebook
    # Scaled to fall within the calibrated thoracic coordinates
    sample_nodules = [
        {"coordX": -145.2, "coordY": -132.4, "coordZ": -270.5, "diameter_mm": 5.4},
        {"coordX": -130.8, "coordY": -110.2, "coordZ": -265.0, "diameter_mm": 8.2},
        {"coordX": -150.1, "coordY": -120.0, "coordZ": -250.0, "diameter_mm": 4.1},
        {"coordX": -110.5, "coordY": -140.2, "coordZ": -280.0, "diameter_mm": 6.8},
        {"coordX": -138.0, "coordY": -105.0, "coordZ": -240.0, "diameter_mm": 7.5},
        {"coordX": -120.4, "coordY": -125.0, "coordZ": -260.0, "diameter_mm": 11.2},
        {"coordX": -135.0, "coordY": -115.0, "coordZ": -275.0, "diameter_mm": 5.0},
        {"coordX": -115.0, "coordY": -130.0, "coordZ": -255.0, "diameter_mm": 6.1},
        {"coordX": -125.0, "coordY": -145.0, "coordZ": -290.0, "diameter_mm": 4.8},
        {"coordX": -142.0, "coordY": -100.0, "coordZ": -245.0, "diameter_mm": 9.4},
        {"coordX": -140.0, "coordY": -135.0, "coordZ": -260.0, "diameter_mm": 3.9},
        {"coordX": -128.0, "coordY": -120.0, "coordZ": -250.0, "diameter_mm": 7.0},
    ]

    # Map physical coords to voxel coords
    annotations = []
    for idx, nodule in enumerate(sample_nodules):
        px, py, pz = nodule["coordX"], nodule["coordY"], nodule["coordZ"]
        vx = int(round((px - origin[0]) / spacing[0]))
        vy = int(round((py - origin[1]) / spacing[1]))
        vz = int(round((pz - origin[2]) / spacing[2]))

        inside = (0 <= vx < x_dim) and (0 <= vy < y_dim) and (0 <= vz < z_dim)
        center_lbl = int(mask_arr[vz, vy, vx]) if inside else None

        annotations.append({
            "nodule_id": f"nodule_{idx + 1:02d}",
            "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518",
            "coordX_mm": px,
            "coordY_mm": py,
            "coordZ_mm": pz,
            "voxel_x": vx,
            "voxel_y": vy,
            "voxel_z": vz,
            "diameter_mm": nodule["diameter_mm"],
            "inside_volume": inside,
            "center_label": center_lbl
        })

    return CTVolume(
        scan_id=scan_id,
        filename="1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518.mhd",
        ct_array=ct_arr,
        spacing_xyz=spacing,
        origin_xyz=origin,
        direction_matrix=dir_mat,
        mask_array=mask_arr,
        labels=[0, 3, 4, 5],
        annotations=annotations,
        geometry_valid=True,
        geometry_mismatches=[]
    )
