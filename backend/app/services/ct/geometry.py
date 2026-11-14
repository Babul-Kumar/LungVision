from typing import List, Tuple
import numpy as np
from backend.app.core.logging import logger


def validate_geometry(
    ct_shape: Tuple[int, int, int],
    mask_shape: Tuple[int, int, int],
    ct_spacing: Tuple[float, float, float],
    mask_spacing: Tuple[float, float, float],
    ct_origin: Tuple[float, float, float],
    mask_origin: Tuple[float, float, float],
    ct_direction: Tuple[float, ...],
    mask_direction: Tuple[float, ...],
    tolerance: float = 1e-3
) -> Tuple[bool, List[str]]:
    """Validates spatial alignment between CT and segmentation mask using floating-point tolerance.
    
    Ported directly from notebook diagnostic Cell 26 & 27:
    Exact floating-point equality (==) can fail on different I/O pipelines, so an absolute
    tolerance (atol=1e-3) is applied to verify sub-millimeter anatomical correspondence.
    """
    mismatches = []

    if ct_shape != mask_shape:
        mismatches.append(f"Array shape mismatch: CT {ct_shape} vs Mask {mask_shape}")

    if not np.allclose(ct_spacing, mask_spacing, atol=tolerance):
        mismatches.append(f"Spacing mismatch: CT {ct_spacing} vs Mask {mask_spacing}")

    if not np.allclose(ct_origin, mask_origin, atol=tolerance):
        mismatches.append(f"Origin mismatch: CT {ct_origin} vs Mask {mask_origin}")

    if not np.allclose(ct_direction, mask_direction, atol=tolerance):
        mismatches.append(f"Direction mismatch: CT {ct_direction} vs Mask {mask_direction}")

    is_valid = len(mismatches) == 0
    return is_valid, mismatches


def physical_to_continuous_index(
    physical_point: Tuple[float, float, float],
    origin: np.ndarray,
    spacing: np.ndarray,
    direction: np.ndarray
) -> np.ndarray:
    """Converts a 3D physical point (coordX, coordY, coordZ in mm) to continuous voxel coordinates (x, y, z).
    
    Formula:
        continuous_xyz = Direction^(-1) @ (physical - origin) / spacing
    """
    pt = np.asarray(physical_point, dtype=float)
    diff = pt - origin
    
    # Solve linear system Direction @ scaled_voxel = diff
    dir_mat = direction.reshape(3, 3)
    scaled_voxel = np.linalg.solve(dir_mat, diff)
    continuous_xyz = scaled_voxel / spacing
    return continuous_xyz


def continuous_index_to_physical(
    continuous_index: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    direction: np.ndarray
) -> np.ndarray:
    """Converts continuous voxel index (x, y, z) back to physical coordinates (X, Y, Z in mm).
    
    Formula:
        physical = origin + Direction @ (continuous_xyz * spacing)
    """
    c_xyz = np.asarray(continuous_index, dtype=float)
    scaled_voxel = c_xyz * spacing
    dir_mat = direction.reshape(3, 3)
    physical = origin + dir_mat @ scaled_voxel
    return physical


def voxel_zyx_to_physical(
    vertices_zyx: np.ndarray,
    origin: np.ndarray,
    spacing_xyz: np.ndarray,
    direction_3x3: np.ndarray,
    step: int = 1,
    pad_offset: int = 0
) -> np.ndarray:
    """Transforms marching cubes vertices from (Z, Y, X) voxel array space into anatomical physical millimeters (X, Y, Z).
    
    Marching cubes in scikit-image produces vertices in array dimension order: (dim0=Z, dim1=Y, dim2=X).
    This function:
      1. Re-indexes vertices from ZYX to XYZ.
      2. Reverses any padding offset applied to ensure watertight surfaces.
      3. Scales by the downsampling step factor.
      4. Multiplies by voxel spacing and the 3x3 direction cosine matrix.
      5. Offsets by the physical origin.
    
    Matches notebook Cell 6, 8, 44 & 45.
    """
    if len(vertices_zyx) == 0:
        return np.empty((0, 3), dtype=float)

    # Convert ZYX vertex coordinates to XYZ
    indices_xyz = (vertices_zyx[:, [2, 1, 0]] - pad_offset) * step

    # Physical millimeters = origin + (direction @ (indices_xyz * spacing).T).T
    scaled = indices_xyz * spacing_xyz
    physical = origin + (direction_3x3 @ scaled.T).T
    return physical


def is_inside_volume(voxel_xyz: Tuple[int, int, int], shape_zyx: Tuple[int, int, int]) -> bool:
    """Verifies whether a discrete voxel coordinate falls within the CT volume boundaries."""
    x, y, z = voxel_xyz
    z_max, y_max, x_max = shape_zyx
    return 0 <= x < x_max and 0 <= y < y_max and 0 <= z < z_max

