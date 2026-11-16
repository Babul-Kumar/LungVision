import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from skimage.measure import marching_cubes
import trimesh
from backend.app.core.logging import logger
from backend.app.services.ct.geometry import voxel_zyx_to_physical
from backend.app.services.ct.loader import CTVolume


class CTMeshService:
    """Extracts, transforms, validates, and exports 3D anatomical surface meshes."""

    @classmethod
    def extract_mesh_for_structure(
        cls,
        volume: CTVolume,
        structure: str = "lungs",
        step: int = 1,
        pad_boundary: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Extracts 3D mesh vertices in physical mm and triangular faces using Marching Cubes.
        
        Implements boundary padding (Cell 44) to prevent open boundary non-manifold edges.
        """
        if volume.mask_array is None:
            raise ValueError(f"CT volume '{volume.scan_id}' has no segmentation mask available for mesh extraction.")

        mask = volume.mask_array
        structure_lower = structure.lower()

        # Select binary region based on requested anatomical structure
        if structure_lower in ("left_lung", "lung_3", "label_3"):
            binary = (mask == 3).astype(np.uint8)
        elif structure_lower in ("right_lung", "lung_4", "label_4"):
            binary = (mask == 4).astype(np.uint8)
        elif structure_lower in ("heart", "cardiac", "label_2"):
            binary = (mask == 2).astype(np.uint8)
        elif structure_lower in ("airways", "trachea", "label_5"):
            binary = (mask == 5).astype(np.uint8)
        elif structure_lower in ("lungs", "both_lungs"):
            binary = np.isin(mask, [3, 4]).astype(np.uint8)
        elif structure_lower in ("all", "all_organs", "lungs_heart"):
            binary = np.isin(mask, [2, 3, 4, 5]).astype(np.uint8)
        else:
            binary = (mask > 0).astype(np.uint8)

        if np.count_nonzero(binary) == 0:
            logger.warning(f"No voxels found for structure '{structure}' in scan {volume.scan_id}")
            return np.empty((0, 3)), np.empty((0, 3)), {"empty": True}

        # Optional 1-voxel padding to ensure watertight surfaces at dataset borders (Notebook Cell 44)
        pad_offset = 1 if pad_boundary else 0
        if pad_boundary:
            padded_binary = np.pad(binary, 1, mode="constant", constant_values=0)
        else:
            padded_binary = binary

        # Marching cubes with step size
        vertices_zyx, faces, normals, _ = marching_cubes(
            padded_binary,
            level=0.5,
            step_size=step,
            allow_degenerate=False,
            method="lewiner"
        )

        # Convert ZYX vertices to anatomical physical coordinates (mm)
        origin = np.asarray(volume.origin_xyz, dtype=float)
        spacing = np.asarray(volume.spacing_xyz, dtype=float)
        dir_mat = volume.direction_matrix

        physical_vertices = voxel_zyx_to_physical(
            vertices_zyx=vertices_zyx,
            origin=origin,
            spacing_xyz=spacing,
            direction_3x3=dir_mat,
            step=step,
            pad_offset=pad_offset
        )

        # Feature-preserving Taubin smoothing to remove marching cubes staircase terracing
        if len(physical_vertices) > 0 and len(faces) > 0:
            try:
                raw_mesh = trimesh.Trimesh(vertices=physical_vertices, faces=faces, process=False)
                if raw_mesh.is_watertight:
                    smoothed_mesh = trimesh.smoothing.filter_taubin(raw_mesh, lamb=0.5, nu=-0.53, iterations=10)
                    physical_vertices = np.asarray(smoothed_mesh.vertices, dtype=float)
            except Exception as e:
                logger.warning(f"Taubin smoothing skipped for mesh: {e}")

        meta = {
            "structure": structure,
            "vertex_count": len(physical_vertices),
            "face_count": len(faces),
            "step": step,
            "padded": pad_boundary
        }

        return physical_vertices, faces, meta

    @classmethod
    def evaluate_mesh_quality(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        voxel_count: int,
        voxel_volume_ml: float
    ) -> Dict[str, Any]:
        """Computes topological and volumetric quality metrics via Trimesh (Notebook Cell 22 & 44)."""
        if len(vertices) == 0 or len(faces) == 0:
            return {"valid": False, "error": "Empty mesh"}

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        is_watertight = bool(mesh.is_watertight)
        mesh_volume_ml = (abs(float(mesh.volume)) / 1000.0) if is_watertight else float("nan")

        vol_diff_pct = float("nan")
        if math.isfinite(mesh_volume_ml) and voxel_volume_ml > 0:
            vol_diff_pct = (abs(mesh_volume_ml - voxel_volume_ml) / voxel_volume_ml) * 100.0

        components = mesh.split(only_watertight=False)
        degenerate = int(np.count_nonzero(mesh.area_faces < 1e-10))

        # Mathematical edge and signed volume audit (Cell 44 & 45)
        signed_vol = float(mesh.volume)
        signed_vol_ml = (signed_vol / 1000.0) if math.isfinite(signed_vol) else None

        edges_unique = mesh.edges_unique
        edges_unique_inverse = mesh.edges_unique_inverse
        edge_counts = np.bincount(edges_unique_inverse, minlength=len(edges_unique))
        boundary_edges = int(np.sum(edge_counts == 1))
        non_manifold_edges = int(np.sum(edge_counts > 2))

        return {
            "watertight": is_watertight,
            "winding_consistent": bool(mesh.is_winding_consistent),
            "is_winding_consistent": bool(mesh.is_winding_consistent),
            "euler_number": int(mesh.euler_number),
            "connected_components": len(components),
            "degenerate_faces": degenerate,
            "degenerate_face_count": degenerate,
            "boundary_edges": boundary_edges,
            "non_manifold_edges": non_manifold_edges,
            "signed_volume_ml": round(signed_vol_ml, 3) if signed_vol_ml is not None else None,
            "signed_volume_positive": bool(signed_vol > 0) if math.isfinite(signed_vol) else False,
            "surface_area_mm2": round(float(mesh.area), 2),
            "bounding_box_volume_mm3": round(float(mesh.bounding_box.volume), 2),
            "mesh_volume_ml": round(mesh_volume_ml, 2) if math.isfinite(mesh_volume_ml) else None,
            "voxel_volume_ml": round(voxel_volume_ml, 2),
            "volume_difference_pct": round(vol_diff_pct, 2) if math.isfinite(vol_diff_pct) else None,
            "volume_discrepancy_percent": round(vol_diff_pct, 2) if math.isfinite(vol_diff_pct) else None
        }

    @classmethod
    def format_as_obj(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        structure_name: str = "lung_surface",
        annotations: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Serializes 3D mesh vertices and triangle faces into standard Wavefront OBJ format."""
        lines = [
            f"# LungVision AI 3D Mesh Export",
            f"# Structure: {structure_name}",
            f"# Vertices: {len(vertices)} | Faces: {len(faces)}",
            f"o {structure_name}"
        ]

        # Vertices (1-based index in OBJ standard)
        for v in vertices:
            lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")

        # Triangular faces
        for f in faces:
            lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")

        # Optional 3D nodule marker annotations as points
        if annotations:
            lines.append(f"o nodule_annotations")
            for ann in annotations:
                if ann.get("inside_volume", False):
                    x = ann.get("coordX_mm", 0.0)
                    y = ann.get("coordY_mm", 0.0)
                    z = ann.get("coordZ_mm", 0.0)
                    lines.append(f"v {x:.4f} {y:.4f} {z:.4f}  # Nodule: {ann.get('nodule_id', '')}")

        return "\n".join(lines)


mesh_service = CTMeshService()

