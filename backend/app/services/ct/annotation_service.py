import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from backend.app.core.logging import logger
from backend.app.services.ct.geometry import is_inside_volume, physical_to_continuous_index
from backend.app.services.ct.loader import CTVolume


class CTAnnotationService:
    """Manages LUNA16 nodule annotations and physical-to-voxel mapping."""

    @staticmethod
    def parse_annotations_csv(csv_content: bytes, series_uid: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parses a LUNA16 annotations CSV file and returns list of nodule coordinates."""
        annotations = []
        text_stream = io.StringIO(csv_content.decode("utf-8", errors="ignore"))
        reader = csv.DictReader(text_stream)

        for row in reader:
            uid = row.get("seriesuid", "")
            if series_uid and uid != series_uid:
                continue

            try:
                px = float(row.get("coordX", 0.0))
                py = float(row.get("coordY", 0.0))
                pz = float(row.get("coordZ", 0.0))
                diam = float(row.get("diameter_mm", 0.0))

                annotations.append({
                    "seriesuid": uid,
                    "coordX_mm": px,
                    "coordY_mm": py,
                    "coordZ_mm": pz,
                    "diameter_mm": diam
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed annotation row: {row} ({e})")

        return annotations

    @classmethod
    def map_annotations_to_volume(
        cls,
        volume: CTVolume,
        raw_annotations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Maps physical LUNA16 annotations into continuous and discrete voxel coordinates.
        
        Matches logic in notebook Cell 4.
        """
        mapped = []
        origin = np.asarray(volume.origin_xyz, dtype=float)
        spacing = np.asarray(volume.spacing_xyz, dtype=float)
        direction = volume.direction_matrix

        for idx, ann in enumerate(raw_annotations):
            pt = (ann["coordX_mm"], ann["coordY_mm"], ann["coordZ_mm"])
            cont_xyz = physical_to_continuous_index(pt, origin, spacing, direction)

            vx, vy, vz = [int(round(v)) for v in cont_xyz]
            inside = is_inside_volume((vx, vy, vz), volume.shape_zyx)

            center_lbl = None
            if inside and volume.mask_array is not None:
                center_lbl = int(volume.mask_array[vz, vy, vx])

            mapped.append({
                "nodule_id": f"nodule_{idx + 1:02d}",
                "seriesuid": ann.get("seriesuid", volume.scan_id),
                "coordX_mm": round(pt[0], 2),
                "coordY_mm": round(pt[1], 2),
                "coordZ_mm": round(pt[2], 2),
                "continuous_x": round(float(cont_xyz[0]), 3),
                "continuous_y": round(float(cont_xyz[1]), 3),
                "continuous_z": round(float(cont_xyz[2]), 3),
                "voxel_x": vx,
                "voxel_y": vy,
                "voxel_z": vz,
                "diameter_mm": round(float(ann.get("diameter_mm", 0.0)), 2),
                "inside_volume": inside,
                "center_label": center_lbl
            })

        return mapped

    @staticmethod
    def filter_nodules_for_slice(
        annotations: List[Dict[str, Any]],
        view: str,
        slice_idx: int,
        tolerance_slices: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Finds nodules that intersect or lie within proximity of a specific 2D slice."""
        intersecting = []
        view = view.lower()

        for ann in annotations:
            if not ann.get("inside_volume", False):
                continue

            if view == "axial":
                target_coord = ann["voxel_z"]
                pos_2d = (ann["voxel_x"], ann["voxel_y"])
            elif view == "coronal":
                target_coord = ann["voxel_y"]
                pos_2d = (ann["voxel_x"], ann["voxel_z"])
            elif view == "sagittal":
                target_coord = ann["voxel_x"]
                pos_2d = (ann["voxel_y"], ann["voxel_z"])
            else:
                target_coord = ann["voxel_z"]
                pos_2d = (ann["voxel_x"], ann["voxel_y"])

            dist = abs(target_coord - slice_idx)
            if dist <= tolerance_slices:
                intersecting.append({
                    "nodule_id": ann["nodule_id"],
                    "pos_2d": pos_2d,
                    "distance_slices": dist,
                    "diameter_mm": ann["diameter_mm"],
                    "is_exact_center": dist == 0
                })

        return intersecting


annotation_service = CTAnnotationService()

