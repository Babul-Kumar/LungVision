from typing import Any, Dict, List
import numpy as np
from backend.app.services.ct.loader import CTVolume


class CTMeasurementService:
    """Computes quantitative anatomical lung volume, symmetry, and segmentation label distributions."""

    @classmethod
    def calculate_volume_metrics(cls, volume: CTVolume) -> Dict[str, Any]:
        """Calculates precise physical volumes in mm³ and mL for segmented lung compartments.
        
        Matches logic in notebook Cell 19 & 20.
        """
        if volume.mask_array is None:
            return {
                "scan_id": volume.scan_id,
                "has_segmentation": False,
                "message": "No segmentation mask available for volumetric measurement."
            }

        mask = volume.mask_array
        total_voxels = mask.size
        sx, sy, sz = volume.spacing_xyz
        voxel_vol_mm3 = volume.voxel_volume_mm3
        voxel_vol_ml = volume.voxel_volume_ml

        # 1. Full Label Distribution Audit
        label_audit = []
        unique_labels = np.unique(mask)

        for lbl in unique_labels:
            lbl_int = int(lbl)
            count = int(np.count_nonzero(mask == lbl))
            vol_mm3 = count * voxel_vol_mm3
            vol_ml = vol_mm3 / 1000.0
            pct = (count / total_voxels) * 100.0

            name = "Background"
            if lbl_int == 3:
                name = "Left Lung (Label 3)"
            elif lbl_int == 4:
                name = "Right Lung (Label 4)"
            elif lbl_int == 5:
                name = "Trachea/Airway (Label 5)"

            label_audit.append({
                "label": lbl_int,
                "name": name,
                "voxel_count": count,
                "volume_mm3": round(vol_mm3, 2),
                "volume_ml": round(vol_ml, 2),
                "percentage_of_scan": round(pct, 4)
            })

        # 2. Individual and Total Lung Metrics
        left_voxels = int(np.count_nonzero(mask == 3))
        right_voxels = int(np.count_nonzero(mask == 4))
        airway_voxels = int(np.count_nonzero(mask == 5))

        left_ml = left_voxels * voxel_vol_ml
        right_ml = right_voxels * voxel_vol_ml
        airway_ml = airway_voxels * voxel_vol_ml
        total_lung_voxels = left_voxels + right_voxels
        total_lung_ml = total_lung_voxels * voxel_vol_ml

        # 3. Symmetry & Balance Analysis (Notebook Cell 20)
        abs_diff_ml = abs(left_ml - right_ml)
        min_lung = min(left_ml, right_ml)
        rel_diff_pct = (abs_diff_ml / min_lung * 100.0) if min_lung > 0 else 0.0

        symmetry_ratio = round(left_ml / right_ml, 3) if right_ml > 0 else 1.0

        return {
            "scan_id": volume.scan_id,
            "has_segmentation": True,
            "geometry_validation_status": "PASSED" if volume.geometry_valid else "FAILED",
            "geometry_mismatches": volume.geometry_mismatches,
            "voxel_spacing_mm": list(volume.spacing_xyz),
            "physical_voxel_volume_mm3": round(voxel_vol_mm3, 6),
            "voxel_volume_mm3": round(voxel_vol_mm3, 6),
            "lungs": {
                "left_lung_ml": round(left_ml, 2),
                "left_lung_voxels": left_voxels,
                "right_lung_ml": round(right_ml, 2),
                "right_lung_voxels": right_voxels,
                "total_lung_volume_ml": round(total_lung_ml, 2),
                "total_lung_voxels": total_lung_voxels,
                "trachea_airway_ml": round(airway_ml, 2),
                "lung_symmetry_ratio": symmetry_ratio,
                "volume_balance": {
                    "absolute_difference_ml": round(abs_diff_ml, 2),
                    "relative_difference_pct": round(rel_diff_pct, 2)
                }
            },
            "airway": {
                "trachea_volume_ml": round(airway_ml, 2),
                "trachea_voxels": airway_voxels
            },
            "label_audit": label_audit,
            "disclaimer": (
                "Quantitative measurements are derived from research segmentation masks. "
                "They are provided for structural exploration and do NOT constitute diagnostic clinical lung function testing."
            )
        }


measurement_service = CTMeasurementService()

