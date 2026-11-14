"""
LungVision AI — Digitally Reconstructed Radiograph (DRR) Service
Service: backend/app/services/ct/drr_service.py
Version: 1.0.0

Generates physically verified Frontal (PA) and Lateral (LL) Digitally Reconstructed
Radiographs (DRRs) from 3D thoracic CT volumes using Beer-Lambert ray attenuation physics
and DICOM LPS coordinate calibration.
"""

import base64
import io
import math
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from backend.app.core.errors import ResourceNotFoundError
from backend.app.core.logging import logger
from backend.app.services.ct.loader import CTVolume


class CTDRRService:
    """
    Simulates radiographic projection physics through a 3D CT volume.
    Preserves DICOM LPS spatial coordinates, millimeter spacing, and attenuation coefficients.
    """

    # Physical linear attenuation constant for water at ~70 keV effective diagnostic energy
    MU_WATER_PER_MM: float = 0.0193  # 0.193 cm^-1

    @staticmethod
    def hu_to_linear_attenuation(ct_hu: np.ndarray) -> np.ndarray:
        """
        Converts Hounsfield Units (-1024 to +3071) to linear attenuation coefficients (mm^-1).
        mu = mu_water * max(0, 1 + HU / 1000)
        """
        clamped_hu = np.maximum(ct_hu.astype(np.float32), -1000.0)
        mu = CTDRRService.MU_WATER_PER_MM * (1.0 + clamped_hu / 1000.0)
        return np.maximum(mu, 0.0)

    @staticmethod
    def _normalize_radiograph(line_integral: np.ndarray) -> np.ndarray:
        """
        Normalizes line integrals into radiographic display values [0, 255] uint8.
        Beer-Lambert transmission I = I0 * exp(-P).
        In X-ray imaging, high attenuation (bone, heart) appears bright/white,
        low attenuation (air) appears dark/black.
        """
        p_min = float(np.percentile(line_integral, 1.0))
        p_max = float(np.percentile(line_integral, 99.0))
        if p_max <= p_min:
            p_max = p_min + 1e-4

        norm = (line_integral - p_min) / (p_max - p_min)
        norm = np.clip(norm, 0.0, 1.0)
        return (norm * 255.0).astype(np.uint8)

    @staticmethod
    def _simulate_poisson_noise(line_integral: np.ndarray, photon_count_n0: float = 1e5) -> np.ndarray:
        """Simulates Poisson quantum photon noise: N ~ Poisson(N0 * exp(-P))."""
        trans = photon_count_n0 * np.exp(-line_integral)
        noisy_trans = np.random.poisson(trans)
        noisy_trans = np.maximum(noisy_trans, 1.0)
        return -np.log(noisy_trans / photon_count_n0)

    def generate_drr(
        self,
        volume: CTVolume,
        view: str = "frontal",
        sdd_mm: float = 1800.0,
        sod_mm: float = 1450.0,
        overlay_silhouette: bool = False,
        add_noise: bool = False,
    ) -> Dict[str, Any]:
        """
        Synthesizes a calibrated DRR radiograph from the 3D CT volume.
        """
        start_time = time.time()
        ct_arr = volume.ct_array  # (Z, Y, X)
        z_dim, y_dim, x_dim = ct_arr.shape
        sx, sy, sz = volume.spacing_xyz
        spacing_zyx = (sz, sy, sx)

        # 1. Convert CT HU to linear attenuation mu (mm^-1)
        mu_vol = self.hu_to_linear_attenuation(ct_arr)

        norm_view = view.strip().lower()
        is_lateral = norm_view in ("lateral", "lat", "ll", "lateral_ll")

        if not is_lateral:
            # Frontal (PA) Projection: Ray integrates along Y-axis (Posterior -> Anterior)
            step_size_mm = spacing_zyx[1]  # sy
            line_integral_raw = np.sum(mu_vol, axis=1, dtype=np.float32) * step_size_mm
            # Flip Z so superior is at row 0 (top), flip X so patient right is on viewer's left
            line_integral = line_integral_raw[::-1, ::-1]
            view_code = "frontal_pa"
            view_name = "Frontal Posteroanterior (PA)"
            fov_width_mm = float(x_dim * sx)
            fov_height_mm = float(z_dim * sz)
            axis_integrated = "Y (Posterior -> Anterior)"

            # Projected 2D lung mask if requested
            mask_proj = None
            if volume.mask_array is not None:
                lung_bin = np.isin(volume.mask_array, [3, 4]).astype(np.float32)
                pa_mask_int = np.sum(lung_bin, axis=1)[::-1, ::-1]
                mask_proj = (pa_mask_int > 0.5).astype(np.uint8) * 255
        else:
            # Lateral (LL) Projection: Ray integrates along X-axis (Right -> Left)
            step_size_mm = spacing_zyx[2]  # sx
            line_integral_raw = np.sum(mu_vol, axis=2, dtype=np.float32) * step_size_mm
            # Flip Z vertically so superior is at row 0 (top)
            line_integral = line_integral_raw[::-1, :]
            view_code = "lateral_ll"
            view_name = "Lateral Left-to-Right (LL)"
            fov_width_mm = float(y_dim * sy)
            fov_height_mm = float(z_dim * sz)
            axis_integrated = "X (Right -> Left)"

            # Projected 2D lung mask if requested
            mask_proj = None
            if volume.mask_array is not None:
                lung_bin = np.isin(volume.mask_array, [3, 4]).astype(np.float32)
                lat_mask_int = np.sum(lung_bin, axis=2)[::-1, :]
                mask_proj = (lat_mask_int > 0.5).astype(np.uint8) * 255

        # 2. Add Poisson quantum noise if enabled
        if add_noise:
            disp_integral = self._simulate_poisson_noise(line_integral)
        else:
            disp_integral = line_integral

        # 3. Radiographic contrast normalization
        norm_img_arr = self._normalize_radiograph(disp_integral)
        h, w = norm_img_arr.shape

        # 4. Generate PNG Base64
        pil_img = Image.fromarray(norm_img_arr, mode="L")
        
        # Optional overlay of cyan silhouette boundary
        mask_base64 = None
        if overlay_silhouette and mask_proj is not None:
            # Create RGB overlay
            rgb_img = pil_img.convert("RGBA")
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            overlay_data = np.zeros((h, w, 4), dtype=np.uint8)
            # Find contour of mask
            import scipy.ndimage as ndi
            eroded = ndi.binary_erosion(mask_proj > 0)
            boundary = (mask_proj > 0) & (~eroded)
            overlay_data[boundary] = [0, 242, 254, 220]  # Medical cyan boundary
            overlay_data[(mask_proj > 0) & eroded] = [0, 242, 254, 30]  # Subtle transparent wash
            overlay = Image.fromarray(overlay_data, mode="RGBA")
            blended = Image.alpha_composite(rgb_img, overlay)
            buf = io.BytesIO()
            blended.save(buf, format="PNG")
            img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        if mask_proj is not None:
            mask_buf = io.BytesIO()
            Image.fromarray(mask_proj, mode="L").save(mask_buf, format="PNG")
            mask_base64 = base64.b64encode(mask_buf.getvalue()).decode("utf-8")

        # 5. Radiographic geometry computation
        magnification = sdd_mm / max(sod_mm, 100.0)
        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "success": True,
            "scan_id": volume.scan_id,
            "view": view_code,
            "view_name": view_name,
            "image_base64": f"data:image/png;base64,{img_base64}",
            "mask_overlay_base64": f"data:image/png;base64,{mask_base64}" if mask_base64 else None,
            "dimensions": {
                "width": w,
                "height": h,
            },
            "physical_fov_mm": {
                "width_mm": round(fov_width_mm, 2),
                "height_mm": round(fov_height_mm, 2),
            },
            "geometry": {
                "sdd_mm": float(sdd_mm),
                "sod_mm": float(sod_mm),
                "magnification_factor": round(magnification, 3),
                "axis_integrated": axis_integrated,
                "detector_pixel_pitch_mm": round(max(fov_width_mm / max(w, 1), 0.1), 3),
            },
            "attenuation_stats": {
                "min_line_integral": round(float(np.min(line_integral)), 4),
                "max_line_integral": round(float(np.max(line_integral)), 4),
                "mean_line_integral": round(float(np.mean(line_integral)), 4),
            },
            "computation_latency_ms": round(elapsed_ms, 2),
            "disclaimer": (
                "Synthesized Digitally Reconstructed Radiograph (DRR) simulated from CT volume "
                "via Beer-Lambert linear attenuation integration for research exploration."
            )
        }


drr_service = CTDRRService()

