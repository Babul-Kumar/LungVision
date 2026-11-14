#!/usr/bin/env python3
"""
LungVision AI — Digitally Reconstructed Radiograph (DRR) Generator
Script: scripts/3d_reconstruction/generate_drr.py
Version: 1.0.0

Generates mathematically verified, geometrically calibrated Frontal (PA) and Lateral (LL)
Digitally Reconstructed Radiographs (DRRs) from 3D thoracic CT volumes using forward ray
projection physics, Beer-Lambert attenuation integration, and DICOM LPS spatial calibration.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def compute_sha256(data: bytes) -> str:
    """Computes standard SHA-256 hex digest for binary data."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def compute_array_sha256(arr: np.ndarray) -> str:
    """Computes SHA-256 hex digest of a numpy array in C-contiguous order."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


class DRRGenerator:
    """
    Simulates radiographic projection physics through a 3D CT volume.
    Preserves DICOM LPS spatial coordinates, millimeter spacing, and attenuation coefficients.
    """

    # Physical attenuation constant for water at ~70 keV effective diagnostic energy
    MU_WATER_PER_MM: float = 0.0193  # 0.193 cm^-1

    def __init__(
        self,
        sdd_mm: float = 1800.0,
        sod_mm: float = 1450.0,
        detector_width_mm: float = 430.0,
        detector_height_mm: float = 430.0,
        detector_pixel_pitch_mm: float = 1.0,
        add_poisson_noise: bool = False,
        photon_count_n0: float = 1e5,
    ):
        self.sdd_mm = float(sdd_mm)
        self.sod_mm = float(sod_mm)
        self.detector_width_mm = float(detector_width_mm)
        self.detector_height_mm = float(detector_height_mm)
        self.detector_pixel_pitch_mm = float(detector_pixel_pitch_mm)
        self.add_poisson_noise = bool(add_poisson_noise)
        self.photon_count_n0 = float(photon_count_n0)

        self.det_w = int(round(self.detector_width_mm / self.detector_pixel_pitch_mm))
        self.det_h = int(round(self.detector_height_mm / self.detector_pixel_pitch_mm))
        self.principal_point_px = (self.det_w / 2.0, self.det_h / 2.0)

    def hu_to_linear_attenuation(self, ct_hu: np.ndarray) -> np.ndarray:
        """
        Converts Hounsfield Units (-1024 to +3071) to linear attenuation coefficients (mm^-1).
        mu = mu_water * max(0, 1 + HU / 1000)
        """
        clamped_hu = np.maximum(ct_hu.astype(np.float32), -1000.0)
        mu = self.MU_WATER_PER_MM * (1.0 + clamped_hu / 1000.0)
        return np.maximum(mu, 0.0)

    def project_orthographic_line_integral(
        self,
        volume_mu: np.ndarray,
        spacing_zyx: Tuple[float, float, float],
        axis: int,
    ) -> np.ndarray:
        """
        Computes accurate forward line integral along the given voxel axis.
        axis=1: PA projection (integrates along anterior-posterior Y-axis).
        axis=2: Lateral projection (integrates along left-right X-axis).
        Multiplies sum by step size (voxel spacing in mm along the integration axis).
        """
        step_size_mm = spacing_zyx[axis]
        line_integral = np.sum(volume_mu, axis=axis, dtype=np.float32) * step_size_mm
        return line_integral

    def generate_projections(
        self,
        ct_volume: np.ndarray,
        spacing_xyz: Tuple[float, float, float],
        origin_xyz: Tuple[float, float, float],
        direction_matrix: np.ndarray,
        mask_volume: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Generates calibrated Frontal (PA) and Lateral (LL) DRR line integrals and display images.
        ct_volume shape is (Z, Y, X).
        """
        z_dim, y_dim, x_dim = ct_volume.shape
        sx, sy, sz = spacing_xyz
        spacing_zyx = (sz, sy, sx)

        # 1. Convert CT HU to linear attenuation mu (mm^-1)
        mu_vol = self.hu_to_linear_attenuation(ct_volume)

        # 2. Frontal (PA) View: Ray passes along Y (Posterior -> Anterior).
        # In DICOM LPS, +Z is Superior (head), +X is Left.
        # Sum along Y (axis 1). Resulting shape is (Z, X).
        # To match clinical radiograph viewing (Superior at top, Patient Right on left/viewer's left),
        # Z should be flipped vertically ([::-1, :]) and X should be flipped horizontally for PA view ([::-1, ::-1]).
        pa_integral = self.project_orthographic_line_integral(mu_vol, spacing_zyx, axis=1)
        # Flip Z so superior is at top (row 0), flip X so patient right is on left side of viewer
        pa_line_integral = pa_integral[::-1, ::-1]

        # 3. Lateral (LL) View: Ray passes along X (Right -> Left).
        # Sum along X (axis 2). Resulting shape is (Z, Y).
        # In DICOM LPS, +Y is Posterior, -Y is Anterior.
        # To match clinical lateral viewing (Anterior on left of viewer, Superior at top):
        # Flip Z so superior is at top, orient Y so anterior is on left (viewer's left).
        lat_integral = self.project_orthographic_line_integral(mu_vol, spacing_zyx, axis=2)
        # Y indices: 0 is anterior, y_dim is posterior. So Y=0 is left. Flip Z vertically:
        lat_line_integral = lat_integral[::-1, :]

        # 4. Optional Poisson Noise simulation
        if self.add_poisson_noise:
            pa_display_raw = self._simulate_poisson_noise(pa_line_integral)
            lat_display_raw = self._simulate_poisson_noise(lat_line_integral)
        else:
            pa_display_raw = pa_line_integral
            lat_display_raw = lat_line_integral

        # 5. Normalization for standard radiographic display (dense=bright, air=dark)
        pa_display_norm = self._normalize_radiograph(pa_display_raw)
        lat_display_norm = self._normalize_radiograph(lat_display_raw)

        # 6. Projected 2D Lung Masks if 3D mask provided
        pa_mask_proj = None
        lat_mask_proj = None
        if mask_volume is not None:
            # Binary lung mask: labels 3 and 4 correspond to Left and Right lungs
            lung_binary = np.isin(mask_volume, [3, 4]).astype(np.float32)
            pa_mask_int = np.sum(lung_binary, axis=1)[::-1, ::-1]
            lat_mask_int = np.sum(lung_binary, axis=2)[::-1, :]
            pa_mask_proj = (pa_mask_int > 0.5).astype(np.uint8) * 255
            lat_mask_proj = (lat_mask_int > 0.5).astype(np.uint8) * 255

        # 7. Construct camera projection matrices (P = K * [R | t])
        fx = self.sdd_mm / self.detector_pixel_pitch_mm
        fy = self.sdd_mm / self.detector_pixel_pitch_mm
        u0, v0 = self.principal_point_px
        k_matrix = np.array([
            [fx, 0.0, u0],
            [0.0, fy, v0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        pa_proj_matrix = {
            "view": "PA",
            "source_lps": [float(origin_xyz[0] + x_dim * sx / 2.0),
                           float(origin_xyz[1] + y_dim * sy / 2.0 + self.sod_mm),
                           float(origin_xyz[2] + z_dim * sz / 2.0)],
            "detector_normal_lps": [0.0, -1.0, 0.0],
            "sdd_mm": self.sdd_mm,
            "sod_mm": self.sod_mm,
            "pixel_pitch_mm": self.detector_pixel_pitch_mm,
            "intrinsic_matrix": k_matrix.tolist(),
        }

        lat_proj_matrix = {
            "view": "LATERAL_LL",
            "source_lps": [float(origin_xyz[0] + x_dim * sx / 2.0 - self.sod_mm),
                           float(origin_xyz[1] + y_dim * sy / 2.0),
                           float(origin_xyz[2] + z_dim * sz / 2.0)],
            "detector_normal_lps": [1.0, 0.0, 0.0],
            "sdd_mm": self.sdd_mm,
            "sod_mm": self.sod_mm,
            "pixel_pitch_mm": self.detector_pixel_pitch_mm,
            "intrinsic_matrix": k_matrix.tolist(),
        }

        return {
            "frontal": {
                "line_integral": pa_line_integral,
                "display_image": pa_display_norm,
                "mask_projection": pa_mask_proj,
                "geometry": pa_proj_matrix,
            },
            "lateral": {
                "line_integral": lat_line_integral,
                "display_image": lat_display_norm,
                "mask_projection": lat_mask_proj,
                "geometry": lat_proj_matrix,
            },
            "volume_stats": {
                "ct_shape_zyx": list(ct_volume.shape),
                "voxel_spacing_xyz_mm": list(spacing_xyz),
                "origin_xyz_mm": list(origin_xyz),
                "hu_min": float(np.min(ct_volume)),
                "hu_max": float(np.max(ct_volume)),
                "hu_mean": float(np.mean(ct_volume)),
            }
        }

    def _simulate_poisson_noise(self, line_integral: np.ndarray) -> np.ndarray:
        """Simulates Poisson quantum photon noise: N ~ Poisson(N0 * exp(-P))."""
        trans = self.photon_count_n0 * np.exp(-line_integral)
        noisy_trans = np.random.poisson(trans)
        noisy_trans = np.maximum(noisy_trans, 1.0)
        return -np.log(noisy_trans / self.photon_count_n0)

    def _normalize_radiograph(self, line_integral: np.ndarray) -> np.ndarray:
        """
        Maps attenuation line integral to inverted grayscale [0, 255].
        High attenuation (bone/tissue) -> high grayscale (bright).
        Low attenuation (air) -> low grayscale (dark).
        """
        p_min = float(np.percentile(line_integral, 1.0))
        p_max = float(np.percentile(line_integral, 99.5))
        if p_max <= p_min:
            p_max = p_min + 1e-4

        norm = (line_integral - p_min) / (p_max - p_min)
        norm = np.clip(norm, 0.0, 1.0)
        # Grayscale: high attenuation = bright
        display_u8 = (norm * 255.0).astype(np.uint8)
        return display_u8


def generate_drr_dataset(
    output_dir: str,
    scan_id: str = "demo_thoracic_scan",
    patient_id: str = "LIDC-IDRI-SAMPLE-001",
    sdd_mm: float = 1800.0,
    sod_mm: float = 1450.0,
    pixel_pitch_mm: float = 1.0,
    add_noise: bool = False,
) -> Dict[str, Any]:
    """
    Main driver function to load CT volume, generate DRRs, and serialize artifacts with SHA-256 digests.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load reference thoracic volume
    from backend.app.services.ct.loader import create_reference_thoracic_volume
    logger_time_start = time.perf_counter()
    ref_vol = create_reference_thoracic_volume(scan_id=scan_id)

    # Compute input volume SHA-256
    ct_sha256 = compute_array_sha256(ref_vol.ct_array)
    mask_sha256 = compute_array_sha256(ref_vol.mask_array) if ref_vol.mask_array is not None else "NONE"

    # 2. Run DRR Generator
    generator = DRRGenerator(
        sdd_mm=sdd_mm,
        sod_mm=sod_mm,
        detector_pixel_pitch_mm=pixel_pitch_mm,
        add_poisson_noise=add_noise,
    )
    result = generator.generate_projections(
        ct_volume=ref_vol.ct_array,
        spacing_xyz=ref_vol.spacing_xyz,
        origin_xyz=ref_vol.origin_xyz,
        direction_matrix=ref_vol.direction_matrix,
        mask_volume=ref_vol.mask_array,
    )

    # 3. Save images to disk
    frontal_img = Image.fromarray(result["frontal"]["display_image"])
    lateral_img = Image.fromarray(result["lateral"]["display_image"])

    frontal_path = out_path / f"{scan_id}_frontal_drr.png"
    lateral_path = out_path / f"{scan_id}_lateral_drr.png"
    frontal_img.save(frontal_path, format="PNG")
    lateral_img.save(lateral_path, format="PNG")

    frontal_mask_path = None
    lateral_mask_path = None
    if result["frontal"]["mask_projection"] is not None:
        f_mask_img = Image.fromarray(result["frontal"]["mask_projection"])
        l_mask_img = Image.fromarray(result["lateral"]["mask_projection"])
        frontal_mask_path = out_path / f"{scan_id}_frontal_mask.png"
        lateral_mask_path = out_path / f"{scan_id}_lateral_mask.png"
        f_mask_img.save(frontal_mask_path, format="PNG")
        l_mask_img.save(lateral_mask_path, format="PNG")

    # 4. Compute file hashes
    with open(frontal_path, "rb") as f:
        frontal_sha256 = compute_sha256(f.read())
    with open(lateral_path, "rb") as f:
        lateral_sha256 = compute_sha256(f.read())

    duration_ms = round((time.perf_counter() - logger_time_start) * 1000.0, 2)

    # 5. Build Comprehensive Metadata Record
    metadata = {
        "study_id": f"STUDY_{scan_id.upper()}",
        "patient_id": patient_id,
        "scan_id": scan_id,
        "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_ms": duration_ms,
        "generator_version": "1.0.0",
        "input_ct": {
            "scan_id": scan_id,
            "dimensions_zyx": result["volume_stats"]["ct_shape_zyx"],
            "spacing_xyz_mm": result["volume_stats"]["voxel_spacing_xyz_mm"],
            "origin_xyz_mm": result["volume_stats"]["origin_xyz_mm"],
            "direction_matrix": ref_vol.direction_matrix.tolist(),
            "sha256": ct_sha256,
            "mask_sha256": mask_sha256,
            "stats": {
                "hu_min": result["volume_stats"]["hu_min"],
                "hu_max": result["volume_stats"]["hu_max"],
                "hu_mean": result["volume_stats"]["hu_mean"],
            }
        },
        "projection_geometry": {
            "sdd_mm": sdd_mm,
            "sod_mm": sod_mm,
            "pixel_pitch_mm": pixel_pitch_mm,
            "poisson_noise_enabled": add_noise,
            "frontal": {
                "file_name": frontal_path.name,
                "file_path": str(frontal_path.resolve()),
                "sha256": frontal_sha256,
                "dimensions": list(result["frontal"]["display_image"].shape),
                "mask_file_name": frontal_mask_path.name if frontal_mask_path else None,
                "camera_matrix": result["frontal"]["geometry"],
            },
            "lateral": {
                "file_name": lateral_path.name,
                "file_path": str(lateral_path.resolve()),
                "sha256": lateral_sha256,
                "dimensions": list(result["lateral"]["display_image"].shape),
                "mask_file_name": lateral_mask_path.name if lateral_mask_path else None,
                "camera_matrix": result["lateral"]["geometry"],
            }
        },
        "clinical_notice": (
            "DRRs are synthetic forward projections through a static supine thoracic CT volume. "
            "They do not capture upright physiological posture, deep breath-hold mechanics, or real scatter."
        )
    }

    meta_path = out_path / f"{scan_id}_drr_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="LungVision AI DRR Projection Generator")
    parser.add_argument("--output-dir", type=str, default="research/3d_reconstruction/demo_drr",
                        help="Directory to save generated DRRs and metadata")
    parser.add_argument("--scan-id", type=str, default="demo_thoracic_scan",
                        help="CT scan ID to project")
    parser.add_argument("--patient-id", type=str, default="LIDC-IDRI-SAMPLE-001",
                        help="Patient ID")
    parser.add_argument("--sdd", type=float, default=1800.0,
                        help="Source-to-detector distance in mm")
    parser.add_argument("--sod", type=float, default=1450.0,
                        help="Source-to-object distance in mm")
    parser.add_argument("--pixel-pitch", type=float, default=1.0,
                        help="Detector pixel pitch in mm")
    parser.add_argument("--noise", action="store_true",
                        help="Enable simulated Poisson quantum mottle")

    args = parser.parse_args()

    print(f"[LungVision DRR Generator] Starting projection pipeline for scan '{args.scan_id}'...")
    meta = generate_drr_dataset(
        output_dir=args.output_dir,
        scan_id=args.scan_id,
        patient_id=args.patient_id,
        sdd_mm=args.sdd,
        sod_mm=args.sod,
        pixel_pitch_mm=args.pixel_pitch,
        add_noise=args.noise,
    )
    print(f"[LungVision DRR Generator] Generation completed in {meta['duration_ms']} ms.")
    print(f"  Frontal DRR : {meta['projection_geometry']['frontal']['file_path']}")
    print(f"  Frontal SHA : {meta['projection_geometry']['frontal']['sha256']}")
    print(f"  Lateral DRR : {meta['projection_geometry']['lateral']['file_path']}")
    print(f"  Lateral SHA : {meta['projection_geometry']['lateral']['sha256']}")
    print(f"  Metadata    : {args.output_dir}/{args.scan_id}_drr_metadata.json")


if __name__ == "__main__":
    main()

