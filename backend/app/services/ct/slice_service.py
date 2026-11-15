import base64
import io
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw
import numpy as np
from backend.app.core.logging import logger
from backend.app.services.ct.annotation_service import annotation_service
from backend.app.services.ct.loader import CTVolume


# Hounsfield Unit Window Presets
WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    "lung": (-600.0, 1500.0),        # Window Center, Window Width
    "mediastinum": (50.0, 350.0),
    "bone": (400.0, 1800.0),
    "abdomen": (40.0, 400.0)
}

# Label Color Palette (RGB)
LABEL_COLORS: Dict[int, Tuple[int, int, int]] = {
    2: (225, 29, 72),    # Rose / Crimson - Heart (Cardiac Silhouette)
    3: (65, 105, 225),   # Royal Blue - Left Lung
    4: (255, 140, 0),    # Dark Orange - Right Lung
    5: (50, 205, 50),    # Lime Green - Trachea / Airway
}


class CTSliceService:
    """Renders 2D multi-planar orthogonal slices with HU windowing, segmentation contours, and nodule markers."""

    _slice_cache: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    _MAX_CACHE_SIZE: int = 512

    @classmethod
    def clear_cache(cls) -> None:
        """Flushes the in-memory slice cache."""
        cls._slice_cache.clear()

    @classmethod
    def render_slice(
        cls,
        volume: CTVolume,
        view: str = "axial",
        slice_idx: int = 32,
        window: str = "lung",
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        overlay_mask: bool = True,
        show_nodules: bool = True,
        alpha: float = 0.35
    ) -> Tuple[str, Dict[str, Any]]:
        """Extracts and renders a cross-sectional CT slice image encoded as Base64 PNG.
        
        Matches interactive visualization logic from notebook Cell 5 & 10.
        Employs fast in-memory LRU caching for instant 60 FPS PACS scrubbing.
        """
        view = view.lower()
        cache_key = f"{volume.scan_id}:{view}:{slice_idx}:{window}:{window_center}:{window_width}:{overlay_mask}:{show_nodules}:{alpha}"
        if cache_key in cls._slice_cache:
            return cls._slice_cache[cache_key]

        shape = volume.shape_zyx  # (Z, Y, X)

        # 1. Determine slice bounds and extract 2D raw HU array
        if view == "axial":
            total_slices = shape[0]
            clamped_idx = max(0, min(slice_idx, total_slices - 1))
            ct_slice = volume.ct_array[clamped_idx, :, :]
            mask_slice = volume.mask_array[clamped_idx, :, :] if volume.mask_array is not None else None
            aspect_ratio = volume.spacing_xyz[1] / volume.spacing_xyz[0]  # sy / sx
        elif view == "coronal":
            total_slices = shape[1]
            clamped_idx = max(0, min(slice_idx, total_slices - 1))
            # Coronal: rows are Z (inverted so row 0 is cranial/superior), cols are X
            ct_slice = volume.ct_array[:, clamped_idx, :][::-1, :]
            mask_slice = volume.mask_array[:, clamped_idx, :][::-1, :] if volume.mask_array is not None else None
            aspect_ratio = volume.spacing_xyz[2] / volume.spacing_xyz[0]  # sz / sx
        elif view == "sagittal":
            total_slices = shape[2]
            clamped_idx = max(0, min(slice_idx, total_slices - 1))
            # Sagittal: rows are Z (inverted so row 0 is cranial/superior), cols are Y
            ct_slice = volume.ct_array[:, :, clamped_idx][::-1, :]
            mask_slice = volume.mask_array[:, :, clamped_idx][::-1, :] if volume.mask_array is not None else None
            aspect_ratio = volume.spacing_xyz[2] / volume.spacing_xyz[1]  # sz / sy
        else:
            view = "axial"
            total_slices = shape[0]
            clamped_idx = max(0, min(slice_idx, total_slices - 1))
            ct_slice = volume.ct_array[clamped_idx, :, :]
            mask_slice = volume.mask_array[clamped_idx, :, :] if volume.mask_array is not None else None
            aspect_ratio = 1.0

        # 2. Apply Hounsfield Windowing
        wc, ww = WINDOW_PRESETS.get(window.lower(), WINDOW_PRESETS["lung"])
        if window_center is not None:
            wc = float(window_center)
        if window_width is not None:
            ww = float(window_width)

        hu_min = wc - (ww / 2.0)
        hu_max = wc + (ww / 2.0)

        # Normalize HU values to 8-bit grayscale [0, 255]
        normalized = np.clip((ct_slice - hu_min) / max(hu_max - hu_min, 1.0), 0.0, 1.0)
        gray_8bit = (normalized * 255.0).astype(np.uint8)

        # Convert to 3-channel RGB image for color overlaying
        rgb_image = np.stack([gray_8bit, gray_8bit, gray_8bit], axis=-1)

        # 3. Apply Segmentation Mask Overlay (if present & requested)
        if overlay_mask and mask_slice is not None:
            for label, color in LABEL_COLORS.items():
                mask_indices = (mask_slice == label)
                if np.any(mask_indices):
                    # Alpha blend
                    for c in range(3):
                        rgb_image[mask_indices, c] = (
                            (1.0 - alpha) * rgb_image[mask_indices, c] +
                            alpha * color[c]
                        ).astype(np.uint8)

        # Convert to PIL Image
        # In medical imaging, axial slice origin is often displayed with patient anterior up/down;
        # For coronal/sagittal, row 0 is cranial or caudal; we maintain consistent orientation
        pil_img = Image.fromarray(rgb_image, mode="RGB")

        # 4. Handle non-isotropic voxel aspect ratio if needed
        if abs(aspect_ratio - 1.0) > 0.05:
            new_w = pil_img.width
            new_h = int(round(pil_img.height * aspect_ratio))
            pil_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        # 5. Draw Nodule Markers if requested
        intersecting_nodules = []
        if show_nodules and volume.annotations:
            intersecting_nodules = annotation_service.filter_nodules_for_slice(
                volume.annotations,
                view=view,
                slice_idx=clamped_idx,
                tolerance_slices=2.0
            )

            draw = ImageDraw.Draw(pil_img)
            for nodule in intersecting_nodules:
                nx, ny = nodule["pos_2d"]
                if view in ("coronal", "sagittal"):
                    ny = (shape[0] - 1) - ny
                    nodule["pos_2d"] = (nx, ny)
                # Adjust for aspect ratio
                ny_scaled = ny * aspect_ratio if abs(aspect_ratio - 1.0) > 0.05 else ny
                r = max(4, int(nodule["diameter_mm"] / 2.0))
                
                # Draw red crosshair & circle
                draw.ellipse(
                    [(nx - r, ny_scaled - r), (nx + r, ny_scaled + r)],
                    outline=(255, 50, 50),
                    width=2
                )
                draw.line([(nx - r - 3, ny_scaled), (nx + r + 3, ny_scaled)], fill=(255, 50, 50), width=1)
                draw.line([(nx, ny_scaled - r - 3), (nx, ny_scaled + r + 3)], fill=(255, 50, 50), width=1)

        # 6. Physical coordinate along active orthogonal axis
        if view == "axial":
            phys_coord = float(volume.origin_xyz[2] + clamped_idx * volume.spacing_xyz[2])
        elif view == "coronal":
            phys_coord = float(volume.origin_xyz[1] + clamped_idx * volume.spacing_xyz[1])
        else:
            phys_coord = float(volume.origin_xyz[0] + clamped_idx * volume.spacing_xyz[0])

        # 7. Encode to Base64 PNG
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        meta = {
            "scan_id": volume.scan_id,
            "view": view,
            "slice_index": clamped_idx,
            "total_slices": total_slices,
            "physical_coordinate_mm": round(phys_coord, 2),
            "dimensions_2d": [pil_img.width, pil_img.height],
            "window_center": wc,
            "window_width": ww,
            "aspect_ratio": round(aspect_ratio, 3),
            "nodules_on_slice": intersecting_nodules
        }

        result = (f"data:image/png;base64,{b64_data}", meta)
        if len(cls._slice_cache) >= cls._MAX_CACHE_SIZE:
            # Evict first 64 entries to maintain bounded memory footprint
            keys_to_remove = list(cls._slice_cache.keys())[:64]
            for k in keys_to_remove:
                cls._slice_cache.pop(k, None)
        cls._slice_cache[cache_key] = result
        return result


slice_service = CTSliceService()

