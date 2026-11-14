import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from backend.app.core.errors import ResourceNotFoundError
from backend.app.core.logging import logger
from backend.app.schemas.ct import (
    CTAnnotationItem,
    CTAnnotationsResponse,
    CTDimensions,
    CTMeshQualityResponse,
    CTMeshResponse,
    CTMeasurementResponse,
    CTMetadataResponse,
    CTSliceResponse,
    CTUploadResponse,
    WindowPreset
)
from backend.app.services.ct.annotation_service import annotation_service
from backend.app.services.ct.geometry import validate_geometry
from backend.app.services.ct.loader import (
    CTVolume,
    create_reference_thoracic_volume,
    load_itk_volume,
    load_volume_from_payload
)
from backend.app.services.ct.measurement_service import measurement_service
from backend.app.services.ct.mesh_service import mesh_service
from backend.app.services.ct.slice_service import WINDOW_PRESETS, slice_service


class CTService:
    """Unified service facade for 3D CT volume ingestion, geometry, slicing, annotations, and meshes."""

    MAX_STORED_SCANS: int = 20

    def __init__(self):
        self.volumes: Dict[str, CTVolume] = {}
        self._initialize_default_scans()

    def _initialize_default_scans(self) -> None:
        """Initializes the verified reference thoracic scan matching notebook LUNA16 analysis."""
        ref_vol = create_reference_thoracic_volume(scan_id="demo_thoracic_scan")
        self.volumes["demo_thoracic_scan"] = ref_vol
        logger.info("Initialized verified reference CT volume 'demo_thoracic_scan' (64x128x128, 12 LUNA16 nodules).")

    def get_volume(self, scan_id: str) -> CTVolume:
        """Retrieves loaded CT volume instance or raises ResourceNotFoundError."""
        if scan_id not in self.volumes:
            raise ResourceNotFoundError("CT Scan", scan_id)
        return self.volumes[scan_id]

    def register_upload(self, filename: str, content: bytes) -> CTUploadResponse:
        """Registers an uploaded CT file, DICOM archive, or MHD/RAW volume."""
        # Prevent unbounded memory growth with FIFO eviction
        if len(self.volumes) >= self.MAX_STORED_SCANS:
            oldest_key = next((k for k in self.volumes if k != "demo_thoracic_scan"), None)
            if oldest_key:
                del self.volumes[oldest_key]
                logger.info(f"Evicted oldest cached CT volume '{oldest_key}' to prevent memory exhaustion.")

        scan_id = f"ct_{uuid.uuid4().hex[:12]}"

        # Parse uploaded payload (MHD, NIfTI, DICOM archive, or calibrated fallback)
        new_vol = load_volume_from_payload(filename=filename, content=content, scan_id=scan_id)

        self.volumes[scan_id] = new_vol
        logger.info(f"Registered CT scan '{scan_id}' ({filename}, {len(content)} bytes)")

        return CTUploadResponse(
            scan_id=scan_id,
            filename=filename,
            size_bytes=len(content)
        )

    def get_metadata(self, scan_id: str) -> CTMetadataResponse:
        """Retrieves spatial dimensions, spacing, origin, direction matrix, labels, and window presets."""
        vol = self.get_volume(scan_id)
        z, y, x = vol.shape_zyx

        window_dict = {
            name: WindowPreset(center=c, width=w)
            for name, (c, w) in WINDOW_PRESETS.items()
        }

        return CTMetadataResponse(
            scan_id=vol.scan_id,
            filename=vol.filename,
            dimensions=CTDimensions(x=x, y=y, z=z),
            spacing_mm=list(vol.spacing_xyz),
            origin_mm=list(vol.origin_xyz),
            direction_matrix=[list(row) for row in vol.direction_matrix],
            supported_views=["axial", "coronal", "sagittal"],
            window_presets=window_dict,
            anatomical_structures=["lungs", "left_lung", "right_lung", "airways"],
            labels_present=vol.labels,
            annotation_count=len(vol.annotations),
            geometry_validation={
                "status": "PASSED" if vol.geometry_valid else "FAILED",
                "mismatches": vol.geometry_mismatches
            }
        )

    def get_slice(
        self,
        scan_id: str,
        view: str = "axial",
        slice_index: int = 32,
        window: str = "lung",
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        overlay_mask: bool = True,
        show_nodules: bool = True
    ) -> CTSliceResponse:
        """Renders an orthogonal cross-sectional slice with optional mask overlay and nodule markers."""
        vol = self.get_volume(scan_id)
        b64_img, meta = slice_service.render_slice(
            volume=vol,
            view=view,
            slice_idx=slice_index,
            window=window,
            window_center=window_center,
            window_width=window_width,
            overlay_mask=overlay_mask,
            show_nodules=show_nodules
        )

        return CTSliceResponse(
            scan_id=scan_id,
            view=meta["view"],
            slice_index=meta["slice_index"],
            total_slices=meta["total_slices"],
            physical_coordinate_mm=meta.get("physical_coordinate_mm"),
            dimensions_2d=meta["dimensions_2d"],
            image_base64=b64_img,
            window_center=meta["window_center"],
            window_width=meta["window_width"],
            aspect_ratio=meta["aspect_ratio"],
            nodules_on_slice=meta["nodules_on_slice"]
        )

    def get_annotations(self, scan_id: str) -> CTAnnotationsResponse:
        """Retrieves all LUNA16 mapped nodule annotations for the scan."""
        vol = self.get_volume(scan_id)
        items = [
            CTAnnotationItem(
                nodule_id=ann["nodule_id"],
                seriesuid=ann["seriesuid"],
                coordX_mm=ann["coordX_mm"],
                coordY_mm=ann["coordY_mm"],
                coordZ_mm=ann["coordZ_mm"],
                continuous_x=ann.get("continuous_x", float(ann["voxel_x"])),
                continuous_y=ann.get("continuous_y", float(ann["voxel_y"])),
                continuous_z=ann.get("continuous_z", float(ann["voxel_z"])),
                voxel_x=ann["voxel_x"],
                voxel_y=ann["voxel_y"],
                voxel_z=ann["voxel_z"],
                diameter_mm=ann["diameter_mm"],
                inside_volume=ann["inside_volume"],
                center_label=ann.get("center_label")
            )
            for ann in vol.annotations
        ]

        inside_count = sum(1 for a in items if a.inside_volume)

        return CTAnnotationsResponse(
            scan_id=scan_id,
            total_annotations=len(items),
            inside_volume_count=inside_count,
            annotations=items
        )

    def generate_surface_mesh(
        self,
        scan_id: str,
        structure: str = "lungs",
        step: int = 1,
        pad_boundary: bool = True
    ) -> Tuple[str, CTMeshResponse]:
        """Extracts 3D surface mesh vertices/faces and serializes to standard Wavefront OBJ."""
        vol = self.get_volume(scan_id)
        verts, faces, meta = mesh_service.extract_mesh_for_structure(
            volume=vol,
            structure=structure,
            step=step,
            pad_boundary=pad_boundary
        )

        obj_text = mesh_service.format_as_obj(
            vertices=verts,
            faces=faces,
            structure_name=structure,
            annotations=vol.annotations
        )

        resp = CTMeshResponse(
            scan_id=scan_id,
            structure=structure,
            format="obj",
            vertex_count=len(verts),
            face_count=len(faces),
            mesh_url=f"/api/v1/ct/{scan_id}/mesh?structure={structure}"
        )
        return obj_text, resp

    def get_mesh_quality(self, scan_id: str, structure: str = "lungs") -> CTMeshQualityResponse:
        """Evaluates mesh watertightness, winding consistency, and Euler characteristic via Trimesh."""
        vol = self.get_volume(scan_id)
        verts, faces, _ = mesh_service.extract_mesh_for_structure(volume=vol, structure=structure, step=1)

        # Determine voxel count for structure
        mask = vol.mask_array if vol.mask_array is not None else np.zeros((1, 1, 1))
        if structure.lower() in ("left_lung", "lung_3", "label_3"):
            vox_cnt = int(np.count_nonzero(mask == 3))
        elif structure.lower() in ("right_lung", "lung_4", "label_4"):
            vox_cnt = int(np.count_nonzero(mask == 4))
        elif structure.lower() in ("airways", "trachea"):
            vox_cnt = int(np.count_nonzero(mask == 5))
        else:
            vox_cnt = int(np.count_nonzero(np.isin(mask, [3, 4])))

        vox_vol_ml = vox_cnt * vol.voxel_volume_ml
        quality = mesh_service.evaluate_mesh_quality(verts, faces, vox_cnt, vox_vol_ml)

        return CTMeshQualityResponse(
            scan_id=scan_id,
            structure=structure,
            watertight=quality.get("watertight", False),
            winding_consistent=quality.get("winding_consistent", False),
            euler_number=quality.get("euler_number", 0),
            connected_components=quality.get("connected_components", 0),
            degenerate_faces=quality.get("degenerate_faces", 0),
            surface_area_mm2=quality.get("surface_area_mm2", 0.0),
            bounding_box_volume_mm3=quality.get("bounding_box_volume_mm3", 0.0),
            mesh_volume_ml=quality.get("mesh_volume_ml"),
            voxel_volume_ml=round(vox_vol_ml, 2),
            volume_difference_pct=quality.get("volume_difference_pct")
        )

    def get_measurements(self, scan_id: str) -> CTMeasurementResponse:
        """Computes anatomical lung volumes, balance ratios, and label audit."""
        vol = self.get_volume(scan_id)
        metrics = measurement_service.calculate_volume_metrics(vol)

        return CTMeasurementResponse(
            scan_id=scan_id,
            has_segmentation=metrics["has_segmentation"],
            geometry_validation_status=metrics.get("geometry_validation_status", "PASSED"),
            geometry_mismatches=metrics.get("geometry_mismatches", []),
            voxel_spacing_mm=metrics.get("voxel_spacing_mm", list(vol.spacing_xyz)),
            physical_voxel_volume_mm3=metrics.get("physical_voxel_volume_mm3", vol.voxel_volume_mm3),
            lungs=metrics.get("lungs", {}),
            airway=metrics.get("airway", {}),
            label_audit=metrics.get("label_audit", [])
        )


ct_service = CTService()
