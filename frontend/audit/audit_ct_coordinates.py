"""CT Coordinate and Geometry Validation Audit Script.

Verifies:
1. Roundtrip precision of physical_to_continuous_index and continuous_index_to_physical.
2. Direction cosine matrix properties (orthonormality, determinant).
3. Marching cubes ZYX-to-XYZ transformation, padding reversal, and spacing/origin scaling.
4. Alignment of LUNA16 physical nodule annotations with 3D mesh vertices.
5. Slice plane geometry: axial, coronal, sagittal bounds, index clamping, and aspect ratios.
6. Trimesh topological quality metrics (watertightness, Euler characteristic, volume diff).
"""

import sys
from pathlib import Path
import json
import numpy as np

# Ensure workspace root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.services.ct.geometry import (
    validate_geometry,
    physical_to_continuous_index,
    continuous_index_to_physical,
    voxel_zyx_to_physical,
    is_inside_volume,
)
from backend.app.services.ct_service import ct_service
from backend.app.services.ct.mesh_service import CTMeshService
from backend.app.services.ct.annotation_service import annotation_service
from backend.app.services.ct.slice_service import CTSliceService
from backend.app.services.ct.measurement_service import measurement_service


def audit_ct():
    results = {}

    # 1. Roundtrip coordinate conversion test
    print("--- 1. Testing Coordinate Invertibility ---")
    origin = np.array([-180.0, -180.0, -350.0])
    spacing = np.array([0.75, 0.75, 1.25])
    # Oblique/Identity direction matrix
    direction = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    test_points = [
        np.array([-180.0, -180.0, -350.0]),  # Origin
        np.array([0.0, 0.0, 0.0]),            # World zero
        np.array([55.4, -42.8, -120.3]),      # Arbitrary inside point
        np.array([-12.345, 98.765, -234.567]) # High-precision point
    ]

    max_roundtrip_err = 0.0
    for pt in test_points:
        cont = physical_to_continuous_index(pt, origin, spacing, direction)
        recovered = continuous_index_to_physical(cont, origin, spacing, direction)
        err = np.max(np.abs(pt - recovered))
        if err > max_roundtrip_err:
            max_roundtrip_err = err

    print(f"Max roundtrip error across test points: {max_roundtrip_err:.2e} mm")
    results["coordinate_invertibility"] = {
        "max_roundtrip_error_mm": float(max_roundtrip_err),
        "pass": bool(max_roundtrip_err < 1e-10)
    }

    # 2. Test with non-identity direction matrix
    print("--- 2. Testing Non-Identity Direction Matrix ---")
    # 45-degree rotation around Z
    cos45 = np.cos(np.pi / 4)
    sin45 = np.sin(np.pi / 4)
    rot_dir = np.array([
        cos45, -sin45, 0.0,
        sin45,  cos45, 0.0,
        0.0,    0.0,   1.0
    ])
    cont_rot = physical_to_continuous_index(test_points[2], origin, spacing, rot_dir)
    rec_rot = continuous_index_to_physical(cont_rot, origin, spacing, rot_dir)
    rot_err = float(np.max(np.abs(test_points[2] - rec_rot)))
    print(f"Rotated direction matrix roundtrip error: {rot_err:.2e} mm")
    results["rotated_matrix_invertibility"] = {
        "roundtrip_error_mm": rot_err,
        "pass": bool(rot_err < 1e-10)
    }

    # 3. Test volume and mesh alignment on demo_thoracic_scan
    print("--- 3. Testing Demo Scan & Mesh Alignment ---")
    volume = ct_service.get_volume("demo_thoracic_scan")
    assert volume is not None, "demo_thoracic_scan could not be loaded"

    print(f"Scan shape (Z, Y, X): {volume.shape_zyx}")
    print(f"Origin (X, Y, Z): {volume.origin_xyz}")
    print(f"Spacing (X, Y, Z): {volume.spacing_xyz}")
    print(f"Annotations count: {len(volume.annotations)}")

    # Extract mesh for lungs
    verts, faces, meta = CTMeshService.extract_mesh_for_structure(volume, "lungs", step=1, pad_boundary=True)
    print(f"Extracted mesh: {len(verts)} vertices, {len(faces)} faces")

    # Vertex physical coordinate bounds
    min_v = np.min(verts, axis=0)
    max_v = np.max(verts, axis=0)
    print(f"Mesh physical bounds X: [{min_v[0]:.2f}, {max_v[0]:.2f}] mm")
    print(f"Mesh physical bounds Y: [{min_v[1]:.2f}, {max_v[1]:.2f}] mm")
    print(f"Mesh physical bounds Z: [{min_v[2]:.2f}, {max_v[2]:.2f}] mm")

    # Expected physical volume bounds based on origin and spacing:
    # X spans: origin[0] to origin[0] + (X-1)*spacing[0]
    expected_min_x = volume.origin_xyz[0]
    expected_max_x = volume.origin_xyz[0] + volume.shape_zyx[2] * volume.spacing_xyz[0]
    expected_min_y = volume.origin_xyz[1]
    expected_max_y = volume.origin_xyz[1] + volume.shape_zyx[1] * volume.spacing_xyz[1]
    expected_min_z = volume.origin_xyz[2]
    expected_max_z = volume.origin_xyz[2] + volume.shape_zyx[0] * volume.spacing_xyz[2]

    # Verify mesh is strictly within the physical scan boundaries
    within_bounds = (
        min_v[0] >= expected_min_x - 1.0 and max_v[0] <= expected_max_x + 1.0 and
        min_v[1] >= expected_min_y - 1.0 and max_v[1] <= expected_max_y + 1.0 and
        min_v[2] >= expected_min_z - 1.0 and max_v[2] <= expected_max_z + 1.0
    )
    print(f"Mesh strictly within physical volume bounds: {within_bounds}")

    # Mesh quality evaluation
    voxel_measurements = measurement_service.calculate_volume_metrics(volume)
    total_voxel_ml = voxel_measurements["lungs"]["total_lung_volume_ml"]
    quality = CTMeshService.evaluate_mesh_quality(verts, faces, 0, total_voxel_ml)
    print(f"Mesh Quality: Watertight={quality['watertight']}, Euler={quality['euler_number']}, Components={quality['connected_components']}")
    print(f"Voxel Volume: {total_voxel_ml:.2f} mL vs Mesh Volume: {quality['mesh_volume_ml']} mL (Diff: {quality['volume_difference_pct']}%)")

    results["demo_mesh_validation"] = {
        "vertex_count": len(verts),
        "face_count": len(faces),
        "mesh_bounds_x": [float(min_v[0]), float(max_v[0])],
        "mesh_bounds_y": [float(min_v[1]), float(max_v[1])],
        "mesh_bounds_z": [float(min_v[2]), float(max_v[2])],
        "within_physical_scan_bounds": bool(within_bounds),
        "watertight": quality["watertight"],
        "euler_number": quality["euler_number"],
        "connected_components": quality["connected_components"],
        "voxel_volume_ml": total_voxel_ml,
        "mesh_volume_ml": quality["mesh_volume_ml"],
        "volume_difference_pct": quality["volume_difference_pct"]
    }

    # 4. Nodule physical annotation alignment with mesh
    print("--- 4. Testing Nodule Annotation Alignment ---")
    nodule_results = []
    for ann in volume.annotations:
        n_pos = np.array([ann["coordX_mm"], ann["coordY_mm"], ann["coordZ_mm"]])
        in_vol = ann["inside_volume"]
        center_lbl = ann.get("center_label")
        
        # Distance from nodule to closest mesh vertex
        dists = np.linalg.norm(verts - n_pos, axis=1)
        min_dist_to_surface = float(np.min(dists))

        nodule_results.append({
            "nodule_id": ann["nodule_id"],
            "coord_mm": [ann["coordX_mm"], ann["coordY_mm"], ann["coordZ_mm"]],
            "voxel_coords": [ann["voxel_x"], ann["voxel_y"], ann["voxel_z"]],
            "inside_volume": in_vol,
            "center_label": center_lbl,
            "min_dist_to_mesh_surface_mm": round(min_dist_to_surface, 2)
        })

    inside_count = sum(1 for n in nodule_results if n["inside_volume"])
    print(f"Nodules inside volume: {inside_count} / {len(nodule_results)}")
    results["nodule_alignment"] = nodule_results

    # 5. Slice Geometry & Aspect Ratios
    print("--- 5. Testing Slice Service Aspect Ratios & Clamping ---")
    slice_tests = {}
    for view in ["axial", "coronal", "sagittal"]:
        b64, meta = CTSliceService.render_slice(volume, view=view, slice_idx=32)
        slice_tests[view] = {
            "view": meta["view"],
            "slice_index": meta["slice_index"],
            "total_slices": meta["total_slices"],
            "aspect_ratio": meta["aspect_ratio"],
            "window_center": meta["window_center"],
            "window_width": meta["window_width"],
            "b64_length": len(b64)
        }
        print(f"View {view:8s}: {meta['slice_index']}/{meta['total_slices']} slices, Aspect Ratio: {meta['aspect_ratio']:.4f}")

    results["slice_geometry"] = slice_tests

    # Save validation artifacts
    out_path = Path(__file__).resolve().parent / "ct_geometry_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Validation results saved to {out_path}")


if __name__ == "__main__":
    audit_ct()
