#!/usr/bin/env python3
"""
LungVision AI — Paired CXR-CT Dataset Manifest Validator
Script: scripts/3d_reconstruction/validate_manifest.py
Version: 1.0.0

Validates paired CXR-CT dataset manifests against the 14-field specification,
enforces mathematical geometry constraints, and verifies zero patient data
leakage across train/val/test splits.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Allowed enumerations
ALLOWED_SOURCE_DATASETS = {
    "LIDC-IDRI", "LUNA16", "IN-CXR", "MIMIC-CXR", "MIDRC", "SYNTHETIC_REFERENCE"
}
ALLOWED_LICENSES = {
    "CC-BY-3.0", "PUBLIC_DOMAIN", "PHYSIONET_DUA", "MIDRC_DUA", "INTERNAL_RESEARCH"
}
ALLOWED_QUALITY_STATUSES = {
    "VERIFIED_DIAGNOSTIC", "ACCEPTABLE_RESEARCH", "REJECTED_MOTION", "REJECTED_TRUNCATION", "REJECTED_IMPLANT"
}
ALLOWED_SPLITS = {"TRAIN", "VAL", "TEST"}

MANDATORY_FIELDS = [
    "study_id",
    "patient_id",
    "ct_path",
    "frontal_drr_path",
    "lateral_drr_path",
    "segmentation_path",
    "voxel_spacing",
    "origin",
    "direction",
    "projection_geometry",
    "source_dataset",
    "source_license",
    "quality_status",
    "split",
]


class ManifestValidationError(Exception):
    """Raised when manifest validation fails."""
    pass


def validate_study_record(idx: int, study: Dict[str, Any], check_files: bool = False, base_dir: Path = None) -> List[str]:
    """Validates a single study record against the 14 mandatory fields and constraints."""
    errors = []
    study_id = study.get("study_id", f"record[{idx}]")

    # 1. Check all 14 mandatory fields exist
    for field in MANDATORY_FIELDS:
        if field not in study or study[field] is None:
            errors.append(f"{study_id}: Missing mandatory field '{field}'")

    if errors:
        return errors  # Cannot do deep type checks if mandatory fields missing

    # 2. Check study_id & patient_id
    if not isinstance(study["study_id"], str) or len(study["study_id"].strip()) < 3:
        errors.append(f"{study_id}: 'study_id' must be a non-empty string of >= 3 chars")
    if not isinstance(study["patient_id"], str) or len(study["patient_id"].strip()) < 3:
        errors.append(f"{study_id}: 'patient_id' must be a non-empty string of >= 3 chars")

    # 3. Check enum fields
    if study["source_dataset"] not in ALLOWED_SOURCE_DATASETS:
        errors.append(f"{study_id}: Invalid 'source_dataset' '{study['source_dataset']}'. Allowed: {ALLOWED_SOURCE_DATASETS}")
    if study["source_license"] not in ALLOWED_LICENSES:
        errors.append(f"{study_id}: Invalid 'source_license' '{study['source_license']}'. Allowed: {ALLOWED_LICENSES}")
    if study["quality_status"] not in ALLOWED_QUALITY_STATUSES:
        errors.append(f"{study_id}: Invalid 'quality_status' '{study['quality_status']}'. Allowed: {ALLOWED_QUALITY_STATUSES}")
    if study["split"] not in ALLOWED_SPLITS:
        errors.append(f"{study_id}: Invalid 'split' '{study['split']}'. Allowed: {ALLOWED_SPLITS}")

    # 4. Check spatial geometry fields
    spacing = study.get("voxel_spacing")
    if not (isinstance(spacing, (list, tuple)) and len(spacing) == 3 and all(isinstance(v, (int, float)) and v > 0 for v in spacing)):
        errors.append(f"{study_id}: 'voxel_spacing' must be a list of 3 positive numbers [sx, sy, sz]")

    origin = study.get("origin")
    if not (isinstance(origin, (list, tuple)) and len(origin) == 3 and all(isinstance(v, (int, float)) for v in origin)):
        errors.append(f"{study_id}: 'origin' must be a list of 3 numbers [ox, oy, oz]")

    direction = study.get("direction")
    if not (isinstance(direction, (list, tuple)) and len(direction) == 9 and all(isinstance(v, (int, float)) for v in direction)):
        errors.append(f"{study_id}: 'direction' must be a flattened 3x3 matrix (list of 9 numbers)")

    # 5. Check projection geometry
    proj = study.get("projection_geometry")
    if not isinstance(proj, dict):
        errors.append(f"{study_id}: 'projection_geometry' must be a dictionary")
    else:
        for p_field in ["sdd_mm", "sod_mm", "pixel_pitch_mm"]:
            if p_field not in proj or not isinstance(proj[p_field], (int, float)) or proj[p_field] <= 0:
                errors.append(f"{study_id}: 'projection_geometry.{p_field}' must be a positive number")

    # 6. Optional file existence verification
    if check_files and base_dir is not None:
        for file_key in ["ct_path", "frontal_drr_path", "lateral_drr_path"]:
            rel_path = study.get(file_key)
            if rel_path:
                full_path = (base_dir / rel_path).resolve()
                if not full_path.exists():
                    errors.append(f"{study_id}: File for '{file_key}' does not exist on disk: {full_path}")

    return errors


def validate_manifest(manifest_path: str, check_files: bool = False) -> Dict[str, Any]:
    """
    Validates full manifest structure, constraints, and patient-disjointness.
    """
    path = Path(manifest_path).resolve()
    if not path.exists():
        raise ManifestValidationError(f"Manifest file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            raise ManifestValidationError(f"Invalid JSON in manifest file {path}: {e}")

    # Top-level structure checks
    required_top = ["manifest_version", "created_at", "total_studies", "total_patients", "studies"]
    for field in required_top:
        if field not in data:
            raise ManifestValidationError(f"Top-level field '{field}' is missing in manifest.")

    studies = data.get("studies", [])
    if not isinstance(studies, list) or len(studies) == 0:
        raise ManifestValidationError("Manifest 'studies' field must be a non-empty list.")

    all_errors = []
    seen_study_ids: Set[str] = set()

    # Track patients per split for strict disjointness check
    split_patients: Dict[str, Set[str]] = {
        "TRAIN": set(),
        "VAL": set(),
        "TEST": set(),
    }
    split_studies: Dict[str, int] = {"TRAIN": 0, "VAL": 0, "TEST": 0}
    dataset_counts: Dict[str, int] = {}
    quality_counts: Dict[str, int] = {}

    base_dir = path.parent

    for idx, study in enumerate(studies):
        study_id = study.get("study_id", f"record[{idx}]")

        # Uniqueness of study_id
        if study_id in seen_study_ids:
            all_errors.append(f"Duplicate study_id '{study_id}' found at index {idx}.")
        seen_study_ids.add(study_id)

        # Field-level validation
        errs = validate_study_record(idx, study, check_files=check_files, base_dir=base_dir)
        all_errors.extend(errs)

        # Split accounting
        split = study.get("split")
        patient_id = study.get("patient_id")
        if split in split_patients and patient_id:
            split_patients[split].add(patient_id)
            split_studies[split] += 1

        source = study.get("source_dataset")
        if source:
            dataset_counts[source] = dataset_counts.get(source, 0) + 1

        quality = study.get("quality_status")
        if quality:
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

    # Strict Patient-Disjoint Verification
    train_val_overlap = split_patients["TRAIN"].intersection(split_patients["VAL"])
    train_test_overlap = split_patients["TRAIN"].intersection(split_patients["TEST"])
    val_test_overlap = split_patients["VAL"].intersection(split_patients["TEST"])

    if train_val_overlap:
        all_errors.append(f"CRITICAL DATA LEAKAGE: Patients appear in both TRAIN and VAL: {sorted(list(train_val_overlap))}")
    if train_test_overlap:
        all_errors.append(f"CRITICAL DATA LEAKAGE: Patients appear in both TRAIN and TEST: {sorted(list(train_test_overlap))}")
    if val_test_overlap:
        all_errors.append(f"CRITICAL DATA LEAKAGE: Patients appear in both VAL and TEST: {sorted(list(val_test_overlap))}")

    all_unique_patients = split_patients["TRAIN"] | split_patients["VAL"] | split_patients["TEST"]

    if all_errors:
        raise ManifestValidationError(
            f"Manifest validation failed with {len(all_errors)} errors:\n" + "\n".join(f"  - {e}" for e in all_errors[:20])
            + (f"\n  ... and {len(all_errors) - 20} more errors." if len(all_errors) > 20 else "")
        )

    return {
        "status": "VALID",
        "manifest_path": str(path),
        "total_studies_recorded": len(studies),
        "total_unique_patients": len(all_unique_patients),
        "split_distribution": {
            "TRAIN": {"studies": split_studies["TRAIN"], "patients": len(split_patients["TRAIN"])},
            "VAL": {"studies": split_studies["VAL"], "patients": len(split_patients["VAL"])},
            "TEST": {"studies": split_studies["TEST"], "patients": len(split_patients["TEST"])},
        },
        "dataset_distribution": dataset_counts,
        "quality_distribution": quality_counts,
        "patient_disjoint_verified": True,
    }


def generate_sample_manifest(output_path: str) -> None:
    """Generates a valid demonstration manifest complying with all 14 mandatory fields."""
    sample = {
        "manifest_version": "1.0.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_studies": 3,
        "total_patients": 3,
        "studies": [
            {
                "study_id": "STUDY_DEMO_THORACIC_001",
                "patient_id": "PATIENT_LUNA_SAMPLE_001",
                "ct_path": "research/3d_reconstruction/demo_drr/demo_thoracic_scan_metadata.json",
                "frontal_drr_path": "research/3d_reconstruction/demo_drr/demo_thoracic_scan_frontal_drr.png",
                "lateral_drr_path": "research/3d_reconstruction/demo_drr/demo_thoracic_scan_lateral_drr.png",
                "segmentation_path": "research/3d_reconstruction/demo_drr/demo_thoracic_scan_frontal_mask.png",
                "voxel_spacing": [0.742188, 0.742188, 1.25],
                "origin": [-193.1, -166.7, -307.25],
                "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "projection_geometry": {
                    "sdd_mm": 1800.0,
                    "sod_mm": 1450.0,
                    "pixel_pitch_mm": 1.0,
                    "camera_matrix": {
                        "fx": 1800.0, "fy": 1800.0, "u0": 215.0, "v0": 215.0
                    }
                },
                "source_dataset": "SYNTHETIC_REFERENCE",
                "source_license": "INTERNAL_RESEARCH",
                "quality_status": "VERIFIED_DIAGNOSTIC",
                "split": "TRAIN"
            },
            {
                "study_id": "STUDY_LIDC_0002",
                "patient_id": "PATIENT_LIDC_0002",
                "ct_path": "data/ct/LIDC-IDRI-0002.nii.gz",
                "frontal_drr_path": "data/drr/LIDC-IDRI-0002_PA.png",
                "lateral_drr_path": "data/drr/LIDC-IDRI-0002_LAT.png",
                "segmentation_path": "data/masks/LIDC-IDRI-0002_lungs.nii.gz",
                "voxel_spacing": [0.703125, 0.703125, 1.25],
                "origin": [-165.0, -140.0, -280.0],
                "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "projection_geometry": {
                    "sdd_mm": 1800.0,
                    "sod_mm": 1450.0,
                    "pixel_pitch_mm": 1.0
                },
                "source_dataset": "LIDC-IDRI",
                "source_license": "CC-BY-3.0",
                "quality_status": "VERIFIED_DIAGNOSTIC",
                "split": "VAL"
            },
            {
                "study_id": "STUDY_LIDC_0003",
                "patient_id": "PATIENT_LIDC_0003",
                "ct_path": "data/ct/LIDC-IDRI-0003.nii.gz",
                "frontal_drr_path": "data/drr/LIDC-IDRI-0003_PA.png",
                "lateral_drr_path": "data/drr/LIDC-IDRI-0003_LAT.png",
                "segmentation_path": "data/masks/LIDC-IDRI-0003_lungs.nii.gz",
                "voxel_spacing": [0.683594, 0.683594, 1.0],
                "origin": [-150.0, -135.0, -295.0],
                "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "projection_geometry": {
                    "sdd_mm": 1800.0,
                    "sod_mm": 1450.0,
                    "pixel_pitch_mm": 1.0
                },
                "source_dataset": "LIDC-IDRI",
                "source_license": "CC-BY-3.0",
                "quality_status": "VERIFIED_DIAGNOSTIC",
                "split": "TEST"
            }
        ]
    }
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)
    print(f"[Manifest Generator] Created verified sample manifest at: {out_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="LungVision AI Dataset Manifest Validator")
    parser.add_argument("manifest_file", type=str, nargs="?", default="research/3d_reconstruction/dataset_acquisition/sample_manifest.json",
                        help="Path to manifest JSON file to validate")
    parser.add_argument("--generate-sample", type=str, default=None,
                        help="Generate a valid demonstration manifest JSON at specified path")
    parser.add_argument("--check-files", action="store_true",
                        help="Verify existence of files referenced in manifest on disk")

    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_manifest(args.generate_sample)
        args.manifest_file = args.generate_sample

    print(f"[LungVision Manifest Validator] Validating: {args.manifest_file}")
    try:
        report = validate_manifest(args.manifest_file, check_files=args.check_files)
        print(f"[LungVision Manifest Validator] SUCCESS: Manifest is valid!")
        print(f"  Total Studies : {report['total_studies_recorded']}")
        print(f"  Total Patients: {report['total_unique_patients']}")
        print(f"  Splits        : {report['split_distribution']}")
        print(f"  Datasets      : {report['dataset_distribution']}")
        print(f"  Patient-Disjoint Check: PASSED (Zero data leakage)")
        sys.exit(0)
    except ManifestValidationError as e:
        print(f"[LungVision Manifest Validator] FAILED:\n{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

