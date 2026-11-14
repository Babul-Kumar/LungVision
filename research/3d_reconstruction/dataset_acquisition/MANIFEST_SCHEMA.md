# LungVision AI — Paired CXR-CT Dataset Manifest Schema
**Document ID:** `LV-RESEARCH-3D-MAN-001`  
**Version:** 1.0.0  
**Status:** Approved for Implementation  
**Auditor/Author:** LungVision AI Data Engineering & Research Governance Group  
**Framework:** Gravitas Engineering Reliability Runtime  

---

## 1. Scope & Objective

This document defines the formal, machine-verifiable manifest schema for paired 3D CT and 2D radiographic datasets in LungVision AI. Every volume, synthetic Digitally Reconstructed Radiograph (DRR), and ground truth anatomical segmentation ingested into the 3D reconstruction and Statistical Shape Model (SSM) pipeline must be indexed by a record strictly conforming to this schema.

The manifest enforces data integrity, spatial traceability, licensing attribution, and strict **patient-disjoint partitioning** to eliminate data leakage across training, validation, and held-out test sets.

---

## 2. Mandatory Fields Specification (14 Fields)

Every manifest entry must contain the following 14 mandatory attributes. Null or missing values in any mandatory field are strictly disallowed.

| Field # | Field Name | Type | Constraints / Allowed Formats | Description |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `study_id` | `string` | Regex: `^[A-Za-z0-9_\-\.]{3,64}$`, Globally unique | Unique radiographic examination / study identifier. |
| **2** | `patient_id` | `string` | Regex: `^[A-Za-z0-9_\-\.]{3,64}$` | De-identified patient / subject identifier. Multiple studies may share a patient ID. |
| **3** | `ct_path` | `string` | File exists or URI; extension `.mhd`, `.raw`, `.nii`, `.nii.gz`, `.dcm` | Path to the 3D thoracic CT volume. |
| **4** | `frontal_drr_path` | `string` | File exists; extension `.png`, `.tif`, `.tiff`, `.dcm` | Path to the calibrated Frontal (PA/AP) projection. |
| **5** | `lateral_drr_path` | `string` | File exists; extension `.png`, `.tif`, `.tiff`, `.dcm` | Path to the calibrated Lateral (LL/RL) projection. |
| **6** | `segmentation_path`| `string` | File exists or null/empty if unsegmented; `.mhd`, `.nii.gz` | Path to the 3D ground truth multi-organ / lung segmentation mask. |
| **7** | `voxel_spacing` | `list[float]` | Exactly 3 positive floats: `[sx, sy, sz]` where $s_i > 0$ mm | CT voxel physical dimensions in millimeters along $(X, Y, Z)$. |
| **8** | `origin` | `list[float]` | Exactly 3 floats: `[ox, oy, oz]` in mm | Physical coordinate of voxel $(0,0,0)$ in DICOM LPS frame. |
| **9** | `direction` | `list[float]` | Exactly 9 floats: flattened orthonormal $3 \times 3$ matrix | Direction cosine matrix relating voxel grid to anatomical LPS axes. |
| **10**| `projection_geometry`| `object` | Contains `sdd_mm`, `sod_mm`, `pixel_pitch_mm`, `camera_matrix` | Complete geometric camera model and source-detector parameters. |
| **11**| `source_dataset` | `string` | Enum: `LIDC-IDRI`, `LUNA16`, `IN-CXR`, `MIMIC-CXR`, `MIDRC`, `SYNTHETIC_REFERENCE` | Originating clinical database or challenge repository. |
| **12**| `source_license` | `string` | Enum: `CC-BY-3.0`, `PUBLIC_DOMAIN`, `PHYSIONET_DUA`, `MIDRC_DUA`, `INTERNAL_RESEARCH` | Legal governing license or Data Use Agreement under which data was accessed. |
| **13**| `quality_status` | `string` | Enum: `VERIFIED_DIAGNOSTIC`, `ACCEPTABLE_RESEARCH`, `REJECTED_MOTION`, `REJECTED_TRUNCATION`, `REJECTED_IMPLANT` | Quality audit verdict; only `VERIFIED` and `ACCEPTABLE` are admitted to training/test. |
| **14**| `split` | `string` | Enum: `TRAIN`, `VAL`, `TEST` | Dataset partition assignment. **Must be patient-disjoint**. |

---

## 3. Strict Patient-Disjoint Partitioning Invariant

Medical imaging models are prone to memorizing patient-specific idiosyncratic anatomy (e.g., unique rib curvatures, aortic calcification patterns, anatomical anomalies). Partitioning data by study or image rather than by patient produces severe data leakage and artificially inflated performance estimates.

The manifest validator rigorously enforces the following set-theoretic invariance:

$$\mathcal{P}_{\text{TRAIN}} \cap \mathcal{P}_{\text{VAL}} = \emptyset$$
$$\mathcal{P}_{\text{TRAIN}} \cap \mathcal{P}_{\text{TEST}} = \emptyset$$
$$\mathcal{P}_{\text{VAL}} \cap \mathcal{P}_{\text{TEST}} = \emptyset$$

where $\mathcal{P}_S = \{ r.\text{patient\_id} \mid r \in \text{Records}, r.\text{split} = S \}$.

If any patient identifier appears in more than one partition, the manifest is automatically rejected with a fatal validation error.

---

## 4. JSON Schema Definition (Draft-07 Compliant)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "LungVisionPairedDatasetManifest",
  "type": "object",
  "required": ["manifest_version", "created_at", "total_studies", "total_patients", "studies"],
  "properties": {
    "manifest_version": { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$" },
    "created_at": { "type": "string", "format": "date-time" },
    "total_studies": { "type": "integer", "minimum": 1 },
    "total_patients": { "type": "integer", "minimum": 1 },
    "studies": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
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
          "split"
        ],
        "properties": {
          "study_id": { "type": "string", "minLength": 3, "maxLength": 64 },
          "patient_id": { "type": "string", "minLength": 3, "maxLength": 64 },
          "ct_path": { "type": "string", "minLength": 1 },
          "frontal_drr_path": { "type": "string", "minLength": 1 },
          "lateral_drr_path": { "type": "string", "minLength": 1 },
          "segmentation_path": { "type": "string" },
          "voxel_spacing": {
            "type": "array",
            "items": { "type": "number", "exclusiveMinimum": 0.0 },
            "minItems": 3,
            "maxItems": 3
          },
          "origin": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 3,
            "maxItems": 3
          },
          "direction": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 9,
            "maxItems": 9
          },
          "projection_geometry": {
            "type": "object",
            "required": ["sdd_mm", "sod_mm", "pixel_pitch_mm"],
            "properties": {
              "sdd_mm": { "type": "number", "minimum": 500.0, "maximum": 3000.0 },
              "sod_mm": { "type": "number", "minimum": 300.0, "maximum": 2500.0 },
              "pixel_pitch_mm": { "type": "number", "minimum": 0.05, "maximum": 5.0 }
            }
          },
          "source_dataset": {
            "type": "string",
            "enum": ["LIDC-IDRI", "LUNA16", "IN-CXR", "MIMIC-CXR", "MIDRC", "SYNTHETIC_REFERENCE"]
          },
          "source_license": {
            "type": "string",
            "enum": ["CC-BY-3.0", "PUBLIC_DOMAIN", "PHYSIONET_DUA", "MIDRC_DUA", "INTERNAL_RESEARCH"]
          },
          "quality_status": {
            "type": "string",
            "enum": ["VERIFIED_DIAGNOSTIC", "ACCEPTABLE_RESEARCH", "REJECTED_MOTION", "REJECTED_TRUNCATION", "REJECTED_IMPLANT"]
          },
          "split": {
            "type": "string",
            "enum": ["TRAIN", "VAL", "TEST"]
          }
        }
      }
    }
  }
}
```

