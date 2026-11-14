# LungVision AI — Product Requirements Document (PRD)

**Document Version:** 1.1.0  
**Status:** Approved / Active Governance  
**Author:** Staff Backend Architect & MLOps Lead  
**Governing Skill:** Gravitas Engineering Reliability Runtime  
**Updated:** 2026-11-12 (Post CT-Notebook Integration Audit)  

---

## 1. Product Vision & Problem Statement

### 1.1 Product Vision
LungVision AI is an experimental AI-assisted decision-support research platform for chest imaging analysis. It provides modular, reproducible, and mathematically rigorous tools for multi-label chest pathology screening on 2D chest radiographs (CXRs), binary abnormality detection, and structural 3D computed tomography (CT) volume exploration, nodule mapping, and lung mesh extraction.

### 1.2 Problem Statement
Clinical interpretation of thoracic imaging faces significant cognitive load, high volumes, and potential for fatigue-induced perceptual misses. While deep learning models offer substantial assistance, many existing prototypes suffer from:
1. Lack of reproducibility and ambiguous preprocessing pipelines.
2. Domain shift when moving across hospital networks (e.g., NIH Clinical Center to CheXpert or VinDr).
3. Conflation of exploratory research outputs with certified medical diagnosis.
4. Tightly coupled monolithic scripts lacking production-ready API boundaries suitable for modern web frontends (React + TypeScript + Three.js).

LungVision AI bridges the gap between frozen research artifacts and robust software engineering, delivering a clean, decoupled backend service with transparent calibration thresholds, confidence metrics, and strict scientific disclaimers.

---

## 2. Target Users & Stakeholders

1. **Medical Imaging Researchers:** Investigating cross-domain robustness, calibration drift, and explainability on standard cohorts (NIH ChestX-ray14, Stanford CheXpert, VinDr-CXR) and structural 3D lung volumes (LUNA16).
2. **Clinical AI Engineers & MLOps:** Building reproducible inference pipelines, model registries, and service layers with strict versioning.
3. **Frontend Developers:** Integrating RESTful APIs with rich React/TypeScript interfaces for image visualization, threshold sliders, and Three.js 3D volumetric mesh viewers.
4. **Auditing & Regulatory Compliance Leads:** Verifying that models are explicitly demarcated as research-only and that patient data boundaries and safety constraints are respected.

---

## 3. Product Objectives & Boundaries

### 3.1 Objectives
- **Zero-Retraining Preservation:** Serve verified research models (`DenseNet121` 6-disease and `ResNet18` binary) exactly as trained.
- **Scientific Honesty:** Clearly display calibrated operational thresholds alongside raw sigmoid probabilities without claiming clinical diagnostic efficacy.
- **Decoupled Architecture:** Provide a stateless FastAPI backend with structured JSON schemas and robust error handling.
- **Frontend Readiness:** Supply comprehensive CORS configuration, file upload validation, structured predictions, and OpenAPI documentation for React + TypeScript integration.
- **Comprehensive 3D CT Exploration:** Fully integrate capabilities discovered in `lungvision-interactive-viewer.ipynb` (multi-planar orthogonal slicing, LUNA16 nodule mapping, watertight marching cubes mesh extraction, and quantitative lung volumetry).

### 3.2 Clear Separation: Existing vs. Proposed Capabilities

| Capability Area | Existing Verified Capability | Implemented Backend Capability |
| :--- | :--- | :--- |
| **2D Multi-Label Classification** | DenseNet121 checkpoint (`densenet121_nih_best.pth`, epoch 4, Macro AUC 0.8206) trained on NIH ChestX-ray14 for 6 pathologies. Calibrated thresholds saved. | High-performance FastAPI endpoint `/api/v1/xray/predict` serving DenseNet121 with sub-second CPU/GPU inference. |
| **2D Binary Classification** | ResNet18 checkpoints (Original & Blur-Augmented, epoch 4 & 1) trained on IN-CXR, evaluated on VinDr-CXR (15,000 images). | Multi-model endpoint `/api/v1/xray/predict-binary` allowing model selection (`original_resnet18`, `blur_augmented_resnet18`). |
| **Threshold Calibration** | NIH validation thresholds (`nih_validation_thresholds.csv`) derived for optimal F1 on NIH validation split. | Automated flag generation (`experimental_flag`: true/false) based on locked calibration thresholds. |
| **3D CT Ingestion & Geometry** | 56-cell interactive notebook (`lungvision-interactive-viewer.ipynb`) using SimpleITK, checking spacing, origin, and direction with `atol=1e-3`. | Modular `ct/loader.py` and `ct/geometry.py` with `validate_geometry` and reference thoracic scan. |
| **LUNA16 Nodule Mapping** | Physical-to-voxel coordinate mapping (`TransformPhysicalPointToContinuousIndex`, bounds checking, mask center label). | Service endpoint `/api/v1/ct/{id}/annotations` and automatic nodule projections onto 2D slice planes. |
| **Interactive CT Viewing** | Notebook Matplotlib / ipywidgets 4-panel dashboard with axial, coronal, sagittal views, HU windowing, and contours. | High-speed slice endpoint `/api/v1/ct/{id}/slice` serving Base64 PNGs with alpha-blended segmentation overlays and crosshairs. |
| **3D Mesh Extraction** | Marching cubes with `voxel_zyx_to_physical` coordinate conversion and 1-voxel padding for watertight surfaces. | Dedicated endpoint `/api/v1/ct/{id}/mesh` streaming standard Wavefront OBJ files in physical coordinates. |
| **Mesh Quality Audit** | Trimesh validation (watertightness, winding consistency, Euler number, surface area, mesh volume vs. voxel volume). | Endpoint `/api/v1/ct/{id}/mesh/quality` returning full Trimesh topological metrics. |
| **Quantitative Measurements** | Voxel volume ($s_x s_y s_z$), left/right lung volumes in mL, lung symmetry ratios, and full label distribution audit. | Endpoint `/api/v1/ct/{id}/measurements` computing anatomical volume metrics. |

---

## 4. Functional Requirements

### 4.1 2D Chest Radiograph Classification
- **FR-2D-01 (Multi-Label Inference):** Accept frontal chest radiographs (PNG, JPEG, WebP) and compute probabilities across the 6 supported thoracic diseases (Cardiomegaly, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax).
- **FR-2D-02 (Preprocessing Integrity):** Apply exact transformation: conversion to 3-channel RGB, direct bilinear resize to $224 \times 224$, ImageNet normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
- **FR-2D-03 (Calibration & Decision Gating):** Return continuous probability, frozen validation threshold, and boolean experimental flag.
- **FR-2D-04 (Mandatory Research Disclaimer):** Include research disclaimer in all prediction responses.

### 4.2 3D Computed Tomography (CT) Exploration
- **FR-3D-01 (CT Volume Ingestion):** Ingest CT volumes (`.zip`, `.nii`, `.nii.gz`, `.mhd`, `.dcm`), validate geometry alignment between CT and mask (`atol=1e-3`), and assign unique `scan_id`.
- **FR-3D-02 (Multi-Planar Orthogonal Slicing):** Serve Axial, Coronal, and Sagittal 2D slice images with adjustable Hounsfield Unit windowing (Lung: -600/1500, Mediastinum: 50/350, Bone: 400/1800, Abdomen: 40/400).
- **FR-3D-03 (Segmentation Overlays):** Support alpha-blended color contours on slices for Left Lung (label 3), Right Lung (label 4), and Trachea/Airway (label 5).
- **FR-3D-04 (LUNA16 Nodule Annotation Mapping):** Convert physical coordinates $(coordX, coordY, coordZ\text{ in mm})$ to continuous and discrete voxel indices, verify `inside_volume`, and project markers onto intersecting 2D slices.
- **FR-3D-05 (Watertight 3D Surface Extraction):** Apply 1-voxel constant zero padding, extract isosurfaces with Marching Cubes, transform vertices from voxel ZYX to physical mm via $\mathbf{P} = \mathbf{origin} + \mathbf{D}(\mathbf{v}_{xyz} \odot \mathbf{spacing})$, and stream standard Wavefront OBJ files.
- **FR-3D-06 (Topological Mesh Audit):** Provide Trimesh quality metrics: watertightness, winding consistency, Euler number, degenerate faces, surface area, and mesh vs. voxel volume discrepancy percentage.
- **FR-3D-07 (Quantitative Volume Calculation):** Return exact physical volumes in mL for individual lungs, combined lungs, trachea, and lung symmetry ratios.
- **FR-3D-08 (Strict Diagnostic Boundary):** Explicitly state that nodule annotations are ground-truth benchmark markers, not model-generated cancer predictions, and that meshes/volumes are structural decision-support tools.

---

## 5. Non-Functional Requirements
- **Performance:** 2D inference $< 250\text{ ms}$ on GPU, $< 1200\text{ ms}$ on CPU. Orthogonal slice generation $< 50\text{ ms}$.
- **Memory Efficiency:** Models loaded as singletons on startup. Slices streamed on-demand.
- **Security:** Strict MIME and magic byte validation. Maximum file sizes ($20\text{ MB}$ 2D, $150\text{ MB}$ 3D). Path traversal immunization.
- **Reproducibility:** 100% automated test pass rate with synthetic and sample fixtures.
