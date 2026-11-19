# LungVision AI — Comprehensive Repository Forensic Audit

**Document Version:** 1.0.0  
**Audit Date:** November 14, 2026  
**Auditor:** Principal Medical AI Product Architect & Senior Medical Imaging Scientist  
**Runtime:** Gravitas Engineering Reliability Runtime  
**Governance Scope:** Read-Only Model Weights, Preserved Historical Research, Verified Checksums  

---

## 1. Executive Summary

A comprehensive forensic audit of the `LungVision AI` repository (`C:\Users\babul\Desktop\computer vision`) was conducted to establish an authoritative inventory of all code, neural network models, datasets, volumetric pipelines, and historical research artifacts.

The audit verified **15,289 files across 1,507 directories** totaling **2.49 GB** on disk. All historical neural network weights, evaluation metrics, and validation records remain intact. No models were retrained, no checkpoints were modified or overwritten, and no datasets were downloaded or deleted during this audit.

### Key Audit Highlights:
- **Trained Neural Models:** 3 primary PyTorch models are registered and verified:
  1. `DenseNet121 NIH 6-Disease Classifier`: 6,960,006 parameters (6.96M), SHA-256 `8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af`.
  2. `Original ResNet18 Baseline Classifier`: 11,177,025 parameters (11.18M), SHA-256 `a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f`.
  3. `Blur-Augmented ResNet18 Robustness Classifier`: 11,177,025 parameters (11.18M), SHA-256 `7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24`.
- **Dataset Provenance:** 5 distinct cohorts are audited:
  - NIH ChestX-ray14 (112,120 radiographs; 25,596 test cases).
  - IN-CXR derivative (12,600 radiographs; 1,260 validation, 1,260 test).
  - Stanford CheXpert (200 clinical validation cases; 13 disease labels).
  - VinDr-CXR (15,000 real hospital cases; external validation cohort).
  - LIDC-IDRI / LUNA16 (888 thoracic CT scans; precomputed segmentation masks).
- **Automated Verification:** 
  - Backend automated test suite: **29 / 29 passed (100%)** in 36.28s.
  - Frontend Vitest suite: **22 / 22 passed (100%)** across 10 test suites in 6.10s.

---

## 2. Repository Architecture & Directory Mapping

The repository is organized into distinct full-stack modules separating backend services, frontend application components, research evaluation archives, and model registries:

```
C:\Users\babul\Desktop\computer vision\
├── backend/                               # FastAPI Python REST Application
│   ├── app/
│   │   ├── api/v1/                        # Endpoint routers (xray, ct, models, health)
│   │   ├── core/                          # Logging, error handlers, security
│   │   ├── schemas/                       # Pydantic request/response data contracts
│   │   └── services/                      # Business logic & scientific pipelines
│   │       ├── ct/                        # CT loader, geometry, MPR slicing, mesh, nodules
│   │       ├── model_loader.py            # PyTorch checkpoint loader with fallback
│   │       ├── preprocessing.py           # Bilinear ImageNet normalization
│   │       ├── registry_service.py        # Model registry & metrics provider
│   │       └── xray_service.py            # 2D inference & calibrated thresholding
│   └── tests/                             # 29 Pytest unit, integration & security tests
├── frontend/                              # Vite + React 18 + TypeScript Workstation
│   ├── src/
│   │   ├── components/                    # Common UI components (Card, Badge, Button)
│   │   ├── features/                      # Domain feature modules
│   │   │   ├── ct/                        # 3D Three.js mesh canvas & MPR slice viewer
│   │   │   ├── dashboard/                 # Intelligence station overview
│   │   │   ├── models/                    # Audited model registry & checkpoint inspector
│   │   │   ├── research/                  # Multi-cohort benchmark evidence & charts
│   │   │   └── xray/                      # 2D single & dual-view CXR screening
│   │   ├── services/api/                  # Axios HTTP client wrappers
│   │   └── store/                         # Zustand global state (health, history, session)
│   └── tests/                             # 22 Vitest unit & integration tests
├── lungVision/                            # NIH & CheXpert evaluation CSVs & artifacts
├── LungVision_AI_Final_Research_Bundle/   # Frozen models, VinDr benchmarks & reports
├── LungVision_AI_Model_Registry/          # Model registry configurations & metadata
├── lungvision_disease_classification/     # DenseNet121 primary checkpoint & training code
├── docs/                                  # Technical documentation & audit inventories
│   └── forensic_inventory/               # 20 machine-readable inventory CSVs & reports
└── scripts/                               # Utility scripts (DRR generator, verification)
```

---

## 3. Audited Neural Model Inventory

| Model ID | Architecture | Task | Parameters | Checkpoint Path | Verified SHA-256 Checksum | Epoch | Training Dataset |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `densenet121_nih_6disease` | DenseNet121 | Multi-Label (6 thoracic diseases) | 6,960,006 | `lungvision_disease_classification/checkpoints/densenet121_nih_best.pth` | `8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af` | 4 | NIH ChestX-ray14 |
| `original_resnet18` | ResNet18 | Binary (Normal vs Abnormal) | 11,177,025 | `LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth` | `a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f` | 4 | IN-CXR (12,600 images) |
| `blur_augmented_resnet18` | ResNet18 | Binary (Normal vs Abnormal) | 11,177,025 | `LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth` | `7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24` | 1 | IN-CXR (Blur + Rotation augmented) |

### Parameter Count Reconciliation:
- Standard ImageNet DenseNet121 with 1,000-class classifier head:
  $$\text{Params} = 6,953,856 + (1024 \times 1000 + 1000) = 7,978,856 \approx 7.98\text{M}$$
- Fine-tuned 6-disease DenseNet121 with 6-class classifier head:
  $$\text{Params} = 6,953,856 + (1024 \times 6 + 6) = 6,960,006 \approx 6.96\text{M}$$
  *(Commonly reported in rounded shorthand as 7.0M parameters).*

---

## 4. Medical Imaging Pipeline Capabilities

### 4.1 2D Chest Radiograph Multi-Label Screening
- **Input:** Single frontal PA/AP radiograph (PNG, JPEG, WebP, DICOM-derived).
- **Preprocessing:** Bilinear interpolation to $224 \times 224 \times 3$, normalization to $[0.0, 1.0]$, ImageNet standardization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Classifier:** DenseNet121 with sigmoid activation per disease head.
- **Calibrated Thresholds:** Locked validation operating points:
  - Cardiomegaly: `0.9502`
  - Edema: `0.8770`
  - Consolidation: `0.7329`
  - Pneumonia: `0.7529`
  - Atelectasis: `0.7495`
  - Pneumothorax: `0.5825`

### 4.2 2D Binary Normal vs Abnormal Classification
- **Classifiers:** Original ResNet18 Baseline vs Blur-Augmented ResNet18.
- **Threshold Calibration:** Validation set 90% target sensitivity calibration:
  - Original ResNet18: `0.176463` (Sensitivity: 90.0%, Specificity: 79.52%).
  - Blur-Augmented ResNet18: `0.187923` (Sensitivity: 90.0%, Specificity: 82.22%).
- **Default Baseline:** `0.5000` uncalibrated midpoint threshold.

### 4.3 3D Volumetric CT Multi-Planar Reconstruction (MPR)
- **Geometry:** Sub-millimeter spatial coordinate mapping using origin, physical spacing, and direction cosine matrix:
  $$\vec{p} = \vec{o} + \mathbf{D} (\vec{i} \odot \vec{s})$$
- **Orthogonal Viewports:** Axial ($XY$), Coronal ($XZ$), and Sagittal ($YZ$).
- **Clinical Windowing:** Hounsfield Unit windowing with standard presets:
  - Lung: Center $-600\text{ HU}$, Width $1500\text{ HU}$
  - Mediastinum: Center $+50\text{ HU}$, Width $350\text{ HU}$
  - Bone: Center $+400\text{ HU}$, Width $1800\text{ HU}$
  - Abdomen: Center $+40\text{ HU}$, Width $400\text{ HU}$

### 4.4 3D Watertight Surface Mesh Generation
- **Algorithm:** Lewiner Marching Cubes (`skimage.measure.marching_cubes`).
- **Boundary Condition:** 1-voxel zero-padding ensuring closed, watertight boundaries ($\partial M = \emptyset$).
- **Export Format:** Standard Wavefront OBJ format streamed via HTTP.
- **Topology Audit:** Evaluated via Trimesh for watertightness, Euler characteristic ($\chi = V - E + F = 2(1 - g)$), winding consistency, and non-manifold edges.

### 4.5 Synthetic DRR Projection Physics
- **Active Pipeline:** `scripts/3d_reconstruction/generate_drr.py`.
- **Physical Geometry:** Source-to-Detector Distance $SDD = 1800.0\text{ mm}$, Source-to-Object Distance $SOD = 1450.0\text{ mm}$, Geometric Magnification $M = SDD / SOD = 1.2414\times$.
- **Attenuation Physics:** Hounsfield Unit conversion to linear attenuation coefficients using $\mu_{\text{water}} = 0.0193\text{ mm}^{-1}$.

---

## 5. Development Status & Roadmap Maturity

| Subsystem / Capability | Current Status | Scientific Limitations | Next Milestone |
| :--- | :--- | :--- | :--- |
| **2D Multi-Label CXR** | **Production Frozen** | Retrospective NIH benchmark; not validated for clinical diagnosis | Multi-hospital clinical prospective validation |
| **2D Binary Robustness** | **Production Frozen** | Calibrated on IN-CXR validation; tested on 15k VinDr cases | Domain adaptation for low-dose portable scanners |
| **CT MPR Slicing** | **Production Frozen** | Requires pre-registered CT studies | Real-time multi-threaded MPR caching |
| **3D Watertight Meshes** | **Production Frozen** | Extracted from segmented CT volumes | Surface smoothing & decimation optimization |
| **LUNA16 Nodule Mapping** | **Production Frozen** | Reference research annotations; not real-time AI detections | Radiomic feature profiling per nodule |
| **Dual-View Radiograph Review** | **Operational Workstation** | Frontal processed by DenseNet; Lateral visual review | Biplanar neural fusion model in Phase 2B |
| **3D Lung Reconstruction (SSM)** | **Research Feasibility (Phase 2B)** | Requires paired multi-patient CXR-CT training cohort | Landmark-guided Statistical Shape Model PCA |

---

## 6. Audit Conclusion

The LungVision AI repository represents a scientifically grounded medical imaging research workstation. All components adhere strictly to medical AI governance:
1. No unverified clinical diagnostic claims are made.
2. Nodule annotations are explicitly disclosed as reference research labels.
3. CT-derived meshes are accurately identified as CT extractions rather than neural X-ray reconstructions.
4. Model weights, hashes, and calibration thresholds are mathematically reconciled and reproducible.

