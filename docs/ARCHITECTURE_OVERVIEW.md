# LungVision AI — Architecture Overview

## 1. System Topology & Mission

**LungVision AI** is an advanced full-stack thoracic medical intelligence and volumetric exploration workstation. Designed for medical imaging research, the system integrates deep learning radiograph triage with clinical computed tomography (CT) visualization.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          LUNGVISION AI PLATFORM TOPOLOGY                        │
├──────────────────────────────────────┬──────────────────────────────────────────┤
│           VITE REACT SPA             │           FASTAPI REST GATEWAY           │
│      (TypeScript / Tailwind CSS)     │      (Python 3.11+ / PyTorch / VTK)      │
├──────────────────────────────────────┼──────────────────────────────────────────┤
│  • App Layout & Responsive Shell     │  • Modular API Routers (/api/v1/*)       │
│  • Route Lazy-Loading & Suspense     │    - /health & /ready probes             │
│  • 2D X-Ray Inspection Canvas       │    - /xray (Multi-label & Binary)        │
│  • 3D CT Orthogonal Slicer (MPR)     │    - /ct (Volumetry, Slices, Annotations)│
│  • Three.js Interactive Mesh Viewer  │    - /models (Active Neural Inventory)   │
│  • Dynamic Benchmark Visualizers     │    - /research (Validation Benchmarks)   │
│  • Ephemeral HIPAA Audit Ledger      │  • In-Memory Checkpoint Runtime          │
│  • Persistent / Session State        │  • Pure In-Memory Nodule Physical-to-    │
│    (Zustand Storage Adapters)        │    Voxel Transformation Pipeline         │
└──────────────────────────────────────┴──────────────────────────────────────────┘
```

---

## 2. Front-End Architectural Design

### Single Page Application (SPA) Framework
- **Framework**: React 18 with Vite 6 build engine.
- **Language**: TypeScript 5.8 in strict mode (`noUnusedLocals`, `noUnusedParameters`).
- **Styling**: Tailwind CSS with custom medical color palette (`medical-bg`, `medical-surface`, `medical-card`, `medical-cyan`, `medical-teal`).
- **State Management**: Zustand lightweight stores:
  - `useAppStore`: System heartbeat, gateway latency, and global backend readiness.
  - `useHistoryStore`: Dual-mode HIPAA compliance ledger (ephemeral `sessionStorage` by default, opt-in `localStorage`).
- **3D Graphics Engine**: Three.js (r182) WebGL pipeline:
  - Orthogonal trackball orbit controls with smooth damping.
  - Directional, ambient, and hemisphere clinical lighting.
  - Dedicated buffer geometry and material disposal handlers on component unmount.
  - Reset camera orientation and toggleable lung mesh visibility.

### Modular Route-Level Code Splitting
To optimize first-contentful-paint (FCP) and prevent bloated bundle sizes, the frontend utilizes `React.lazy()` with `React.Suspense` wrapping routes:
- `DashboardPage` (~9.1 kB)
- `XrayPage` (~17.9 kB)
- `CTExplorerPage` (~57.7 kB)
- `ResearchPage` (~16.5 kB)
- `ModelsPage` (~9.7 kB)
- `HistoryPage` (~7.3 kB)
- `SettingsPage` (~7.9 kB)
- Vendor libraries are separated into distinct chunks (`three` ~479 kB, `recharts` ~423 kB, `vendor` ~262 kB).

---

## 3. Back-End Architectural Design

### Modular FastAPI Service Layer
The backend is structured into clean, decoupled layers adhering to clean architecture principles:

```
backend/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── ct.py                  # 3D CT volumetry, slice extraction, annotations
│   │   │   ├── health.py              # Liveness and model readiness probes
│   │   │   ├── models.py              # Checkpoint inventory and calibration metadata
│   │   │   ├── research.py            # NIH, CheXpert, VinDr validation benchmarks
│   │   │   └── xray.py                # DenseNet121 & ResNet18 inference endpoints
│   │   └── router.py                  # Master v1 API router aggregation
│   ├── core/
│   │   ├── config.py                  # Pydantic v2 settings with path fallback resolution
│   │   ├── errors.py                  # Standardized RFC 7807 error responses
│   │   └── security.py                # Input validation and file sanitization
│   ├── schemas/                       # Pydantic request/response contract models
│   ├── services/
│   │   ├── ct/                        # LUNA16 coordinate conversion & volumetric analysis
│   │   ├── model_loader.py            # Thread-safe PyTorch checkpoint loader
│   │   ├── registry_service.py        # Model inventory & performance metadata
│   │   ├── research_service.py        # Audited validation metrics parser
│   │   └── xray_service.py            # Image preprocessing and calibrated inference
│   └── main.py                        # FastAPI application entrypoint and middleware
└── tests/                             # Comprehensive 29-test pytest verification suite
```

### In-Memory Coordinate Conversion Pipeline
The CT subsystem performs physical-to-voxel coordinate transformation without file writes:
$$\begin{bmatrix} x_{\text{voxel}} \\ y_{\text{voxel}} \\ z_{\text{voxel}} \end{bmatrix} = \begin{bmatrix} \frac{x_{\text{phys}} - x_{\text{origin}}}{s_x} \\ \frac{y_{\text{phys}} - y_{\text{origin}}}{s_y} \\ \frac{z_{\text{phys}} - z_{\text{origin}}}{s_z} \end{bmatrix}$$
where $[s_x, s_y, s_z]$ is the anisotropic voxel spacing (mm) and $[x_{\text{origin}}, y_{\text{origin}}, z_{\text{origin}}]$ is the scan physical origin.

---

## 4. Security & Privacy Guarantees

1. **HIPAA Privacy & Zero Server-Side Retention**:
   - The FastAPI backend performs in-memory analysis of uploaded radiographs and CT archives.
   - Uploaded files are processed in ephemeral byte buffers and immediately discarded after inference.
   - Filenames in the client-side audit log are sanitized to strip local disk directories and operating system paths.
2. **Strict Medical Disclaimers**:
   - Every diagnostic screen features non-diagnostic disclaimers highlighting investigational status.
   - Probability outputs display calibrated operating points rather than binary diagnoses.

