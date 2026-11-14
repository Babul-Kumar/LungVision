# LungVision AI — Frontend Research Workstation

A clinical-grade, modern web workstation for thoracic imaging intelligence, combining multi-label chest radiograph AI screening with interactive 3D computed tomography (CT) volumetric exploration.

Built with **React 18**, **TypeScript 5.6**, **Vite 6**, **Tailwind CSS**, **Three.js**, **Recharts**, and **Zustand**. Fully integrated with the production-verified LungVision FastAPI backend gateway.

---

## 1. System Architecture & Tech Stack

| Layer | Technologies | Purpose |
|---|---|---|
| **Core Framework** | React 18.3, TypeScript 5.6, Vite 6 | Type-safe single-page reactive application with lightning HMR |
| **Styling & Design** | Tailwind CSS 3.4, Lucide React | Medical dark workstation theme (`#070b14`, `#0d1424`, `#06b6d4`, `#14b8a6`) |
| **3D Mesh Rendering** | Three.js 0.170, OrbitControls, OBJLoader | GPU-accelerated WebGL rendering of Marching Cubes lung surfaces & LUNA16 nodule spheres |
| **Orthogonal Slicing** | Canvas 2D MPR Engine | Axial, coronal, and sagittal slice visualization with clinical HU window presets |
| **Statistical Charts** | Recharts 2.15 | AUROC, AUPRC, radar sensitivity profiling, and cross-dataset generalization |
| **State Management** | Zustand 5 | Modular stores for gateway telemetry, active scan selection, and local history |
| **API Client** | Axios 1.7 | Typed HTTP client with RFC 7807 problem details error handling |
| **Testing** | Vitest 2.1, React Testing Library, JSDOM | Comprehensive unit and component testing suite |

---

## 2. Key Diagnostic & Research Modules

### 2.1 Station Dashboard (`/dashboard`)
- Live gateway telemetry polling `/api/v1/health` and `/api/v1/ready`.
- PyTorch execution device reporting (`CPU` / `CUDA`).
- Quick-action launcher cards for 2D X-ray, 3D CT, and Research.
- Registered model summary with parameter counts and validation AUCs.

### 2.2 2D Chest Radiograph Screening (`/xray`)
- Drag-and-drop file upload with client-side format and 20 MB size validation.
- **DenseNet121 Multi-Label Screening**:
  - Locked 6-disease NIH label ordering:
    1. `Cardiomegaly` (Threshold: `0.9502`)
    2. `Edema` (Threshold: `0.8770`)
    3. `Consolidation` (Threshold: `0.7329`)
    4. `Pneumonia` (Threshold: `0.7529`)
    5. `Atelectasis` (Threshold: `0.7495`)
    6. `Pneumothorax` (Threshold: `0.5825`)
  - Probability meters with calibrated threshold tick markers and positive alert badges.
- **ResNet18 Binary Screening**:
  - Checkpoint selector: *Original ResNet18* (Threshold `0.1765`, 90% Sensitivity operating point) vs. *Blur-Augmented ResNet18* (Threshold `0.1879`, robust against motion blur artifacts).

### 2.3 3D CT Volume Explorer (`/ct-explorer`)
- **Spatial Geometry Strip**: Scan ID, voxel dimensions `(Z, Y, X)`, physical voxel spacing `(mm)`, origin `(mm)`, and direction matrix.
- **Interactive Three.js 3D Mesh Canvas**:
  - Streams Wavefront OBJ data directly from `GET /api/v1/ct/{id}/mesh`.
  - OrbitControls for intuitive pan, rotation, and zoom.
  - Centers mesh bounding box at origin.
  - Renders 12 LUNA16 ground-truth nodule spheres positioned in physical millimeters `(X, Y, Z)` with interactive selection.
  - Dedicated Structure Switcher (`lung_surface`, `airway`, `nodules`).
- **Multi-Planar Reconstruction (MPR) Orthogonal Slicer**:
  - Synchronized sliders for **Axial**, **Coronal**, and **Sagittal** planes.
  - Clinical Hounsfield Unit (HU) window presets:
    - **Lung Window**: Level `-600 HU`, Width `1500 HU`
    - **Mediastinum**: Level `+40 HU`, Width `400 HU`
    - **Bone Window**: Level `+300 HU`, Width `1500 HU`
  - Toggleable segmentation mask overlays and nodule crosshair markers.
- **Trimesh Mesh Quality Audit**:
  - Watertightness verification, Euler characteristic, connected components, and volume comparison.
- **Quantitative Volumetry**:
  - Right lung, left lung, total volume `(mL)`, and mean attenuation `(HU)`.

### 2.4 Research Benchmarks & Evidence Registry (`/research`)
- Historical audited evaluation metrics:
  - **NIH ChestX-ray14**: Held-out test set AUROC, AUPRC, Sensitivity, Specificity, Precision, F1.
  - **Stanford CheXpert**: Cross-hospital domain transferability evaluation.
  - **VinDr-CXR (15,000 images)**: External evaluation of ResNet18 baseline vs blur-augmented models with Brier scores and Expected Calibration Error (ECE).
- Recharts bar graphs and radar sensitivity profiling.
- One-click CSV export for downstream research pipelines.

### 2.5 Audited Model Registry (`/models`)
- Deep inspection of all frozen PyTorch checkpoints.
- Input tensor shapes `[224, 224, 3]`, parameter counts, training datasets, and checkpoint epochs.
- Cryptographic SHA-256 hashes and calibration rationales.

### 2.6 Ephemeral Session History (`/history`)
- Client-side inference ledger storing recent X-ray inferences and CT sessions.
- Privacy-preserving architecture: cached exclusively in local HTML5 storage, zero patient data retained on backend.

### 2.7 Settings & Diagnostics (`/settings`)
- Backend gateway health probe with real-time round-trip latency (ms).
- Client hardware diagnostics (WebGL2 GPU vendor and renderer detection).
- Full safety, ethical, and regulatory research disclaimers.

---

## 3. Quick Start & Setup

### Prerequisites
- Node.js >= 18.0.0 (Tested on Node v24.19.0)
- npm >= 9.0.0
- Running LungVision AI backend on `http://localhost:8000`

### Installation
```bash
cd frontend
npm install
```

### Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default configuration:
```env
VITE_API_BASE_URL=http://localhost:8000
```

### Development Server
```bash
npm run dev
```
The application will launch at `http://localhost:5173`.

### Automated Testing
Run the Vitest test suite:
```bash
npm test
```
Or run once:
```bash
npx vitest run
```

### Production Build
```bash
npm run build
```
Generates optimized, minified production assets in `dist/`.

---

## 4. Regulatory & Ethical Disclaimer

> [!WARNING]
> **Investigational Research Platform — Not for Primary Diagnostic Use**
>
> LungVision AI is an experimental computer-aided detection and exploration research tool developed solely for investigational, scientific, and educational purposes. It is **NOT cleared by the US FDA, CE-marked, or certified** under medical device software regulations for primary diagnostic use. Predictions and volumetric calculations must be verified by board-certified radiologists in conjunction with complete clinical records.

