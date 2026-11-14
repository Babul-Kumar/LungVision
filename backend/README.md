# LungVision AI — High-Performance Backend

**Production-Ready Backend Service for 2D Chest Radiograph Screening & 3D CT Volumetric Exploration**  
*Built with FastAPI, PyTorch, Pydantic, and OpenCV/Pillow.*

---

## 1. Overview

LungVision AI Backend is a decoupled, reproducible Python service designed to serve frozen deep-learning research models for frontal chest X-ray screening and structural 3D CT exploration. It provides RESTful APIs tailored for modern web clients (React + TypeScript).

### Key Features
- **DenseNet121 6-Disease Multi-Label Classifier:** Sub-second inference across Cardiomegaly, Edema, Consolidation, Pneumonia, Atelectasis, and Pneumothorax with verified NIH calibration thresholds.
- **ResNet18 Binary Normal/Abnormal Classifiers:** Dual-model support for baseline and blur-augmented checkpoints.
- **3D CT Exploration:** Orthogonal MPR slicing (axial, coronal, sagittal) and 3D surface mesh generation (Wavefront OBJ) for interactive Three.js/Babylon.js rendering.
- **Historical Benchmark Metrics:** Direct API access to verified evaluation metrics from NIH, CheXpert, and VinDr cohorts.
- **Robust Security & Input Validation:** Magic byte inspection, file size limits, and RFC 7807 error formatting.
- **Scientific Integrity:** Strict non-diagnostic disclaimers and explicit separation between research outputs and medical diagnoses.

---

## 2. Directory Structure

```text
backend/
├── app/
│   ├── main.py                  # FastAPI application entrypoint & lifespan
│   ├── config.py                # Environment configuration (Pydantic BaseSettings)
│   ├── api/                     # REST API routers (/api/v1)
│   │   ├── router.py            # Aggregated router
│   │   └── v1/
│   │       ├── health.py        # /health, /ready probes
│   │       ├── xray.py          # /xray/predict, /xray/predict-binary
│   │       ├── ct.py            # /ct/upload, /ct/{id}/metadata, /ct/{id}/mesh
│   │       ├── models.py        # /models, /models/{id}
│   │       └── research.py      # /research/metrics
│   ├── core/                    # Core infrastructure
│   │   ├── errors.py            # Custom exceptions & handlers
│   │   ├── security.py          # File validation & path sanitization
│   │   └── logging.py           # Structured logging
│   ├── schemas/                 # Pydantic request/response models
│   │   ├── common.py
│   │   ├── xray.py
│   │   ├── ct.py
│   │   └── models.py
│   └── services/                # Business logic & ML inference
│       ├── model_loader.py      # Singleton checkpoint manager
│       ├── preprocessing.py     # Verified training-matched transforms
│       ├── xray_service.py      # Multi-label & binary inference
│       ├── ct_service.py        # CT volume slicing & OBJ mesh export
│       └── registry_service.py  # Model registry & metrics parser
├── tests/                       # Comprehensive pytest suite
├── requirements.txt             # Dependency specification
├── .env.example                 # Configuration template
├── ARCHITECTURE.md              # System design & component diagrams
├── API_DOCUMENTATION.md         # Endpoint specifications & curl examples
├── MODEL_INVENTORY.md           # Model weights, hashes, and datasets
└── IMPLEMENTATION_REPORT.md     # Engineering report & audit evidence
```

---

## 3. Quickstart & Installation (Windows / Linux / macOS)

### 3.1 Prerequisites
- Python 3.10+ (Tested on Python 3.14.7)
- System memory: $\ge 4\text{ GB}$ RAM
- CUDA GPU optional (automatic CPU fallback supported)

### 3.2 Setup Environment
```bash
# Navigate to project root
cd "computer vision"

# Install dependencies
pip install -r backend/requirements.txt
```

### 3.3 Run the Development Server
```bash
# Start server with Uvicorn
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 4. Running Tests
```bash
python -m pytest backend/tests -v
```

---

## 5. Important Research & Ethical Disclaimer
*LungVision AI is an experimental decision-support research platform. It is NOT certified as a medical device and must NOT be used for autonomous diagnosis, clinical triage, or patient treatment decisions. All predictions must be evaluated by a certified radiologist or healthcare provider.*

