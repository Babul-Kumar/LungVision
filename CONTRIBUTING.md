# Contributing to LungVision AI

Thank you for your interest in contributing to the LungVision AI medical research platform.

LungVision AI is a rigorous medical-imaging deep learning research framework comprising:
1. Multi-label thoracic pathology detection (DenseNet121).
2. Binary normal vs. abnormal screening & blur robustness (ResNet18).
3. 3D Computed Tomography (CT) volumetric exploration & LUNA16 ground-truth nodule mapping.
4. Audited external validation across 40,798 radiographs (NIH ChestX-ray14, CheXpert, VinDr-CXR).

---

## 1. Scientific & Ethical Guidelines

- **Investigational Research Standard**: All features, models, and interfaces must clearly indicate their non-diagnostic, investigational status.
- **Strict Label Ordering**: Multi-label classification models must strictly adhere to the locked NIH 6-disease order:
  `['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax']`.
- **Calibrated Decision Thresholds**: Thresholds established during validation are frozen research baselines and must not be altered without documented external validation evidence.
- **HIPAA Privacy & Zero-Retention**: All user-facing features must operate ephemerally by default. The backend does not persist uploaded patient DICOM or radiograph data to persistent disks.

---

## 2. Development Setup

### Backend (Python 3.11+ / PyTorch)
```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux / macOS

# Install dependencies
pip install -r backend/requirements.txt

# Run backend test suite
python -m pytest backend/tests -v

# Launch local FastAPI development gateway
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend (Node.js 18+ / React 18 / Vite / TypeScript)
```bash
cd frontend

# Install packages
npm install

# Run static TypeScript type check
npx tsc --noEmit

# Run unit and integration tests
npm test -- --run

# Run production build
npm run build

# Start local Vite development server
npm run dev
```

---

## 3. Pull Request Protocol

1. **Reconnaissance First**: Inspect existing governance documents (`project_governance/PRD.md`, `ARCHITECTURE.md`, `RULES.md`).
2. **Never Claim Without Verification**: All pull requests must include concrete verification logs (`pytest`, `tsc --noEmit`, Vitest).
3. **No Dead or Corrupt Binaries**: Do not commit empty or dummy zip archives, scratch binaries, or uncalibrated weights.
4. **Clean Code Style**:
   - Python: Follow PEP 8 with type hints on all public functions.
   - TypeScript: Strict typing enabled (`noUnusedLocals`, `noUnusedParameters`).

