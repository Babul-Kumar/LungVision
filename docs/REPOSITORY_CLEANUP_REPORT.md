# LungVision AI — Repository Cleanup & Hardening Report

## 1. Executive Summary

As part of Phase 3 (Repository Cleanup & Engineering Hardening) under the Gravitas Engineering Reliability Runtime, a systematic audit of the repository was conducted to remove redundant artifacts, eliminate empty binary stubs, enforce strict Git hygiene, and build robust fallback paths.

All deletions were verified against code imports, file hashes, and test dependencies prior to execution.

---

## 2. Eliminated Redundant & Dummy Artifacts

During Phase 1 reconnaissance, 4 empty dummy ZIP archives and 1 root duplicate CSV were cataloged:

| Target File | Initial Size | SHA-256 Hash | Deletion Justification | Status |
| :--- | :--- | :--- | :--- | :--- |
| `LungVision_AI_Disease_Classification_result.zip` | 22 bytes | `8739c76e...cd30` | Empty PK zip header (0 content files), unreferenced stub | **Deleted** |
| `LungVision_AI_Final_Research_Bundle.zip` | 22 bytes | `8739c76e...cd30` | Empty PK zip header (0 content files), unreferenced stub | **Deleted** |
| `results (1).zip` | 22 bytes | `8739c76e...cd30` | Empty PK zip header (0 content files), unreferenced stub | **Deleted** |
| `results (2).zip` | 22 bytes | `8739c76e...cd30` | Empty PK zip header (0 content files), unreferenced stub | **Deleted** |
| `nih_test_results.csv` (Root) | 10.9 kB | `132bbf2d...8b0b` | Byte-for-byte duplicate of `lungVision/nih_test_results.csv` | **Deleted** |

> [!NOTE]
> `results.zip` (858 MB) was audited and explicitly **preserved** as the authoritative research dataset bundle. It is tracked in the repository artifact manifest and excluded from Git tracking via `.gitignore`.

---

## 3. Git Hygiene & Artifact Exclusion

A root `.gitignore` file was created to prevent inadvertent tracking of development cache, build outputs, and large binary blobs:

- **Python**: Ignored `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `*.egg-info/`, `.venv/`, `.env`.
- **Node/Vite**: Ignored `node_modules/`, `frontend/dist/`, `frontend/.vite/`, `npm-debug.log*`.
- **Operating System & IDE**: Ignored `.DS_Store`, `Thumbs.db`, `.vscode/`, `.idea/`.
- **Medical Uploads & Caches**: Ephemeral storage `backend/data/uploads/*` and `backend/data/cache/*` ignored, preserving `.gitkeep`.
- **Large Archives**: All `*.zip`, `*.tar.gz`, `*.7z` archives ignored from version control.

---

## 4. Fallback Path Resolution

To ensure zero breakage when moving or running the application across varied environments, `backend/app/config.py` was enhanced with dynamic fallback properties:

- `RESOLVED_DENSENET121_PATH`: Dynamically falls back between `lungvision_disease_classification/checkpoints/densenet121_nih_best.pth`, `checkpoints/densenet121_nih_best.pth`, and `LungVision_AI_Model_Registry/models/densenet121_nih_6disease/densenet121_nih_best.pth`.
- `RESOLVED_RESNET18_ORIGINAL_PATH`: Dynamically falls back between `LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth` and `LungVision_AI_Model_Registry/models/original_resnet18/model.pth`.
- `RESOLVED_RESNET18_AUGMENTED_PATH`: Dynamically falls back between `LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth` and `LungVision_AI_Model_Registry/models/blur_augmented_resnet18/model.pth`.

The service layers `model_loader.py` and `registry_service.py` were updated to utilize these resolved paths.

---

## 5. Post-Cleanup Verification Audit

Following the cleanup and hardening:
1. **Frontend TypeScript Check**: `npx tsc --noEmit` exited cleanly with code 0.
2. **Frontend Vitest Suite**: All 8 test suites (15/15 tests) passed 100%.
3. **Frontend Production Build**: `npm run build` completed in 15.17s with code-split chunks.
4. **Backend Pytest Suite**: All 29 tests passed in 43.79s.

