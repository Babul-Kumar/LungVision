# LungVision AI — Mandatory Engineering Rules

**Document Version:** 1.0.0  
**Status:** In Force / Active Governance  
**Author:** Staff Backend Architect & MLOps Lead  
**Governing Skill:** Gravitas Engineering Reliability Runtime  
**Created:** 2026-11-12  

---

## 1. Core Engineering Invariants (Gravitas Principles)

1. **Ground Before Mutating:** Read target files, dependencies, manifests, and tests before writing or editing. Never make architectural or code modifications based on assumed state.
2. **Stay in Declared Scope:** All new backend code must live strictly within `backend/`. All governance documents must live within `LungVision_AI_Final_Research_Bundle/project_governance/`. Never modify original research bundles or frontend source trees without explicit instruction.
3. **Completion Needs Real Evidence:** A task or feature is complete ONLY when executed commands, unit tests, or runtime outputs verify its functionality. Stale evidence or self-attestation is forbidden.
4. **Never Exact-Retry a Failure:** If a tool call, build, or test fails, diagnose the root cause and alter strategy before retrying. Byte-identical retries are blocked.
5. **Satisfy the Contract:** Every acceptance criterion declared in `contract.json` must be met before project completion.
6. **Say What You Cannot Verify:** State limitations, missing dependencies, or experimental boundaries clearly. Never fabricate certainty, accuracy scores, or clinical capabilities.

---

## 2. Machine Learning & Model Preservation Rules

1. **Never Retrain Existing Checkpoints:** Frozen research checkpoints are final historical artifacts. Do not initiate retraining, fine-tuning, or parameter overwrites.
2. **Never Overwrite Original Checkpoints:**
   - `lungvision_disease_classification/checkpoints/densenet121_nih_best.pth`
   - `LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth`
   - `LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth`
   These files are read-only.
3. **Never Delete Research Evidence:** CSV manifests, training history, evaluation tables, and ROC/PR plots in `lungVision/` and `LungVision_AI_Final_Research_Bundle/` must remain intact.
4. **Never Fabricate Model Capabilities or Metrics:** Any reported AUROC, AUPRC, F1, sensitivity, or specificity must originate directly from verified audit files.
5. **Never Silently Modify Preprocessing:** The 2D image preprocessing pipeline must strictly preserve:
   - Bilinear resize to $224 \times 224$.
   - Conversion to 3-channel RGB.
   - ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.
6. **Preserve Exact Output Label Ordering:**
   The DenseNet121 classifier output indices must remain strictly:
   - `0`: Cardiomegaly
   - `1`: Edema
   - `2`: Consolidation
   - `3`: Pneumonia
   - `4`: Atelectasis
   - `5`: Pneumothorax
7. **Preserve Verified Calibration Thresholds:**
   The locked NIH validation thresholds must be strictly applied:
   - Cardiomegaly: `0.9501953125`
   - Edema: `0.876953125`
   - Consolidation: `0.73291015625`
   - Pneumonia: `0.7529296875`
   - Atelectasis: `0.74951171875`
   - Pneumothorax: `0.58251953125`
8. **Decouple 2D and 3D Pipelines:** The 2D X-ray classification engine and 3D CT exploration services must remain completely independent modules. Failure or absence of 3D CT dependencies must never compromise 2D X-ray inference.
9. **Never Describe Experimental Outputs as Medical Diagnoses:** Every prediction response and user-facing text must include an explicit disclaimer stating the tool is an experimental research prototype and not a clinical diagnosis.

---

## 3. Backend Implementation & Code Quality Rules

1. **Modular Architecture:** Maintain clean separation between routers (`api/`), business logic (`services/`), ML loaders (`services/model_loader.py`), schemas (`schemas/`), and configuration (`core/`).
2. **Strict Typing:** All Python code must use explicit type annotations (`typing`, Pydantic models).
3. **Input Validation:** Validate all incoming HTTP payloads, query params, and multipart files (size, MIME type, image dimensions).
4. **Environment-Driven Configuration:** All environment settings (ports, hosts, CORS origins, checkpoint paths) must be managed via Pydantic `BaseSettings` reading from `.env` or system environment variables.
5. **No Hardcoded Secrets or Paths:** Never commit absolute filesystem paths or secret credentials.
6. **Fail-Fast Singleton Model Loading:** Load models into memory once at application lifespan startup. Do not reload model checkpoints on individual request calls.
7. **Graceful Degradation:** The application must start cleanly on both CPU and GPU (CUDA) environments. If a GPU is absent, fall back to CPU without crashing.

---

## 4. Documentation & Governance Maintenance Rules

1. **Continuous Synchronization:**
   - Update `TASK.md` immediately upon completing or blocking any task.
   - Record significant discoveries, architectural decisions, and bugfixes in `MEMORY.md`.
   - Update `DESIGN.md` if any endpoint schema or data contract is adjusted.
   - Update `ARCHITECTURE.md` if system topology changes.
2. **Evidence-Based Reporting:** Every claim in `IMPLEMENTATION_REPORT.md` must be supported by command logs, test run results, or file hashes.

