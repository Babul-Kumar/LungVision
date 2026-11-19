# LungVision AI — Phase 7 Model Integration & Architecture

## 1. System Overview

Phase 7 integrates the experimentally validated **MultiScaleDenseNet121** deep neural network into LungVision AI.

The integrated architecture provides multi-scale feature aggregation, combining fine spatial details from early dense blocks with semantic macroscopic context from late blocks.

```mermaid
flowchart TD
    subgraph Backbone ["DenseNet121 Backbone"]
        Stem["Stem (conv0 + pool0)<br/>Stride 4"]
        B1["DenseBlock 1 (p2)<br/>256 ch, Stride 4"]
        T1["Transition 1"]
        B2["DenseBlock 2 (p3)<br/>512 ch, Stride 8"]
        T2["Transition 2"]
        B3["DenseBlock 3 (p4)<br/>1024 ch, Stride 16"]
        T3["Transition 3"]
        B4["DenseBlock 4 + Norm5<br/>1024 ch, Stride 32"]
    end

    Stem --> B1 --> T1 --> B2 --> T2 --> B3 --> T3 --> B4

    subgraph Lateral ["Lateral 1x1 Projections"]
        L2["lat2: 256 -> 128"]
        L3["lat3: 512 -> 128"]
        L4["lat4: 1024 -> 128"]
    end

    B1 --> L2
    B2 --> L3
    B3 --> L4

    subgraph Fusion ["Multi-Scale Spatial Fusion & Pooling"]
        Interp["Bilinear Upsampling to p2 Spatial Size (56x56)"]
        L3 --> Interp
        L4 --> Interp
        Concat["Concat [L2, up(L3), up(L4)]<br/>384 channels"]
        L2 --> Concat
        Interp --> Concat
        FuseConv["Fuse: Conv 3x3 + BN + ReLU<br/>256 channels (56x56)"]
        Concat --> FuseConv
        DualPool["Dual Pooling: GAP (256) + GMP (256)<br/>Flattened to 512 channels"]
        FuseConv --> DualPool
        Drop["Dropout (p = 0.25)"]
        DualPool --> Drop
    end

    subgraph Heads ["6 Decoupled Pathology Heads"]
        H1["Linear(512, 1) -> Cardiomegaly"]
        H2["Linear(512, 1) -> Edema"]
        H3["Linear(512, 1) -> Consolidation"]
        H4["Linear(512, 1) -> Pneumonia"]
        H5["Linear(512, 1) -> Atelectasis"]
        H6["Linear(512, 1) -> Pneumothorax"]
    end

    Drop --> H1
    Drop --> H2
    Drop --> H3
    Drop --> H4
    Drop --> H5
    Drop --> H6

    Out["Logits [Batch, 6] -> Sigmoid -> Continuous Probabilities"]
    H1 --> Out
    H2 --> Out
    H3 --> Out
    H4 --> Out
    H5 --> Out
    H6 --> Out
```

---

## 2. Checkpoint Specifications & Integrity

- **Filename**: `msd121_224_weighted_bce_direct.pth`
- **Location**: `models/multiscale_densenet121/msd121_224_weighted_bce_direct.pth`
- **SHA-256**: `7dc735ee00d9e57ece14cd33f2bdbd1e7fd226adca41cdb692ba0af2a33a9b23`
- **Filesize**: 32,904,058 bytes (31.38 MB)
- **Trainable Parameters**: Exactly 8,071,942
- **Tensors in State Dict**: 749 tensors (100% verified finite, strict loading without missing or unexpected keys)
- **Validation Macro AUROC**: **0.819726** (Best performing candidate in Phase 7 recovery)

---

## 3. End-to-End Prediction Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor Client as User / Frontend UI
    participant Gateway as FastAPI Router (/api/v1/predictions)
    participant Quality as Image Quality Checker
    participant Preproc as Phase 7 Preprocessor
    participant Model as MultiScaleDenseNet121
    participant Uncertainty as Uncertainty Analyzer
    participant CAM as High-Res Grad-CAM

    Client->>Gateway: POST /api/v1/predictions (radiograph file)
    Gateway->>Quality: evaluate(image_bytes)
    alt Quality Insufficient (corrupt, <128px, blank)
        Quality-->>Gateway: QualityReport (can_proceed=False)
        Gateway-->>Client: 200 OK (status="REJECTED", reason)
    else Quality Passed / Warning
        Quality-->>Gateway: QualityReport (can_proceed=True)
        Gateway->>Preproc: preprocess(pil_img, device)
        Preproc-->>Gateway: Tensor [1, 3, 224, 224] (exact ImageNet norm)
        Gateway->>Model: forward(tensor) [torch.inference_mode]
        Model-->>Gateway: Logits [1, 6] -> Sigmoid -> Probs [1, 6]
        Gateway->>Uncertainty: assess_disease(probs, thresholds)
        Uncertainty-->>Gateway: UncertaintySummary & margins
        opt generate_cam == True
            Gateway->>CAM: generate(model, tensor, target_disease)
            CAM-->>Gateway: Base64 Heatmap & Colormap Overlay
        end
        Gateway-->>Client: 200 OK (status="COMPLETED", findings, uncertainty, gradcam)
    end
```

---

## 4. Software Architecture Organization

```
backend/
├── app/
│   ├── models/
│   │   ├── multiscale_densenet.py     # Phase7MultiScaleDenseNet121 & legacy FPN
│   │   └── ...
│   ├── inference/
│   │   ├── preprocessing.py          # Exact direct 224 bilinear ImageNet preprocessor
│   │   ├── quality_checker.py        # Clinical decodability, blur & dynamic range checks
│   │   ├── calibration.py            # Operating Points A (F1), B (0.5), C (High-Sens)
│   │   ├── uncertainty.py            # Margin-based certainty & abstention recommendations
│   │   ├── gradcam.py                # 56x56 High-resolution fused Grad-CAM
│   │   └── prediction_engine.py      # Thread-safe singleton model inference engine
│   ├── services/
│   │   ├── registry_service.py       # Central model catalog & benchmark metrics
│   │   └── xray_service.py           # Multi-label & binary 2D screening coordinator
│   └── api/v1/
│       ├── predictions.py            # POST /predictions, GET /model/health, GET /model/info
│       └── xray.py                   # POST /xray/predict (supporting model_id)
```

---

## 5. Verification Matrix

| Verification Criterion | Method | Requirement | Observed Status |
|---|---|---|:---:|
| **Checkpoint Integrity** | SHA-256 calculation | `7dc735ee...` byte match | **PASSED** |
| **Model Architecture** | Strict state dict load | 8,071,942 parameters, 0 missing | **PASSED** |
| **Preprocessing Parity** | `torch.allclose` comparison | Max absolute diff $< 1\times 10^{-6}$ | **PASSED (diff = 0.0)** |
| **Quality Gate** | Ingestion boundary testing | Reject unreadable, $<128$, blank | **PASSED** |
| **Prediction Contract** | Shape & domain check | Output shape $[1, 6]$, probs $\in (0, 1)$ | **PASSED** |
| **Explainability** | Grad-CAM activation test | $56 \times 56$ feature map, base64 overlay | **PASSED** |
| **Regression Stability** | Pytest test suite | 159/159 tests passing | **PASSED** |
| **Frontend Integration** | Vitest & TypeScript build | 39/39 vitest passing, Vite build | **PASSED** |
