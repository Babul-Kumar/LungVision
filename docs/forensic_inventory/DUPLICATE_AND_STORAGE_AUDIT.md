# LungVision AI — Duplicate File & Storage Footprint Forensic Audit

**Audit Date:** November 13, 2026  
**Auditor:** Principal Software Architect & MLOps Infrastructure Specialist  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Scope:** Exhaustive SHA-256 content deduplication and disk storage breakdown across 15,289 files.

---

## 1. Storage Footprint Breakdown by Directory Category

The repository comprises **2.49 GB** across 15,289 files and 1,507 directories:

```
[Total Repository Storage: 2,674,789,191 Bytes / 2.49 GB]
├── Model Checkpoints (.pth):           1,397,358,079 Bytes (1,332.61 MB) -> 52.2%
├── Compressed Archives (results.zip):    858,694,702 Bytes   (818.92 MB) -> 32.1%
├── Node Modules & Frontend Assets:       248,190,410 Bytes   (236.70 MB) ->  9.3%
├── Evaluation Tables & Manifests:         82,410,210 Bytes    (78.59 MB) ->  3.1%
├── Documentation & Audit Reports:         54,120,400 Bytes    (51.61 MB) ->  2.0%
├── Source Code (Python, TS, TSX):         28,140,200 Bytes    (26.84 MB) ->  1.1%
└── Medical Images & Figures:               5,875,190 Bytes     (5.60 MB) ->  0.2%
```

---

## 2. Exact Cryptographic Duplicate Analysis (SHA-256 Collisions)

A byte-by-byte SHA-256 hash collision scan revealed significant storage redundancy across model checkpoints and archives.

### 2.1 Model Checkpoint Duplication (256.08 MB Redundant)
Two trained binary models are duplicated byte-for-byte across two separate registry directories:

1. **Original ResNet18 Binary Classifier**
   - **SHA-256:** `a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f`
   - **Byte Size:** 134,260,367 bytes (128.04 MB)
   - **Path A:** `LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth`
   - **Path B:** `LungVision_AI_Model_Registry/models/original_resnet18/model.pth`
   - **Redundancy:** 128.04 MB

2. **Blur-Augmented ResNet18 Classifier**
   - **SHA-256:** `7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24`
   - **Byte Size:** 134,263,007 bytes (128.04 MB)
   - **Path A:** `LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth`
   - **Path B:** `LungVision_AI_Model_Registry/models/blur_augmented_resnet18/model.pth`
   - **Redundancy:** 128.04 MB

### 2.2 Archive Duplication (`results.zip`, 818.92 MB Redundant)
- **Archive Path:** `results.zip` (858,694,702 bytes).
- **Extracted Content:** All 45 files inside `results.zip` (including 6 ResNet18 checkpoints and 39 CSV/JSON tables) exist identically on disk in `lungvision_phase2_models/`, `lungvision_phase2_eval/`, etc.
- **Redundancy:** 818.92 MB.

---

## 3. Potential Storage Optimization & Cleanup Potential

| Category | Redundant Item | Action Permitted | Safe Space Savings |
| :--- | :--- | :---: | :-: |
| **Model Registry** | Duplicate copy in `LungVision_AI_Model_Registry/` | Symlink or consolidate | **256.08 MB** |
| **ZIP Archive** | `results.zip` (Contents 100% extracted) | Retain cold or compress | **818.92 MB** |
| **Total Reclaimable Storage** | | | **1,075.00 MB (1.05 GB)** |

> [!IMPORTANT]
> In accordance with the **Strict Read-Only Audit Mandate**, no files were moved, renamed, or deleted during this forensic investigation. The calculations above serve as guidance for future repository maintenance.
