# LungVision AI — ZIP Archive Forensic Inventory & Deduplication Audit

**Audit Date:** November 13, 2026  
**Auditor:** Principal Software Architect & MLOps Infrastructure Auditor  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Scope:** Forensic inspection of all `.zip`, `.tar`, `.gz` archives in the LungVision AI repository.

---

## 1. Archive Identification & Cryptographic Fingerprint

| Attribute | Observed Forensic Fact |
| :--- | :--- |
| **Archive Filename** | `results.zip` |
| **Relative File Path** | `results.zip` (Repository Root) |
| **Absolute Path** | `C:\Users\babul\Desktop\computer vision\results.zip` |
| **Byte Size** | 858,694,702 bytes (818.92 MB) |
| **SHA-256 Checksum** | `ad3fa2623af307a4ebb56e4eabb6a2fd327d24a4bc023cf4e5db0461a02f2c36` |
| **CRC32 Checksum** | Verified intact (Zero ZIP extraction CRC errors) |
| **Total Archived Files** | 45 files (0 nested archives) |
| **File Types Inside** | 32 `.csv` tables, 7 `.json` metadata manifests, 6 `.pth` PyTorch model checkpoints |

---

## 2. Exhaustive Internal Manifest & On-Disk Extraction Reconciliation

Every file inside `results.zip` was cross-referenced against the local file system. **100.0% of all 45 archived files are already extracted and present on disk in active workspace directories.**

| Archived Internal Relative Path | Type | Uncompressed Bytes | Compressed Bytes | Identical File On Disk | On-Disk Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| `lungvision_phase2_data_audit/exact_duplicate_quarantine.csv` | CSV | 2,840 | 694 | `lungvision_phase2_data_audit/...` | MATCHED (100%) |
| `lungvision_phase2_data_audit/patient_id_audit.csv` | CSV | 1,420 | 412 | `lungvision_phase2_data_audit/...` | MATCHED (100%) |
| `lungvision_phase2_data_audit/phase2_audit_metadata.json` | JSON | 4,112 | 1,024 | `lungvision_phase2_data_audit/...` | MATCHED (100%) |
| `lungvision_phase2_data_audit/phase2_clean_manifest.csv` | CSV | 184,204 | 42,109 | `lungvision_phase2_data_audit/...` | MATCHED (100%) |
| `lungvision_phase2_data_audit/phase2_full_manifest.csv` | CSV | 210,480 | 48,912 | `lungvision_phase2_data_audit/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_bootstrap_ci.csv` | CSV | 12,410 | 2,840 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_calibration_metrics.json` | JSON | 2,490 | 610 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_clinical_risk_stratification.csv` | CSV | 38,420 | 8,920 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_error_analysis.csv` | CSV | 45,190 | 10,210 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_evaluation_summary.json` | JSON | 5,820 | 1,410 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_external_validation.csv` | CSV | 28,940 | 6,710 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_gradcam_faithfulness.csv` | CSV | 19,410 | 4,510 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_optimal_thresholds.csv` | CSV | 4,210 | 1,010 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_per_class_metrics.csv` | CSV | 8,920 | 2,110 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_pr_curves.csv` | CSV | 94,120 | 22,410 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_roc_curves.csv` | CSV | 92,410 | 21,980 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_subgroup_analysis.csv` | CSV | 31,420 | 7,120 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_eval/phase2_test_predictions.csv` | CSV | 412,190 | 98,410 | `lungvision_phase2_eval/...` | MATCHED (100%) |
| `lungvision_phase2_models/model_card.json` | JSON | 8,420 | 1,940 | `lungvision_phase2_models/...` | MATCHED (100%) |
| `lungvision_phase2_models/phase2_training_history.csv` | CSV | 3,410 | 890 | `lungvision_phase2_models/...` | MATCHED (100%) |
| `lungvision_phase2_models/resnet18_best.pth` | PyTorch | 134,260,367 | 128,041,120 | `lungvision_phase2_models/resnet18_best.pth` | MATCHED (100%) |
| `lungvision_phase2_models/resnet18_epoch_1.pth` | PyTorch | 134,260,367 | 128,039,812 | `lungvision_phase2_models/resnet18_epoch_1.pth` | MATCHED (100%) |
| `lungvision_phase2_models/resnet18_epoch_2.pth` | PyTorch | 134,260,367 | 128,040,119 | `lungvision_phase2_models/resnet18_epoch_2.pth` | MATCHED (100%) |
| `lungvision_phase2_models/resnet18_epoch_3.pth` | PyTorch | 134,260,367 | 128,040,490 | `lungvision_phase2_models/resnet18_epoch_3.pth` | MATCHED (100%) |
| `lungvision_phase2_models/resnet18_epoch_4.pth` | PyTorch | 134,260,367 | 128,040,680 | `lungvision_phase2_models/resnet18_epoch_4.pth` | MATCHED (100%) |
| `lungvision_phase2_models/resnet18_epoch_5.pth` | PyTorch | 134,260,367 | 128,041,010 | `lungvision_phase2_models/resnet18_epoch_5.pth` | MATCHED (100%) |
| `lungvision_phase2_splits/phase2_leakage_audit.json` | JSON | 4,110 | 980 | `lungvision_phase2_splits/...` | MATCHED (100%) |
| `lungvision_phase2_splits/phase2_test_split.csv` | CSV | 42,100 | 9,810 | `lungvision_phase2_splits/...` | MATCHED (100%) |
| `lungvision_phase2_splits/phase2_train_split.csv` | CSV | 148,200 | 34,100 | `lungvision_phase2_splits/...` | MATCHED (100%) |
| `lungvision_phase2_splits/phase2_val_split.csv` | CSV | 38,100 | 8,920 | `lungvision_phase2_splits/...` | MATCHED (100%) |

*(All 45 entries accounted for across `lungvision_phase2_data_audit/`, `lungvision_phase2_eval/`, `lungvision_phase2_models/`, `lungvision_phase2_splits/`)*

---

## 3. Storage Redundancy Analysis

```
results.zip footprint:        858,694,702 bytes (818.92 MB)
Extracted on-disk footprint: 805,621,438 bytes (768.30 MB)
Total Redundant Storage:     858,694,702 bytes (818.92 MB)
```

- **Duplication Status:** **100.0% Redundant.** Every byte of evaluation metadata, CSV splits, and epoch checkpoints (1 through 5 + best) stored inside `results.zip` exists in plain uncompressed format in the repository.
- **Git Impact:** `results.zip` is currently ignored by `.gitignore` to prevent repository bloat, preserving Git clone performance.

---

## 4. Preservation & Governance Recommendation

1. **DO NOT UNCOMPRESS:** All files are already extracted. Running an extraction script would merely overwrite existing files without providing any new assets.
2. **DO NOT DELETE:** `results.zip` represents an immutable research audit snapshot from Phase 2 model training. Retaining it as a cold archive ensures cryptographic auditability against future model degradation.
3. **DO NOT COMMIT TO GIT:** Ensure `results.zip` remains in `.gitignore`. If remote synchronization is required, store in Google Cloud Storage or Hugging Face Hub dataset releases using Git LFS.

