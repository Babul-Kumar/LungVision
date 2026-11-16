"""
Automated Backend Tests for External Dataset Validation & Misclassification Audit.
Verifies compliance with Gravitas engineering invariants and prompt completion criteria:
- Checkpoint hash immutability
- Delivery of all 8 evaluation artifacts
- Patient-level split disjointness (Zero leakage)
- Metric mathematical bounds
- Production threshold preservation
"""

import csv
import json
from pathlib import Path
import pytest
from backend.app.config import BASE_DIR, settings
from backend.app.services.xray_service import DEFAULT_NIH_THRESHOLDS, LABELS_6_DISEASE

EVAL_DIR = BASE_DIR / "research" / "evaluation"
CHECKPOINT_HASH = "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af"


def test_all_eight_deliverables_exist():
    """Verifies that all 8 required research evaluation deliverables exist."""
    expected_files = [
        "manual_test_manifest.csv",
        "manual_test_reproduction.json",
        "external_dataset_manifest.csv",
        "classwise_metrics.json",
        "confusion_matrices.json",
        "threshold_comparison.json",
        "error_analysis_report.md",
        "EXTERNAL_VALIDATION_REPORT.md"
    ]
    for filename in expected_files:
        path = EVAL_DIR / filename
        assert path.exists(), f"Missing required deliverable: {filename}"
        assert path.stat().st_size > 100, f"File {filename} is unexpectedly empty"


def test_densenet121_checkpoint_hash_integrity():
    """Verifies that the DenseNet121 model checkpoint hash matches verified SHA-256."""
    import hashlib
    ckpt_path = settings.RESOLVED_DENSENET121_PATH
    assert ckpt_path.exists(), f"Checkpoint missing at {ckpt_path}"
    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    assert hasher.hexdigest().lower() == CHECKPOINT_HASH.lower()


def test_production_thresholds_strictly_preserved():
    """Verifies that production thresholds in xray_service match verified constants."""
    expected_thresholds = {
        "Cardiomegaly": 0.9501953125,
        "Edema": 0.876953125,
        "Consolidation": 0.73291015625,
        "Pneumonia": 0.7529296875,
        "Atelectasis": 0.74951171875,
        "Pneumothorax": 0.58251953125
    }
    for disease, expected in expected_thresholds.items():
        actual = DEFAULT_NIH_THRESHOLDS[disease]
        assert abs(actual - expected) < 1e-4, f"Production threshold mutated for {disease}: {actual} vs {expected}"


def test_zero_patient_leakage_in_external_manifest():
    """Verifies strict patient-level separation between calibration and test splits."""
    manifest_path = EVAL_DIR / "external_dataset_manifest.csv"
    calib_patients = set()
    test_patients = set()

    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            patient_id = row["patient_id"]
            if split == "calibration":
                calib_patients.add(patient_id)
            elif split == "test":
                test_patients.add(patient_id)

    assert len(calib_patients) > 0, "No calibration patients found"
    assert len(test_patients) > 0, "No test patients found"
    intersection = calib_patients.intersection(test_patients)
    assert len(intersection) == 0, f"PATIENT LEAKAGE DETECTED: {intersection}"


def test_classwise_metrics_mathematical_bounds():
    """Verifies that computed metrics in classwise_metrics.json obey mathematical axioms."""
    json_path = EVAL_DIR / "classwise_metrics.json"
    with open(json_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    for cohort_key in ["full_external_cohort_locked_nih", "held_out_test_split_locked_nih"]:
        cohort = data[cohort_key]
        for disease, m in cohort["classwise"].items():
            assert 0.0 <= m["auroc"] <= 1.0, f"AUROC out of bounds for {disease}: {m['auroc']}"
            assert 0.0 <= m["auprc"] <= 1.0, f"AUPRC out of bounds for {disease}: {m['auprc']}"
            assert 0.0 <= m["sensitivity"] <= 1.0
            assert 0.0 <= m["specificity"] <= 1.0
            assert 0.0 <= m["precision"] <= 1.0
            assert 0.0 <= m["f1_score"] <= 1.0
            assert 0.0 <= m["brier_score"] <= 1.0

            # Mathematical complements
            assert abs((m["sensitivity"] + m["false_negative_rate"]) - 1.0) < 1e-3
            assert abs((m["specificity"] + m["false_positive_rate"]) - 1.0) < 1e-3

            # Confusion matrix sum equals total cases
            counts = m["confusion_counts"]
            total_sum = counts["tp"] + counts["fp"] + counts["tn"] + counts["fn"]
            assert total_sum == m["total_cases"]


def test_manual_test_manifest_integrity():
    """Verifies the manual test manifest contains all 6 test records with valid status."""
    csv_path = EVAL_DIR / "manual_test_manifest.csv"
    with open(csv_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 6, f"Expected 6 manual test records, found {len(rows)}"
    valid_statuses = [
        "Concordant (Correct Negative)",
        "Concordant (Correct Positive)",
        "Discordant (False Negative / Missed Pneumothorax)",
        "Discordant (False Negative for Edema, False Positive for Consolidation)",
        "Discordant (False Negative for Consolidation, False Positive for Pneumonia)",
        "Discordant (False Negative / Missed Atelectasis)"
    ]
    for r in rows:
        assert r["screening_concordance"] in valid_statuses
        assert r["model_checkpoint_hash"] == CHECKPOINT_HASH
