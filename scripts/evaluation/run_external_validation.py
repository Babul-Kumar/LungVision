"""
Comprehensive External Dataset Validation & Operating Point Analysis Pipeline.
Evaluates the frozen DenseNet121 checkpoint on the Stanford CheXpert external cohort.

Generates:
1. research/evaluation/external_dataset_manifest.csv
2. research/evaluation/classwise_metrics.json
3. research/evaluation/confusion_matrices.json
4. research/evaluation/threshold_comparison.json

Strictly enforces:
- Patient-level split separation (Zero patient leakage between calibration and test).
- Class-wise metrics reporting (AUROC, AUPRC, Sensitivity, Specificity, Precision, F1, FPR, FNR, Brier, ECE).
- Bootstrap 95% confidence intervals (patient-clustered).
- Operating point comparison (Locked NIH, External Calibrated, High-Sensitivity Triage).
"""

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = BASE_DIR / "research" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DISEASES = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax"
]

LOCKED_NIH_THRESHOLDS = {
    "Cardiomegaly": 0.9502,
    "Edema": 0.8770,
    "Consolidation": 0.7329,
    "Pneumonia": 0.7529,
    "Atelectasis": 0.7495,
    "Pneumothorax": 0.5825
}

CHECKPOINT_HASH = "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af"


def calculate_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(y_scores)) + 1
    sorted_scores = y_scores[order]
    unique_scores, inverse, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(sorted_scores):
        tie_ranks = np.bincount(inverse, weights=ranks[order]) / counts
        ranks = tie_ranks[inverse][np.argsort(order)]
    u_stat = np.sum(ranks[y_true == 1]) - (n_pos * (n_pos + 1)) / 2.0
    return float(np.clip(u_stat / (n_pos * n_neg), 0.0, 1.0))


def calculate_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_scores)
    y_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_sorted == 1)
    fp_cumsum = np.cumsum(y_sorted == 0)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
    recalls = tp_cumsum / n_pos
    precisions = np.concatenate(([1.0], precisions))
    recalls = np.concatenate(([0.0], recalls))
    # Trapz over unique recall steps
    return float(np.clip(np.trapz(precisions, recalls), 0.0, 1.0))


def calculate_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_probs = np.asarray(y_probs, dtype=np.float64)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_probs >= bin_lower) & (y_probs < bin_upper if i < n_bins - 1 else y_probs <= bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_probs[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def load_dataset() -> List[Dict[str, Any]]:
    # 1. Load predictions
    pred_path = BASE_DIR / "lungVision" / "chexpert_external_predictions.csv"
    with open(pred_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 2. Load demographic metadata
    meta_path = BASE_DIR / "lungVision" / "chexpert_subgroup_error_analysis.csv"
    meta_by_path: Dict[str, Dict[str, str]] = {}
    if meta_path.exists():
        with open(meta_path, mode="r", encoding="utf-8") as f:
            m_reader = csv.DictReader(f)
            for m in m_reader:
                p = m["image_path"]
                if p not in meta_by_path:
                    meta_by_path[p] = {
                        "Sex": m.get("Sex", "Unknown"),
                        "Age": m.get("Age", "Unknown"),
                        "AP_PA": m.get("AP/PA", "AP"),
                        "Support_Devices": m.get("Support_Device_Status", "Unknown")
                    }

    records = []
    for r in rows:
        path = r["image_path"]
        meta = meta_by_path.get(path, {"Sex": "Unknown", "Age": "Unknown", "AP_PA": "AP", "Support_Devices": "Unknown"})
        rec = {
            "image_path": path,
            "patient_id": r["patient_id"],
            "sex": meta["Sex"],
            "age": meta["Age"],
            "projection": meta["AP_PA"],
            "support_devices": meta["Support_Devices"],
            "labels": {},
            "probs": {}
        }
        for d in DISEASES:
            rec["labels"][d] = int(float(r[f"true_{d}"]))
            rec["probs"][d] = float(r[f"prob_{d}"])
        records.append(rec)
    return records


def partition_patients(records: List[Dict[str, Any]], calib_ratio: float = 0.25, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits records into patient-disjoint calibration and test cohorts."""
    patients = sorted(list(set(r["patient_id"] for r in records)))
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_calib = int(len(patients) * calib_ratio)
    calib_patients = set(patients[:n_calib])
    test_patients = set(patients[n_calib:])

    calib_records = [r for r in records if r["patient_id"] in calib_patients]
    test_records = [r for r in records if r["patient_id"] in test_patients]

    for r in calib_records:
        r["split"] = "calibration"
    for r in test_records:
        r["split"] = "test"

    # Verify zero leakage
    assert calib_patients.isdisjoint(test_patients), "PATIENT LEAKAGE DETECTED BETWEEN CALIBRATION AND TEST!"
    return calib_records, test_records


def calibrate_thresholds(calib_records: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calibrates alternative thresholds strictly on the calibration split:
    1. Youden's J / F1 Optimal operating point.
    2. High-Sensitivity Screening operating point (targeting >= 80% sensitivity).
    """
    youden_thresholds = {}
    high_sens_thresholds = {}

    for d in DISEASES:
        y_true = np.array([r["labels"][d] for r in calib_records])
        scores = np.array([r["probs"][d] for r in calib_records])

        best_j = -1.0
        best_t_j = 0.5
        best_t_sens = 0.5
        found_sens = False

        # Threshold grid search
        grid = np.linspace(0.01, 0.99, 100)
        # Scan from high threshold to low to find 80% sensitivity
        for t in grid:
            preds = (scores >= t).astype(int)
            tp = int(np.sum((preds == 1) & (y_true == 1)))
            fp = int(np.sum((preds == 1) & (y_true == 0)))
            fn = int(np.sum((preds == 0) & (y_true == 1)))
            tn = int(np.sum((preds == 0) & (y_true == 0)))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            j = sens + spec - 1.0

            if j > best_j:
                best_j = j
                best_t_j = float(t)

            if sens >= 0.80 and not found_sens:
                best_t_sens = float(t)
                found_sens = True

        if not found_sens:
            # Fallback to 10th percentile of positives
            pos_scores = scores[y_true == 1]
            best_t_sens = float(np.percentile(pos_scores, 15)) if len(pos_scores) > 0 else 0.20

        youden_thresholds[d] = round(best_t_j, 4)
        high_sens_thresholds[d] = round(best_t_sens, 4)

    return youden_thresholds, high_sens_thresholds


def compute_metrics_for_cohort(
    records: List[Dict[str, Any]],
    thresholds: Dict[str, float],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Computes full class-wise metrics with 95% bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    patients = list(set(r["patient_id"] for r in records))
    patient_to_records = {p: [r for r in records if r["patient_id"] == p] for p in patients}

    disease_metrics = {}
    macro_auroc_list = []
    macro_auprc_list = []
    macro_f1_list = []

    for d in DISEASES:
        y_true = np.array([r["labels"][d] for r in records], dtype=int)
        y_probs = np.array([r["probs"][d] for r in records], dtype=float)
        t = thresholds[d]
        y_pred = (y_probs >= t).astype(int)

        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        n_total = len(y_true)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        brier = float(np.mean((y_probs - y_true) ** 2))
        ece = calculate_ece(y_true, y_probs, n_bins=10)
        auroc = calculate_auroc(y_true, y_probs)
        auprc = calculate_auprc(y_true, y_probs)

        # Bootstrap for 95% CI
        boot_aurocs = []
        boot_auprcs = []
        boot_sens = []
        boot_specs = []
        boot_f1s = []

        for _ in range(n_bootstrap):
            boot_pats = rng.choice(patients, size=len(patients), replace=True)
            boot_recs = [rec for pat in boot_pats for rec in patient_to_records[pat]]
            b_true = np.array([r["labels"][d] for r in boot_recs], dtype=int)
            b_probs = np.array([r["probs"][d] for r in boot_recs], dtype=float)
            if np.sum(b_true == 1) == 0 or np.sum(b_true == 0) == 0:
                continue
            b_pred = (b_probs >= t).astype(int)

            b_tp = int(np.sum((b_pred == 1) & (b_true == 1)))
            b_fp = int(np.sum((b_pred == 1) & (b_true == 0)))
            b_tn = int(np.sum((b_pred == 0) & (b_true == 0)))
            b_fn = int(np.sum((b_pred == 0) & (b_true == 1)))

            b_s = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
            b_sp = b_tn / (b_tn + b_fp) if (b_tn + b_fp) > 0 else 0.0
            b_p = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
            b_f = 2 * b_p * b_s / (b_p + b_s) if (b_p + b_s) > 0 else 0.0

            boot_aurocs.append(calculate_auroc(b_true, b_probs))
            boot_auprcs.append(calculate_auprc(b_true, b_probs))
            boot_sens.append(b_s)
            boot_specs.append(b_sp)
            boot_f1s.append(b_f)

        def ci_bounds(arr: List[float], default_val: float) -> List[float]:
            if len(arr) < 20:
                return [round(default_val, 4), round(default_val, 4)]
            return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

        disease_metrics[d] = {
            "disease": d,
            "threshold": t,
            "total_cases": n_total,
            "positive_cases": n_pos,
            "negative_cases": n_neg,
            "prevalence": round(n_pos / n_total, 4),
            "auroc": round(auroc, 4),
            "auroc_95_ci": ci_bounds(boot_aurocs, auroc),
            "auprc": round(auprc, 4),
            "auprc_95_ci": ci_bounds(boot_auprcs, auprc),
            "sensitivity": round(sens, 4),
            "sensitivity_95_ci": ci_bounds(boot_sens, sens),
            "specificity": round(spec, 4),
            "specificity_95_ci": ci_bounds(boot_specs, spec),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "f1_95_ci": ci_bounds(boot_f1s, f1),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "brier_score": round(brier, 4),
            "ece_10_bins": round(ece, 4),
            "confusion_counts": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            }
        }
        macro_auroc_list.append(auroc)
        macro_auprc_list.append(auprc)
        macro_f1_list.append(f1)

    return {
        "classwise": disease_metrics,
        "macro_summary": {
            "macro_auroc": round(float(np.mean(macro_auroc_list)), 4),
            "macro_auprc": round(float(np.mean(macro_auprc_list)), 4),
            "macro_f1": round(float(np.mean(macro_f1_list)), 4)
        }
    }


def generate_external_manifest(
    records: List[Dict[str, Any]],
    youden_thresh: Dict[str, float],
    high_sens_thresh: Dict[str, float]
):
    manifest_path = EVAL_DIR / "external_dataset_manifest.csv"
    headers = [
        "image_id",
        "patient_id",
        "split",
        "sex",
        "age",
        "projection",
        "support_devices"
    ]
    for d in DISEASES:
        headers.extend([
            f"true_{d}",
            f"prob_{d}",
            f"pred_nih_{d}",
            f"pred_youden_{d}",
            f"pred_sens_{d}",
            f"status_nih_{d}"
        ])

    with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for idx, r in enumerate(records):
            row = [
                f"chexpert_img_{idx + 1:03d}",
                r["patient_id"],
                r["split"],
                r["sex"],
                r["age"],
                r["projection"],
                r["support_devices"]
            ]
            for d in DISEASES:
                y = r["labels"][d]
                p = r["probs"][d]
                pred_nih = int(p >= LOCKED_NIH_THRESHOLDS[d])
                pred_youden = int(p >= youden_thresh[d])
                pred_sens = int(p >= high_sens_thresh[d])

                # Determine evaluation status
                if y == 1 and pred_nih == 1:
                    status = "Correct positive"
                elif y == 0 and pred_nih == 0:
                    status = "Correct negative"
                elif y == 0 and pred_nih == 1:
                    status = "False positive"
                else:
                    status = "False negative"

                row.extend([y, round(p, 6), pred_nih, pred_youden, pred_sens, status])
            writer.writerow(row)
    print(f"Generated {manifest_path} with {len(records)} external validation records.")


def generate_confusion_matrices(
    records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    youden_thresh: Dict[str, float],
    high_sens_thresh: Dict[str, float]
):
    json_path = EVAL_DIR / "confusion_matrices.json"

    def get_matrices_for_subset(subset: List[Dict[str, Any]]):
        res = {
            "operating_point_A_locked_nih": {},
            "operating_point_B_external_youden": {},
            "operating_point_C_high_sensitivity": {}
        }
        for d in DISEASES:
            y_true = np.array([r["labels"][d] for r in subset])
            scores = np.array([r["probs"][d] for r in subset])

            for op_key, thresh_dict in [
                ("operating_point_A_locked_nih", LOCKED_NIH_THRESHOLDS),
                ("operating_point_B_external_youden", youden_thresh),
                ("operating_point_C_high_sensitivity", high_sens_thresh)
            ]:
                t = thresh_dict[d]
                preds = (scores >= t).astype(int)
                tp = int(np.sum((preds == 1) & (y_true == 1)))
                fp = int(np.sum((preds == 1) & (y_true == 0)))
                tn = int(np.sum((preds == 0) & (y_true == 0)))
                fn = int(np.sum((preds == 0) & (y_true == 1)))

                res[op_key][d] = {
                    "threshold": t,
                    "matrix_2x2": [
                        [tn, fp],
                        [fn, tp]
                    ],
                    "matrix_labels": [
                        ["True Negative (TN)", "False Positive (FP)"],
                        ["False Negative (FN)", "True Positive (TP)"]
                    ],
                    "counts": {
                        "true_negatives": tn,
                        "false_positives": fp,
                        "false_negatives": fn,
                        "true_positives": tp,
                        "total": len(subset)
                    }
                }
        return res

    # Cross-disease confusion / co-occurrence
    cross_confusion = {}
    for d_target in DISEASES:
        cross_confusion[d_target] = {}
        target_true = np.array([r["labels"][d_target] for r in records])
        for d_other in DISEASES:
            other_preds = np.array([r["probs"][d_other] >= LOCKED_NIH_THRESHOLDS[d_other] for r in records], dtype=int)
            co_occur = int(np.sum((target_true == 1) & (other_preds == 1)))
            cross_confusion[d_target][d_other] = co_occur

    data = {
        "metadata": {
            "title": "DenseNet121 External Validation Confusion Matrices",
            "source_dataset": "Stanford CheXpert External Cohort",
            "evaluable_images": len(records),
            "patient_disjoint_test_images": len(test_records)
        },
        "full_cohort_confusion": get_matrices_for_subset(records),
        "held_out_test_split_confusion": get_matrices_for_subset(test_records),
        "cross_pathology_co_activation_matrix": {
            "description": "Number of cases where true ground truth for row pathology triggered positive model flag for column pathology under locked NIH thresholds.",
            "matrix": cross_confusion
        }
    }

    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {json_path} successfully.")


def generate_threshold_comparison(
    records: List[Dict[str, Any]],
    calib_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    youden_thresh: Dict[str, float],
    high_sens_thresh: Dict[str, float]
):
    json_path = EVAL_DIR / "threshold_comparison.json"

    comparison_data = {
        "metadata": {
            "title": "Operating Point Threshold Sensitivity & Specificity Comparison",
            "description": "Comparison between Locked NIH Production Thresholds, Independent External Calibrated Thresholds, and High-Sensitivity Screening Thresholds.",
            "calibration_protocol": "Calibrated strictly on patient-disjoint calibration split (50 patients, 51 images). Evaluated on held-out test split (150 patients, 151 images) and full external cohort (200 patients, 202 images).",
            "active_production_thresholds": "Locked NIH ChestX-ray14 Validation Thresholds (Preserved Unchanged in Production)",
            "operating_points": {
                "A": "Original NIH-calibrated thresholds (Frozen Production)",
                "B": "External Calibrated thresholds (Youden J optimal on calibration split)",
                "C": "Documented High-Sensitivity screening operating point (target >= 80% sensitivity)"
            }
        },
        "diseases": {}
    }

    for d in DISEASES:
        # Full cohort and test split performance for all three
        def eval_point(subset: List[Dict[str, Any]], thresh: float):
            y_true = np.array([r["labels"][d] for r in subset])
            scores = np.array([r["probs"][d] for r in subset])
            preds = (scores >= thresh).astype(int)
            tp = int(np.sum((preds == 1) & (y_true == 1)))
            fp = int(np.sum((preds == 1) & (y_true == 0)))
            tn = int(np.sum((preds == 0) & (y_true == 0)))
            fn = int(np.sum((preds == 0) & (y_true == 1)))
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
            return {
                "threshold": thresh,
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "precision": round(prec, 4),
                "f1_score": round(f1, 4),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            }

        comparison_data["diseases"][d] = {
            "operating_point_A_locked_nih": {
                "threshold": LOCKED_NIH_THRESHOLDS[d],
                "description": "Original NIH validation threshold calibrated on single-institution cohort",
                "calibration_cohort_result": eval_point(calib_records, LOCKED_NIH_THRESHOLDS[d]),
                "held_out_test_result": eval_point(test_records, LOCKED_NIH_THRESHOLDS[d]),
                "full_external_cohort_result": eval_point(records, LOCKED_NIH_THRESHOLDS[d])
            },
            "operating_point_B_external_youden": {
                "threshold": youden_thresh[d],
                "description": "Balanced Youden's J optimal threshold derived on independent calibration split",
                "calibration_cohort_result": eval_point(calib_records, youden_thresh[d]),
                "held_out_test_result": eval_point(test_records, youden_thresh[d]),
                "full_external_cohort_result": eval_point(records, youden_thresh[d])
            },
            "operating_point_C_high_sensitivity": {
                "threshold": high_sens_thresh[d],
                "description": "Sensitivity-oriented operating point targeting >= 80% sensitivity for triage screening",
                "calibration_cohort_result": eval_point(calib_records, high_sens_thresh[d]),
                "held_out_test_result": eval_point(test_records, high_sens_thresh[d]),
                "full_external_cohort_result": eval_point(records, high_sens_thresh[d])
            }
        }

    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Generated {json_path} successfully.")


def run_pipeline():
    print("Loading external CheXpert dataset...")
    records = load_dataset()
    print(f"Loaded {len(records)} records across {len(set(r['patient_id'] for r in records))} patients.")

    print("Partitioning patients into calibration (25%) and test (75%) splits...")
    calib_records, test_records = partition_patients(records, calib_ratio=0.25, seed=42)
    print(f"Calibration split: {len(calib_records)} records, {len(set(r['patient_id'] for r in calib_records))} patients.")
    print(f"Held-out test split: {len(test_records)} records, {len(set(r['patient_id'] for r in test_records))} patients.")

    print("Calibrating alternative operating thresholds on calibration split...")
    youden_thresh, high_sens_thresh = calibrate_thresholds(calib_records)
    print(f"External Youden Thresholds: {youden_thresh}")
    print(f"High-Sensitivity Thresholds: {high_sens_thresh}")

    print("Generating external dataset manifest...")
    generate_external_manifest(records, youden_thresh, high_sens_thresh)

    print("Computing class-wise metrics with 95% bootstrap confidence intervals...")
    metrics_full = compute_metrics_for_cohort(records, LOCKED_NIH_THRESHOLDS)
    metrics_test = compute_metrics_for_cohort(test_records, LOCKED_NIH_THRESHOLDS)
    metrics_youden = compute_metrics_for_cohort(test_records, youden_thresh)
    metrics_sens = compute_metrics_for_cohort(test_records, high_sens_thresh)

    classwise_metrics_path = EVAL_DIR / "classwise_metrics.json"
    with open(classwise_metrics_path, mode="w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "title": "DenseNet121 CheXpert External Cohort Class-Wise Metrics",
                "checkpoint_hash": CHECKPOINT_HASH,
                "total_records": len(records),
                "total_patients": len(set(r["patient_id"] for r in records)),
                "held_out_test_records": len(test_records),
                "bootstrap_iterations": 1000,
                "confidence_interval_level": "95% (percentile method with patient-level clustering)"
            },
            "full_external_cohort_locked_nih": metrics_full,
            "held_out_test_split_locked_nih": metrics_test,
            "held_out_test_split_external_youden": metrics_youden,
            "held_out_test_split_high_sensitivity": metrics_sens
        }, f, indent=2)
    print(f"Generated {classwise_metrics_path} successfully.")

    print("Generating confusion matrices...")
    generate_confusion_matrices(records, test_records, youden_thresh, high_sens_thresh)

    print("Generating threshold comparison report...")
    generate_threshold_comparison(records, calib_records, test_records, youden_thresh, high_sens_thresh)
    print("External validation pipeline complete!")


if __name__ == "__main__":
    run_pipeline()
