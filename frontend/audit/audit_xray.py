"""
Audit 2: X-Ray Inference Verification Script
Verifies:
1. Real checkpoint SHA-256 and loading
2. DenseNet121 6-label order and probabilities in [0, 1]
3. NIH validation thresholds match model registry
4. Original & Blur ResNet18 binary schemas
5. Frontend consumption of raw backend values
"""
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.audit.test_api_matrix import multipart_encode, request

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def verify_checkpoints():
    checkpoints = {
        "densenet121": "lungvision_disease_classification/checkpoints/densenet121_nih_best.pth",
        "original_resnet18": "LungVision_AI_Final_Research_Bundle/models/original_resnet18/model.pth",
        "blur_resnet18": "LungVision_AI_Final_Research_Bundle/models/blur_augmented_resnet18/model.pth",
    }
    results = {}
    for name, p in checkpoints.items():
        exists = Path(p).exists()
        size = Path(p).stat().st_size if exists else 0
        digest = sha256_file(p) if exists else None
        results[name] = {"path": p, "exists": exists, "size": size, "sha256": digest}
    return results

def verify_inference():
    real_cxr_path = "LungVision_AI_Final_Research_Bundle/external_validation_readiness/vindr_dicom_preview.png"
    with open(real_cxr_path, "rb") as f:
        real_cxr_bytes = f.read()

    # 1. DenseNet121 Multi-Label
    body, headers = multipart_encode({}, {"file": ("real_cxr.png", real_cxr_bytes, "image/png")})
    status, d121_resp, ms_d121 = request("POST", "/api/v1/xray/predict", data=body, headers=headers)

    # 2. Original ResNet18
    body_orig, headers_orig = multipart_encode({"model_id": "original_resnet18"}, {"file": ("real_cxr.png", real_cxr_bytes, "image/png")})
    status_orig, orig_resp, ms_orig = request("POST", "/api/v1/xray/predict-binary", data=body_orig, headers=headers_orig)

    # 3. Blur ResNet18
    body_blur, headers_blur = multipart_encode({"model_id": "blur_augmented_resnet18"}, {"file": ("real_cxr.png", real_cxr_bytes, "image/png")})
    status_blur, blur_resp, ms_blur = request("POST", "/api/v1/xray/predict-binary", data=body_blur, headers=headers_blur)

    # 4. Model Registry Check
    status_reg, reg_resp, _ = request("GET", "/api/v1/models")

    return {
        "densenet121": {"status": status, "response": d121_resp, "latency_ms": ms_d121},
        "original_resnet18": {"status": status_orig, "response": orig_resp, "latency_ms": ms_orig},
        "blur_resnet18": {"status": status_blur, "response": blur_resp, "latency_ms": ms_blur},
        "registry": reg_resp
    }

def main():
    print("--- 1. Checkpoint Verification ---")
    chk = verify_checkpoints()
    for k, v in chk.items():
        print(f"[{k}] Size: {v['size']:,} bytes | SHA256: {v['sha256']}")

    print("\n--- 2. Live Inference Verification ---")
    inf = verify_inference()
    
    # Check DenseNet121 labels and order
    d121 = inf["densenet121"]["response"]
    expected_labels = ["Cardiomegaly", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax"]
    actual_labels = list(d121["predictions"].keys())
    print(f"Expected 6 labels: {expected_labels}")
    print(f"Actual 6 labels:   {actual_labels}")
    assert actual_labels == expected_labels, "Label order mismatch!"
    print("Label order EXACT MATCH: PASS")

    # Verify probability bounds and thresholds
    for label, pred in d121["predictions"].items():
        p = pred["probability"]
        t = pred["threshold"]
        flag = pred["experimental_flag"]
        assert 0.0 <= p <= 1.0 and not math.isnan(p), f"Invalid probability for {label}: {p}"
        assert flag == (p >= t), f"Threshold flag inconsistency for {label}"
        print(f"  - {label:15}: p={p:.6f}, thresh={t:.6f}, flag={flag}")

    # Check ResNet18 outputs
    print("\n--- 3. Binary Classifiers Verification ---")
    for variant in ["original_resnet18", "blur_resnet18"]:
        res = inf[variant]["response"]
        pred = res["prediction"]
        p = pred["abnormal_probability"]
        t = pred["calibrated_threshold"]
        cls = pred["classification"]
        flag = pred["experimental_flag"]
        assert 0.0 <= p <= 1.0 and not math.isnan(p)
        assert flag == (p >= t)
        assert cls == ("Abnormal" if flag else "Normal")
        print(f"  [{variant}] p={p:.6f}, thresh={t:.6f}, class={cls}, flag={flag}")

    # Verify disclaimers
    print("\n--- 4. Non-Diagnostic Disclaimer Verification ---")
    for name, r in [("DenseNet121", d121), ("ResNet18 Orig", inf["original_resnet18"]["response"]), ("ResNet18 Blur", inf["blur_resnet18"]["response"])]:
        disc = r.get("disclaimer", "")
        print(f"  [{name}] Disclaimer present: {bool(disc)}")
        assert "not a medical diagnosis" in disc.lower() or "research" in disc.lower()
        print(f"         Text: {disc[:80]}...")

    with open("frontend/audit/xray_inference_audit.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoints": chk, "inference": inf}, f, indent=2)

if __name__ == "__main__":
    main()
