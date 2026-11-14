"""
API Contract Verification Matrix Test
Tests all routes for:
1. Valid 200/201 response with schema validation
2. 422 / 400 on invalid input
3. 404 on missing resource
4. RFC 7807 problem details error structure
"""
import urllib.request
import urllib.error
import json
import time

BASE = "http://127.0.0.1:8000"

def request(method, path, data=None, headers=None):
    headers = headers or {}
    url = f"{BASE}{path}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000
            body = resp.read()
            try:
                parsed = json.loads(body.decode("utf-8"))
            except Exception:
                parsed = body.decode("utf-8")
            return resp.status, parsed, elapsed_ms
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        body = e.read()
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception:
            parsed = body.decode("utf-8")
        return e.code, parsed, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return 0, str(e), elapsed_ms

def multipart_encode(fields, files):
    boundary = "----WebKitFormBoundaryLungVisionAudit"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8"))
    for name, (filename, content, mime) in files.items():
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}

def run_tests():
    matrix = []
    
    # 1. Root
    status, body, ms = request("GET", "/")
    matrix.append({
        "route": "GET /",
        "scenario": "Discovery root metadata",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "name" in body and "version" in body
    })
    
    # 2. Health
    status, body, ms = request("GET", "/api/v1/health")
    matrix.append({
        "route": "GET /api/v1/health",
        "scenario": "Liveness probe",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and body.get("status") == "healthy"
    })
    
    # 3. Ready
    status, body, ms = request("GET", "/api/v1/ready")
    matrix.append({
        "route": "GET /api/v1/ready",
        "scenario": "Readiness probe",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and body.get("status") == "ready" and all(body.get("models_loaded", {}).values())
    })
    
    # 4. Models List
    status, body, ms = request("GET", "/api/v1/models")
    matrix.append({
        "route": "GET /api/v1/models",
        "scenario": "List registered models",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and len(body.get("models", [])) == 3
    })
    
    # 5. Model Detail (Valid)
    status, body, ms = request("GET", "/api/v1/models/densenet121_nih_6disease")
    matrix.append({
        "route": "GET /api/v1/models/{model_id}",
        "scenario": "Get DenseNet121 detail",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and body.get("architecture") == "DenseNet121"
    })
    
    # 6. Model Detail (404 Missing)
    status, body, ms = request("GET", "/api/v1/models/nonexistent_model_id")
    matrix.append({
        "route": "GET /api/v1/models/{model_id}",
        "scenario": "Non-existent model lookup",
        "expected": 404, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "error" in body and body["error"].get("code") == "RESOURCE_NOT_FOUND"
    })
    
    # 7. Research Metrics
    status, body, ms = request("GET", "/api/v1/research/metrics")
    matrix.append({
        "route": "GET /api/v1/research/metrics",
        "scenario": "Historical research benchmarks",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "nih_test_metrics" in body and "vindr_external_metrics" in body
    })
    
    # 8. CT Scan Summary (Valid)
    status, body, ms = request("GET", "/api/v1/ct/demo_thoracic_scan")
    matrix.append({
        "route": "GET /api/v1/ct/{scan_id}",
        "scenario": "CT scan summary descriptor",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and body.get("scan_id") == "demo_thoracic_scan"
    })
    
    # 9. CT Scan Summary (404 Missing)
    status, body, ms = request("GET", "/api/v1/ct/missing_scan_id")
    matrix.append({
        "route": "GET /api/v1/ct/{scan_id}",
        "scenario": "Missing CT scan lookup",
        "expected": 404, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "error" in body and body["error"].get("code") == "RESOURCE_NOT_FOUND"
    })
    
    # 10. CT Metadata
    status, body, ms = request("GET", "/api/v1/ct/demo_thoracic_scan/metadata")
    matrix.append({
        "route": "GET /api/v1/ct/{scan_id}/metadata",
        "scenario": "Full geometric metadata",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "dimensions" in body and "spacing_mm" in body
    })
    
    # 11. CT Slice (Valid Axial)
    status, body, ms = request("GET", "/api/v1/ct/demo_thoracic_scan/slice?view=axial&slice_index=32&window=lung")
    matrix.append({
        "route": "GET /api/v1/ct/{scan_id}/slice",
        "scenario": "Extract axial slice 32 with lung window",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "image_base64" in body and body.get("view") == "axial"
    })
    
    # 12. CT Slice (422 Invalid View)
    status, body, ms = request("GET", "/api/v1/ct/demo_thoracic_scan/slice?view=oblique&slice_index=32")
    matrix.append({
        "route": "GET /api/v1/ct/{scan_id}/slice",
        "scenario": "Invalid anatomical view plane",
        "expected": 422, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "detail" in body
    })
    
    # 13. CT Mesh Stream (Valid OBJ)
    status, body, ms = request("GET", "/api/v1/ct/demo_thoracic_scan/mesh?structure=lungs&step=2&pad_boundary=true")
    matrix.append({
        "route": "GET /api/v1/ct/{scan_id}/mesh",
        "scenario": "Stream Wavefront OBJ 3D mesh",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, str) and body.startswith("# LungVision AI 3D Mesh Export") and "v " in body and "f " in body
    })
    
    # 14. CT Mesh Quality
    status, body, ms = request("GET", "/api/v1/ct/demo_thoracic_scan/mesh/quality?structure=lungs")
    matrix.append({
        "route": "GET /api/v1/ct/{scan_id}/mesh/quality",
        "scenario": "Trimesh manifold & watertight audit",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "watertight" in body and "euler_number" in body
    })
    
    # 15. CT Annotations
    status, body, ms = request("GET", "/api/v1/ct/demo_thoracic_scan/annotations")
    matrix.append({
        "route": "GET /api/v1/ct/{scan_id}/annotations",
        "scenario": "LUNA16 mapped nodule coordinates",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "annotations" in body and body.get("total_annotations", 0) > 0
    })
    
    # 16. CT Measurements
    status, body, ms = request("GET", "/api/v1/ct/demo_thoracic_scan/measurements")
    matrix.append({
        "route": "GET /api/v1/ct/{scan_id}/measurements",
        "scenario": "Quantitative lung volumetry & airway analysis",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "lungs" in body and "airway" in body and "label_audit" in body
    })
    
    # 17. CT Segmentation
    status, body, ms = request("GET", "/api/v1/ct/demo_thoracic_scan/segmentation")
    matrix.append({
        "route": "GET /api/v1/ct/{scan_id}/segmentation",
        "scenario": "Structural segmentation metadata descriptor",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and body.get("scan_id") == "demo_thoracic_scan"
    })
    
    # 18. X-Ray Predict (Valid with real CXR image)
    real_cxr_path = "LungVision_AI_Final_Research_Bundle/external_validation_readiness/vindr_dicom_preview.png"
    with open(real_cxr_path, "rb") as f:
        real_cxr_bytes = f.read()
    
    body_data, headers = multipart_encode({}, {"file": ("vindr_test.png", real_cxr_bytes, "image/png")})
    status, body, ms = request("POST", "/api/v1/xray/predict", data=body_data, headers=headers)
    matrix.append({
        "route": "POST /api/v1/xray/predict",
        "scenario": "DenseNet121 6-disease inference on real 2544x2516 CXR",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and body.get("model", {}).get("architecture") == "DenseNet121" and len(body.get("predictions", {})) == 6
    })
    
    # 19. X-Ray Predict (400 Corrupted Image)
    corrupted_data, headers = multipart_encode({}, {"file": ("corrupted.png", b"\x89PNG\r\n\x1a\ncorrupted", "image/png")})
    status, body, ms = request("POST", "/api/v1/xray/predict", data=corrupted_data, headers=headers)
    matrix.append({
        "route": "POST /api/v1/xray/predict",
        "scenario": "Reject corrupted image file",
        "expected": 400, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "error" in body and body["error"].get("code") in ["FILE_VALIDATION_ERROR", "INVALID_IMAGE"]
    })
    
    # 20. X-Ray Binary (Valid with Original ResNet18)
    body_data, headers = multipart_encode({"model_id": "original_resnet18"}, {"file": ("vindr_test.png", real_cxr_bytes, "image/png")})
    status, body, ms = request("POST", "/api/v1/xray/predict-binary", data=body_data, headers=headers)
    matrix.append({
        "route": "POST /api/v1/xray/predict-binary",
        "scenario": "Original ResNet18 binary inference on real CXR",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and body.get("prediction", {}).get("classification") in ["Normal", "Abnormal"]
    })
    
    # 21. X-Ray Binary (Valid with Blur-Augmented ResNet18)
    body_data, headers = multipart_encode({"model_id": "blur_augmented_resnet18"}, {"file": ("vindr_test.png", real_cxr_bytes, "image/png")})
    status, body, ms = request("POST", "/api/v1/xray/predict-binary", data=body_data, headers=headers)
    matrix.append({
        "route": "POST /api/v1/xray/predict-binary",
        "scenario": "Blur ResNet18 binary inference on real CXR",
        "expected": 200, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and body.get("prediction", {}).get("classification") in ["Normal", "Abnormal"]
    })
    
    # 22. X-Ray Binary (404 Unknown Model Name)
    body_data, headers = multipart_encode({"model_id": "unknown_model"}, {"file": ("vindr_test.png", real_cxr_bytes, "image/png")})
    status, body, ms = request("POST", "/api/v1/xray/predict-binary", data=body_data, headers=headers)
    matrix.append({
        "route": "POST /api/v1/xray/predict-binary",
        "scenario": "Reject unrecognized binary model name",
        "expected": 404, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "error" in body and body["error"].get("code") == "RESOURCE_NOT_FOUND"
    })
    
    # 23. CT Upload (400 Non-archive payload)
    body_data, headers = multipart_encode({}, {"file": ("fake.zip", b"Not a real zip archive content", "application/zip")})
    status, body, ms = request("POST", "/api/v1/ct/upload", data=body_data, headers=headers)
    matrix.append({
        "route": "POST /api/v1/ct/upload",
        "scenario": "Reject fake CT archive missing ZIP magic bytes",
        "expected": 400, "actual": status, "latency_ms": round(ms, 2),
        "valid_schema": isinstance(body, dict) and "error" in body and body["error"].get("code") in ["FILE_VALIDATION_ERROR", "INVALID_ARCHIVE"]
    })

    print(f"Executed {len(matrix)} comprehensive route scenario checks.")
    passed = 0
    for idx, item in enumerate(matrix, 1):
        verdict = "PASS" if item["actual"] == item["expected"] and item["valid_schema"] else "FAIL"
        if verdict == "PASS":
            passed += 1
        print(f"[{verdict}] Test {idx:2}: {item['route']:32} | Scenario: {item['scenario'][:42]:42} | Exp: {item['expected']} Got: {item['actual']} ({item['latency_ms']}ms)")
    
    print(f"\nFinal Result: {passed}/{len(matrix)} Passed ({passed/len(matrix)*100:.1f}%)")
    
    with open("frontend/audit/api_verification_matrix.json", "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

if __name__ == "__main__":
    run_tests()
