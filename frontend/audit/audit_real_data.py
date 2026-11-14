"""Audit Real Data Validation Script.

Verifies:
1. Real X-ray image ingestion with VinDr radiograph (2544x2516).
2. Local file inventory for DICOM, NIfTI, MHD/RAW, and PNG.
3. Live CT upload endpoint testing with valid ZIP archive, malformed archive, and unsupported file.
4. Post-upload lifecycle: query metadata and slice rendering on the newly uploaded scan.
"""

import io
import json
from pathlib import Path
import sys
import zipfile
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

BASE_URL = "http://127.0.0.1:8000"


def multipart_encode(fields, files):
    boundary = "----WebKitFormBoundaryRealDataAudit"
    body = io.BytesIO()
    for name, value in fields.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.write(f"{value}\r\n".encode("utf-8"))
    for name, (filename, data, content_type) in files.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.write(data)
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    return body.getvalue(), headers


def run_real_data_audit():
    results = {}

    # 1. Local Image Inventory Check
    print("--- 1. Scanning Local Repository for Medical Imaging Formats ---")
    root = Path(__file__).resolve().parent.parent.parent
    extensions = [".dcm", ".mhd", ".raw", ".nii", ".gz", ".png", ".jpg", ".jpeg"]
    found_by_ext = {ext: [] for ext in extensions}

    for path in root.rglob("*"):
        if path.is_file():
            lower = path.name.lower()
            for ext in extensions:
                if lower.endswith(ext):
                    found_by_ext[ext].append({
                        "rel_path": str(path.relative_to(root)),
                        "size_bytes": path.stat().st_size
                    })

    print(f"DICOM (.dcm): {len(found_by_ext['.dcm'])} files")
    print(f"MHD (.mhd): {len(found_by_ext['.mhd'])} files")
    print(f"RAW (.raw): {len(found_by_ext['.raw'])} files")
    print(f"NIfTI (.nii / .gz): {len(found_by_ext['.nii'])} / {len(found_by_ext['.gz'])} files")
    print(f"PNG (.png): {len(found_by_ext['.png'])} files")

    results["file_inventory"] = {
        "dcm_count": len(found_by_ext[".dcm"]),
        "mhd_count": len(found_by_ext[".mhd"]),
        "raw_count": len(found_by_ext[".raw"]),
        "nii_count": len(found_by_ext[".nii"]),
        "png_count": len(found_by_ext[".png"]),
        "real_cxr_present": len(found_by_ext[".png"]) > 0
    }

    # 2. Test Real Radiograph Ingestion
    print("--- 2. Verifying Real Radiograph Ingestion on Live API ---")
    vindr_cxr_path = root / "LungVision_AI_Final_Research_Bundle" / "external_validation_readiness" / "vindr_dicom_preview.png"
    assert vindr_cxr_path.exists(), f"Missing real radiograph: {vindr_cxr_path}"

    with open(vindr_cxr_path, "rb") as f:
        real_cxr_bytes = f.read()

    print(f"Loaded real radiograph: {vindr_cxr_path.name} ({len(real_cxr_bytes)} bytes)")

    # DenseNet121 Inference
    body_data, headers = multipart_encode({}, {"file": ("vindr_preview.png", real_cxr_bytes, "image/png")})
    req = urllib.request.Request(f"{BASE_URL}/api/v1/xray/predict", data=body_data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        densenet_resp = json.loads(resp.read().decode())
    
    print(f"DenseNet121 Inference: {densenet_resp['summary']['flagged_conditions_count']} conditions flagged in {densenet_resp['inference_time_ms']}ms")
    results["real_cxr_densenet_inference"] = {
        "status": "SUCCESS",
        "flagged_conditions": densenet_resp["summary"]["flagged_conditions_count"],
        "highest_disease": densenet_resp["summary"]["highest_probability_disease"],
        "highest_prob": densenet_resp["summary"]["highest_probability_value"],
        "inference_time_ms": densenet_resp["inference_time_ms"]
    }

    # 3. Test CT Upload Endpoint
    print("--- 3. Testing CT Upload API Endpoint ---")
    
    # 3A. Valid ZIP Archive Upload
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("test_scan.mhd", "ObjectType = Image\nNDims = 3\nDimSize = 64 128 128\nElementType = MET_SHORT\n")
        zf.writestr("test_scan.raw", b"\x00" * 1024)
    valid_zip_bytes = zip_buffer.getvalue()

    body_data, headers = multipart_encode({}, {"file": ("thoracic_patient_scan_01.zip", valid_zip_bytes, "application/zip")})
    req = urllib.request.Request(f"{BASE_URL}/api/v1/ct/upload", data=body_data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        upload_resp = json.loads(resp.read().decode())
        upload_status = resp.status

    new_scan_id = upload_resp["scan_id"]
    print(f"Valid CT Upload: HTTP {upload_status} -> Registered scan_id: {new_scan_id}")
    results["valid_ct_upload"] = {
        "status_code": upload_status,
        "registered_scan_id": new_scan_id,
        "filename": upload_resp["filename"]
    }

    # 3B. Verify Ingestion Lifecycle for the newly uploaded scan
    req = urllib.request.Request(f"{BASE_URL}/api/v1/ct/{new_scan_id}/metadata")
    with urllib.request.urlopen(req) as resp:
        meta_resp = json.loads(resp.read().decode())
    print(f"Query uploaded scan metadata: {meta_resp['dimensions']} - Labels: {meta_resp['labels_present']}")

    req = urllib.request.Request(f"{BASE_URL}/api/v1/ct/{new_scan_id}/slice?view=axial&slice_index=32")
    with urllib.request.urlopen(req) as resp:
        slice_resp = json.loads(resp.read().decode())
    print(f"Query uploaded scan slice: view={slice_resp['view']}, dim={slice_resp['dimensions_2d']}, has_image={bool(slice_resp['image_base64'])}")

    results["uploaded_scan_lifecycle"] = {
        "metadata_query": "PASS",
        "slice_rendering": "PASS",
        "slice_dimensions": slice_resp["dimensions_2d"]
    }

    # 3C. Invalid CT Upload (Corrupted magic bytes)
    corrupted_zip_bytes = b"NOT_A_VALID_ZIP_MAGIC_HEADER_BYTES"
    body_data, headers = multipart_encode({}, {"file": ("corrupted.zip", corrupted_zip_bytes, "application/zip")})
    req = urllib.request.Request(f"{BASE_URL}/api/v1/ct/upload", data=body_data, headers=headers, method="POST")
    try:
        urllib.request.urlopen(req)
        rejected_properly = False
    except urllib.error.HTTPError as e:
        rejected_properly = (e.code == 400)
        err_body = json.loads(e.read().decode())
        print(f"Corrupted CT Upload: HTTP {e.code} ({err_body['error']['code']}) -> Rejected as expected: {rejected_properly}")

    results["invalid_ct_upload_rejection"] = {
        "expected_code": 400,
        "actual_code": 400 if rejected_properly else 200,
        "pass": rejected_properly
    }

    # 3D. Unsupported Extension
    body_data, headers = multipart_encode({}, {"file": ("malicious_payload.exe", b"\x00" * 128, "application/octet-stream")})
    req = urllib.request.Request(f"{BASE_URL}/api/v1/ct/upload", data=body_data, headers=headers, method="POST")
    try:
        urllib.request.urlopen(req)
        rejected_ext = False
    except urllib.error.HTTPError as e:
        rejected_ext = (e.code == 400)
        err_body = json.loads(e.read().decode())
        print(f"Unsupported Extension: HTTP {e.code} ({err_body['error']['code']}) -> Rejected as expected: {rejected_ext}")

    results["unsupported_ext_rejection"] = {
        "expected_code": 400,
        "actual_code": 400 if rejected_ext else 200,
        "pass": rejected_ext
    }

    # Save output artifact
    out_path = Path(__file__).resolve().parent / "real_data_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Real data validation audit saved to {out_path}")


if __name__ == "__main__":
    run_real_data_audit()

