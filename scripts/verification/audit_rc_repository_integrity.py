import hashlib
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import settings

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):  # 8MB chunks
            sha.update(chunk)
    return sha.hexdigest()

def main():
    report = {
        "checkpoints": {},
        "results_zip": {},
        "research_artifacts": {},
        "notebooks": {},
        "config_resolution": {},
        "status": "PASS"
    }
    
    # 1. Verify Checkpoints
    expected_checkpoints = {
        "densenet121_nih_best.pth": {
            "path": BASE_DIR / "lungvision_disease_classification" / "checkpoints" / "densenet121_nih_best.pth",
            "expected_sha256": "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af",
            "expected_size": 84457897
        },
        "original_resnet18": {
            "path": BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "models" / "original_resnet18" / "model.pth",
            "expected_sha256": "a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f",
            "expected_size": 134260367
        },
        "blur_augmented_resnet18": {
            "path": BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "models" / "blur_augmented_resnet18" / "model.pth",
            "expected_sha256": "7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24",
            "expected_size": 134263007
        }
    }

    for name, info in expected_checkpoints.items():
        p = info["path"]
        if not p.exists():
            report["checkpoints"][name] = {"status": "FAIL", "error": f"File missing at {p}"}
            report["status"] = "FAIL"
            continue
        size = p.stat().st_size
        sha = compute_sha256(p)
        match_sha = (sha == info["expected_sha256"])
        match_size = (size == info["expected_size"])
        report["checkpoints"][name] = {
            "path": str(p.relative_to(BASE_DIR)),
            "size_bytes": size,
            "expected_size": info["expected_size"],
            "sha256": sha,
            "expected_sha256": info["expected_sha256"],
            "sha_match": match_sha,
            "size_match": match_size,
            "status": "PASS" if (match_sha and match_size) else "FAIL"
        }
        if not (match_sha and match_size):
            report["status"] = "FAIL"

    # 2. Verify results.zip
    results_zip = BASE_DIR / "results.zip"
    if results_zip.exists():
        size = results_zip.stat().st_size
        sha = compute_sha256(results_zip)
        report["results_zip"] = {
            "path": "results.zip",
            "size_bytes": size,
            "sha256": sha,
            "status": "PASS"
        }
    else:
        report["results_zip"] = {"status": "FAIL", "error": "results.zip missing"}
        report["status"] = "FAIL"

    # 3. Verify Notebooks
    notebook_paths = [
        BASE_DIR / "backend" / "lungvision-interactive-viewer.ipynb"
    ]
    for nb in notebook_paths:
        exists = nb.exists()
        size = nb.stat().st_size if exists else 0
        report["notebooks"][str(nb.relative_to(BASE_DIR))] = {
            "exists": exists,
            "size_bytes": size,
            "status": "PASS" if exists and size > 0 else "FAIL"
        }
        if not exists:
            report["status"] = "FAIL"

    # 4. Verify Critical Research Manifests & Tables
    artifacts = [
        BASE_DIR / "lungVision" / "nih_validation_thresholds.csv",
        BASE_DIR / "lungVision" / "nih_test_calibrated_metrics.csv",
        BASE_DIR / "lungVision" / "chexpert_external_metrics.csv",
        BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "vindr_external_evaluation" / "vindr_external_metrics.csv",
        BASE_DIR / "LungVision_AI_Model_Registry" / "configuration" / "model_registry.json",
        BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "project_governance" / "PRD.md",
        BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "project_governance" / "ARCHITECTURE.md",
        BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "project_governance" / "RULES.md",
        BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "project_governance" / "DESIGN.md",
        BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "project_governance" / "TASK.md",
        BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "project_governance" / "MEMORY.md",
        BASE_DIR / "docs" / "ARCHITECTURE_OVERVIEW.md",
        BASE_DIR / "docs" / "DEVELOPMENT_SETUP.md",
        BASE_DIR / "docs" / "MODEL_ARTIFACT_INVENTORY.md",
        BASE_DIR / "docs" / "REGRESSION_TEST_REPORT.md",
        BASE_DIR / "docs" / "REPOSITORY_CLEANUP_REPORT.md",
        BASE_DIR / "docs" / "REPOSITORY_STRUCTURE.md",
        BASE_DIR / "docs" / "UI_UX_IMPROVEMENT_REPORT.md"
    ]
    for art in artifacts:
        exists = art.exists()
        size = art.stat().st_size if exists else 0
        report["research_artifacts"][str(art.relative_to(BASE_DIR))] = {
            "exists": exists,
            "size_bytes": size,
            "status": "PASS" if exists and size > 0 else "FAIL"
        }
        if not exists:
            report["status"] = "FAIL"

    # 5. Checkpoint Resolution via Backend Settings
    resolved_densenet = settings.RESOLVED_DENSENET121_PATH
    resolved_res_orig = settings.RESOLVED_RESNET18_ORIGINAL_PATH
    resolved_res_aug = settings.RESOLVED_RESNET18_AUGMENTED_PATH

    report["config_resolution"] = {
        "densenet121": {
            "configured_path": str(settings.DENSENET121_CHECKPOINT_PATH),
            "resolved_path": str(resolved_densenet),
            "exists": resolved_densenet.exists()
        },
        "original_resnet18": {
            "configured_path": str(settings.RESNET18_ORIGINAL_PATH),
            "resolved_path": str(resolved_res_orig),
            "exists": resolved_res_orig.exists()
        },
        "blur_augmented_resnet18": {
            "configured_path": str(settings.RESNET18_AUGMENTED_PATH),
            "resolved_path": str(resolved_res_aug),
            "exists": resolved_res_aug.exists()
        }
    }
    for k, v in report["config_resolution"].items():
        if not v["exists"]:
            report["status"] = "FAIL"

    out_file = BASE_DIR / "scripts" / "verification" / "rc_repository_integrity_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
