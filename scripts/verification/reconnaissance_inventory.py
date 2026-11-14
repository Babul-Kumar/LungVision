"""Reconnaissance Inventory Script for LungVision AI.

Audits:
1. Complete directory tree and file inventory.
2. Checkpoint hashes and duplicate binary discovery.
3. Large binary artifacts (>1MB).
4. Code import references and path dependencies across backend, frontend, and tests.
5. Duplicate CSV and manifest discovery by SHA-256.
6. Git status and tracked file state.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent.parent


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run():
    print(f"Starting complete repository reconnaissance from: {ROOT}")
    
    # 1. Directory Tree & File Inventory
    all_files = []
    dir_summary = {}
    large_artifacts = []
    hash_to_files = {}

    # File extensions of interest for deep hashing
    hash_exts = {".pth", ".zip", ".png", ".csv", ".json", ".mhd", ".raw", ".dcm", ".ipynb"}

    for path in ROOT.rglob("*"):
        # Skip node_modules and .git for speed and clean reporting
        parts = path.parts
        if "node_modules" in parts or ".git" in parts or "__pycache__" in parts or ".pytest_cache" in parts or "dist" in parts:
            continue
        
        if path.is_file():
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            size = path.stat().st_size
            top_dir = rel.split("/")[0] if "/" in rel else "root"
            
            dir_summary[top_dir] = dir_summary.get(top_dir, 0) + 1
            all_files.append({"path": rel, "size": size})

            if size > 1_000_000: # > 1MB
                large_artifacts.append({"path": rel, "size_mb": round(size / (1024*1024), 2)})

            if path.suffix.lower() in hash_exts and size < 200_000_000: # Hash files < 200MB
                sha = get_sha256(path)
                hash_to_files.setdefault(sha, []).append({"path": rel, "size": size})

    # Find duplicates
    duplicates = {sha: files for sha, files in hash_to_files.items() if len(files) > 1}

    # 2. Checkpoint Hashes specifically
    checkpoints = {}
    for item in all_files:
        if item["path"].endswith(".pth"):
            full_p = ROOT / item["path"]
            checkpoints[item["path"]] = {
                "size_bytes": item["size"],
                "sha256": get_sha256(full_p)
            }

    # 3. Path references in code
    code_references = {}
    search_terms = [
        "lungVision",
        "LungVision_AI_Final_Research_Bundle",
        "LungVision_AI_Model_Registry",
        "lungvision_disease_classification",
        "densenet121_nih_best.pth",
        "original_resnet18",
        "blur_augmented_resnet18",
        "demo_thoracic_scan",
        "results.zip"
    ]

    for term in search_terms:
        matches = []
        for path in (ROOT / "backend").rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if term in text:
                    matches.append(str(path.relative_to(ROOT)).replace("\\", "/"))
            except Exception:
                pass
        for path in (ROOT / "frontend" / "src").rglob("*.ts*"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if term in text:
                    matches.append(str(path.relative_to(ROOT)).replace("\\", "/"))
            except Exception:
                pass
        code_references[term] = matches

    # 4. Git status
    try:
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=str(ROOT), text=True)
    except Exception as e:
        git_status = f"Git error: {e}"

    report = {
        "directory_summary": dir_summary,
        "total_files_scanned": len(all_files),
        "checkpoints": checkpoints,
        "duplicate_hash_groups_count": len(duplicates),
        "duplicate_files_sample": {sha: [f["path"] for f in files] for sha, files in list(duplicates.items())[:15]},
        "large_artifacts": sorted(large_artifacts, key=lambda x: x["size_mb"], reverse=True),
        "code_references": code_references,
        "git_status_lines": len(git_status.splitlines()) if isinstance(git_status, str) else 0
    }

    out_file = ROOT / "scripts" / "verification" / "reconnaissance_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Reconnaissance completed. Report saved to: {out_file}")
    print(f"Total non-ignored files: {len(all_files)}")
    print(f"Top-level directories and file counts:")
    for d, c in sorted(dir_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  {d:40s}: {c:4d} files")
    print(f"\nModel checkpoints identified:")
    for p, meta in checkpoints.items():
        print(f"  {p} ({meta['size_bytes'] / (1024*1024):.1f} MB) -> SHA-256: {meta['sha256'][:16]}...")


if __name__ == "__main__":
    run()

