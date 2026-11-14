#!/usr/bin/env python3
"""
LungVision AI — Comprehensive Repository Forensic Scanner
Script: scripts/verification/forensic_repository_scanner.py
Version: 1.0.0

Executes exhaustive, read-only forensic inspection across all directories,
files, models, archives, notebooks, datasets, and architecture components.
Outputs structured JSON for generating authoritative audit reports.
"""

import hashlib
import json
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def get_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash in 1MB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def scan_directory_tree(root: Path) -> Dict[str, Any]:
    """Exhaustive recursive scan of all directories and files."""
    dir_records = []
    file_records = []
    hash_map = defaultdict(list)
    ext_counter = Counter()

    total_bytes = 0
    total_files = 0
    total_dirs = 0

    # Walk repository
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""

        # Skip scanning deep inside node_modules or .git for individual file details,
        # but record the directory summary
        is_node_modules = "node_modules" in rel_dir
        is_git = ".git" in rel_dir or rel_dir == ".git"

        dir_files_count = len(filenames)
        dir_subdirs_count = len(dirnames)
        dir_bytes = 0

        dir_extensions = Counter()
        notable_files = []

        for f in filenames:
            file_path = Path(dirpath) / f
            try:
                stat = file_path.stat()
                sz = stat.st_size
                dir_bytes += sz
                total_bytes += sz
                total_files += 1

                ext = file_path.suffix.lower()
                dir_extensions[ext] += 1
                ext_counter[ext] += 1

                # If not inside node_modules or .git, track file details
                if not is_node_modules and not is_git:
                    # Compute sha256 for non-huge files (< 200MB) or models/archives
                    should_hash = sz < 100 * 1024 * 1024 or ext in {".pth", ".pt", ".zip", ".png", ".json", ".csv"}
                    file_sha = get_sha256(file_path) if should_hash else "SKIPPED_LARGE_FILE"

                    rel_file = str(file_path.relative_to(root)).replace("\\", "/")
                    if file_sha != "SKIPPED_LARGE_FILE":
                        hash_map[file_sha].append(rel_file)

                    if ext in {".pth", ".pt", ".zip", ".ipynb", ".csv", ".json", ".mhd", ".dcm", ".png"}:
                        notable_files.append({"name": f, "size": sz, "sha256": file_sha})

                    file_records.append({
                        "path": rel_file,
                        "size": sz,
                        "ext": ext,
                        "sha256": file_sha
                    })

            except Exception as e:
                pass

        total_dirs += 1
        dir_records.append({
            "rel_path": rel_dir if rel_dir else ".",
            "file_count": dir_files_count,
            "subdir_count": dir_subdirs_count,
            "size_bytes": dir_bytes,
            "extensions": dict(dir_extensions),
            "notable_files": notable_files[:10],
            "is_node_modules": is_node_modules,
            "is_git": is_git
        })

    return {
        "total_dirs": total_dirs,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "extensions_summary": dict(ext_counter.most_common(30)),
        "directories": dir_records,
        "files": file_records,
        "hash_map": {k: v for k, v in hash_map.items() if len(v) > 1}  # Only duplicates
    }


def scan_model_checkpoints(root: Path) -> List[Dict[str, Any]]:
    """Inspects all model weight files in repository."""
    model_exts = {".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".h5", ".keras", ".joblib", ".pkl", ".engine", ".bin"}
    models = []

    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in model_exts:
            if "node_modules" in str(file_path) or ".git" in str(file_path):
                continue

            rel_path = str(file_path.relative_to(root)).replace("\\", "/")
            sz = file_path.stat().st_size
            sha = get_sha256(file_path)

            info = {
                "rel_path": rel_path,
                "filename": file_path.name,
                "format": file_path.suffix.lower().replace(".", "").upper(),
                "size_bytes": sz,
                "size_mb": round(sz / (1024 * 1024), 2),
                "sha256": sha,
                "arch": "Unknown",
                "params_count": None,
                "keys_summary": [],
                "active_in_backend": False,
                "active_in_frontend": False
            }

            # Identify architecture and role from filename/path
            lower_path = rel_path.lower()
            if "densenet121" in lower_path:
                info["arch"] = "DenseNet121"
                info["model_id"] = "densenet121_nih"
                info["name"] = "NIH Six-Disease Classifier (DenseNet121)"
                info["active_in_backend"] = True
                info["active_in_frontend"] = True
            elif "blur_augmented" in lower_path:
                info["arch"] = "ResNet18"
                info["model_id"] = "blur_augmented_resnet18"
                info["name"] = "Blur-Augmented Binary Classifier (ResNet18)"
                info["active_in_backend"] = True
                info["active_in_frontend"] = True
            elif "original_resnet18" in lower_path:
                info["arch"] = "ResNet18"
                info["model_id"] = "original_resnet18"
                info["name"] = "Baseline Binary Classifier (ResNet18)"
                info["active_in_backend"] = True
                info["active_in_frontend"] = True
            elif "phase2_models" in lower_path or "resnet18_epoch" in lower_path:
                info["arch"] = "ResNet18"
                info["model_id"] = file_path.stem
                info["name"] = f"Phase 2 Intermediate Checkpoint ({file_path.stem})"
                info["active_in_backend"] = False
                info["active_in_frontend"] = False
            else:
                info["model_id"] = file_path.stem
                info["name"] = file_path.stem

            # Try PyTorch inspection safely without executing untrusted code
            if file_path.suffix.lower() in {".pth", ".pt"}:
                try:
                    import torch
                    ckpt = torch.load(file_path, map_location="cpu", weights_only=True)
                    if isinstance(ckpt, dict):
                        info["keys_summary"] = list(ckpt.keys())[:15]
                        if "state_dict" in ckpt:
                            sd = ckpt["state_dict"]
                            info["params_count"] = sum(p.numel() for p in sd.values() if hasattr(p, "numel"))
                        elif any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                            info["params_count"] = sum(p.numel() for p in ckpt.values() if hasattr(p, "numel"))
                    elif hasattr(ckpt, "state_dict"):
                        info["params_count"] = sum(p.numel() for p in ckpt.parameters())
                except Exception as e:
                    info["load_error"] = str(e)

            models.append(info)

    return models


def scan_archives(root: Path) -> List[Dict[str, Any]]:
    """Inspects all ZIP archives in repository."""
    archives = []
    for file_path in root.rglob("*.zip"):
        if "node_modules" in str(file_path) or ".git" in str(file_path):
            continue

        rel_path = str(file_path.relative_to(root)).replace("\\", "/")
        sz = file_path.stat().st_size
        sha = get_sha256(file_path)

        archive_info = {
            "path": rel_path,
            "filename": file_path.name,
            "size_bytes": sz,
            "size_mb": round(sz / (1024 * 1024), 2),
            "sha256": sha,
            "total_entries": 0,
            "entries_by_ext": Counter(),
            "has_models": False,
            "has_datasets": False,
            "has_notebooks": False,
            "sample_entries": [],
            "all_entries": []
        }

        try:
            with zipfile.ZipFile(file_path, "r") as z:
                nl = z.namelist()
                archive_info["total_entries"] = len(nl)
                archive_info["sample_entries"] = nl[:25]
                archive_info["all_entries"] = nl

                for name in nl:
                    ext = Path(name).suffix.lower()
                    archive_info["entries_by_ext"][ext] += 1
                    if ext in {".pth", ".pt", ".onnx"}:
                        archive_info["has_models"] = True
                    if ext in {".csv", ".tsv", ".json", ".parquet", ".mhd", ".raw"}:
                        archive_info["has_datasets"] = True
                    if ext == ".ipynb":
                        archive_info["has_notebooks"] = True

                archive_info["entries_by_ext"] = dict(archive_info["entries_by_ext"])

        except Exception as e:
            archive_info["error"] = str(e)

        archives.append(archive_info)

    return archives


def scan_notebooks(root: Path) -> List[Dict[str, Any]]:
    """Analyzes every Jupyter Notebook in the repository and Downloads directory."""
    notebooks = []
    candidate_paths = list(root.rglob("*.ipynb"))
    # Also check user Downloads for LUNA16 notebooks
    downloads_dir = Path("C:/Users/babul/Downloads")
    if downloads_dir.exists():
        for nb_path in downloads_dir.glob("*.ipynb"):
            if "luna" in nb_path.name.lower() or "lungvision" in nb_path.name.lower():
                candidate_paths.append(nb_path)

    for file_path in candidate_paths:
        if "node_modules" in str(file_path) or ".git" in str(file_path):
            continue

        try:
            rel_path = str(file_path.relative_to(root)).replace("\\", "/")
        except ValueError:
            rel_path = str(file_path).replace("\\", "/")

        sz = file_path.stat().st_size
        sha = get_sha256(file_path)

        nb_info = {
            "path": rel_path,
            "filename": file_path.name,
            "size_kb": round(sz / 1024, 1),
            "sha256": sha,
            "cells_count": 0,
            "code_cells_count": 0,
            "markdown_cells_count": 0,
            "cells_with_outputs": 0,
            "kaggle_paths_found": [],
            "models_referenced": [],
            "datasets_referenced": [],
            "has_executed_outputs": False,
            "summary_titles": []
        }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cells = data.get("cells", [])
            nb_info["cells_count"] = len(cells)

            kaggle_paths = set()
            models_ref = set()
            datasets_ref = set()

            for idx, c in enumerate(cells):
                ctype = c.get("cell_type")
                if ctype == "code":
                    nb_info["code_cells_count"] += 1
                elif ctype == "markdown":
                    nb_info["markdown_cells_count"] += 1

                outs = c.get("outputs", [])
                if outs:
                    nb_info["cells_with_outputs"] += 1
                    nb_info["has_executed_outputs"] = True

                src = "".join(c.get("source", []))
                for line in src.splitlines():
                    line_clean = line.strip()
                    if line_clean.startswith("#"):
                        if len(nb_info["summary_titles"]) < 8 and len(line_clean) > 3:
                            nb_info["summary_titles"].append(line_clean[:80])
                    if "/kaggle/input" in line:
                        for token in line.split():
                            if "/kaggle/input" in token:
                                clean_token = token.strip("\"',()[]{}")
                                kaggle_paths.add(clean_token)
                    if "densenet" in line.lower():
                        models_ref.add("DenseNet121")
                    if "resnet" in line.lower():
                        models_ref.add("ResNet18")
                    for d_kw in ["luna16", "nih", "chexpert", "vindr", "mimic", "in-cxr"]:
                        if d_kw in line.lower():
                            datasets_ref.add(d_kw.upper())

            nb_info["kaggle_paths_found"] = sorted(list(kaggle_paths))
            nb_info["models_referenced"] = sorted(list(models_ref))
            nb_info["datasets_referenced"] = sorted(list(datasets_ref))

        except Exception as e:
            nb_info["error"] = str(e)

        notebooks.append(nb_info)

    return notebooks


def scan_datasets_and_tables(root: Path) -> List[Dict[str, Any]]:
    """Analyzes all tabular data manifests, ground truth CSVs, and annotations."""
    tables = []
    for file_path in root.rglob("*.csv"):
        if "node_modules" in str(file_path) or ".git" in str(file_path):
            continue

        rel_path = str(file_path.relative_to(root)).replace("\\", "/")
        sz = file_path.stat().st_size
        sha = get_sha256(file_path)

        table_info = {
            "path": rel_path,
            "filename": file_path.name,
            "size_kb": round(sz / 1024, 2),
            "sha256": sha,
            "row_count": 0,
            "col_count": 0,
            "columns": []
        }

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.readline().strip().split(",")
                table_info["columns"] = header
                table_info["col_count"] = len(header)
                # Count lines
                cnt = 1 + sum(1 for _ in f)
                table_info["row_count"] = cnt
        except Exception as e:
            table_info["error"] = str(e)

        tables.append(table_info)

    return tables


def scan_medical_images(root: Path) -> List[Dict[str, Any]]:
    """Inspects all medical imaging files and figures."""
    img_exts = {".dcm", ".mhd", ".raw", ".zraw", ".nii", ".gz", ".png", ".jpg", ".jpeg"}
    images = []

    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in img_exts:
            if "node_modules" in str(file_path) or ".git" in str(file_path):
                continue

            rel_path = str(file_path.relative_to(root)).replace("\\", "/")
            sz = file_path.stat().st_size
            ext = file_path.suffix.lower()

            # Skip tiny web icons < 1KB
            if sz < 1024 and ext in {".png", ".jpg"}:
                continue

            images.append({
                "path": rel_path,
                "filename": file_path.name,
                "ext": ext,
                "size_kb": round(sz / 1024, 2),
                "is_drr": "drr" in rel_path.lower(),
                "is_clinical": "vindr" in rel_path.lower() or "cxr" in rel_path.lower(),
                "is_figure": "results" in rel_path.lower() or "figure" in rel_path.lower() or "report" in rel_path.lower()
            })

    return images


def scan_backend_routes(root: Path) -> List[Dict[str, Any]]:
    """Parses FastAPI routes from backend source code."""
    routes = []
    main_py = root / "backend" / "app" / "main.py"
    api_py = root / "backend" / "app" / "api" / "v1" / "api.py"

    endpoints_dir = root / "backend" / "app" / "api" / "v1" / "endpoints"
    if endpoints_dir.exists():
        for ep_file in endpoints_dir.glob("*.py"):
            if ep_file.name == "__init__.py":
                continue
            with open(ep_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                line_s = line.strip()
                if line_s.startswith("@router."):
                    method = line_s.split("@router.")[1].split("(")[0].upper()
                    path_part = line_s.split("(")[1].split(")")[0].strip("\"'")
                    # Find function name
                    func_name = "unknown"
                    for next_line in lines[idx+1:idx+6]:
                        if "def " in next_line:
                            func_name = next_line.split("def ")[1].split("(")[0].strip()
                            break

                    routes.append({
                        "file": ep_file.name,
                        "method": method,
                        "endpoint": path_part,
                        "handler": func_name
                    })

    return routes


def scan_frontend_components(root: Path) -> Dict[str, Any]:
    """Inspects frontend pages, routes, and tests."""
    frontend_dir = root / "frontend"
    pages = []
    components = []
    tests = []

    if frontend_dir.exists():
        src_dir = frontend_dir / "src"
        if src_dir.exists():
            for p in src_dir.rglob("*.tsx"):
                rel = str(p.relative_to(frontend_dir)).replace("\\", "/")
                if "page" in p.name.lower():
                    pages.append(rel)
                else:
                    components.append(rel)

        tests_dir = frontend_dir / "tests"
        if tests_dir.exists():
            for t in tests_dir.rglob("*.test.tsx"):
                tests.append(str(t.relative_to(frontend_dir)).replace("\\", "/"))

    return {
        "pages": pages,
        "components_count": len(components),
        "tests": tests
    }


def main():
    print(f"[Forensic Scanner] Initiating comprehensive scan from root: {REPO_ROOT}")
    start_time = time.perf_counter()

    tree_data = scan_directory_tree(REPO_ROOT)
    models = scan_model_checkpoints(REPO_ROOT)
    archives = scan_archives(REPO_ROOT)
    notebooks = scan_notebooks(REPO_ROOT)
    tables = scan_datasets_and_tables(REPO_ROOT)
    images = scan_medical_images(REPO_ROOT)
    routes = scan_backend_routes(REPO_ROOT)
    frontend_data = scan_frontend_components(REPO_ROOT)

    duration = round(time.perf_counter() - start_time, 2)
    print(f"[Forensic Scanner] Scan completed in {duration} s.")
    print(f"  Directories : {tree_data['total_dirs']}")
    print(f"  Files       : {tree_data['total_files']}")
    print(f"  Total Size  : {tree_data['total_bytes'] / (1024*1024*1024):.2f} GB")
    print(f"  Models      : {len(models)}")
    print(f"  Archives    : {len(archives)}")
    print(f"  Notebooks   : {len(notebooks)}")
    print(f"  CSV Tables  : {len(tables)}")
    print(f"  Med Images  : {len(images)}")
    print(f"  API Routes  : {len(routes)}")

    output_dir = REPO_ROOT / "docs" / "forensic_inventory"
    output_dir.mkdir(parents=True, exist_ok=True)

    master_data = {
        "scan_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scan_duration_seconds": duration,
        "repo_root": str(REPO_ROOT),
        "summary": {
            "total_directories": tree_data["total_dirs"],
            "total_files": tree_data["total_files"],
            "total_size_bytes": tree_data["total_bytes"],
            "total_size_gb": round(tree_data["total_bytes"] / (1024 * 1024 * 1024), 2),
            "model_checkpoints_count": len(models),
            "archives_count": len(archives),
            "notebooks_count": len(notebooks),
            "csv_tables_count": len(tables),
            "medical_images_count": len(images),
            "backend_routes_count": len(routes),
            "frontend_pages_count": len(frontend_data["pages"]),
            "frontend_tests_count": len(frontend_data["tests"])
        },
        "extensions_summary": tree_data["extensions_summary"],
        "models": models,
        "archives": archives,
        "notebooks": notebooks,
        "tables": tables,
        "images": images,
        "backend_routes": routes,
        "frontend": frontend_data,
        "duplicates": tree_data["hash_map"]
    }

    # Save master machine-readable inventory.json
    inv_json = output_dir / "inventory.json"
    with open(inv_json, "w", encoding="utf-8") as f:
        json.dump(master_data, f, indent=2)
    print(f"[Forensic Scanner] Exported machine-readable master inventory: {inv_json}")


if __name__ == "__main__":
    main()

