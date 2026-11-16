#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.xray_service import XRayService
from backend.app.services.model_loader import model_loader

model_loader.load_all_models()
svc = XRayService()

test_files = [
    'test/Atelectasis.jpg',
    'test/Cardiomegaly.jpg',
    'test/Edema.jpg',
    'test/Lung-Consolidation-x-ray-image.webp',
    'test/pneumonia.png',
    'test/Pneumothorax.jpeg'
]

print(f"{'Image':<38} | {'Cardio':<7} | {'Edema':<7} | {'Consol':<7} | {'Pneumo':<7} | {'Atelec':<7} | {'Pnx':<7} | Positive Flags")
print("-" * 110)

for tf in test_files:
    p = Path(tf)
    data = p.read_bytes()
    res = svc.predict_multilabel(data)
    preds = {d: p_obj.probability for d, p_obj in res.predictions.items()}
    flags = [d for d, p_obj in res.predictions.items() if p_obj.experimental_flag]
    print(f"{p.name:<38} | {preds['Cardiomegaly']:>6.2%} | {preds['Edema']:>6.2%} | {preds['Consolidation']:>6.2%} | {preds['Pneumonia']:>6.2%} | {preds['Atelectasis']:>6.2%} | {preds['Pneumothorax']:>6.2%} | {flags}")
