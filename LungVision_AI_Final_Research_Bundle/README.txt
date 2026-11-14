LUNGVISION AI — FINAL RESEARCH BUNDLE
====================================

Generated UTC:
2026-11-07T00:13:28.438453+00:00

PROJECT STATUS
--------------

Experimental AI-assisted chest X-ray research platform.

The included binary models are frozen research checkpoints.
They are not clinically validated diagnostic systems.

CONTENTS
--------

final_research_evidence/
    Consolidated report, artifact inventory, hashes.

vindr_external_evaluation/
    Metrics, bootstrap intervals, threshold transfer,
    paired comparisons, and plots.

vindr_finding_level_analysis/
    Finding-level miss rates, consensus analysis,
    radiologist boxes, Grad-CAM, and occlusion experiments.

external_validation_readiness/
    VinDr dataset manifest and preprocessing audits.

vindr_frozen_external_inference/
    Complete 15,000-image prediction CSV and related outputs.

models/
    Original ResNet18 checkpoint.
    Blur-Augmented ResNet18 checkpoint.

model_registry_documentation/
    Available small model registry documentation.

bundle_file_manifest.csv
    SHA-256 integrity manifest for included files.

SCIENTIFIC LIMITATIONS
----------------------

1. The main classifier predicts Normal versus Abnormal.
2. VinDr binary labels are derived from three-reader training
   annotations, not the separate five-reader consensus test set.
3. VinDr confidence intervals are image-level because the
   inspected DICOM metadata did not provide patient identifiers.
4. External performance is modest and demonstrates domain
   generalization limitations.
5. Grad-CAM and occlusion experiments are exploratory.
6. The 3D research branch has not been validated using paired
   patient-level X-ray and CT data.
7. No prospective clinical validation has been completed.

The archived models and metrics must not be represented as
clinically validated diagnostic performance.

For reproducibility, verify file hashes using
bundle_file_manifest.csv.
