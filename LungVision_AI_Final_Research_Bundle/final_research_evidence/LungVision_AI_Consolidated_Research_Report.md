
# LungVision AI — Consolidated Research Report

**Generated:** 2026-11-07T00:10:19.084782+00:00

**Status:** Experimental research pipeline; not clinically validated.

---

## 1. Research objective

LungVision AI investigates deep-learning-based chest X-ray
classification, robustness, external generalization, explainability,
and a separate CT-based 3D research component.

The binary classification task predicts Normal versus Abnormal.
Finding-level analyses use radiologist annotations to characterize
model errors. They do not establish disease-specific diagnosis.

## 2. Dataset evidence

### IN-CXR

- 12,600 PNG images.
- Balanced Normal and Abnormal classes.
- Used for initial ResNet18 development and internal evaluation.

### NIH expert-reviewed cohort

- 810 expert-reviewed PA images.
- Five independent reader assessments per image.
- Three designated readers contribute to the supplied
  majority-vote ground truth.
- A patient-disjoint subset was used for the earlier evaluation.

### VinDr-CXR

- 18,000 DICOM images available.
- 15,000 annotated training images used for external evaluation.
- 67,914 annotation rows.
- 17 radiologists contributed annotations.
- Three readers per labelled image.
- Derived binary cohort: 10,606 Normal and 4,394 Abnormal.
- 14 finding categories plus No Finding.
- 14,577 images had Window Center/Width.
- 423 images required a percentile fallback.

The VinDr binary label is derived from the three-reader training
annotations. It is not the separate five-reader consensus
ground truth for the competition test set.

No patient identifiers were available in the inspected VinDr
DICOM metadata. VinDr confidence intervals are therefore
image-level, not patient-clustered.

## 3. Frozen model configurations

The evaluated configurations are:

1. Original ResNet18 + Baseline.
2. Original ResNet18 + CLAHE.
3. Blur-Augmented ResNet18 + Baseline.
4. Blur-Augmented ResNet18 + CLAHE.

No retraining was performed during the VinDr external inference
or threshold-transfer experiments.

## 4. VinDr external evaluation

### Full-cohort metrics

| model | ROC_AUC | Average_Precision | Sensitivity | Specificity | Accuracy | F1 | Brier_Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original + Baseline | 0.6015 | 0.3467 | 0.2726 | 0.7782 | 0.6301 | 0.3016 | 0.2585 |
| Original + CLAHE | 0.6005 | 0.3516 | 0.4508 | 0.6491 | 0.5910 | 0.3924 | 0.3002 |
| Augmented + Baseline | 0.6043 | 0.3451 | 0.5919 | 0.5589 | 0.5686 | 0.4456 | 0.3288 |
| Augmented + CLAHE | 0.6134 | 0.3594 | 0.7328 | 0.4449 | 0.5293 | 0.4770 | 0.3767 |

### Locked threshold transfer

| model | threshold_source | threshold | sensitivity | specificity | precision | F1 | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original + Baseline | Default 0.50 | 0.5000 | 0.2726 | 0.7782 | 0.3375 | 0.3016 | 0.6301 |
| Original + Baseline | IN-CXR locked | 0.1765 | 0.6329 | 0.5292 | 0.3577 | 0.4571 | 0.5596 |
| Original + CLAHE | Default 0.50 | 0.5000 | 0.4508 | 0.6491 | 0.3474 | 0.3924 | 0.5910 |
| Augmented + Baseline | Default 0.50 | 0.5000 | 0.5919 | 0.5589 | 0.3573 | 0.4456 | 0.5686 |
| Augmented + Baseline | IN-CXR locked | 0.1879 | 0.8466 | 0.3472 | 0.3495 | 0.4947 | 0.4935 |
| Augmented + CLAHE | Default 0.50 | 0.5000 | 0.7328 | 0.4449 | 0.3536 | 0.4770 | 0.5293 |

Thresholds were transferred from IN-CXR validation without
recalibration. They are research thresholds, not clinical
operating thresholds.

## 5. Finding-level error analysis

The binary classifier was evaluated among images carrying
individual radiologist finding annotations.

Three-reader support indicates that all three training readers
annotated the particular finding. It does not imply pathology
confirmation or five-reader adjudication.

### Augmented ResNet18 + CLAHE, three-reader subset

| finding | positive_images | positive_images_missed | miss_rate_pct |
| --- | --- | --- | --- |
| Cardiomegaly | 1297 | 445 | 34.3099 |
| Aortic enlargement | 1744 | 551 | 31.5940 |
| Pneumothorax | 45 | 11 | 24.4444 |
| Nodule/Mass | 181 | 38 | 20.9945 |
| Pleural effusion | 439 | 91 | 20.7289 |
| Pulmonary fibrosis | 633 | 102 | 16.1137 |
| Lung Opacity | 167 | 26 | 15.5689 |
| Other lesion | 127 | 19 | 14.9606 |
| Calcification | 57 | 8 | 14.0351 |
| Atelectasis | 18 | 2 | 11.1111 |
| Infiltration | 91 | 10 | 10.9890 |
| Pleural thickening | 365 | 39 | 10.6849 |
| Consolidation | 31 | 3 | 9.6774 |
| ILD | 71 | 2 | 2.8169 |

These miss rates are not disease-specific diagnostic
sensitivities. A single image may carry multiple findings.

## 6. Bounding-box inspection

Selected false-negative images were visualized with radiologist
bounding boxes.

- Ten selected image-finding records.
- Nine unique images.
- Thirty-five annotation boxes checked.
- Thirty-five valid boxes.
- Zero invalid boxes.

The visualizations show annotated regions, not model attention.

## 7. Grad-CAM attribution audit

Grad-CAM was generated for the model's Normal-class decision
on the selected false-negative cases.

### CAM mass inside radiologist boxes

| model | mean_cam_mass_inside_boxes | median_cam_mass_inside_boxes | mean_box_area | mean_cam_mass_enrichment |
| --- | --- | --- | --- | --- |
| Augmented + CLAHE | 6.7903 | 5.2322 | 6.8525 | 1.2255 |
| Original + CLAHE | 7.6152 | 5.1958 | 6.8525 | 1.2806 |

The small selected sample and modest overlap do not establish
validated localization or explanation reliability.

## 8. Occlusion faithfulness experiment

Nine unique images were tested using:

- High-CAM occlusion.
- Low-CAM occlusion.
- Random occlusion controls.

Seven of 49 image patches were masked in each occlusion
condition. Twenty random controls were generated per
image and model.

### Mean abnormal-probability changes

| model | method | cases | mean_delta_abnormal_probability | median_delta_abnormal_probability | positive_probability_change_pct |
| --- | --- | --- | --- | --- | --- |
| Augmented + CLAHE | High-CAM occlusion | 9 | 0.4462 | 0.3535 | 100.0000 |
| Augmented + CLAHE | Low-CAM occlusion | 9 | 0.0412 | 0.0326 | 100.0000 |
| Augmented + CLAHE | Random occlusion mean | 9 | 0.4245 | 0.4382 | 100.0000 |
| Original + CLAHE | High-CAM occlusion | 9 | 0.2254 | 0.2020 | 100.0000 |
| Original + CLAHE | Low-CAM occlusion | 9 | 0.0213 | 0.0123 | 100.0000 |
| Original + CLAHE | Random occlusion mean | 9 | 0.2647 | 0.2120 | 100.0000 |

### Paired high-CAM versus random comparison

| model | mean_high_minus_random | median_high_minus_random | images_high_cam_better | images_evaluated |
| --- | --- | --- | --- | --- |
| Augmented + CLAHE | 0.0217 | 0.1128 | 5 | 9 |
| Original + CLAHE | -0.0393 | -0.0030 | 3 | 9 |

High-CAM occlusion had a larger effect than low-CAM occlusion,
but did not consistently outperform random occlusion.
Occlusion may also introduce out-of-distribution artifacts.

The Grad-CAM results must therefore be treated as exploratory.

## 9. Model checkpoint integrity

| model | exists | size_bytes | sha256 |
| --- | --- | --- | --- |
| Original ResNet18 | True | 134260367 | a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f |
| Blur-Augmented ResNet18 | True | 134263007 | 7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24 |

## 10. Research limitations

1. External ROC-AUC remains modest.
2. The current classifier is binary, not a 14-disease diagnostic model.
3. Dataset shift affects both probability calibration and threshold
   transfer.
4. VinDr lacks patient identifiers in the inspected DICOM metadata.
5. VinDr binary ground truth is derived from three-reader annotations.
6. Finding-level errors are not disease-specific diagnostic
   sensitivities.
7. Grad-CAM overlap is not validated localization.
8. Occlusion results use a small, selected false-negative sample.
9. The 2D-to-3D branch has not been validated using paired
   patient-level X-ray and CT data.
10. No prospective clinical evaluation or clinical deployment
    has been completed.

## 11. Overall conclusion

LungVision AI has a functioning research pipeline for binary
chest X-ray classification, model robustness analysis, external
evaluation and finding-level error characterization.

The strongest result is the systematic identification of
cross-dataset generalization limitations.

The system remains experimental. Further model development,
independent external validation, clinician evaluation and
appropriate clinical and regulatory review would be required
before any clinical use.

---

## 12. Evidence inventory

The accompanying JSON file contains the saved artifact inventory,
checkpoint hashes, and report generation metadata.

**Generated by Cell 28.28.**
