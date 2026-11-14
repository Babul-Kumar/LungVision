
# LungVision AI — Model Registry

## Project
LungVision AI

Version: 0.1.0

## Purpose
Research prototype for binary classification of frontal chest X-rays
into Normal and Abnormal categories.

## Models

### Original ResNet18
The baseline model trained using the IN-CXR dataset.

### Blur-Augmented ResNet18
Fine-tuned from the original model using mild Gaussian blur
augmentation and random rotation.

## Class mapping

- 0: Normal
- 1: Abnormal

## Input configuration

- Image dimensions: 224 × 224
- Channels: 3
- Color conversion: RGB
- Normalization: ImageNet mean and standard deviation

## Calibration

Thresholds were selected on the validation dataset with an
exploratory target sensitivity of 90%.

These thresholds are not clinically validated.

## Important limitations

1. Binary Normal/Abnormal classification is not disease-specific.
2. Abnormal predictions do not establish TB or lung cancer.
3. Patient-level data independence has not been established.
4. External clinical validation has not been completed.
5. Blur experiments are artificial robustness experiments.
6. Grad-CAM and occlusion maps are model explanations, not
   lesion segmentations or confirmed disease locations.

## Intended use

Research and development only.

This model is not approved or validated for autonomous
clinical diagnosis or patient-management decisions.

A qualified clinician must interpret medical imaging and use
appropriate confirmatory testing.

## Preservation

The original and augmented model checkpoints are preserved
separately. Neither checkpoint overwrites the other.
