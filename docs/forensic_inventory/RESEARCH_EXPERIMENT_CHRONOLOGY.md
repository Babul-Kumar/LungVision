# LungVision AI — Research Experiment Chronology & Scientific Evolution

**Audit Date:** November 13, 2026  
**Auditor:** Principal Medical AI Scientist & Research Historian  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Scope:** Chronological reconstruction of research hypotheses, methodology, and breakthroughs across Experiments E1–E10, Control-Meta, Phase 1 SSM, and Phase 2A Pilot.

---

## 1. Chronological Timeline of Scientific Experiments

```
[Phase 1: 2D Baseline Exploration]
E1 (DenseNet121) ──> E2 (ResNet18) ──> E3 (Blur Robustness) ──> E4 (Grad-CAM Audit)
                                                                       │
[Phase 2: Multimodal & 3D Evolution]                                   ▼
E8 (Domain Gap) <── E7 (Kendall-Gal Fusion) <── E6 (Shortcut Audit) <── E5 (Biplanar 2D)
       │
       ▼
E9 (External VinDr) ──> E10 (SSM Feasibility) ──> Phase 1 SSM ──> Phase 2A Real Pilot
```

---

## 2. In-Depth Narrative of Each Research Milestone

### Experiment E1: Multi-Label Thoracic Pathology Screening Baseline
- **Objective:** Establish baseline screening accuracy on frontal chest radiographs across 6 major thoracic pathologies.
- **Architecture:** DenseNet121 with feature transfer from CheXNet / NIH ChestX-ray14.
- **Key Finding:** Achieved mean macro ROC-AUC of 0.887 across Cardiomegaly, Effusion, Infiltration, Mass, Nodule, and Pneumothorax. Revealed class imbalance vulnerability in rare nodule detection.

### Experiment E2: Binary Thoracic Abnormality Classifier
- **Objective:** Train a lightweight binary normal vs. abnormal classifier for emergency triage.
- **Architecture:** ResNet18 backbone with binary cross-entropy loss.
- **Key Finding:** Achieved ROC-AUC of 0.892. Model was highly sensitive to diffuse opacities but suffered false positives from subtle radiographic positioning differences.

### Experiment E3: Motion & Blur Robustness Augmentation
- **Objective:** Harden ResNet18 against patient respiratory motion blur and portable bedside imaging artifacts.
- **Methodology:** Injected randomized Gaussian blur ($\sigma \in [0.5, 2.0]$) and defocus kernels during training.
- **Key Finding:** Maintained ROC-AUC of 0.871 under severe blur, outperforming the baseline model by +14.2% in retention accuracy under degraded resolution.

### Experiment E4: Grad-CAM Saliency Faithfulness Audit
- **Objective:** Evaluate whether Grad-CAM heatmaps genuinely track pathology or latch onto spurious background artifacts.
- **Key Finding:** Found that while dense feature block 4 reliably highlighted cardiomegaly and large effusions, it frequently attributed mass predictions to peripheral rib borders or clavicle edges, demonstrating the need for anatomical landmark guidance.

### Experiment E5: Biplanar (Frontal + Lateral) 2D Representation Fusion
- **Objective:** Measure diagnostic gain when pairing Frontal (PA) and Lateral (LAT) chest radiographs.
- **Methodology:** Dual-stream ResNet backbones with cross-attention feature pooling.
- **Key Finding:** Biplanar fusion increased retrocardiac mass detection sensitivity from 64.2% to 88.5%, proving that lateral views eliminate blind spots occluded by the cardiac silhouette.

### Experiment E6: The Critical Scan-Geometry Shortcut Discovery
- **Objective:** Audit the 3D volume reconstruction prototype (`luna16_x2ct_v4_final.ipynb`).
- **The Shortcut:** The prototype claimed an exceptional lung volume estimation $R^2 > 0.94$. However, rigorous ablation revealed that the network had learned to read the `spacing_zyx` voxel header metadata directly rather than deducing 3D volume from image projections!
- **Scientific Impact:** When voxel spacing was randomized, $R^2$ dropped to 0.31. This landmark discovery caused the team to discard naive 3D regressors and institute true physics-based loss formulations.

### Experiment E7: Kendall & Gal Multi-Task Uncertainty Weighting
- **Objective:** Eliminate manual hyperparameter tuning when balancing 3D silhouette loss, landmark loss, and mesh smoothness.
- **Methodology:** Implemented learnable homoscedastic observation noise variance $\sigma_i^2$.
- **Key Finding:** Dynamically prevented landmark loss gradients from destabilizing the Laplacian mesh smoother, yielding smooth, anatomically plausible surfaces.

### Experiment E8: Cross-Study Domain Gap Analysis
- **Objective:** Evaluate generalizability across different clinical institutions (NIH vs CheXpert vs VinDr).
- **Key Finding:** Detected a 7.8% drop in AUROC caused by differences in manufacturer acquisition windowing, emphasizing the need for standard Hounsfield and density calibration.

### Experiment E9: VinDr-CXR External Clinical Validation
- **Objective:** Test model behavior on 16-bit hospital DICOMs from Southeast Asian patient cohorts.
- **Key Finding:** Highlighted importance of photometric interpretation handling (`MONOCHROME1` vs `MONOCHROME2`) and confirmed high calibration fidelity.

### Experiment E10: 3D Lung Reconstruction Feasibility Study
- **Objective:** Comprehensive evaluation of direct voxel synthesis vs Statistical Shape Models (SSM).
- **Governing Decision:** Direct 2D-to-3D voxel generation is ill-posed and prone to hallucination; **Statistical Shape Model (SSM) deformation constrained by biplanar silhouettes is scientifically superior and clinically safe.**

### Phase 1 SSM & Phase 2A Pilot: Real LUNA16 CT Ingestion
- **Objective:** Establish the foundation for real CT-to-DRR synthesis and mesh extraction.
- **Key Finding:** Validated Beer-Lambert ray casting ($D_{\text{SOD}}=1000\,\text{mm}$, $D_{\text{SID}}=1400\,\text{mm}$) against calibrated thoracic scan, extracting 2-manifold closed surface meshes at ISO -400 HU.
