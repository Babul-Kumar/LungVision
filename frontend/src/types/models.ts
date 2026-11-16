/**
 * Model registry and research benchmark schemas matching backend/app/schemas/models.py
 */

export interface ModelInfo {
  id: string;
  name: string;
  architecture: string;
  task: string;
  checkpoint_path?: string;
  sha256?: string | null;
  input_size?: number[];
  labels: string[];
  status?: string;
  active_threshold_set?: string;
  metrics?: Record<string, any>;
  calibration?: Record<string, any>;
  training_dataset?: string | null;
  validation_auc?: number | null;
  parameters_count?: number | null;
}

export interface ModelListResponse {
  success: boolean;
  message?: string;
  timestamp?: string;
  total_models?: number;
  models: ModelInfo[];
}

export interface NihMetricRow {
  Disease: string;
  Threshold: number;
  AUROC: number;
  AUPRC: number;
  Precision: number;
  Sensitivity: number;
  F1: number;
  Specificity: number;
  TP?: number;
  FP?: number;
  TN?: number;
  FN?: number;
}

export interface ChexpertMetricRow {
  Disease: string;
  Threshold: number;
  Evaluable: number;
  Positive: number;
  Negative: number;
  AUROC: number;
  AUPRC: number;
  Precision: number;
  Sensitivity: number;
  F1: number;
  Specificity: number;
  TP?: number;
  FP?: number;
  TN?: number;
  FN?: number;
}

export interface VindrMetricRow {
  model: string;
  cohort: string;
  ROC_AUC: number;
  Average_Precision: number;
  Accuracy: number;
  Balanced_Accuracy: number;
  Precision: number;
  Sensitivity: number;
  Specificity: number;
  F1: number;
  Brier_Score: number;
  ECE_10_bins: number;
  threshold: number;
  images?: number;
  normal_images?: number;
  abnormal_images?: number;
  positive_prevalence?: number;
}

export interface ManualTestAuditItem {
  test_id: number;
  intended_category: string;
  image_filename: string;
  dataset_source: string;
  patient_study_id: string;
  image_metadata: {
    dimensions: string;
    bit_depth: string;
    format: string;
    orientation: string;
    artifacts_present: string;
  };
  model_inference: {
    reproduced_probabilities: Record<string, number>;
    per_disease_evaluation: Record<string, {
      raw_probability: number;
      locked_threshold: number;
      flagged_positive: boolean;
      delta_to_threshold: number;
    }>;
    observed_ui_summary: string;
    final_prediction_flag: string;
    screening_status: string;
  };
  radiological_investigation: string;
}

export interface Phase4ExperimentManifest {
  experiment_id: string;
  governing_standard: string;
  evaluation_timestamp: string;
  baseline_model: Record<string, any>;
  experimental_model: Record<string, any>;
  experimental_cohorts: Record<string, any>;
  reproducible_seeds: Record<string, any>;
}

export interface ResolutionComparison {
  objective: string;
  methodology: string;
  scientific_preamble: string;
  resolutions_evaluated: Record<string, {
    resolution: string;
    interpolation: string;
    aspect_ratio_mode: string;
    receptive_field_pixel_step: number;
    macro_auroc: number;
    pneumothorax_auroc: number;
    atelectasis_auroc: number;
    cardiomegaly_auroc: number;
    pleural_line_attenuation: string;
    inference_latency_cpu_ms: number;
    peak_ram_mb: number;
  }>;
  reproduction_findings: {
    reported_pneumothorax_auroc_baseline: number;
    reproduced_pneumothorax_auroc_highres_multiscale: number;
    delta_auroc: string;
    paired_bootstrap_p_value: number;
    statistical_verdict: string;
  };
}

export interface Phase4ClasswiseMetrics {
  evaluation_cohort: string;
  models_evaluated: {
    baseline_densenet121_224: Record<string, any>;
    multiscale_densenet121_fpn_512: Record<string, any>;
  };
  statistical_comparison: Record<string, {
    delta_auroc: number;
    delta_auprc: number;
    delta_f1: number;
    paired_permutation_p_value: number;
    statistically_significant: boolean;
  }>;
  macro_summary: {
    baseline_macro_auroc: number;
    multiscale_macro_auroc: number;
    macro_auroc_gain: number;
  };
}

export interface ResearchMetricsResponse {
  success: boolean;
  message?: string;
  timestamp?: string;
  nih_test_metrics: NihMetricRow[];
  chexpert_external_metrics: ChexpertMetricRow[];
  vindr_external_metrics: VindrMetricRow[];
  external_validation_metrics?: Record<string, any> | null;
  threshold_comparison?: Record<string, any> | null;
  manual_tests_audit?: ManualTestAuditItem[] | null;
  expanded_cohort_comparison?: Record<string, any> | null;
  threshold_optimization_report?: Record<string, any> | null;
  uncertain_label_sensitivity?: Record<string, any> | null;
  calibration_analysis?: Record<string, any> | null;
  resolution_ablation?: Record<string, any> | null;
  phase4_experiment_manifest?: Phase4ExperimentManifest | null;
  resolution_comparison?: ResolutionComparison | null;
  loss_function_comparison?: Record<string, any> | null;
  phase4_classwise_metrics?: Phase4ClasswiseMetrics | null;
  phase4_calibration_report?: Record<string, any> | null;
  active_threshold_set?: string;
  notes: string[];
  disclaimer?: string;
}
