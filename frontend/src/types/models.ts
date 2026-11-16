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
  active_threshold_set?: string;
  notes: string[];
  disclaimer?: string;
}
