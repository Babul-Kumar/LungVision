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
  metrics?: Record<string, any>;
  calibration?: Record<string, number>;
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

export interface ResearchMetricsResponse {
  success: boolean;
  message?: string;
  timestamp?: string;
  nih_test_metrics: NihMetricRow[];
  chexpert_external_metrics: ChexpertMetricRow[];
  vindr_external_metrics: VindrMetricRow[];
  notes: string[];
  disclaimer?: string;
}
