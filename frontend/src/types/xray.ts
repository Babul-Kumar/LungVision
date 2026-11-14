/**
 * 2D Chest X-Ray inference schemas matching backend/app/schemas/xray.py
 */

export const EXPECTED_6_DISEASES = [
  'Cardiomegaly',
  'Edema',
  'Consolidation',
  'Pneumonia',
  'Atelectasis',
  'Pneumothorax',
] as const;

export type DiseaseName = (typeof EXPECTED_6_DISEASES)[number];

export interface ModelDescriptor {
  id: string;
  architecture: string;
  task: string;
  training_dataset: string;
  checkpoint_epoch: number;
}

export interface DiseasePrediction {
  probability: number;
  threshold: number;
  experimental_flag: boolean;
}

export interface PredictionSummary {
  flagged_conditions_count: number;
  highest_probability_disease: string;
  highest_probability_value: number;
}

export interface XRayPredictionResponse {
  success: boolean;
  model: ModelDescriptor;
  predictions: Record<DiseaseName, DiseasePrediction>;
  summary: PredictionSummary;
  inference_time_ms: number;
  disclaimer: string;
  timestamp: string;
}

export interface BinaryPrediction {
  abnormal_probability: number;
  calibrated_threshold: number;
  classification: 'Normal' | 'Abnormal';
  experimental_flag: boolean;
}

export interface XRayBinaryResponse {
  success: boolean;
  model: ModelDescriptor;
  prediction: BinaryPrediction;
  inference_time_ms: number;
  disclaimer: string;
  timestamp: string;
}

