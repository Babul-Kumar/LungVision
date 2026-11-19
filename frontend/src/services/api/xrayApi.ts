import { apiClient } from './client';
import { XRayBinaryResponse, XRayPredictionResponse } from '@/types/xray';

export interface Phase7PredictionResponse {
  status: 'COMPLETED' | 'REJECTED';
  findings?: Record<
    string,
    {
      disease: string;
      probability: number;
      threshold: number;
      is_positive: boolean;
      margin: number;
      confidence_level: 'HIGH' | 'MODERATE' | 'LOW' | 'BORDERLINE';
      clinical_note: string;
    }
  >;
  uncertainty?: {
    overall_confidence: string;
    mean_margin: number;
    borderline_diseases: string[];
    positive_findings: string[];
    recommendation: string;
    recommendation_reason: string;
  };
  quality_report?: {
    status: string;
    can_proceed: boolean;
    metrics: Record<string, any>;
    warnings: string[];
    rejection_reason?: string;
  };
  gradcam?: {
    disease: string;
    target_idx: number;
    overlay_base64: string;
    heatmap_base64: string;
  };
  duration_ms: number;
  model_metadata?: Record<string, any>;
  timestamp?: string;
  error?: string;
}

export const xrayApi = {
  predictMultiLabel: async (
    file: File,
    modelId: string = 'lungvision_phase7_msd121_224_direct'
  ): Promise<XRayPredictionResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', modelId);

    const res = await apiClient.post<XRayPredictionResponse>('/api/v1/xray/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  predictPhase7: async (
    file: File,
    operatingPoint: string = 'operating_point_A_f1_optimal',
    generateCam: boolean = false,
    camTargetDisease?: string
  ): Promise<Phase7PredictionResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('operating_point', operatingPoint);
    formData.append('generate_cam', String(generateCam));
    if (camTargetDisease) {
      formData.append('cam_target_disease', camTargetDisease);
    }

    const res = await apiClient.post<Phase7PredictionResponse>('/api/v1/predictions', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  predictBinary: async (
    file: File,
    modelId: 'original_resnet18' | 'blur_augmented_resnet18' = 'original_resnet18'
  ): Promise<XRayBinaryResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', modelId);

    const res = await apiClient.post<XRayBinaryResponse>('/api/v1/xray/predict-binary', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  getModelHealth: async () => {
    const res = await apiClient.get('/api/v1/model/health');
    return res.data;
  },

  getModelInfo: async () => {
    const res = await apiClient.get('/api/v1/model/info');
    return res.data;
  },
};
