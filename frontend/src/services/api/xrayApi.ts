import { apiClient } from './client';
import { XRayBinaryResponse, XRayPredictionResponse } from '@/types/xray';

export const xrayApi = {
  predictMultiLabel: async (file: File): Promise<XRayPredictionResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await apiClient.post<XRayPredictionResponse>('/api/v1/xray/predict', formData, {
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
};

