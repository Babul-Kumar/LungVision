import { apiClient } from './client';
import { ModelInfo, ModelListResponse } from '@/types/models';

export const modelApi = {
  listModels: async (): Promise<ModelListResponse> => {
    const res = await apiClient.get<ModelListResponse>('/api/v1/models');
    return res.data;
  },

  getModel: async (modelId: string): Promise<ModelInfo> => {
    const res = await apiClient.get<ModelInfo>(`/api/v1/models/${modelId}`);
    return res.data;
  },
};

