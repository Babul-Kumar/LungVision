import { apiClient } from './client';
import { HealthResponse, ReadinessResponse } from '@/types/api';

export const healthApi = {
  getHealth: async (): Promise<HealthResponse> => {
    const res = await apiClient.get<HealthResponse>('/api/v1/health');
    return res.data;
  },

  getReady: async (): Promise<ReadinessResponse> => {
    const res = await apiClient.get<ReadinessResponse>('/api/v1/ready');
    return res.data;
  },
};

