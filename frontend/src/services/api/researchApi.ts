import { apiClient } from './client';
import { ResearchMetricsResponse } from '@/types/models';

export const researchApi = {
  getResearchMetrics: async (): Promise<ResearchMetricsResponse> => {
    const res = await apiClient.get<ResearchMetricsResponse>('/api/v1/research/metrics');
    return res.data;
  },
};

