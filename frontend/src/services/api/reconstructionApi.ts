import { apiClient } from './client';
import {
  ExportOBJRequest,
  SSMDeformRequest,
  SSMDeformResponse,
  SSMModelInfo,
  SSMReconstructionRequest,
  SSMReconstructionResponse,
} from '@/types/reconstruction';

export const reconstructionApi = {
  getModelInfo: async (): Promise<SSMModelInfo> => {
    const res = await apiClient.get<SSMModelInfo>('/api/v1/reconstruction/ssm/model-info');
    return res.data;
  },

  deformShape: async (params: SSMDeformRequest): Promise<SSMDeformResponse> => {
    const res = await apiClient.post<SSMDeformResponse>('/api/v1/reconstruction/ssm/deform', params);
    return res.data;
  },

  fitReconstruction: async (params: SSMReconstructionRequest): Promise<SSMReconstructionResponse> => {
    const res = await apiClient.post<SSMReconstructionResponse>('/api/v1/reconstruction/ssm/fit', params);
    return res.data;
  },

  getTemplateMeshObj: async (): Promise<string> => {
    const res = await apiClient.get<string>('/api/v1/reconstruction/ssm/template-mesh', {
      headers: { Accept: 'text/plain' },
      responseType: 'text',
    });
    return res.data;
  },

  exportReconstructedObj: async (shapeCoefficients: number[]): Promise<string> => {
    const payload: ExportOBJRequest = { shape_coefficients: shapeCoefficients };
    const res = await apiClient.post<string>('/api/v1/reconstruction/ssm/export-obj', payload, {
      headers: { Accept: 'text/plain' },
      responseType: 'text',
    });
    return res.data;
  },
};


