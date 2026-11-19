import { apiClient } from './client';
import {
  CTAnnotationsResponse,
  CTMeasurementResponse,
  CTMeshQualityResponse,
  CTMetadataResponse,
  CTSliceResponse,
  CTUploadResponse,
  DRRProjectionResponse,
} from '@/types/ct';

export const ctApi = {
  uploadCT: async (file: File): Promise<CTUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await apiClient.post<CTUploadResponse>('/api/v1/ct/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  getMetadata: async (scanId: string): Promise<CTMetadataResponse> => {
    const res = await apiClient.get<CTMetadataResponse>(`/api/v1/ct/${scanId}/metadata`);
    return res.data;
  },

  getSlice: async (
    scanId: string,
    view: 'axial' | 'coronal' | 'sagittal' = 'axial',
    sliceIndex = 32,
    window = 'lung',
    overlayMask = true,
    showNodules = true
  ): Promise<CTSliceResponse> => {
    const params = new URLSearchParams({
      view,
      slice_index: sliceIndex.toString(),
      window,
      overlay_mask: overlayMask.toString(),
      show_nodules: showNodules.toString(),
    });

    const res = await apiClient.get<CTSliceResponse>(`/api/v1/ct/${scanId}/slice?${params.toString()}`);
    return res.data;
  },

  getMeshObj: async (
    scanId: string,
    structure = 'lungs',
    step = 2,
    padBoundary = true
  ): Promise<string> => {
    const params = new URLSearchParams({
      structure,
      step: step.toString(),
      pad_boundary: padBoundary.toString(),
    });

    // Important: responseType is text for Wavefront OBJ streams
    const res = await apiClient.get<string>(`/api/v1/ct/${scanId}/mesh?${params.toString()}`, {
      responseType: 'text',
      headers: {
        Accept: 'text/plain',
      },
    });
    return res.data;
  },

  getMeshQuality: async (scanId: string, structure = 'left_lung'): Promise<CTMeshQualityResponse> => {
    const res = await apiClient.get<CTMeshQualityResponse>(
      `/api/v1/ct/${scanId}/mesh/quality?structure=${structure}`
    );
    return res.data;
  },

  getAnnotations: async (scanId: string): Promise<CTAnnotationsResponse> => {
    const res = await apiClient.get<CTAnnotationsResponse>(`/api/v1/ct/${scanId}/annotations`);
    return res.data;
  },

  getMeasurements: async (scanId: string): Promise<CTMeasurementResponse> => {
    const res = await apiClient.get<CTMeasurementResponse>(`/api/v1/ct/${scanId}/measurements`);
    return res.data;
  },

  getSegmentation: async (scanId: string): Promise<{ status: string; disclaimer: string }> => {
    const res = await apiClient.get<{ status: string; disclaimer: string }>(`/api/v1/ct/${scanId}/segmentation`);
    return res.data;
  },

  getDRR: async (
    scanId: string,
    view: 'frontal' | 'lateral' = 'frontal',
    sdd_mm = 1800.0,
    sod_mm = 1450.0,
    overlay_silhouette = false,
    add_noise = false
  ): Promise<DRRProjectionResponse> => {
    const params = new URLSearchParams({
      view,
      sdd_mm: sdd_mm.toString(),
      sod_mm: sod_mm.toString(),
      overlay_silhouette: overlay_silhouette.toString(),
      add_noise: add_noise.toString(),
    });

    const res = await apiClient.get<DRRProjectionResponse>(`/api/v1/ct/${scanId}/drr?${params.toString()}`);
    return res.data;
  },

  getInteractiveViewerUrl: (scanId: string): string => {
    const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
    return `${baseUrl}/api/v1/ct/${scanId}/interactive-viewer`;
  },
};


