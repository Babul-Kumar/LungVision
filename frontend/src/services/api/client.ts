import axios, { AxiosError } from 'axios';
import { ApiErrorResponse } from '@/types/api';

// Configurable API base URL, defaulting to IPv4 loopback 127.0.0.1:8000 to prevent IPv6 localhost resolution mismatch on Windows
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30s timeout for heavy volumetric operations
  headers: {
    Accept: 'application/json',
  },
});

export class ApiError extends Error {
  code: string;
  timestamp?: string;
  details?: Record<string, any>;
  statusCode?: number;

  constructor(message: string, code = 'API_ERROR', statusCode?: number, details?: Record<string, any>, timestamp?: string) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
    this.timestamp = timestamp;
  }
}

// Global response interceptor to normalize errors compliant with RFC 7807
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiErrorResponse>) => {
    if (error.response?.data?.error) {
      const err = error.response.data.error;
      throw new ApiError(err.message, err.code, error.response.status, err.details, err.timestamp);
    }
    
    if (error.code === 'ECONNABORTED') {
      throw new ApiError('Request timed out while communicating with medical backend', 'TIMEOUT_ERROR', 408);
    }
    
    if (!error.response) {
      throw new ApiError(
        `Unable to connect to LungVision AI backend at ${API_BASE_URL}. Ensure FastAPI is running on port 8000.`,
        'NETWORK_ERROR',
        503
      );
    }

    throw new ApiError(error.message || 'An unexpected API error occurred', 'HTTP_ERROR', error.response.status);
  }
);

