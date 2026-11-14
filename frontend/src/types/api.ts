/**
 * API common types and RFC 7807 problem details
 */

export interface ErrorDetails {
  code: string;
  message: string;
  timestamp: string;
  details?: Record<string, any>;
}

export interface ApiErrorResponse {
  error: ErrorDetails;
}

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
}

export interface ReadinessResponse {
  status: 'ready' | 'degraded';
  device: string;
  models_loaded: {
    densenet121_6disease: boolean;
    resnet18_original: boolean;
    resnet18_augmented: boolean;
  };
  timestamp: string;
}

