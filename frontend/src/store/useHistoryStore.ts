import { create } from 'zustand';
import { XRayBinaryResponse, XRayPredictionResponse } from '@/types/xray';

export interface HistoryItem {
  id: string;
  type: 'xray_multilabel' | 'xray_binary' | 'ct_session';
  timestamp: string;
  filename: string;
  resultSummary: string;
  data: XRayPredictionResponse | XRayBinaryResponse | any;
}

interface HistoryState {
  history: HistoryItem[];
  isPersistent: boolean;
  addHistoryItem: (item: Omit<HistoryItem, 'id' | 'timestamp'>) => void;
  clearHistory: () => void;
  setPersistence: (enabled: boolean) => void;
}

const STORAGE_KEY = 'lungvision_session_history';
const PREF_KEY = 'lungvision_persist_pref';

function sanitizeHistoryData(data: any): any {
  if (!data || typeof data !== 'object') return data;
  if (Array.isArray(data)) {
    return data.slice(0, 50).map(sanitizeHistoryData);
  }
  const clean: Record<string, any> = {};
  for (const [key, value] of Object.entries(data)) {
    // Exclude large base64 strings and bulky 3D geometry arrays to prevent QuotaExceededError
    if (
      key === 'image_base64' ||
      key === 'heatmap_base64' ||
      key === 'overlay_base64' ||
      key === 'drr_image_base64' ||
      key === 'previewUrl' ||
      key === 'vertices' ||
      key === 'faces'
    ) {
      continue;
    }
    if (typeof value === 'string' && (value.startsWith('data:image/') || value.length > 2048)) {
      continue;
    }
    if (typeof value === 'object' && value !== null) {
      clean[key] = sanitizeHistoryData(value);
    } else {
      clean[key] = value;
    }
  }
  return clean;
}

const getInitialPersistence = (): boolean => {
  try {
    return localStorage.getItem(PREF_KEY) === 'true';
  } catch {
    return false;
  }
};

const loadSavedHistory = (): HistoryItem[] => {
  try {
    const isPersistent = getInitialPersistence();
    const raw = isPersistent
      ? localStorage.getItem(STORAGE_KEY)
      : sessionStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
};

export const useHistoryStore = create<HistoryState>((set, get) => ({
  history: loadSavedHistory(),
  isPersistent: getInitialPersistence(),

  setPersistence: (enabled: boolean) => {
    try {
      localStorage.setItem(PREF_KEY, String(enabled));
      if (!enabled) {
        localStorage.removeItem(STORAGE_KEY);
        // Transfer to session storage
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(get().history));
      } else {
        sessionStorage.removeItem(STORAGE_KEY);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(get().history));
      }
    } catch {
      // Ignored
    }
    set({ isPersistent: enabled });
  },

  addHistoryItem: (item) => {
    // Sanitize filename to avoid storing raw client filesystem paths
    const cleanFilename = item.filename
      ? item.filename.replace(/^.*[\\/]/, '')
      : 'unnamed_scan.png';

    const newItem: HistoryItem = {
      ...item,
      data: sanitizeHistoryData(item.data),
      filename: cleanFilename,
      id: Math.random().toString(36).substring(2, 9),
      timestamp: new Date().toISOString(),
    };

    set((state) => {
      const updated = [newItem, ...state.history].slice(0, 50); // Keep last 50 items
      try {
        const payload = JSON.stringify(updated);
        if (state.isPersistent) {
          localStorage.setItem(STORAGE_KEY, payload);
        } else {
          sessionStorage.setItem(STORAGE_KEY, payload);
        }
      } catch {
        // Fallback if quota exceeded
      }
      return { history: updated };
    });
  },

  clearHistory: () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
      sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignored
    }
    set({ history: [] });
  },
}));
