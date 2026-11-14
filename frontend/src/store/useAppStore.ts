import { create } from 'zustand';
import { HealthResponse, ReadinessResponse } from '@/types/api';
import { healthApi } from '@/services/api/healthApi';

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
}

interface AppState {
  health: HealthResponse | null;
  readiness: ReadinessResponse | null;
  isBackendOnline: boolean;
  activeScanId: string;
  userExperienceMode: 'patient' | 'doctor';
  notifications: Notification[];
  fetchHealth: () => Promise<void>;
  setActiveScanId: (id: string) => void;
  setUserExperienceMode: (mode: 'patient' | 'doctor') => void;
  addNotification: (type: Notification['type'], message: string) => void;
  removeNotification: (id: string) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  health: null,
  readiness: null,
  isBackendOnline: false,
  activeScanId: 'demo_thoracic_scan',
  userExperienceMode: 'patient',
  notifications: [],

  fetchHealth: async () => {
    try {
      const [h, r] = await Promise.all([
        healthApi.getHealth().catch(() => null),
        healthApi.getReady().catch(() => null),
      ]);
      set({
        health: h,
        readiness: r,
        isBackendOnline: !!(h && h.status === 'healthy'),
      });
    } catch {
      set({
        health: null,
        readiness: null,
        isBackendOnline: false,
      });
    }
  },

  setActiveScanId: (id: string) => set({ activeScanId: id }),

  setUserExperienceMode: (mode: 'patient' | 'doctor') => set({ userExperienceMode: mode }),

  addNotification: (type, message) => {
    const id = Math.random().toString(36).substring(2, 9);
    set((state) => ({
      notifications: [...state.notifications, { id, type, message }],
    }));
    setTimeout(() => {
      get().removeNotification(id);
    }, 5000);
  },

  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),
}));

