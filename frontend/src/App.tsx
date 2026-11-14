import React, { useEffect, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppLayout } from '@/components/layout/AppLayout';
import { useAppStore } from '@/store/useAppStore';

const DashboardPage = lazy(() => import('@/features/dashboard/DashboardPage').then(m => ({ default: m.DashboardPage })));
const XrayPage = lazy(() => import('@/features/xray/XrayPage').then(m => ({ default: m.XrayPage })));
const DualViewXRayPage = lazy(() => import('@/features/xray/DualViewXRayPage').then(m => ({ default: m.DualViewXRayPage })));
const CTExplorerPage = lazy(() => import('@/features/ct/CTExplorerPage').then(m => ({ default: m.CTExplorerPage })));
const ResearchPage = lazy(() => import('@/features/research/ResearchPage').then(m => ({ default: m.ResearchPage })));
const ModelsPage = lazy(() => import('@/features/models/ModelsPage').then(m => ({ default: m.ModelsPage })));
const HistoryPage = lazy(() => import('@/features/history/HistoryPage').then(m => ({ default: m.HistoryPage })));
const SettingsPage = lazy(() => import('@/features/settings/SettingsPage').then(m => ({ default: m.SettingsPage })));

export const App: React.FC = () => {
  const { fetchHealth } = useAppStore();

  useEffect(() => {
    // Initial health check on application boot
    fetchHealth();

    // Heartbeat polling every 30 seconds
    const interval = setInterval(() => {
      fetchHealth();
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchHealth]);

  return (
    <BrowserRouter
      future={{
        v7_relativeSplatPath: true,
        v7_startTransition: true,
      }}
    >
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/xray" element={<XrayPage />} />
          <Route path="/dual-view" element={<DualViewXRayPage />} />
          <Route path="/ct-explorer" element={<CTExplorerPage />} />
          <Route path="/research" element={<ResearchPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;

