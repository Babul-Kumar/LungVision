import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ExternalLink, Settings as SettingsIcon } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';

export const Navbar: React.FC = () => {
  const { health, isBackendOnline, fetchHealth } = useAppStore();

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000); // 15s poll
    return () => clearInterval(interval);
  }, [fetchHealth]);

  return (
    <header className="h-16 border-b border-medical-border bg-medical-surface/90 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-40">
      {/* Brand & Logo */}
      <div className="flex items-center gap-3">
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-medical-cyan to-medical-teal flex items-center justify-center text-medical-bg font-bold shadow-md shadow-cyan-950/40">
            LV
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight text-medical-text group-hover:text-medical-cyan transition-colors">
              LUNGVISION <span className="text-medical-cyan font-normal text-xs uppercase ml-1 px-1.5 py-0.5 rounded bg-medical-card border border-medical-border">AI</span>
            </h1>
            <p className="text-[10px] text-medical-muted tracking-wider uppercase">Thoracic Imaging Intelligence</p>
          </div>
        </Link>
      </div>

      {/* Right Controls: Health & Links */}
      <div className="flex items-center gap-4">
        {/* Backend Health Indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-medical-bg border border-medical-border text-xs">
          <div
            className={`w-2 h-2 rounded-full ${
              isBackendOnline ? 'bg-medical-emerald animate-pulse' : 'bg-medical-rose'
            }`}
          />
          <span className="text-medical-muted text-[11px] font-mono">
            {isBackendOnline ? (
              <span className="text-emerald-400">API v{health?.version || '1.0.0'} Online</span>
            ) : (
              <span className="text-rose-400">Backend Offline</span>
            )}
          </span>
        </div>

        {/* API Docs Link */}
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noreferrer"
          className="hidden sm:flex items-center gap-1.5 text-xs text-medical-muted hover:text-medical-cyan transition-colors px-2.5 py-1.5 rounded-lg hover:bg-medical-card"
        >
          <ExternalLink className="w-3.5 h-3.5" />
          <span>Swagger Docs</span>
        </a>

        {/* Settings Icon */}
        <Link
          to="/settings"
          className="p-2 rounded-lg text-medical-muted hover:text-medical-text hover:bg-medical-card transition-colors"
          title="System Settings"
        >
          <SettingsIcon className="w-4 h-4" />
        </Link>
      </div>
    </header>
  );
};
