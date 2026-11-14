import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ExternalLink,
  Settings as SettingsIcon,
  Stethoscope,
  UserCheck,
  Clock,
  Activity,
} from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';

export const Navbar: React.FC = () => {
  const { health, isBackendOnline, fetchHealth, userExperienceMode, setUserExperienceMode } =
    useAppStore();
  const [timeString, setTimeString] = useState<string>('');

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000); // 15s poll
    return () => clearInterval(interval);
  }, [fetchHealth]);

  // Live UTC Clock
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeString(now.toUTCString().slice(17, 25) + ' UTC');
    };
    updateTime();
    const clockInterval = setInterval(updateTime, 1000);
    return () => clearInterval(clockInterval);
  }, []);

  const togglePersonaMode = () => {
    setUserExperienceMode(userExperienceMode === 'patient' ? 'doctor' : 'patient');
  };

  return (
    <header className="h-16 border-b border-medical-border bg-medical-surface/90 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-40">
      {/* Brand & Logo */}
      <div className="flex items-center gap-3">
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-medical-cyan to-medical-teal flex items-center justify-center text-medical-bg font-bold shadow-md shadow-cyan-950/40 group-hover:scale-105 group-hover:shadow-cyan-500/20 transition-all">
            LV
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight text-medical-text group-hover:text-medical-cyan transition-colors flex items-center">
              LUNGVISION
              <span className="text-medical-cyan font-normal text-xs uppercase ml-1.5 px-1.5 py-0.5 rounded bg-medical-card border border-medical-border font-mono">
                AI
              </span>
            </h1>
            <p className="text-[10px] text-medical-muted tracking-wider uppercase font-mono">
              Thoracic Imaging Intelligence
            </p>
          </div>
        </Link>
      </div>

      {/* Middle: Live Workstation Clock */}
      <div className="hidden lg:flex items-center gap-2 bg-black/40 px-3 py-1 rounded-full border border-white/5 text-[11px] font-mono text-slate-400">
        <Clock className="w-3.5 h-3.5 text-cyan-400" />
        <span>{timeString || '00:00:00 UTC'}</span>
        <span className="text-white/20">|</span>
        <span className="text-emerald-400 flex items-center gap-1">
          <Activity className="w-3 h-3 animate-pulse" />
          Station Synchronized
        </span>
      </div>

      {/* Right Controls: Persona Switcher, Health & Links */}
      <div className="flex items-center gap-3 sm:gap-4">
        {/* Experience Mode Toggle Button */}
        <button
          onClick={togglePersonaMode}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-medium transition-all ${
            userExperienceMode === 'doctor'
              ? 'bg-cyan-950/60 border-cyan-500/40 text-cyan-300 shadow-sm shadow-cyan-950/50'
              : 'bg-emerald-950/50 border-emerald-500/30 text-emerald-300'
          }`}
          title="Toggle between Doctor/Researcher Technical Mode and Patient-Friendly Mode"
        >
          {userExperienceMode === 'doctor' ? (
            <>
              <Stethoscope className="w-3.5 h-3.5 text-cyan-400" />
              <span className="hidden sm:inline">Clinician View</span>
            </>
          ) : (
            <>
              <UserCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span className="hidden sm:inline">Patient View</span>
            </>
          )}
        </button>

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
          <span>Docs</span>
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
