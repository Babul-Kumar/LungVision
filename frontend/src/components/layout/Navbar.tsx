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
    <header className="h-16 border-b border-[#e2e8f0] bg-white/95 backdrop-blur-xl px-6 flex items-center justify-between sticky top-0 z-40 shadow-xs">
      {/* Brand & Logo */}
      <div className="flex items-center gap-3">
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-500 via-amber-500 to-amber-600 flex items-center justify-center text-white font-bold shadow-md shadow-amber-500/20 group-hover:scale-105 group-hover:shadow-amber-500/30 transition-all">
            LV
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight text-[#1a082e] group-hover:text-amber-700 transition-colors flex items-center">
              LUNGVISION
              <span className="text-amber-900 font-semibold text-xs uppercase ml-1.5 px-1.5 py-0.5 rounded bg-amber-100/90 border border-amber-300 font-mono">
                AI
              </span>
            </h1>
            <p className="text-[10px] text-[#5e4b75] tracking-wider uppercase font-mono">
              Thoracic Imaging Intelligence
            </p>
          </div>
        </Link>
      </div>

      {/* Middle: Live Workstation Clock */}
      <div className="hidden lg:flex items-center gap-2 bg-[#f8fafc] px-3 py-1 rounded-full border border-[#e2e8f0] text-[11px] font-mono text-[#1a082e]">
        <Clock className="w-3.5 h-3.5 text-amber-600" />
        <span>{timeString || '00:00:00 UTC'}</span>
        <span className="text-slate-300">|</span>
        <span className="text-emerald-700 font-medium flex items-center gap-1">
          <Activity className="w-3 h-3 text-emerald-600 animate-pulse" />
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
              ? 'bg-amber-50 border-amber-300 text-amber-900 shadow-xs'
              : 'bg-emerald-50 border-emerald-300 text-emerald-900'
          }`}
          title="Toggle between Doctor/Researcher Technical Mode and Patient-Friendly Mode"
        >
          {userExperienceMode === 'doctor' ? (
            <>
              <Stethoscope className="w-3.5 h-3.5 text-amber-600" />
              <span className="hidden sm:inline">Clinician View</span>
            </>
          ) : (
            <>
              <UserCheck className="w-3.5 h-3.5 text-emerald-600" />
              <span className="hidden sm:inline">Patient View</span>
            </>
          )}
        </button>

        {/* Backend Health Indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#f8fafc] border border-[#e2e8f0] text-xs">
          <div
            className={`w-2 h-2 rounded-full ${
              isBackendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
            }`}
          />
          <span className="text-[#5e4b75] text-[11px] font-mono">
            {isBackendOnline ? (
              <span className="text-emerald-700 font-medium">API v{health?.version || '1.0.0'} Online</span>
            ) : (
              <span className="text-rose-600 font-medium">Backend Offline</span>
            )}
          </span>
        </div>

        {/* API Docs Link */}
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noreferrer"
          className="hidden sm:flex items-center gap-1.5 text-xs text-[#5e4b75] hover:text-[#1a082e] transition-colors px-2.5 py-1.5 rounded-lg hover:bg-slate-100"
        >
          <ExternalLink className="w-3.5 h-3.5" />
          <span>Docs</span>
        </a>

        {/* Settings Icon */}
        <Link
          to="/settings"
          className="p-2 rounded-lg text-[#5e4b75] hover:text-[#1a082e] hover:bg-slate-100 transition-colors"
          title="System Settings"
        >
          <SettingsIcon className="w-4 h-4" />
        </Link>
      </div>
    </header>
  );
};
