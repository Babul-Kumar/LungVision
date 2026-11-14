import React, { Suspense } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { Navbar } from './Navbar';
import { Sidebar } from './Sidebar';
import { DisclaimerBanner } from '../common/DisclaimerBanner';
import { RouteLoadingSkeleton } from '../common/RouteLoadingSkeleton';
import {
  LayoutDashboard,
  Scan,
  Box,
  LineChart,
  Cpu,
} from 'lucide-react';

export const AppLayout: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-medical-bg text-medical-text">
      {/* Top Navigation */}
      <Navbar />

      {/* Main Container */}
      <div className="flex-1 flex overflow-hidden">
        {/* Desktop Sidebar */}
        <Sidebar />

        {/* Content Viewport */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 flex flex-col justify-between">
          <div className="max-w-7xl w-full mx-auto space-y-6">
            <Suspense fallback={<RouteLoadingSkeleton />}>
              <Outlet />
            </Suspense>
          </div>

          {/* Persistent Clinical Disclaimer Footer */}
          <footer className="mt-12 pt-6 border-t border-medical-border/60 max-w-7xl w-full mx-auto">
            <DisclaimerBanner compact />
            <div className="flex flex-col sm:flex-row items-center justify-between mt-3 text-[11px] text-medical-subtle gap-2">
              <span>© 2026 LungVision AI Research Group. All rights reserved.</span>
              <span>FastAPI REST Gateway · Three.js 3D CT Engine · Vite React SPA</span>
            </div>
          </footer>
        </main>
      </div>

      {/* Mobile Bottom Navigation Bar */}
      <div className="md:hidden border-t border-medical-border bg-medical-surface flex items-center justify-around py-2 px-3 sticky bottom-0 z-40">
        <NavLink
          to="/dashboard"
          className={({ isActive }) =>
            `flex flex-col items-center gap-1 text-[10px] ${
              isActive ? 'text-medical-cyan font-semibold' : 'text-medical-muted'
            }`
          }
        >
          <LayoutDashboard className="w-4 h-4" />
          <span>Dashboard</span>
        </NavLink>
        <NavLink
          to="/xray"
          className={({ isActive }) =>
            `flex flex-col items-center gap-1 text-[10px] ${
              isActive ? 'text-medical-cyan font-semibold' : 'text-medical-muted'
            }`
          }
        >
          <Scan className="w-4 h-4" />
          <span>X-Ray</span>
        </NavLink>
        <NavLink
          to="/ct-explorer"
          className={({ isActive }) =>
            `flex flex-col items-center gap-1 text-[10px] ${
              isActive ? 'text-medical-cyan font-semibold' : 'text-medical-muted'
            }`
          }
        >
          <Box className="w-4 h-4" />
          <span>CT 3D</span>
        </NavLink>
        <NavLink
          to="/research"
          className={({ isActive }) =>
            `flex flex-col items-center gap-1 text-[10px] ${
              isActive ? 'text-medical-cyan font-semibold' : 'text-medical-muted'
            }`
          }
        >
          <LineChart className="w-4 h-4" />
          <span>Research</span>
        </NavLink>
        <NavLink
          to="/models"
          className={({ isActive }) =>
            `flex flex-col items-center gap-1 text-[10px] ${
              isActive ? 'text-medical-cyan font-semibold' : 'text-medical-muted'
            }`
          }
        >
          <Cpu className="w-4 h-4" />
          <span>Models</span>
        </NavLink>
      </div>
    </div>
  );
};

