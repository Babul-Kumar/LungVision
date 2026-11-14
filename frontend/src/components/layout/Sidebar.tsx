import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Scan,
  SplitSquareVertical,
  Box,
  LineChart,
  Cpu,
  History,
  Settings,
} from 'lucide-react';

const NAV_ITEMS = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/xray', label: '2D X-Ray Analysis', icon: Scan },
  { path: '/dual-view', label: 'Dual-View CXR Lab', icon: SplitSquareVertical },
  { path: '/ct-explorer', label: '3D CT Explorer', icon: Box },
  { path: '/research', label: 'Research Benchmarks', icon: LineChart },
  { path: '/models', label: 'Model Registry', icon: Cpu },
  { path: '/history', label: 'Session History', icon: History },
  { path: '/settings', label: 'Settings', icon: Settings },
];

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-64 border-r border-medical-border bg-medical-surface/50 p-4 flex flex-col justify-between shrink-0 hidden md:flex">
      <nav className="space-y-1.5">
        <div className="text-[11px] font-semibold text-medical-subtle tracking-wider uppercase px-3 py-2">
          Diagnostic Modules
        </div>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-medical-card text-medical-cyan border border-medical-border/80 shadow-sm font-semibold'
                    : 'text-medical-muted hover:text-medical-text hover:bg-medical-hover/60'
                }`
              }
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* Sidebar Footer Info */}
      <div className="p-3 rounded-xl bg-medical-card/60 border border-medical-border/40 text-[11px] text-medical-subtle hud-corner-bracket">
        <div className="flex items-center justify-between mb-1">
          <span className="font-semibold text-slate-300">Model Runtime</span>
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        </div>
        <p className="text-[10px] font-mono text-cyan-400">PyTorch 2.14.1 · SimpleITK 2.5</p>
        <p className="mt-1 text-[10px] text-medical-muted leading-tight">
          4 Locked Checkpoints: DenseNet121, 2x ResNet18, SSM Lung PCA (20 Modes)
        </p>
      </div>
    </aside>
  );
};

