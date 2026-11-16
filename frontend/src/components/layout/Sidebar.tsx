import React, { useEffect } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
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
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard, shortcut: '1' },
  { path: '/xray', label: '2D X-Ray Analysis', icon: Scan, shortcut: '2' },
  { path: '/dual-view', label: 'Dual-View CXR Lab', icon: SplitSquareVertical, shortcut: '3', badge: '3D SSM' },
  { path: '/ct-explorer', label: '3D CT Explorer', icon: Box, shortcut: '4', badge: '12 Nodules' },
  { path: '/research', label: 'Research Benchmarks', icon: LineChart, shortcut: '5' },
  { path: '/models', label: 'Model Registry', icon: Cpu, shortcut: '6', badge: '4 Models' },
  { path: '/history', label: 'Session History', icon: History, shortcut: '7' },
  { path: '/settings', label: 'Settings', icon: Settings, shortcut: '8' },
];

export const Sidebar: React.FC = () => {
  const navigate = useNavigate();

  // Global keyboard shortcuts (Alt+1 through Alt+8 or number keys when not focusing inputs)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in input, textarea, or contentEditable
      const target = e.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      // Check for Alt+Number or raw number
      if (e.altKey && e.key >= '1' && e.key <= '8') {
        e.preventDefault();
        const index = parseInt(e.key, 10) - 1;
        if (NAV_ITEMS[index]) {
          navigate(NAV_ITEMS[index].path);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [navigate]);

  return (
    <aside className="w-64 border-r border-[#e2e8f0] bg-white/95 backdrop-blur-xl p-4 flex flex-col justify-between shrink-0 hidden md:flex shadow-xs">
      <nav className="space-y-1.5">
        <div className="flex items-center justify-between px-3 py-2">
          <span className="text-[11px] font-semibold text-[#5e4b75] tracking-wider uppercase">
            Diagnostic Modules
          </span>
          <span className="text-[10px] font-mono text-[#5e4b75] bg-slate-100 px-1.5 py-0.5 rounded border border-[#e2e8f0]">
            Alt+1..8
          </span>
        </div>

        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center justify-between px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all group ${
                  isActive
                    ? 'bg-gradient-to-r from-amber-500/15 via-amber-500/5 to-transparent text-[#92400e] border-l-2 border-l-amber-500 border-t border-b border-r border-amber-300/40 shadow-xs font-semibold'
                    : 'text-[#4c3a63] hover:text-[#1a082e] hover:bg-slate-50 border border-transparent'
                }`
              }
            >
              <div className="flex items-center gap-3">
                <Icon className="w-4 h-4 shrink-0 group-hover:text-amber-600 transition-colors" />
                <span>{item.label}</span>
              </div>

              <div className="flex items-center gap-1.5">
                {item.badge && (
                  <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-amber-100 text-amber-900 border border-amber-300">
                    {item.badge}
                  </span>
                )}
                <span className="hidden group-hover:inline text-[9px] font-mono text-[#5e4b75] bg-slate-100 px-1 py-0.5 rounded border border-[#e2e8f0]">
                  {item.shortcut}
                </span>
              </div>
            </NavLink>
          );
        })}
      </nav>

      {/* Sidebar Footer Info */}
      <div className="p-3.5 rounded-xl bg-[#f8fafc] border border-[#e2e8f0] text-[11px]">
        <div className="flex items-center justify-between mb-1">
          <span className="font-semibold text-[#1a082e]">Model Runtime</span>
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
        </div>
        <p className="text-[10px] font-mono text-amber-800 font-medium">PyTorch 2.14.1 · SimpleITK 2.5</p>
        <p className="mt-1 text-[10px] text-[#5e4b75] leading-tight">
          4 Locked Checkpoints: DenseNet121, 2x ResNet18, SSM Lung PCA (20 Modes)
        </p>
      </div>
    </aside>
  );
};
