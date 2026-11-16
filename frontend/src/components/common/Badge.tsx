import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'emerald' | 'amber' | 'rose' | 'cyan' | 'blue' | 'slate' | 'teal' | 'marigold' | 'silver' | 'aubergine';
  size?: 'sm' | 'md';
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'slate',
  size = 'md',
  className,
}) => {
  const variantStyles = {
    emerald: 'bg-emerald-950/70 text-emerald-300 border-emerald-700/50 shadow-xs',
    amber: 'bg-amber-950/70 text-amber-300 border-amber-600/50 shadow-[0_0_12px_rgba(245,158,11,0.2)]',
    marigold: 'bg-amber-950/70 text-amber-300 border-amber-600/50 shadow-[0_0_12px_rgba(245,158,11,0.2)]',
    rose: 'bg-rose-950/70 text-rose-300 border-rose-700/50 shadow-xs',
    cyan: 'bg-amber-950/70 text-amber-300 border-amber-600/50 shadow-[0_0_10px_rgba(245,158,11,0.15)]',
    blue: 'bg-blue-950/70 text-blue-300 border-blue-700/50 shadow-xs',
    slate: 'bg-slate-900/80 text-slate-200 border-white/10 shadow-xs',
    silver: 'bg-slate-800/60 text-slate-100 border-slate-400/30 shadow-xs',
    teal: 'bg-slate-800/60 text-slate-100 border-slate-400/30 shadow-xs',
    aubergine: 'bg-[#2b1642]/80 text-[#e9d5ff] border-[#6b359e]/50',
  };

  const sizeStyles = {
    sm: 'text-[11px] px-2 py-0.5',
    md: 'text-xs px-2.5 py-1',
  };

  return (
    <span
      className={twMerge(
        clsx(
          'inline-flex items-center gap-1 font-medium rounded-full border',
          variantStyles[variant],
          sizeStyles[size],
          className
        )
      )}
    >
      {children}
    </span>
  );
};
