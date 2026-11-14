import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'emerald' | 'amber' | 'rose' | 'cyan' | 'blue' | 'slate' | 'teal';
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
    emerald: 'bg-emerald-950/60 text-emerald-400 border-emerald-800/60',
    amber: 'bg-amber-950/60 text-amber-400 border-amber-800/60',
    rose: 'bg-rose-950/60 text-rose-400 border-rose-800/60',
    cyan: 'bg-cyan-950/60 text-cyan-400 border-cyan-800/60',
    blue: 'bg-blue-950/60 text-blue-400 border-blue-800/60',
    slate: 'bg-slate-900/80 text-slate-300 border-slate-700/60',
    teal: 'bg-teal-950/60 text-teal-400 border-teal-800/60',
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
