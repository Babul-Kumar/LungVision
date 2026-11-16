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
    emerald: 'bg-emerald-50 text-emerald-800 border-emerald-300/80 shadow-xs',
    amber: 'bg-amber-50 text-amber-900 border-amber-300 shadow-xs',
    marigold: 'bg-amber-50 text-amber-900 border-amber-300 shadow-xs',
    rose: 'bg-rose-50 text-rose-800 border-rose-300/80 shadow-xs',
    cyan: 'bg-amber-50 text-amber-900 border-amber-300 shadow-xs',
    blue: 'bg-amber-50 text-amber-900 border-amber-300 shadow-xs',
    slate: 'bg-slate-100 text-[#1a082e] border-slate-300 shadow-xs',
    silver: 'bg-[#f1f5f9] text-[#1a082e] border-[#cbd5e1] shadow-xs',
    teal: 'bg-[#f1f5f9] text-[#1a082e] border-[#cbd5e1] shadow-xs',
    aubergine: 'bg-[#f5effc] text-[#2a0e4a] border-[#d8c3ea] shadow-xs',
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
