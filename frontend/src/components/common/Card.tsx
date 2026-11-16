import React, { HTMLAttributes } from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  icon?: React.ReactNode;
  headerAction?: React.ReactNode;
}

export const Card: React.FC<CardProps> = ({
  children,
  className,
  title,
  subtitle,
  icon,
  headerAction,
  ...props
}) => {
  return (
    <div
      className={twMerge(
        clsx(
          'bg-[#160c26]/80 backdrop-blur-xl border border-white/10 rounded-2xl p-5 sm:p-6 shadow-xl shadow-black/40 transition-all duration-300 hover:border-amber-400/30',
          className
        )
      )}
      {...props}
    >
      {(title || icon || headerAction) && (
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-white/[0.08]">
          <div className="flex items-center gap-3">
            {icon && <div className="text-amber-400">{icon}</div>}
            <div>
              {title && <h3 className="text-base font-semibold text-slate-100 tracking-tight">{title}</h3>}
              {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
            </div>
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      )}
      {children}
    </div>
  );
};

