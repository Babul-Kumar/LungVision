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
          'bg-white/95 backdrop-blur-xl border border-[#e2e8f0] rounded-2xl p-5 sm:p-6 shadow-[0_4px_20px_-4px_rgba(26,8,46,0.06),0_1px_3px_rgba(26,8,46,0.04)] transition-all duration-300 hover:border-amber-400/40 hover:shadow-[0_8px_30px_-4px_rgba(26,8,46,0.1)]',
          className
        )
      )}
      {...props}
    >
      {(title || icon || headerAction) && (
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-[#f1f5f9]">
          <div className="flex items-center gap-3">
            {icon && <div className="text-amber-600">{icon}</div>}
            <div>
              {title && <h3 className="text-base font-semibold text-[#1a082e] tracking-tight">{title}</h3>}
              {subtitle && <p className="text-xs text-[#5e4b75] mt-0.5">{subtitle}</p>}
            </div>
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      )}
      {children}
    </div>
  );
};

