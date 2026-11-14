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
          'bg-medical-card border border-medical-border rounded-xl p-5 shadow-sm transition-all',
          className
        )
      )}
      {...props}
    >
      {(title || icon || headerAction) && (
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-medical-border/60">
          <div className="flex items-center gap-3">
            {icon && <div className="text-medical-cyan">{icon}</div>}
            <div>
              {title && <h3 className="text-base font-semibold text-medical-text tracking-tight">{title}</h3>}
              {subtitle && <p className="text-xs text-medical-muted mt-0.5">{subtitle}</p>}
            </div>
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      )}
      {children}
    </div>
  );
};

