import React, { ButtonHTMLAttributes } from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  className,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  disabled,
  ...props
}) => {
  const baseStyles = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#0c0614] disabled:opacity-50 disabled:cursor-not-allowed';

  const variantStyles = {
    primary: 'bg-gradient-to-r from-amber-400 via-amber-500 to-amber-600 text-slate-950 hover:from-amber-300 hover:to-amber-500 focus:ring-amber-400 shadow-md shadow-amber-950/40 border border-amber-300/40 font-semibold',
    secondary: 'bg-[#201135] text-slate-100 border border-white/10 hover:bg-[#2b1747] hover:border-white/20 focus:ring-white/20',
    outline: 'border border-amber-500/40 text-amber-300 hover:bg-amber-500/10 hover:border-amber-400 focus:ring-amber-500',
    danger: 'bg-rose-600 text-white hover:bg-rose-500 focus:ring-rose-500 shadow-md shadow-rose-950/40',
    ghost: 'text-slate-300 hover:text-white hover:bg-white/[0.06]',
  };

  const sizeStyles = {
    sm: 'text-xs px-2.5 py-1.5 gap-1.5',
    md: 'text-sm px-4 py-2 gap-2',
    lg: 'text-base px-5 py-2.5 gap-2.5',
  };

  return (
    <button
      className={twMerge(clsx(baseStyles, variantStyles[variant], sizeStyles[size], className))}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading && (
        <svg className="animate-spin h-4 w-4 text-current" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
      )}
      {children}
    </button>
  );
};

