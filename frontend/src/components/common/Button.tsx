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
  const baseStyles = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white disabled:opacity-50 disabled:cursor-not-allowed';

  const variantStyles = {
    primary: 'bg-gradient-to-r from-amber-500 via-amber-500 to-amber-600 text-white hover:from-amber-400 hover:to-amber-500 focus:ring-amber-500 shadow-md shadow-amber-500/25 border border-amber-400/40 font-semibold',
    secondary: 'bg-white text-[#1a082e] border border-[#e2e8f0] hover:bg-[#f8fafc] hover:border-[#cbd5e1] focus:ring-amber-400/30 shadow-xs font-medium',
    outline: 'border border-amber-600/40 text-[#92400e] bg-amber-50/50 hover:bg-amber-50 hover:border-amber-500 focus:ring-amber-500 font-medium',
    danger: 'bg-rose-600 text-white hover:bg-rose-500 focus:ring-rose-500 shadow-md shadow-rose-500/25 font-semibold',
    ghost: 'text-[#4c3a63] hover:text-[#1a082e] hover:bg-[#f1f5f9]',
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

