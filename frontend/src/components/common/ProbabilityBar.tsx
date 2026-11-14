import React from 'react';
import { Badge } from './Badge';

interface ProbabilityBarProps {
  label: string;
  probability: number;
  threshold: number;
  experimentalFlag: boolean;
}

export const ProbabilityBar: React.FC<ProbabilityBarProps> = ({
  label,
  probability,
  threshold,
  experimentalFlag,
}) => {
  const probPercent = Math.min(Math.max(probability * 100, 0), 100);
  const threshPercent = Math.min(Math.max(threshold * 100, 0), 100);

  return (
    <div className="bg-medical-surface/80 border border-medical-border/60 rounded-lg p-3.5 hover:border-medical-border transition-colors">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-medical-text tracking-wide">{label}</span>
          {experimentalFlag ? (
            <Badge variant="amber" size="sm">
              Above Threshold
            </Badge>
          ) : (
            <Badge variant="emerald" size="sm">
              Negative
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <span className="text-medical-muted">
            Thresh: <strong className="text-slate-300">{(threshold * 100).toFixed(1)}%</strong>
          </span>
          <span
            className={`font-semibold text-sm ${
              experimentalFlag ? 'text-medical-amber' : 'text-medical-cyan'
            }`}
          >
            {(probability * 100).toFixed(2)}%
          </span>
        </div>
      </div>

      {/* Progress Track with Threshold Marker */}
      <div className="relative h-2.5 w-full bg-medical-bg rounded-full overflow-hidden border border-medical-border/40">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            experimentalFlag
              ? 'bg-gradient-to-r from-amber-500 to-rose-500 shadow-sm shadow-amber-500/50'
              : 'bg-gradient-to-r from-teal-500 to-cyan-500'
          }`}
          style={{ width: `${probPercent}%` }}
        />
        {/* Calibrated NIH Threshold Line Indicator */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-sm shadow-white z-10"
          style={{ left: `${threshPercent}%` }}
          title={`Calibrated Threshold: ${(threshold * 100).toFixed(2)}%`}
        />
      </div>
    </div>
  );
};

