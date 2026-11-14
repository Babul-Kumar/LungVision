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
    <div
      className={`rounded-xl p-3.5 transition-all hud-corner-bracket relative overflow-hidden group ${
        experimentalFlag
          ? 'bg-amber-950/20 border border-amber-500/40 shadow-sm shadow-amber-950/30'
          : 'bg-medical-surface/80 border border-medical-border/60 hover:border-cyan-500/30'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-medical-text tracking-wide group-hover:text-cyan-300 transition-colors">
            {label}
          </span>
          {experimentalFlag ? (
            <Badge variant="amber" size="sm" className="animate-pulse">
              Above Threshold
            </Badge>
          ) : (
            <Badge variant="emerald" size="sm">
              Negative
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-3.5 text-xs font-mono">
          <span className="text-medical-muted">
            Thresh: <strong className="text-slate-300">{(threshold * 100).toFixed(1)}%</strong>
          </span>
          <span
            className={`font-bold text-sm tracking-tight ${
              experimentalFlag ? 'text-amber-400 drop-shadow-[0_0_6px_rgba(245,158,11,0.5)]' : 'text-cyan-400'
            }`}
          >
            {(probability * 100).toFixed(2)}%
          </span>
        </div>
      </div>

      {/* Progress Track with Threshold Marker */}
      <div className="relative h-2.5 w-full bg-slate-950 rounded-full overflow-hidden border border-medical-border/40 p-[1px]">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${
            experimentalFlag
              ? 'bg-gradient-to-r from-amber-500 via-rose-500 to-rose-400 shadow-sm shadow-rose-500/50'
              : 'bg-gradient-to-r from-teal-500 via-cyan-500 to-sky-400'
          }`}
          style={{ width: `${probPercent}%` }}
        />
        {/* Calibrated NIH Threshold Line Indicator */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_6px_#ffffff] z-10"
          style={{ left: `${threshPercent}%` }}
          title={`Calibrated Operating Threshold: ${(threshold * 100).toFixed(2)}%`}
        />
      </div>

      {/* Subtle relative position bar sub-label */}
      <div className="flex justify-between items-center text-[10px] text-medical-subtle mt-1 font-mono">
        <span>0%</span>
        <span className="text-slate-400">Cutoff: {(threshold * 100).toFixed(1)}%</span>
        <span>100%</span>
      </div>
    </div>
  );
};
