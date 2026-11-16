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
      className={`rounded-xl p-3.5 transition-all relative overflow-hidden group ${
        experimentalFlag
          ? 'bg-amber-950/30 border border-amber-500/40 shadow-md shadow-amber-950/30'
          : 'bg-[#150a24]/80 border border-white/10 hover:border-amber-400/30'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-100 tracking-wide group-hover:text-amber-300 transition-colors">
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
          <span className="text-slate-400">
            Thresh: <strong className="text-slate-200">{(threshold * 100).toFixed(1)}%</strong>
          </span>
          <span
            className={`font-bold text-sm tracking-tight ${
              experimentalFlag ? 'text-amber-400 drop-shadow-[0_0_6px_rgba(245,158,11,0.5)]' : 'text-slate-100'
            }`}
          >
            {(probability * 100).toFixed(2)}%
          </span>
        </div>
      </div>

      {/* Progress Track with Threshold Marker */}
      <div className="relative h-2.5 w-full bg-black/60 rounded-full overflow-hidden border border-white/10 p-[1px]">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${
            experimentalFlag
              ? 'bg-gradient-to-r from-amber-500 via-rose-500 to-rose-400 shadow-sm shadow-rose-500/50'
              : 'bg-gradient-to-r from-slate-600 via-slate-300 to-white shadow-xs'
          }`}
          style={{ width: `${probPercent}%` }}
        />
        {/* Calibrated NIH Threshold Line Indicator */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.8)] z-10"
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
