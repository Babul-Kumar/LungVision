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
          ? 'bg-amber-50/80 border border-amber-300 shadow-xs'
          : 'bg-[#f8fafc] border border-[#e2e8f0] hover:border-amber-400/50'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-[#1a082e] tracking-wide group-hover:text-amber-800 transition-colors">
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
          <span className="text-[#5e4b75]">
            Thresh: <strong className="text-[#1a082e]">{(threshold * 100).toFixed(1)}%</strong>
          </span>
          <span
            className={`font-bold text-sm tracking-tight ${
              experimentalFlag ? 'text-amber-800' : 'text-[#1a082e]'
            }`}
          >
            {(probability * 100).toFixed(2)}%
          </span>
        </div>
      </div>

      {/* Progress Track with Threshold Marker */}
      <div className="relative h-2.5 w-full bg-slate-200/80 rounded-full overflow-hidden border border-slate-300/60 p-[1px]">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${
            experimentalFlag
              ? 'bg-gradient-to-r from-amber-500 via-rose-500 to-rose-600 shadow-xs'
              : 'bg-gradient-to-r from-amber-400 to-amber-600 shadow-xs'
          }`}
          style={{ width: `${probPercent}%` }}
        />
        {/* Calibrated NIH Threshold Line Indicator */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-amber-700 shadow-xs z-10"
          style={{ left: `${threshPercent}%` }}
          title={`Calibrated Operating Threshold: ${(threshold * 100).toFixed(2)}%`}
        />
      </div>

      {/* Subtle relative position bar sub-label */}
      <div className="flex justify-between items-center text-[10px] text-[#7e6c96] mt-1 font-mono">
        <span>0%</span>
        <span className="text-[#5e4b75]">Cutoff: {(threshold * 100).toFixed(1)}%</span>
        <span>100%</span>
      </div>
    </div>
  );
};
