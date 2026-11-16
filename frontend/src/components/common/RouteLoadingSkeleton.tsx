import React from 'react';
import { Loader2 } from 'lucide-react';

export const RouteLoadingSkeleton: React.FC = () => {
  return (
    <div className="w-full min-h-[400px] flex flex-col items-center justify-center p-8 space-y-4">
      <div className="relative flex items-center justify-center">
        <div className="w-12 h-12 rounded-full border border-amber-400/40 animate-ping absolute" />
        <div className="w-10 h-10 rounded-xl bg-white border border-amber-300 flex items-center justify-center text-amber-600 shadow-md shadow-amber-500/10">
          <Loader2 className="w-5 h-5 animate-spin" />
        </div>
      </div>
      <div className="text-center space-y-1">
        <p className="text-xs font-mono uppercase tracking-widest text-amber-800 font-semibold">
          Initializing Workstation
        </p>
        <p className="text-[11px] text-[#5e4b75]">
          Loading diagnostic module & dynamic assets...
        </p>
      </div>
    </div>
  );
};

export default RouteLoadingSkeleton;

