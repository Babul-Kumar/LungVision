import React, { useState } from 'react';
import {
  X,
  BookOpen,
  Info,
  ShieldAlert,
  AlertTriangle,
  CheckCircle,
  Sliders,
  Activity,
  Layers,
} from 'lucide-react';
import { DiseaseName, EXPECTED_6_DISEASES } from '@/types/xray';
import { DISEASE_CLINICAL_PROFILES } from '@/types/explainability';
import { Badge } from '@/components/common/Badge';

interface DiseaseExplainabilityModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialDisease?: DiseaseName;
  currentProbabilities?: Record<DiseaseName, { probability: number; threshold: number; experimental_flag: boolean }>;
}

export const DiseaseExplainabilityModal: React.FC<DiseaseExplainabilityModalProps> = ({
  isOpen,
  onClose,
  initialDisease = 'Cardiomegaly',
  currentProbabilities,
}) => {
  const [selectedDisease, setSelectedDisease] = useState<DiseaseName>(initialDisease);

  if (!isOpen) return null;

  const profile = DISEASE_CLINICAL_PROFILES[selectedDisease];
  const liveProb = currentProbabilities?.[selectedDisease];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-[#140a22] border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-5 border-b border-white/10 flex items-center justify-between bg-[#0e0719]/90">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-amber-950/60 border border-amber-800/60 flex items-center justify-center text-amber-400">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-white">Clinical Pathology Guide & AI Explainability</h3>
                <Badge variant="amber" size="sm">Medically Reviewed</Badge>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Understanding what the AI evaluates, how thresholds work, and why model scores are not diagnoses.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Disease Tabs Navigation */}
        <div className="flex items-center gap-1.5 px-5 py-2.5 bg-black/30 border-b border-white/10 overflow-x-auto scrollbar-thin">
          {EXPECTED_6_DISEASES.map((d) => (
            <button
              key={d}
              onClick={() => setSelectedDisease(d)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                selectedDisease === d
                  ? 'bg-gradient-to-r from-amber-400 via-amber-500 to-amber-600 text-[#0c0614] font-bold shadow-md shadow-amber-950/40 border border-amber-300/40'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              {d}
            </button>
          ))}
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 text-sm">
          {/* Active Condition Overview Header */}
          <div className="p-4 rounded-xl bg-medical-surface border border-white/10 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="text-[11px] font-mono text-amber-400 uppercase tracking-wider">
                Thoracic Pathology Profile
              </div>
              <h4 className="text-xl font-bold text-white mt-0.5">
                {profile.name} <span className="text-sm font-normal text-slate-400">({profile.patientFriendlyName})</span>
              </h4>
            </div>

            {liveProb && (
              <div className="flex items-center gap-4 bg-[#180d28]/80 px-4 py-2 rounded-xl border border-white/10">
                <div>
                  <span className="text-[10px] text-slate-400 block uppercase">Current Scan Score</span>
                  <span className="text-sm font-mono font-bold text-white">
                    {(liveProb.probability * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="h-6 w-px bg-white/10" />
                <div>
                  <span className="text-[10px] text-slate-400 block uppercase">Threshold</span>
                  <span className="text-sm font-mono text-slate-300">
                    {(liveProb.threshold * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="h-6 w-px bg-white/10" />
                <div>
                  <span className="text-[10px] text-slate-400 block uppercase">Flag Status</span>
                  {liveProb.experimental_flag ? (
                    <span className="inline-flex items-center text-xs font-semibold text-amber-400">
                      <AlertTriangle className="w-3.5 h-3.5 mr-1" /> Above Threshold
                    </span>
                  ) : (
                    <span className="inline-flex items-center text-xs font-semibold text-emerald-400">
                      <CheckCircle className="w-3.5 h-3.5 mr-1" /> Below Threshold
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Section 1: Clinical Definition & Meaning */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-medical-card/50 border border-white/10 space-y-2">
              <h5 className="font-semibold text-white flex items-center gap-2 text-xs uppercase tracking-wider text-amber-400">
                <Info className="w-4 h-4" /> What is {profile.name}?
              </h5>
              <p className="text-xs text-slate-300 leading-relaxed">{profile.whatItIs}</p>
            </div>

            <div className="p-4 rounded-xl bg-medical-card/50 border border-white/10 space-y-2">
              <h5 className="font-semibold text-white flex items-center gap-2 text-xs uppercase tracking-wider text-slate-200">
                <Activity className="w-4 h-4" /> What does it mean clinically?
              </h5>
              <p className="text-xs text-slate-300 leading-relaxed">{profile.clinicalMeaning}</p>
            </div>
          </div>

          {/* Section 2: AI Model Score & Experimental Threshold */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-800/40 space-y-2">
              <h5 className="font-semibold text-amber-300 flex items-center gap-2 text-xs uppercase tracking-wider">
                <Sliders className="w-4 h-4" /> What the Model Score Represents
              </h5>
              <p className="text-xs text-slate-300 leading-relaxed">{profile.modelScoreMeaning}</p>
            </div>

            <div className="p-4 rounded-xl bg-[#1c102f] border border-white/10 space-y-2">
              <h5 className="font-semibold text-slate-200 flex items-center gap-2 text-xs uppercase tracking-wider">
                <Layers className="w-4 h-4" /> What the Experimental Threshold Represents
              </h5>
              <p className="text-xs text-slate-300 leading-relaxed">{profile.thresholdMeaning}</p>
            </div>
          </div>

          {/* Section 3: Diagnostic Caveat & Why It Is NOT A Diagnosis */}
          <div className="p-4 rounded-xl bg-amber-950/25 border border-amber-800/50 space-y-2">
            <h5 className="font-semibold text-amber-300 flex items-center gap-2 text-xs uppercase tracking-wider">
              <ShieldAlert className="w-4 h-4 text-amber-400" /> Crucial Clinical & Diagnostic Caveats
            </h5>
            <p className="text-xs text-amber-100/90 leading-relaxed">{profile.diagnosticCaveat}</p>
          </div>

          {/* Section 4: Key Radiographic Signs */}
          <div className="p-4 rounded-xl bg-medical-surface border border-white/10 space-y-2.5">
            <h5 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Hallmark Radiographic Visual Features Looked for by Radiologists:
            </h5>
            <ul className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {profile.radiographicSigns.map((sign, idx) => (
                <li
                  key={idx}
                  className="p-2.5 rounded-lg bg-medical-card border border-white/[0.08] text-xs text-slate-300 flex items-start gap-2"
                >
                  <span className="w-4 h-4 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center shrink-0 text-[10px] font-bold">
                    {idx + 1}
                  </span>
                  <span>{sign}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Four-Way Differentiation Matrix */}
          <div className="p-4 rounded-xl bg-medical-card border border-white/10 space-y-3">
            <h5 className="text-xs font-semibold uppercase tracking-wider text-white">
              Understanding the Difference: 4 Distinct Levels
            </h5>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-2.5 text-xs">
              <div className="p-2.5 rounded-lg bg-medical-surface border border-white/[0.08]">
                <strong className="text-amber-400 block mb-1">1. Model Probability</strong>
                <p className="text-[11px] text-slate-400">
                  Continuous number (0.00 to 1.00) indicating visual pattern match strength inside the neural network.
                </p>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-white/[0.08]">
                <strong className="text-slate-200 block mb-1">2. Experimental Threshold</strong>
                <p className="text-[11px] text-slate-400">
                  Mathematical cutoff derived from NIH benchmark validation to separate high from low activations.
                </p>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-white/[0.08]">
                <strong className="text-amber-300 block mb-1">3. Threshold Flag</strong>
                <p className="text-[11px] text-slate-400">
                  Simple binary signal (Yes/No) that triggers only when the probability equals or exceeds the threshold.
                </p>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-white/[0.08]">
                <strong className="text-emerald-400 block mb-1">4. Clinical Diagnosis</strong>
                <p className="text-[11px] text-slate-400">
                  A comprehensive determination made exclusively by a doctor evaluating patient history, exams, and labs.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-white/10 bg-[#0e0719]/90 flex items-center justify-between text-xs text-slate-400">
          <span>LungVision AI · Investigational Decision Support Architecture</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-medical-surface border border-white/10 text-white hover:border-amber-400/40 hover:bg-medical-card transition-colors text-xs font-medium"
          >
            Close Guide
          </button>
        </div>
      </div>
    </div>
  );
};
