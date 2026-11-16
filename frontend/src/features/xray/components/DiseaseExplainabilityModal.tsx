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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#1a082e]/50 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-white border border-[#e2e8f0] rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-5 border-b border-[#e2e8f0] flex items-center justify-between bg-slate-50/90">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-amber-50 border border-amber-200 flex items-center justify-center text-amber-700 shadow-xs">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-[#1a082e]">Clinical Pathology Guide & AI Explainability</h3>
                <Badge variant="amber" size="sm">Medically Reviewed</Badge>
              </div>
              <p className="text-xs text-[#5e4b75] mt-0.5">
                Understanding what the AI evaluates, how thresholds work, and why model scores are not diagnoses.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-[#5e4b75] hover:text-[#1a082e] hover:bg-slate-200/50 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Disease Tabs Navigation */}
        <div className="flex items-center gap-1.5 px-5 py-2.5 bg-slate-100/70 border-b border-[#e2e8f0] overflow-x-auto scrollbar-thin">
          {EXPECTED_6_DISEASES.map((d) => (
            <button
              key={d}
              onClick={() => setSelectedDisease(d)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                selectedDisease === d
                  ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-white font-bold shadow-xs border border-amber-600'
                  : 'text-[#5e4b75] hover:text-[#1a082e] hover:bg-slate-200/50'
              }`}
            >
              {d}
            </button>
          ))}
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 text-sm">
          {/* Active Condition Overview Header */}
          <div className="p-4 rounded-xl bg-slate-50 border border-[#e2e8f0] flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="text-[11px] font-mono text-amber-700 uppercase tracking-wider font-semibold">
                Thoracic Pathology Profile
              </div>
              <h4 className="text-xl font-bold text-[#1a082e] mt-0.5">
                {profile.name} <span className="text-sm font-normal text-[#5e4b75]">({profile.patientFriendlyName})</span>
              </h4>
            </div>

            {liveProb && (
              <div className="flex items-center gap-4 bg-white px-4 py-2 rounded-xl border border-[#e2e8f0] shadow-xs">
                <div>
                  <span className="text-[10px] text-[#5e4b75] block uppercase font-medium">Current Scan Score</span>
                  <span className="text-sm font-mono font-bold text-[#1a082e]">
                    {(liveProb.probability * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="h-6 w-px bg-slate-200" />
                <div>
                  <span className="text-[10px] text-[#5e4b75] block uppercase font-medium">Threshold</span>
                  <span className="text-sm font-mono text-[#5e4b75]">
                    {(liveProb.threshold * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="h-6 w-px bg-slate-200" />
                <div>
                  <span className="text-[10px] text-[#5e4b75] block uppercase font-medium">Flag Status</span>
                  {liveProb.experimental_flag ? (
                    <span className="inline-flex items-center text-xs font-semibold text-amber-700">
                      <AlertTriangle className="w-3.5 h-3.5 mr-1 text-amber-600" /> Above Threshold
                    </span>
                  ) : (
                    <span className="inline-flex items-center text-xs font-semibold text-emerald-700">
                      <CheckCircle className="w-3.5 h-3.5 mr-1 text-emerald-600" /> Below Threshold
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Section 1: Clinical Definition & Meaning */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] shadow-xs space-y-2">
              <h5 className="font-semibold text-amber-800 flex items-center gap-2 text-xs uppercase tracking-wider">
                <Info className="w-4 h-4 text-amber-600" /> What is {profile.name}?
              </h5>
              <p className="text-xs text-[#5e4b75] leading-relaxed">{profile.whatItIs}</p>
            </div>

            <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] shadow-xs space-y-2">
              <h5 className="font-semibold text-[#1a082e] flex items-center gap-2 text-xs uppercase tracking-wider">
                <Activity className="w-4 h-4 text-[#5e4b75]" /> What does it mean clinically?
              </h5>
              <p className="text-xs text-[#5e4b75] leading-relaxed">{profile.clinicalMeaning}</p>
            </div>
          </div>

          {/* Section 2: AI Model Score & Experimental Threshold */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-amber-50/70 border border-amber-200 shadow-xs space-y-2">
              <h5 className="font-semibold text-amber-900 flex items-center gap-2 text-xs uppercase tracking-wider">
                <Sliders className="w-4 h-4 text-amber-600" /> What the Model Score Represents
              </h5>
              <p className="text-xs text-[#5e4b75] leading-relaxed">{profile.modelScoreMeaning}</p>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-[#e2e8f0] shadow-xs space-y-2">
              <h5 className="font-semibold text-[#1a082e] flex items-center gap-2 text-xs uppercase tracking-wider">
                <Layers className="w-4 h-4 text-[#5e4b75]" /> What the Experimental Threshold Represents
              </h5>
              <p className="text-xs text-[#5e4b75] leading-relaxed">{profile.thresholdMeaning}</p>
            </div>
          </div>

          {/* Section 3: Diagnostic Caveat & Why It Is NOT A Diagnosis */}
          <div className="p-4 rounded-xl bg-amber-50 border border-amber-300 shadow-xs space-y-2">
            <h5 className="font-semibold text-amber-900 flex items-center gap-2 text-xs uppercase tracking-wider">
              <ShieldAlert className="w-4 h-4 text-amber-600" /> Crucial Clinical & Diagnostic Caveats
            </h5>
            <p className="text-xs text-amber-950/90 leading-relaxed">{profile.diagnosticCaveat}</p>
          </div>

          {/* Section 4: Key Radiographic Signs */}
          <div className="p-4 rounded-xl bg-slate-50 border border-[#e2e8f0] space-y-2.5">
            <h5 className="text-xs font-semibold uppercase tracking-wider text-[#5e4b75]">
              Hallmark Radiographic Visual Features Looked for by Radiologists:
            </h5>
            <ul className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {profile.radiographicSigns.map((sign, idx) => (
                <li
                  key={idx}
                  className="p-2.5 rounded-lg bg-white border border-[#e2e8f0] text-xs text-[#5e4b75] flex items-start gap-2 shadow-xs"
                >
                  <span className="w-4 h-4 rounded-full bg-amber-100 text-amber-900 flex items-center justify-center shrink-0 text-[10px] font-bold">
                    {idx + 1}
                  </span>
                  <span>{sign}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Four-Way Differentiation Matrix */}
          <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] shadow-xs space-y-3">
            <h5 className="text-xs font-semibold uppercase tracking-wider text-[#1a082e]">
              Understanding the Difference: 4 Distinct Levels
            </h5>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-2.5 text-xs">
              <div className="p-2.5 rounded-lg bg-slate-50 border border-[#e2e8f0]">
                <strong className="text-amber-800 block mb-1">1. Model Probability</strong>
                <p className="text-[11px] text-[#5e4b75]">
                  Continuous number (0.00 to 1.00) indicating visual pattern match strength inside the neural network.
                </p>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-50 border border-[#e2e8f0]">
                <strong className="text-[#1a082e] block mb-1">2. Experimental Threshold</strong>
                <p className="text-[11px] text-[#5e4b75]">
                  Mathematical cutoff derived from NIH benchmark validation to separate high from low activations.
                </p>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-50 border border-[#e2e8f0]">
                <strong className="text-amber-700 block mb-1">3. Threshold Flag</strong>
                <p className="text-[11px] text-[#5e4b75]">
                  Simple binary signal (Yes/No) that triggers only when the probability equals or exceeds the threshold.
                </p>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-50 border border-[#e2e8f0]">
                <strong className="text-emerald-700 block mb-1">4. Clinical Diagnosis</strong>
                <p className="text-[11px] text-[#5e4b75]">
                  A comprehensive determination made exclusively by a doctor evaluating patient history, exams, and labs.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-[#e2e8f0] bg-slate-50/90 flex items-center justify-between text-xs text-[#5e4b75]">
          <span>LungVision AI · Investigational Decision Support Architecture</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-white border border-[#e2e8f0] text-[#1a082e] hover:border-amber-400 hover:bg-slate-50 transition-colors text-xs font-semibold shadow-xs"
          >
            Close Guide
          </button>
        </div>
      </div>
    </div>
  );
};
