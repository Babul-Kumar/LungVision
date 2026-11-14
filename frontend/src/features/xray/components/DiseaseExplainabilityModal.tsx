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
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-medical-card border border-medical-border rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-5 border-b border-medical-border flex items-center justify-between bg-medical-surface/80">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-cyan-950/60 border border-cyan-800/60 flex items-center justify-center text-medical-cyan">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-white">Clinical Pathology Guide & AI Explainability</h3>
                <Badge variant="cyan" size="sm">Medically Reviewed</Badge>
              </div>
              <p className="text-xs text-medical-muted mt-0.5">
                Understanding what the AI evaluates, how thresholds work, and why model scores are not diagnoses.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-medical-muted hover:text-white hover:bg-medical-surface transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Disease Tabs Navigation */}
        <div className="flex items-center gap-1.5 px-5 py-2.5 bg-medical-surface/40 border-b border-medical-border/80 overflow-x-auto scrollbar-thin">
          {EXPECTED_6_DISEASES.map((d) => (
            <button
              key={d}
              onClick={() => setSelectedDisease(d)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                selectedDisease === d
                  ? 'bg-medical-cyan text-slate-950 font-semibold shadow-sm'
                  : 'text-medical-muted hover:text-white hover:bg-medical-surface'
              }`}
            >
              {d}
            </button>
          ))}
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 text-sm">
          {/* Active Condition Overview Header */}
          <div className="p-4 rounded-xl bg-medical-surface border border-medical-border flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="text-[11px] font-mono text-medical-cyan uppercase tracking-wider">
                Thoracic Pathology Profile
              </div>
              <h4 className="text-xl font-bold text-white mt-0.5">
                {profile.name} <span className="text-sm font-normal text-medical-muted">({profile.patientFriendlyName})</span>
              </h4>
            </div>

            {liveProb && (
              <div className="flex items-center gap-4 bg-medical-card/80 px-4 py-2 rounded-xl border border-medical-border">
                <div>
                  <span className="text-[10px] text-medical-muted block uppercase">Current Scan Score</span>
                  <span className="text-sm font-mono font-bold text-white">
                    {(liveProb.probability * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="h-6 w-px bg-medical-border" />
                <div>
                  <span className="text-[10px] text-medical-muted block uppercase">Threshold</span>
                  <span className="text-sm font-mono text-slate-300">
                    {(liveProb.threshold * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="h-6 w-px bg-medical-border" />
                <div>
                  <span className="text-[10px] text-medical-muted block uppercase">Flag Status</span>
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
            <div className="p-4 rounded-xl bg-medical-card/50 border border-medical-border space-y-2">
              <h5 className="font-semibold text-white flex items-center gap-2 text-xs uppercase tracking-wider text-medical-cyan">
                <Info className="w-4 h-4" /> What is {profile.name}?
              </h5>
              <p className="text-xs text-medical-muted leading-relaxed">{profile.whatItIs}</p>
            </div>

            <div className="p-4 rounded-xl bg-medical-card/50 border border-medical-border space-y-2">
              <h5 className="font-semibold text-white flex items-center gap-2 text-xs uppercase tracking-wider text-medical-teal">
                <Activity className="w-4 h-4" /> What does it mean clinically?
              </h5>
              <p className="text-xs text-medical-muted leading-relaxed">{profile.clinicalMeaning}</p>
            </div>
          </div>

          {/* Section 2: AI Model Score & Experimental Threshold */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-cyan-950/20 border border-cyan-800/40 space-y-2">
              <h5 className="font-semibold text-cyan-300 flex items-center gap-2 text-xs uppercase tracking-wider">
                <Sliders className="w-4 h-4" /> What the Model Score Represents
              </h5>
              <p className="text-xs text-slate-300 leading-relaxed">{profile.modelScoreMeaning}</p>
            </div>

            <div className="p-4 rounded-xl bg-teal-950/20 border border-teal-800/40 space-y-2">
              <h5 className="font-semibold text-teal-300 flex items-center gap-2 text-xs uppercase tracking-wider">
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
          <div className="p-4 rounded-xl bg-medical-surface border border-medical-border space-y-2.5">
            <h5 className="text-xs font-semibold uppercase tracking-wider text-medical-subtle">
              Hallmark Radiographic Visual Features Looked for by Radiologists:
            </h5>
            <ul className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {profile.radiographicSigns.map((sign, idx) => (
                <li
                  key={idx}
                  className="p-2.5 rounded-lg bg-medical-card border border-medical-border/60 text-xs text-slate-300 flex items-start gap-2"
                >
                  <span className="w-4 h-4 rounded-full bg-medical-cyan/20 text-medical-cyan flex items-center justify-center shrink-0 text-[10px] font-bold">
                    {idx + 1}
                  </span>
                  <span>{sign}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Four-Way Differentiation Matrix */}
          <div className="p-4 rounded-xl bg-medical-card border border-medical-border/80 space-y-3">
            <h5 className="text-xs font-semibold uppercase tracking-wider text-white">
              Understanding the Difference: 4 Distinct Levels
            </h5>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-2.5 text-xs">
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border">
                <strong className="text-medical-cyan block mb-1">1. Model Probability</strong>
                <p className="text-[11px] text-medical-muted">
                  Continuous number (0.00 to 1.00) indicating visual pattern match strength inside the neural network.
                </p>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border">
                <strong className="text-medical-teal block mb-1">2. Experimental Threshold</strong>
                <p className="text-[11px] text-medical-muted">
                  Mathematical cutoff derived from NIH benchmark validation to separate high from low activations.
                </p>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border">
                <strong className="text-amber-300 block mb-1">3. Threshold Flag</strong>
                <p className="text-[11px] text-medical-muted">
                  Simple binary signal (Yes/No) that triggers only when the probability equals or exceeds the threshold.
                </p>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border">
                <strong className="text-emerald-400 block mb-1">4. Clinical Diagnosis</strong>
                <p className="text-[11px] text-medical-muted">
                  A comprehensive determination made exclusively by a doctor evaluating patient history, exams, and labs.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-medical-border bg-medical-surface/80 flex items-center justify-between text-xs text-medical-muted">
          <span>LungVision AI · Investigational Decision Support Architecture</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-medical-surface border border-medical-border text-white hover:bg-medical-card transition-colors text-xs font-medium"
          >
            Close Guide
          </button>
        </div>
      </div>
    </div>
  );
};
