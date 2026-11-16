import React, { useState } from 'react';
import {
  HelpCircle,
  CheckCircle2,
  AlertTriangle,
  Info,
  ChevronDown,
  ChevronUp,
  Stethoscope,
} from 'lucide-react';
import {
  EXPECTED_6_DISEASES,
  DiseaseName,
  XRayPredictionResponse,
  XRayBinaryResponse,
} from '@/types/xray';
import { DISEASE_CLINICAL_PROFILES } from '@/types/explainability';
import { Badge } from '@/components/common/Badge';

interface PatientFriendlyXRayViewProps {
  fileName?: string;
  fileSize?: number;
  multiLabelResult: XRayPredictionResponse | null;
  binaryResult: XRayBinaryResponse | null;
  activeModelType: 'multilabel' | 'binary';
  onOpenExplainability: (disease?: DiseaseName) => void;
  onOpenGlossary: () => void;
}

export const PatientFriendlyXRayView: React.FC<PatientFriendlyXRayViewProps> = ({
  fileName,
  fileSize,
  multiLabelResult,
  binaryResult,
  activeModelType,
  onOpenExplainability,
  onOpenGlossary,
}) => {
  const [showDetailedScores, setShowDetailedScores] = useState(false);

  // If no analysis executed
  if (!multiLabelResult && !binaryResult) {
    return (
      <div className="p-8 text-center border border-dashed border-medical-border rounded-2xl bg-medical-card/40 space-y-3">
        <HelpCircle className="w-10 h-10 text-medical-subtle mx-auto" />
        <h4 className="text-base font-semibold text-white">No Patient Summary Available</h4>
        <p className="text-xs text-medical-muted max-w-md mx-auto">
          Please upload a frontal chest radiograph on the left and select "Analyze" to generate a simple, plain-language patient explanation.
        </p>
      </div>
    );
  }

  // 1. Multi-label DenseNet121 Patient Explanation
  if (activeModelType === 'multilabel' && multiLabelResult) {
    const summary = multiLabelResult.summary;
    const highestDisease = summary.highest_probability_disease as DiseaseName;
    const highestScore = summary.highest_probability_value;
    const highestProfile = DISEASE_CLINICAL_PROFILES[highestDisease];
    const highestThreshold = multiLabelResult.predictions[highestDisease]?.threshold ?? highestProfile?.calibratedThreshold;
    const hasFlagged = summary.flagged_conditions_count > 0;

    return (
      <div className="space-y-5 animate-in fade-in duration-300">
        {/* Banner: Analysis Completed Summary */}
        <div
          className={`p-5 rounded-2xl border ${
            hasFlagged
              ? 'bg-amber-950/20 border-amber-800/40'
              : 'bg-[#160c26]/80 border-white/10'
          }`}
        >
          <div className="flex items-start gap-3.5">
            <div
              className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 ${
                hasFlagged
                  ? 'bg-amber-950/60 text-amber-400 border border-amber-800/60'
                  : 'bg-white/10 text-slate-200 border border-white/20'
              }`}
            >
              {hasFlagged ? <AlertTriangle className="w-5 h-5" /> : <CheckCircle2 className="w-5 h-5" />}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono uppercase tracking-wider text-amber-400">
                  Plain-Language Summary
                </span>
                <Badge variant={hasFlagged ? 'amber' : 'silver'} size="sm">
                  {hasFlagged ? `${summary.flagged_conditions_count} of 6 Above Threshold` : 'All Below Experimental Thresholds'}
                </Badge>
              </div>
              <h3 className="text-lg font-bold text-white mt-1">
                AI Analysis Completed
              </h3>
              <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                The AI model evaluated this image for <strong>six specific lung conditions</strong>.
                {hasFlagged
                  ? ` The model produced a score above its experimental decision cutoff for ${summary.flagged_conditions_count} of the 6 conditions.`
                  : ' None of the six evaluated conditions exceeded the model’s experimental decision thresholds.'}
              </p>
            </div>
          </div>
        </div>

        {/* The 6 Core Plain-Language Patient Questions */}
        <div className="space-y-4">
          {/* Question 1: What image was analyzed? */}
          <div className="p-4 rounded-xl bg-medical-surface border border-white/10 space-y-1.5">
            <h4 className="text-xs font-bold text-white flex items-center gap-2 uppercase tracking-wider text-amber-400">
              <span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[11px] font-bold">1</span>
              What image was analyzed?
            </h4>
            <p className="text-xs text-medical-muted leading-relaxed pl-7">
              You uploaded <strong className="text-slate-200">{fileName || 'chest radiograph'}</strong>
              {fileSize ? ` (${(fileSize / (1024 * 1024)).toFixed(2)} MB)` : ''}.
              This is a standard 2D frontal chest X-ray image showing the ribs, lungs, and heart silhouette.
            </p>
          </div>

          {/* Question 2: What did the AI evaluate? */}
          <div className="p-4 rounded-xl bg-medical-surface border border-white/10 space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold text-white flex items-center gap-2 uppercase tracking-wider text-slate-200">
                <span className="w-5 h-5 rounded-full bg-white/10 text-slate-200 flex items-center justify-center text-[11px] font-bold">2</span>
                What did the AI evaluate?
              </h4>
              <button
                onClick={onOpenGlossary}
                className="text-[11px] text-amber-400 hover:underline flex items-center gap-1"
              >
                <HelpCircle className="w-3.5 h-3.5" /> Medical Glossary
              </button>
            </div>
            <p className="text-xs text-medical-muted leading-relaxed pl-7">
              The AI was trained exclusively on 6 specific conditions:
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pl-7 pt-1">
              {EXPECTED_6_DISEASES.map((d) => (
                <button
                  key={d}
                  onClick={() => onOpenExplainability(d)}
                  className="p-2 rounded-lg bg-medical-card border border-white/10 text-left hover:border-amber-400/40 transition-colors"
                >
                  <span className="text-xs font-semibold text-white block">{d}</span>
                  <span className="text-[10px] text-medical-muted block truncate">
                    {DISEASE_CLINICAL_PROFILES[d]?.patientFriendlyName}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Question 3: What did the model return? */}
          <div className="p-4 rounded-xl bg-medical-surface border border-white/10 space-y-1.5">
            <h4 className="text-xs font-bold text-white flex items-center gap-2 uppercase tracking-wider text-slate-200">
              <span className="w-5 h-5 rounded-full bg-[#2a1744] text-amber-300 border border-amber-500/30 flex items-center justify-center text-[11px] font-bold">3</span>
              What did the model return?
            </h4>
            <div className="pl-7 text-xs text-slate-300 space-y-2">
              <p>
                The model calculated a continuous mathematical activation (between 0% and 100%) for each of the six conditions.
              </p>
              {/* Callout: Highest Model Score */}
              <div className="p-3.5 rounded-xl bg-medical-card border border-white/10 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white">
                    Highest Model Score: <span className="text-amber-400">{highestDisease}</span>
                  </span>
                  <span className="font-mono text-xs font-bold text-amber-400">
                    {(highestScore * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="text-[11px] text-medical-muted leading-relaxed">
                  The model’s experimental cutoff threshold for {highestDisease} is{' '}
                  <strong className="text-slate-200">{(highestThreshold * 100).toFixed(2)}%</strong>.
                  Because {(highestScore * 100).toFixed(2)}% is{' '}
                  {highestScore >= highestThreshold ? 'above' : 'below'} this threshold, it is{' '}
                  {highestScore >= highestThreshold ? (
                    <strong className="text-amber-400">flagged for doctor review</strong>
                  ) : (
                    <span className="text-emerald-400">not flagged</span>
                  )}.
                </div>
                <p className="text-[10px] text-medical-subtle pt-1 border-t border-white/[0.08] italic">
                  * Note: "Highest model score" means this was simply the highest mathematical score among the 6 tested conditions. It is NOT a confirmed diagnosis, suspicion, or certainty.
                </p>
              </div>
            </div>
          </div>

          {/* Question 4: What do these results mean? (Strictly No "All Clear" or "Normal") */}
          <div className="p-4 rounded-xl bg-medical-surface border border-white/10 space-y-2">
            <h4 className="text-xs font-bold text-white flex items-center gap-2 uppercase tracking-wider text-amber-400">
              <span className="w-5 h-5 rounded-full bg-amber-950/60 text-amber-400 border border-amber-800/60 flex items-center justify-center text-[11px] font-bold">4</span>
              What do these results mean?
            </h4>
            <div className="pl-7 text-xs text-slate-300 space-y-2 leading-relaxed">
              {!hasFlagged ? (
                <>
                  <p>
                    The model did <strong>not</strong> produce a threshold-positive result for the six specific conditions it evaluates.
                  </p>
                  <div className="p-3 bg-amber-950/20 border border-amber-800/40 rounded-lg text-amber-200 text-xs">
                    <strong>Important Medical Notice:</strong> This does <u>NOT</u> prove that the lungs are completely healthy, normal, or "all clear."
                  </div>
                  <p className="text-medical-muted text-xs">
                    Many conditions—such as pulmonary nodules, interstitial lung disease, rib fractures, tuberculosis, vascular anomalies, or early infections—are outside the scope of this experimental model and can still be present.
                  </p>
                </>
              ) : (
                <>
                  <p>
                    The AI detected visual patterns that matched its training examples for {summary.flagged_conditions_count} condition(s) above the experimental threshold.
                  </p>
                  <p className="text-medical-muted text-xs">
                    This is an algorithmic flag created for medical research. It indicates an area warranting closer inspection by a doctor, NOT a definitive illness.
                  </p>
                </>
              )}
            </div>
          </div>

          {/* Question 5: What are the limitations? */}
          <div className="p-4 rounded-xl bg-medical-surface border border-white/10 space-y-1.5">
            <h4 className="text-xs font-bold text-white flex items-center gap-2 uppercase tracking-wider text-slate-300">
              <span className="w-5 h-5 rounded-full bg-[#2a1744] text-amber-300 border border-amber-500/30 flex items-center justify-center text-[11px] font-bold">5</span>
              What are the limitations of this analysis?
            </h4>
            <ul className="pl-7 space-y-1.5 text-xs text-medical-muted list-disc list-outside">
              <li>This AI is an investigational research tool, NOT an FDA-cleared diagnostic medical device.</li>
              <li>Plain 2D chest radiographs compress 3D human anatomy into a flat image; subtle abnormalities can be hidden behind the heart or ribs.</li>
              <li>The AI does not have access to your symptoms, vital signs, physical exam, blood tests, or medical history.</li>
            </ul>
          </div>

          {/* Question 6: What should the user understand next? */}
          <div className="p-4 rounded-xl bg-[#1c102f] border border-amber-500/30 space-y-2">
            <h4 className="text-xs font-bold text-amber-300 flex items-center gap-2 uppercase tracking-wider">
              <Stethoscope className="w-4 h-4 text-amber-400" />
              What should you do next?
            </h4>
            <div className="text-xs text-slate-200 space-y-1.5 leading-relaxed pl-6">
              <p>
                <strong>Always consult a qualified doctor or radiologist:</strong> A licensed physician will review the actual high-resolution DICOM X-ray alongside your medical history, symptoms (such as cough, shortness of breath, or fever), and other clinical findings.
              </p>
              <p className="text-medical-muted text-[11px]">
                Never change your medications, start treatments, or ignore symptoms based on these AI research scores.
              </p>
            </div>
          </div>
        </div>

        {/* Expandable Technical Details for Inquisitive Patients */}
        <div className="border border-white/10 rounded-xl overflow-hidden bg-medical-surface/40">
          <button
            onClick={() => setShowDetailedScores(!showDetailedScores)}
            className="w-full px-4 py-3 flex items-center justify-between text-xs text-medical-muted hover:text-white transition-colors"
          >
            <span className="flex items-center gap-2 font-medium">
              <Info className="w-4 h-4 text-amber-400" />
              {showDetailedScores ? 'Hide Detailed Numbers' : 'View Detailed Numbers for All 6 Conditions'}
            </span>
            {showDetailedScores ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showDetailedScores && (
            <div className="p-4 border-t border-white/10 space-y-2.5 bg-medical-card/30">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                {EXPECTED_6_DISEASES.map((disease) => {
                  const pred = multiLabelResult.predictions[disease];
                  return (
                    <div
                      key={disease}
                      className="p-3 rounded-lg bg-medical-surface border border-white/10 flex items-center justify-between"
                    >
                      <div>
                        <span className="font-semibold text-white block">{disease}</span>
                        <span className="text-[10px] text-medical-muted">
                          Threshold: {(pred.threshold * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-right">
                        <span className="font-mono font-bold text-white block">
                          {(pred.probability * 100).toFixed(1)}%
                        </span>
                        <span
                          className={`text-[10px] font-medium ${
                            pred.experimental_flag ? 'text-amber-400' : 'text-emerald-400'
                          }`}
                        >
                          {pred.experimental_flag ? 'Above Cutoff' : 'Below Cutoff'}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // 2. Binary ResNet18 Patient Explanation
  if (activeModelType === 'binary' && binaryResult) {
    const isAbnormal = binaryResult.prediction.classification === 'Abnormal';
    return (
      <div className="space-y-5 animate-in fade-in duration-300">
        <div
          className={`p-5 rounded-2xl border ${
            isAbnormal ? 'bg-amber-950/20 border-amber-800/40' : 'bg-[#160c26]/80 border-white/10'
          }`}
        >
          <div className="flex items-start gap-3.5">
            <div
              className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 ${
                isAbnormal
                  ? 'bg-amber-950/60 text-amber-400 border border-amber-800/60'
                  : 'bg-white/10 text-slate-200 border border-white/20'
              }`}
            >
              {isAbnormal ? <AlertTriangle className="w-5 h-5" /> : <CheckCircle2 className="w-5 h-5" />}
            </div>
            <div className="flex-1">
              <span className="text-xs font-mono uppercase tracking-wider text-amber-400">
                Plain-Language Triage Summary
              </span>
              <h3 className="text-lg font-bold text-white mt-1">
                Binary Screening Model Output: {isAbnormal ? 'Flagged as Abnormal' : 'Flagged as Normal'}
              </h3>
              <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                {isAbnormal
                  ? 'The AI detected patterns commonly associated with abnormal chest findings in its training data, exceeding the calibrated threshold.'
                  : 'The AI model did not detect abnormal visual patterns above its threshold. However, this is an automated screening tool and does NOT guarantee healthy lungs.'}
              </p>
            </div>
          </div>
        </div>

        {/* Plain Language Interpretation */}
        <div className="p-4 rounded-xl bg-medical-surface border border-white/10 space-y-2 text-xs">
          <h4 className="font-bold text-white uppercase tracking-wider text-amber-400">
            What this binary score means
          </h4>
          <p className="text-slate-300 leading-relaxed">
            The model was calibrated at <strong>90% sensitivity</strong> on its benchmark dataset.
            The abnormal probability was{' '}
            <strong className="text-white">{(binaryResult.prediction.abnormal_probability * 100).toFixed(2)}%</strong>{' '}
            compared to the calibrated decision cutoff of{' '}
            <strong className="text-slate-300">{(binaryResult.prediction.calibrated_threshold * 100).toFixed(2)}%</strong>.
          </p>
          <p className="text-medical-muted text-[11px] pt-1 border-t border-white/[0.08]">
            A "Normal" result from a binary AI model does not exclude subtle or non-pulmonary abnormalities. A qualified radiologist must evaluate the complete chest examination.
          </p>
        </div>
      </div>
    );
  }

  return null;
};
