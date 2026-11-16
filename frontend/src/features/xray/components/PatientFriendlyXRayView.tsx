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
      <div className="p-8 text-center border border-dashed border-[#cbd5e1] rounded-2xl bg-white space-y-3 shadow-xs">
        <HelpCircle className="w-10 h-10 text-[#7e6c96] mx-auto" />
        <h4 className="text-base font-semibold text-[#1a082e]">No Patient Summary Available</h4>
        <p className="text-xs text-[#5e4b75] max-w-md mx-auto">
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
              ? 'bg-amber-50/80 border-amber-300 shadow-xs'
              : 'bg-white border-[#e2e8f0] shadow-xs'
          }`}
        >
          <div className="flex items-start gap-3.5">
            <div
              className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 ${
                hasFlagged
                  ? 'bg-amber-100 text-amber-800 border border-amber-300'
                  : 'bg-slate-100 text-[#1a082e] border border-[#e2e8f0]'
              }`}
            >
              {hasFlagged ? <AlertTriangle className="w-5 h-5 text-amber-700" /> : <CheckCircle2 className="w-5 h-5 text-emerald-600" />}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono uppercase tracking-wider text-amber-900 font-semibold">
                  Plain-Language Summary
                </span>
                <Badge variant={hasFlagged ? 'amber' : 'silver'} size="sm">
                  {hasFlagged ? `${summary.flagged_conditions_count} of 6 Above Threshold` : 'All Below Experimental Thresholds'}
                </Badge>
              </div>
              <h3 className="text-lg font-bold text-[#1a082e] mt-1">
                AI Analysis Completed
              </h3>
              <p className="text-xs text-[#5e4b75] mt-1 leading-relaxed">
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
          <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] space-y-1.5 shadow-xs">
            <h4 className="text-xs font-bold text-[#1a082e] flex items-center gap-2 uppercase tracking-wider text-amber-900">
              <span className="w-5 h-5 rounded-full bg-amber-100 text-amber-900 flex items-center justify-center text-[11px] font-bold">1</span>
              What image was analyzed?
            </h4>
            <p className="text-xs text-[#5e4b75] leading-relaxed pl-7">
              You uploaded <strong className="text-[#1a082e]">{fileName || 'chest radiograph'}</strong>
              {fileSize ? ` (${(fileSize / (1024 * 1024)).toFixed(2)} MB)` : ''}.
              This is a standard 2D frontal chest X-ray image showing the ribs, lungs, and heart silhouette.
            </p>
          </div>

          {/* Question 2: What did the AI evaluate? */}
          <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] space-y-2 shadow-xs">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold text-[#1a082e] flex items-center gap-2 uppercase tracking-wider">
                <span className="w-5 h-5 rounded-full bg-slate-100 text-[#1a082e] border border-[#e2e8f0] flex items-center justify-center text-[11px] font-bold">2</span>
                What did the AI evaluate?
              </h4>
              <button
                onClick={onOpenGlossary}
                className="text-[11px] text-amber-700 hover:underline flex items-center gap-1 font-semibold"
              >
                <HelpCircle className="w-3.5 h-3.5" /> Medical Glossary
              </button>
            </div>
            <p className="text-xs text-[#5e4b75] leading-relaxed pl-7">
              The AI was trained exclusively on 6 specific conditions:
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pl-7 pt-1">
              {EXPECTED_6_DISEASES.map((d) => (
                <button
                  key={d}
                  onClick={() => onOpenExplainability(d)}
                  className="p-2 rounded-lg bg-[#f8fafc] border border-[#e2e8f0] text-left hover:border-amber-400 transition-colors shadow-xs"
                >
                  <span className="text-xs font-semibold text-[#1a082e] block">{d}</span>
                  <span className="text-[10px] text-[#5e4b75] block truncate">
                    {DISEASE_CLINICAL_PROFILES[d]?.patientFriendlyName}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Question 3: What did the model return? */}
          <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] space-y-1.5 shadow-xs">
            <h4 className="text-xs font-bold text-[#1a082e] flex items-center gap-2 uppercase tracking-wider">
              <span className="w-5 h-5 rounded-full bg-amber-100 text-amber-900 border border-amber-300 flex items-center justify-center text-[11px] font-bold">3</span>
              What did the model return?
            </h4>
            <div className="pl-7 text-xs text-[#5e4b75] space-y-2">
              <p>
                The model calculated a continuous mathematical activation (between 0% and 100%) for each of the six conditions.
              </p>
              {/* Callout: Highest Model Score */}
              <div className="p-3.5 rounded-xl bg-[#f8fafc] border border-[#e2e8f0] space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-[#1a082e]">
                    Highest Model Score: <span className="text-amber-800">{highestDisease}</span>
                  </span>
                  <span className="font-mono text-xs font-bold text-amber-800">
                    {(highestScore * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="text-[11px] text-[#5e4b75] leading-relaxed">
                  The model’s experimental cutoff threshold for {highestDisease} is{' '}
                  <strong className="text-[#1a082e]">{(highestThreshold * 100).toFixed(2)}%</strong>.
                  Because {(highestScore * 100).toFixed(2)}% is{' '}
                  {highestScore >= highestThreshold ? 'above' : 'below'} this threshold, it is{' '}
                  {highestScore >= highestThreshold ? (
                    <strong className="text-amber-800">flagged for doctor review</strong>
                  ) : (
                    <span className="text-emerald-700 font-semibold">not flagged</span>
                  )}.
                </div>
                <p className="text-[10px] text-[#5e4b75] pt-1 border-t border-[#e2e8f0] italic font-medium">
                  * Note: "Highest model score" means this was simply the highest mathematical score among the 6 tested conditions. It is NOT a confirmed diagnosis, suspicion, or certainty.
                </p>
              </div>
            </div>
          </div>

          {/* Question 4: What do these results mean? (Strictly No "All Clear" or "Normal") */}
          <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] space-y-2 shadow-xs">
            <h4 className="text-xs font-bold text-[#1a082e] flex items-center gap-2 uppercase tracking-wider text-amber-800">
              <span className="w-5 h-5 rounded-full bg-amber-100 text-amber-900 border border-amber-300 flex items-center justify-center text-[11px] font-bold">4</span>
              What do these results mean?
            </h4>
            <div className="pl-7 text-xs text-[#5e4b75] space-y-2 leading-relaxed">
              {!hasFlagged ? (
                <>
                  <p>
                    The model did <strong>not</strong> produce a threshold-positive result for the six specific conditions it evaluates.
                  </p>
                  <div className="p-3 bg-amber-50 border border-amber-300 rounded-lg text-amber-950 text-xs">
                    <strong>Important Medical Notice:</strong> This does <u>NOT</u> prove that the lungs are completely healthy, normal, or "all clear."
                  </div>
                  <p className="text-[#5e4b75] text-xs">
                    Many conditions—such as pulmonary nodules, interstitial lung disease, rib fractures, tuberculosis, vascular anomalies, or early infections—are outside the scope of this experimental model and can still be present.
                  </p>
                </>
              ) : (
                <>
                  <p>
                    The AI detected visual patterns that matched its training examples for {summary.flagged_conditions_count} condition(s) above the experimental threshold.
                  </p>
                  <p className="text-[#5e4b75] text-xs">
                    This is an algorithmic flag created for medical research. It indicates an area warranting closer inspection by a doctor, NOT a definitive illness.
                  </p>
                </>
              )}
            </div>
          </div>

          {/* Question 5: What are the limitations? */}
          <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] space-y-1.5 shadow-xs">
            <h4 className="text-xs font-bold text-[#1a082e] flex items-center gap-2 uppercase tracking-wider">
              <span className="w-5 h-5 rounded-full bg-amber-100 text-amber-900 border border-amber-300 flex items-center justify-center text-[11px] font-bold">5</span>
              What are the limitations of this analysis?
            </h4>
            <ul className="pl-7 space-y-1.5 text-xs text-[#5e4b75] list-disc list-outside">
              <li>This AI is an investigational research tool, NOT an FDA-cleared diagnostic medical device.</li>
              <li>Plain 2D chest radiographs compress 3D human anatomy into a flat image; subtle abnormalities can be hidden behind the heart or ribs.</li>
              <li>The AI does not have access to your symptoms, vital signs, physical exam, blood tests, or medical history.</li>
            </ul>
          </div>

          {/* Question 6: What should the user understand next? */}
          <div className="p-4 rounded-xl bg-amber-50/60 border border-amber-300 space-y-2 shadow-xs">
            <h4 className="text-xs font-bold text-amber-900 flex items-center gap-2 uppercase tracking-wider">
              <Stethoscope className="w-4 h-4 text-amber-700" />
              What should you do next?
            </h4>
            <div className="text-xs text-[#1a082e] space-y-1.5 leading-relaxed pl-6">
              <p>
                <strong>Always consult a qualified doctor or radiologist:</strong> A licensed physician will review the actual high-resolution DICOM X-ray alongside your medical history, symptoms (such as cough, shortness of breath, or fever), and other clinical findings.
              </p>
              <p className="text-[#5e4b75] text-[11px]">
                Never change your medications, start treatments, or ignore symptoms based on these AI research scores.
              </p>
            </div>
          </div>
        </div>

        {/* Expandable Technical Details for Inquisitive Patients */}
        <div className="border border-[#e2e8f0] rounded-xl overflow-hidden bg-white shadow-xs">
          <button
            onClick={() => setShowDetailedScores(!showDetailedScores)}
            className="w-full px-4 py-3 flex items-center justify-between text-xs text-[#1a082e] hover:bg-slate-50 transition-colors"
          >
            <span className="flex items-center gap-2 font-medium">
              <Info className="w-4 h-4 text-amber-600" />
              {showDetailedScores ? 'Hide Detailed Numbers' : 'View Detailed Numbers for All 6 Conditions'}
            </span>
            {showDetailedScores ? <ChevronUp className="w-4 h-4 text-amber-600" /> : <ChevronDown className="w-4 h-4 text-[#5e4b75]" />}
          </button>

          {showDetailedScores && (
            <div className="p-4 border-t border-[#e2e8f0] space-y-2.5 bg-[#f8fafc]">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                {EXPECTED_6_DISEASES.map((disease) => {
                  const pred = multiLabelResult.predictions[disease];
                  return (
                    <div
                      key={disease}
                      className="p-3 rounded-lg bg-white border border-[#e2e8f0] flex items-center justify-between shadow-2xs"
                    >
                      <div>
                        <span className="font-semibold text-[#1a082e] block">{disease}</span>
                        <span className="text-[10px] text-[#5e4b75]">
                          Threshold: {(pred.threshold * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-right">
                        <span className="font-mono font-bold text-[#1a082e] block">
                          {(pred.probability * 100).toFixed(1)}%
                        </span>
                        <span
                          className={`text-[10px] font-medium ${
                            pred.experimental_flag ? 'text-amber-800' : 'text-emerald-700'
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
            isAbnormal ? 'bg-amber-50/80 border-amber-300 shadow-xs' : 'bg-white border-[#e2e8f0] shadow-xs'
          }`}
        >
          <div className="flex items-start gap-3.5">
            <div
              className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 ${
                isAbnormal
                  ? 'bg-amber-100 text-amber-800 border border-amber-300'
                  : 'bg-slate-100 text-[#1a082e] border border-[#e2e8f0]'
              }`}
            >
              {isAbnormal ? <AlertTriangle className="w-5 h-5 text-amber-700" /> : <CheckCircle2 className="w-5 h-5 text-emerald-600" />}
            </div>
            <div className="flex-1">
              <span className="text-xs font-mono uppercase tracking-wider text-amber-900 font-semibold">
                Plain-Language Triage Summary
              </span>
              <h3 className="text-lg font-bold text-[#1a082e] mt-1">
                Binary Screening Model Output: {isAbnormal ? 'Flagged as Abnormal' : 'Flagged as Normal'}
              </h3>
              <p className="text-xs text-[#5e4b75] mt-1 leading-relaxed">
                {isAbnormal
                  ? 'The AI detected patterns commonly associated with abnormal chest findings in its training data, exceeding the calibrated threshold.'
                  : 'The AI model did not detect abnormal visual patterns above its threshold. However, this is an automated screening tool and does NOT guarantee healthy lungs.'}
              </p>
            </div>
          </div>
        </div>

        {/* Plain Language Interpretation */}
        <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] space-y-2 text-xs shadow-xs">
          <h4 className="font-bold text-[#1a082e] uppercase tracking-wider text-amber-900">
            What this binary score means
          </h4>
          <p className="text-[#5e4b75] leading-relaxed">
            The model was calibrated at <strong>90% sensitivity</strong> on its benchmark dataset.
            The abnormal probability was{' '}
            <strong className="text-[#1a082e]">{(binaryResult.prediction.abnormal_probability * 100).toFixed(2)}%</strong>{' '}
            compared to the calibrated decision cutoff of{' '}
            <strong className="text-[#1a082e]">{(binaryResult.prediction.calibrated_threshold * 100).toFixed(2)}%</strong>.
          </p>
          <p className="text-[#7e6c96] text-[11px] pt-1 border-t border-[#e2e8f0]">
            A "Normal" result from a binary AI model does not exclude subtle or non-pulmonary abnormalities. A qualified radiologist must evaluate the complete chest examination.
          </p>
        </div>
      </div>
    );
  }

  return null;
};
