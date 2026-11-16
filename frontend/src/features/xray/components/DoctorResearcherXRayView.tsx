import React, { useState } from 'react';
import {
  Download,
  Printer,
  FileCode,
  Clock,
  Cpu,
  ShieldAlert,
  AlertCircle,
  CheckCircle2,
  HelpCircle,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import {
  EXPECTED_6_DISEASES,
  DiseaseName,
  XRayPredictionResponse,
  XRayBinaryResponse,
} from '@/types/xray';
import { ProbabilityBar } from '@/components/common/ProbabilityBar';
import { Badge } from '@/components/common/Badge';

interface DoctorResearcherXRayViewProps {
  fileName?: string;
  fileSize?: number;
  multiLabelResult: XRayPredictionResponse | null;
  binaryResult: XRayBinaryResponse | null;
  activeModelType: 'multilabel' | 'binary';
  onOpenExplainability: (disease?: DiseaseName) => void;
  onOpenGlossary: () => void;
}

export const DoctorResearcherXRayView: React.FC<DoctorResearcherXRayViewProps> = ({
  fileName,
  fileSize,
  multiLabelResult,
  binaryResult,
  activeModelType,
  onOpenExplainability,
  onOpenGlossary,
}) => {
  const [showRawJson, setShowRawJson] = useState(false);
  const [referenceCaseId, setReferenceCaseId] = useState<string>('unverified');

  const REFERENCE_CASES: Record<string, { name: string; labels: Record<DiseaseName, 0 | 1> }> = {
    unverified: {
      name: 'Unverified Reference / Ad-hoc Upload',
      labels: {
        Cardiomegaly: 0,
        Edema: 0,
        Consolidation: 0,
        Pneumonia: 0,
        Atelectasis: 0,
        Pneumothorax: 0,
      },
    },
    test1_normal: {
      name: 'Test 1: Normal (Healthy Thoracic Baseline)',
      labels: {
        Cardiomegaly: 0,
        Edema: 0,
        Consolidation: 0,
        Pneumonia: 0,
        Atelectasis: 0,
        Pneumothorax: 0,
      },
    },
    test2_cardiomegaly: {
      name: 'Test 2: Cardiomegaly (CTR > 0.62)',
      labels: {
        Cardiomegaly: 1,
        Edema: 0,
        Consolidation: 0,
        Pneumonia: 0,
        Atelectasis: 0,
        Pneumothorax: 0,
      },
    },
    test3_pneumothorax: {
      name: 'Test 3: Pneumothorax (Apical Pleural Line)',
      labels: {
        Cardiomegaly: 0,
        Edema: 0,
        Consolidation: 0,
        Pneumonia: 0,
        Atelectasis: 0,
        Pneumothorax: 1,
      },
    },
    test4_edema: {
      name: 'Test 4: Edema (Perihilar Bat-Wing Infiltrate)',
      labels: {
        Cardiomegaly: 0,
        Edema: 1,
        Consolidation: 0,
        Pneumonia: 0,
        Atelectasis: 0,
        Pneumothorax: 0,
      },
    },
    test5_consolidation: {
      name: 'Test 5: Consolidation (Dense Lobar Opacity)',
      labels: {
        Cardiomegaly: 0,
        Edema: 0,
        Consolidation: 1,
        Pneumonia: 0,
        Atelectasis: 0,
        Pneumothorax: 0,
      },
    },
    test6_atelectasis: {
      name: 'Test 6: Atelectasis (Discoid Basilar Collapse)',
      labels: {
        Cardiomegaly: 0,
        Edema: 0,
        Consolidation: 0,
        Pneumonia: 0,
        Atelectasis: 1,
        Pneumothorax: 0,
      },
    },
  };

  // If no analysis executed
  if (!multiLabelResult && !binaryResult) {
    return (
      <div className="p-8 text-center border border-dashed border-[#cbd5e1] rounded-2xl bg-white space-y-3 shadow-xs">
        <Cpu className="w-10 h-10 text-medical-subtle mx-auto" />
        <h4 className="text-base font-semibold text-[#1a082e]">No Telemetry Available</h4>
        <p className="text-xs text-[#5e4b75] max-w-md mx-auto">
          Upload a frontal chest radiograph on the left and execute inference to display full research-grade telemetry, raw logits, threshold flags, and validation benchmarks.
        </p>
      </div>
    );
  }

  // Export JSON handler
  const handleExportJson = () => {
    const payload = activeModelType === 'multilabel' ? multiLabelResult : binaryResult;
    if (!payload) return;

    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute(
      'download',
      `lungvision_${activeModelType}_${Date.now()}.json`
    );
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Print/PDF Handler
  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="space-y-5 animate-in fade-in duration-300">
      {/* Top Telemetry & Export Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-xl bg-white border border-[#e2e8f0] shadow-xs">
        <div className="flex items-center gap-3">
          <Badge variant="amber" size="md">
            Doctor / Researcher Telemetry
          </Badge>
          <span className="text-xs font-mono text-[#5e4b75]">
            {fileName ? `${fileName} (${fileSize ? (fileSize / (1024 * 1024)).toFixed(2) : '0'} MB) · ` : ''}
            Format: Exact Float Output (Uncalibrated to Clinical Risk)
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleExportJson}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white border border-[#e2e8f0] hover:border-amber-400 text-[#1a082e] hover:text-amber-800 transition-all shadow-xs"
            title="Download Raw Output JSON"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span>Export JSON</span>
          </button>
          <button
            onClick={handlePrint}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white border border-[#e2e8f0] hover:border-slate-400 text-[#1a082e] transition-all shadow-xs"
            title="Print or Save as PDF"
          >
            <Printer className="w-3.5 h-3.5 text-slate-500" />
            <span>Print Report</span>
          </button>
          <button
            onClick={onOpenGlossary}
            className="p-1.5 rounded-lg text-[#5e4b75] hover:text-[#1a082e] hover:bg-slate-50 bg-white border border-[#e2e8f0] shadow-xs"
            title="Terminology Glossary"
          >
            <HelpCircle className="w-4 h-4" />
          </button>
        </div>
      </div>

      {activeModelType === 'multilabel' && multiLabelResult && (
        <div className="space-y-4">
          {/* Scientific Specification Card */}
          <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] space-y-3 shadow-xs">
            <div className="text-[11px] font-mono uppercase tracking-wider text-amber-800 font-semibold">
              Deep Learning Model Provenance
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#5e4b75] block uppercase">Architecture</span>
                <span className="font-mono font-semibold text-[#1a082e] block mt-0.5">
                  {multiLabelResult.model.architecture}
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#5e4b75] block uppercase">Training Corpus</span>
                <span className="font-mono font-semibold text-[#1a082e] block mt-0.5">
                  {multiLabelResult.model.training_dataset}
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#5e4b75] block uppercase">Checkpoint State</span>
                <span className="font-mono font-semibold text-[#1a082e] block mt-0.5">
                  Epoch {multiLabelResult.model.checkpoint_epoch} (Best Val AUC)
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#5e4b75] block uppercase">Inference Latency</span>
                <span className="font-mono font-semibold text-amber-800 block mt-0.5 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {multiLabelResult.inference_time_ms.toFixed(1)} ms
                </span>
              </div>
            </div>

            <div className="text-[11px] font-mono text-[#5e4b75] bg-[#f8fafc] px-3 py-2 rounded-lg border border-[#e2e8f0] space-y-0.5">
              <div>
                <strong className="text-[#1a082e]">Checkpoint Identifier:</strong> <code>densenet121_nih_best.pth</code> (SHA-256: <code>8d3690cd...</code>)
              </div>
              <div>
                <strong className="text-[#1a082e]">Input Preprocessing:</strong> 224×224 bilinear interpolation, 3-channel RGB, ImageNet normalization (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]).
              </div>
            </div>
          </div>

          {/* 6 Pathologies in Strict NIH Order */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold uppercase tracking-wider text-[#1a082e]">
                Six Disease Output Activations (Strict NIH ChestX-ray14 Ordering)
              </h4>
              <span className="text-[11px] text-[#5e4b75] font-mono font-medium">
                Flagged: <strong className="text-[#1a082e] font-bold">{multiLabelResult.summary.flagged_conditions_count}</strong> / 6
              </span>
            </div>

            <div className="space-y-2">
              {EXPECTED_6_DISEASES.map((disease) => {
                const pred = multiLabelResult.predictions[disease];
                return (
                  <div key={disease} className="space-y-1">
                    <ProbabilityBar
                      label={disease}
                      probability={pred.probability}
                      threshold={pred.threshold}
                      experimentalFlag={pred.experimental_flag}
                    />
                    <div className="flex items-center justify-between text-[11px] font-mono text-[#5e4b75] px-1">
                      <span>
                        Raw Probability: <strong className="text-[#1a082e] font-bold">{pred.probability.toFixed(6)}</strong> · Threshold: <strong className="text-[#1a082e] font-bold">{pred.threshold.toFixed(6)}</strong>
                      </span>
                      <button
                        onClick={() => onOpenExplainability(disease)}
                        className="text-amber-800 hover:text-amber-950 hover:underline font-semibold text-[11px] transition-colors"
                      >
                        Clinical Details & Signs →
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Research Evaluation Panel */}
          <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] space-y-3.5 shadow-xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#e2e8f0] pb-3">
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-[#1a082e] flex items-center gap-1.5">
                  <ShieldAlert className="w-4 h-4 text-amber-600" />
                  Research Evaluation & Clinical Concordance Panel
                </h4>
                <p className="text-[11px] text-[#5e4b75] mt-0.5">
                  Separates continuous model activation scores from locked experimental thresholds and reference dataset labels.
                </p>
              </div>

              {/* Reference Dataset Case Selector */}
              <div className="flex items-center gap-2">
                <label className="text-[11px] font-semibold text-[#5e4b75]">Reference Case:</label>
                <select
                  value={referenceCaseId}
                  onChange={(e) => setReferenceCaseId(e.target.value)}
                  className="text-xs font-mono bg-[#f8fafc] border border-[#cbd5e1] rounded-lg px-2.5 py-1 text-[#1a082e] focus:outline-none focus:ring-1 focus:ring-amber-500"
                >
                  {Object.entries(REFERENCE_CASES).map(([key, item]) => (
                    <option key={key} value={key}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse font-mono">
                <thead>
                  <tr className="border-b border-[#e2e8f0] text-[#5e4b75] bg-slate-50 font-semibold uppercase text-[10px] tracking-wider">
                    <th className="py-2.5 px-3">Pathology</th>
                    <th className="py-2.5 px-3">Model Activation Score</th>
                    <th className="py-2.5 px-3">Locked Threshold</th>
                    <th className="py-2.5 px-3">Predicted Flag</th>
                    <th className="py-2.5 px-3">Reference Label</th>
                    <th className="py-2.5 px-3 font-sans">Evaluation Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#e2e8f0] text-[#1a082e]">
                  {EXPECTED_6_DISEASES.map((disease) => {
                    const pred = multiLabelResult.predictions[disease];
                    const rawScore = pred.probability;
                    const threshold = pred.threshold;
                    const flag = pred.experimental_flag;

                    const hasRef = referenceCaseId !== 'unverified';
                    const refLabel = hasRef ? REFERENCE_CASES[referenceCaseId].labels[disease] : null;

                    let status = 'Unverified reference';
                    let statusBadgeClass = 'bg-slate-100 text-slate-700 border-slate-300';

                    if (hasRef && refLabel !== null) {
                      if (refLabel === 1 && flag) {
                        status = 'Correct positive';
                        statusBadgeClass = 'bg-emerald-100 text-emerald-800 border-emerald-300 font-bold';
                      } else if (refLabel === 0 && !flag) {
                        status = 'Correct negative';
                        statusBadgeClass = 'bg-slate-100 text-slate-700 border-slate-300';
                      } else if (refLabel === 0 && flag) {
                        status = 'False positive';
                        statusBadgeClass = 'bg-amber-100 text-amber-900 border-amber-300 font-bold';
                      } else if (refLabel === 1 && !flag) {
                        status = 'False negative';
                        statusBadgeClass = 'bg-rose-100 text-rose-900 border-rose-300 font-bold';
                      }
                    }

                    return (
                      <tr key={disease} className="hover:bg-amber-50/40 transition-colors">
                        <td className="py-2.5 px-3 font-semibold text-[#1a082e]">{disease}</td>
                        <td className="py-2.5 px-3 font-bold text-amber-900">
                          {rawScore.toFixed(6)}
                          <span className="text-[10px] text-[#5e4b75] block font-sans font-normal">
                            Uncalibrated Scalar
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-[#5e4b75]">
                          {threshold.toFixed(6)}
                          <span className="text-[10px] text-[#5e4b75] block font-sans font-normal">
                            NIH Locked
                          </span>
                        </td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`inline-block px-2 py-0.5 rounded text-[11px] font-semibold border ${
                              flag
                                ? 'bg-amber-100 text-amber-900 border-amber-300'
                                : 'bg-slate-100 text-[#5e4b75] border-[#cbd5e1]'
                            }`}
                          >
                            {flag ? 'Flagged (Positive)' : 'Below Threshold (Negative)'}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-[#5e4b75]">
                          {hasRef ? (
                            refLabel === 1 ? (
                              <span className="text-amber-800 font-bold">Positive (1)</span>
                            ) : (
                              <span>Negative (0)</span>
                            )
                          ) : (
                            <span className="text-slate-400 italic">Unspecified</span>
                          )}
                        </td>
                        <td className="py-2.5 px-3 font-sans">
                          <span className={`inline-block px-2 py-0.5 rounded text-[11px] border ${statusBadgeClass}`}>
                            {status}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="p-3 bg-amber-50/70 border border-amber-200/80 rounded-lg text-[11px] text-[#5e4b75] leading-relaxed">
              <strong className="text-amber-950">Scientific Integrity Notice:</strong> Model activation scores are raw continuous mathematical values $[0.0, 1.0]$ derived from deep convolutional feature representations on retrospective data. They do <strong>NOT</strong> represent clinical diagnostic probabilities, disease likelihood percentages, or prognostic risks.
            </div>
          </div>

          {/* Technical Caveats & Domain Generalization */}
          <div className="p-4 rounded-xl bg-amber-50/90 border border-amber-300 space-y-2 text-xs shadow-xs">
            <div className="font-bold text-amber-950 flex items-center gap-2 uppercase tracking-wider text-[11px]">
              <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0" />
              Evaluation Benchmark & Domain Generalization Caveats
            </div>
            <p className="text-amber-950/90 leading-relaxed text-[11px]">
              <strong className="text-amber-950 font-bold">External Generalization (CheXpert Drop):</strong> When evaluating this model on external Stanford CheXpert radiographs with locked NIH calibration thresholds, macro AUROC dropped from <strong className="text-amber-950 font-bold">0.8206</strong> to <strong className="text-amber-950 font-bold">0.6277</strong> (-23.5% relative drop). This demonstrates high sensitivity to hospital-specific beam hardware, projection protocols (AP vs PA), and patient demographics.
            </p>
            <p className="text-amber-950/85 leading-relaxed text-[11px]">
              <strong className="text-amber-950 font-bold">Non-Diagnostic Notice:</strong> Never treat output probabilities as clinical risk percentages. Model outputs represent relative visual activation vectors on retrospective public datasets and have NOT been validated in prospective clinical trials.
            </p>
          </div>
        </div>
      )}

      {activeModelType === 'binary' && binaryResult && (
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] space-y-3 shadow-xs">
            <div className="text-[11px] font-mono uppercase tracking-wider text-[#1a082e] font-semibold">
              Binary ResNet18 Architecture Telemetry
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#5e4b75] block uppercase">Model Variant</span>
                <span className="font-mono font-semibold text-[#1a082e] block mt-0.5">
                  {binaryResult.model.id}
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#5e4b75] block uppercase">Training Corpus</span>
                <span className="font-mono font-semibold text-[#1a082e] block mt-0.5">
                  {binaryResult.model.training_dataset}
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#5e4b75] block uppercase">Target Sensitivity</span>
                <span className="font-mono font-semibold text-[#1a082e] block mt-0.5">
                  90.0% Calibrated
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                <span className="text-[10px] text-[#5e4b75] block uppercase">Inference Latency</span>
                <span className="font-mono font-semibold text-amber-800 block mt-0.5 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {binaryResult.inference_time_ms.toFixed(1)} ms
                </span>
              </div>
            </div>
          </div>

          <div className="p-5 rounded-xl bg-white border border-[#e2e8f0] flex flex-col items-center justify-center text-center shadow-xs">
            <span className="text-xs text-[#5e4b75] uppercase tracking-wider font-semibold">
              Binary Classification State
            </span>
            <div className="mt-2.5">
              <Badge
                variant={binaryResult.prediction.classification === 'Abnormal' ? 'amber' : 'emerald'}
                size="md"
                className="text-lg px-6 py-2 font-mono font-bold"
              >
                {binaryResult.prediction.classification === 'Abnormal' ? (
                  <AlertCircle className="w-5 h-5 mr-2 text-amber-700" />
                ) : (
                  <CheckCircle2 className="w-5 h-5 mr-2 text-emerald-600" />
                )}
                {binaryResult.prediction.classification}
              </Badge>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-6 text-center font-mono">
              <div>
                <span className="text-[10px] text-[#5e4b75] uppercase block">Abnormal Probability</span>
                <span className="text-lg font-bold text-[#1a082e]">
                  {binaryResult.prediction.abnormal_probability.toFixed(6)}
                </span>
              </div>
              <div>
                <span className="text-[10px] text-[#5e4b75] uppercase block">Calibrated Threshold</span>
                <span className="text-lg font-bold text-[#5e4b75]">
                  {binaryResult.prediction.calibrated_threshold.toFixed(6)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Raw JSON Accordion for Technical Verification */}
      <div className="border border-[#e2e8f0] rounded-xl overflow-hidden bg-white shadow-xs">
        <button
          onClick={() => setShowRawJson(!showRawJson)}
          className="w-full px-4 py-3 flex items-center justify-between text-xs text-[#1a082e] hover:bg-slate-50 transition-colors font-medium"
        >
          <span className="flex items-center gap-2 font-mono font-semibold">
            <FileCode className="w-4 h-4 text-amber-600" />
            {showRawJson ? 'Hide Raw Backend JSON Payload' : 'Inspect Raw Backend JSON Payload'}
          </span>
          {showRawJson ? <ChevronUp className="w-4 h-4 text-amber-600" /> : <ChevronDown className="w-4 h-4 text-[#5e4b75]" />}
        </button>

        {showRawJson && (
          <div className="p-4 border-t border-[#e2e8f0] bg-[#140a22] font-mono text-[11px] text-emerald-400 overflow-x-auto max-h-80 scrollbar-thin">
            <pre>
              {JSON.stringify(
                activeModelType === 'multilabel' ? multiLabelResult : binaryResult,
                null,
                2
              )}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};
