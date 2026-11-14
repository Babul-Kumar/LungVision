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

  // If no analysis executed
  if (!multiLabelResult && !binaryResult) {
    return (
      <div className="p-8 text-center border border-dashed border-medical-border rounded-2xl bg-medical-card/40 space-y-3">
        <Cpu className="w-10 h-10 text-medical-subtle mx-auto" />
        <h4 className="text-base font-semibold text-white">No Telemetry Available</h4>
        <p className="text-xs text-medical-muted max-w-md mx-auto">
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
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-xl bg-medical-surface border border-medical-border">
        <div className="flex items-center gap-3">
          <Badge variant="cyan" size="md">
            Doctor / Researcher Telemetry
          </Badge>
          <span className="text-xs font-mono text-medical-muted">
            {fileName ? `${fileName} (${fileSize ? (fileSize / (1024 * 1024)).toFixed(2) : '0'} MB) · ` : ''}
            Format: Exact Float Output (Uncalibrated to Clinical Risk)
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleExportJson}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-medical-card border border-medical-border hover:border-medical-cyan text-white hover:text-medical-cyan transition-all"
            title="Download Raw Output JSON"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export JSON</span>
          </button>
          <button
            onClick={handlePrint}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-medical-card border border-medical-border hover:border-medical-teal text-white hover:text-medical-teal transition-all"
            title="Print or Save as PDF"
          >
            <Printer className="w-3.5 h-3.5" />
            <span>Print Report</span>
          </button>
          <button
            onClick={onOpenGlossary}
            className="p-1.5 rounded-lg text-medical-muted hover:text-white bg-medical-card border border-medical-border"
            title="Terminology Glossary"
          >
            <HelpCircle className="w-4 h-4" />
          </button>
        </div>
      </div>

      {activeModelType === 'multilabel' && multiLabelResult && (
        <div className="space-y-4">
          {/* Scientific Specification Card */}
          <div className="p-4 rounded-xl bg-medical-card border border-medical-border/80 space-y-3">
            <div className="text-[11px] font-mono uppercase tracking-wider text-medical-cyan font-semibold">
              Deep Learning Model Provenance
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border/60">
                <span className="text-[10px] text-medical-muted block uppercase">Architecture</span>
                <span className="font-mono font-semibold text-white block mt-0.5">
                  {multiLabelResult.model.architecture}
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border/60">
                <span className="text-[10px] text-medical-muted block uppercase">Training Corpus</span>
                <span className="font-mono font-semibold text-white block mt-0.5">
                  {multiLabelResult.model.training_dataset}
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border/60">
                <span className="text-[10px] text-medical-muted block uppercase">Checkpoint State</span>
                <span className="font-mono font-semibold text-white block mt-0.5">
                  Epoch {multiLabelResult.model.checkpoint_epoch} (Best Val AUC)
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border/60">
                <span className="text-[10px] text-medical-muted block uppercase">Inference Latency</span>
                <span className="font-mono font-semibold text-medical-cyan block mt-0.5 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {multiLabelResult.inference_time_ms.toFixed(1)} ms
                </span>
              </div>
            </div>

            <div className="text-[11px] font-mono text-medical-muted bg-medical-surface px-3 py-2 rounded-lg border border-medical-border/60 space-y-0.5">
              <div>
                <strong>Checkpoint Identifier:</strong> <code>densenet121_nih_best.pth</code> (SHA-256: <code>8d3690cd...</code>)
              </div>
              <div>
                <strong>Input Preprocessing:</strong> 224×224 bilinear interpolation, 3-channel RGB, ImageNet normalization (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]).
              </div>
            </div>
          </div>

          {/* 6 Pathologies in Strict NIH Order */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold uppercase tracking-wider text-white">
                Six Disease Output Activations (Strict NIH ChestX-ray14 Ordering)
              </h4>
              <span className="text-[11px] text-medical-muted font-mono">
                Flagged: {multiLabelResult.summary.flagged_conditions_count} / 6
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
                    <div className="flex items-center justify-between text-[10px] font-mono text-medical-muted px-1">
                      <span>
                        Raw Probability: <strong>{pred.probability.toFixed(6)}</strong> · Threshold: <strong>{pred.threshold.toFixed(6)}</strong>
                      </span>
                      <button
                        onClick={() => onOpenExplainability(disease)}
                        className="text-medical-cyan hover:underline text-[10px]"
                      >
                        Clinical Details & Signs →
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Technical Caveats & Domain Generalization */}
          <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-800/40 space-y-2 text-xs">
            <div className="font-bold text-amber-300 flex items-center gap-2 uppercase tracking-wider text-[11px]">
              <ShieldAlert className="w-4 h-4 text-amber-400" />
              Evaluation Benchmark & Domain Generalization Caveats
            </div>
            <p className="text-amber-100/90 leading-relaxed text-[11px]">
              <strong>External Generalization (CheXpert Drop):</strong> When evaluating this model on external Stanford CheXpert radiographs with locked NIH calibration thresholds, macro AUROC dropped from <strong>0.8206</strong> to <strong>0.6277</strong> (-23.5% relative drop). This demonstrates high sensitivity to hospital-specific beam hardware, projection protocols (AP vs PA), and patient demographics.
            </p>
            <p className="text-amber-100/80 leading-relaxed text-[11px]">
              <strong>Non-Diagnostic Notice:</strong> Never treat output probabilities as clinical risk percentages. Model outputs represent relative visual activation vectors on retrospective public datasets and have NOT been validated in prospective clinical trials.
            </p>
          </div>
        </div>
      )}

      {activeModelType === 'binary' && binaryResult && (
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-medical-card border border-medical-border/80 space-y-3">
            <div className="text-[11px] font-mono uppercase tracking-wider text-medical-teal font-semibold">
              Binary ResNet18 Architecture Telemetry
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border/60">
                <span className="text-[10px] text-medical-muted block uppercase">Model Variant</span>
                <span className="font-mono font-semibold text-white block mt-0.5">
                  {binaryResult.model.id}
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border/60">
                <span className="text-[10px] text-medical-muted block uppercase">Training Corpus</span>
                <span className="font-mono font-semibold text-white block mt-0.5">
                  {binaryResult.model.training_dataset}
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border/60">
                <span className="text-[10px] text-medical-muted block uppercase">Target Sensitivity</span>
                <span className="font-mono font-semibold text-medical-teal block mt-0.5">
                  90.0% Calibrated
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border/60">
                <span className="text-[10px] text-medical-muted block uppercase">Inference Latency</span>
                <span className="font-mono font-semibold text-medical-cyan block mt-0.5 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {binaryResult.inference_time_ms.toFixed(1)} ms
                </span>
              </div>
            </div>
          </div>

          <div className="p-5 rounded-xl bg-medical-surface border border-medical-border flex flex-col items-center justify-center text-center">
            <span className="text-xs text-medical-muted uppercase tracking-wider font-semibold">
              Binary Classification State
            </span>
            <div className="mt-2.5">
              <Badge
                variant={binaryResult.prediction.classification === 'Abnormal' ? 'amber' : 'emerald'}
                size="md"
                className="text-lg px-6 py-2 font-mono font-bold"
              >
                {binaryResult.prediction.classification === 'Abnormal' ? (
                  <AlertCircle className="w-5 h-5 mr-2 text-amber-400" />
                ) : (
                  <CheckCircle2 className="w-5 h-5 mr-2 text-emerald-400" />
                )}
                {binaryResult.prediction.classification}
              </Badge>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-6 text-center font-mono">
              <div>
                <span className="text-[10px] text-medical-muted uppercase block">Abnormal Probability</span>
                <span className="text-lg font-bold text-white">
                  {binaryResult.prediction.abnormal_probability.toFixed(6)}
                </span>
              </div>
              <div>
                <span className="text-[10px] text-medical-muted uppercase block">Calibrated Threshold</span>
                <span className="text-lg font-bold text-slate-300">
                  {binaryResult.prediction.calibrated_threshold.toFixed(6)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Raw JSON Accordion for Technical Verification */}
      <div className="border border-medical-border/80 rounded-xl overflow-hidden bg-medical-surface/40">
        <button
          onClick={() => setShowRawJson(!showRawJson)}
          className="w-full px-4 py-3 flex items-center justify-between text-xs text-medical-muted hover:text-white transition-colors"
        >
          <span className="flex items-center gap-2 font-mono font-medium">
            <FileCode className="w-4 h-4 text-medical-cyan" />
            {showRawJson ? 'Hide Raw Backend JSON Payload' : 'Inspect Raw Backend JSON Payload'}
          </span>
          {showRawJson ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showRawJson && (
          <div className="p-4 border-t border-medical-border bg-black/60 font-mono text-[11px] text-emerald-400 overflow-x-auto max-h-80 scrollbar-thin">
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
