import React, { useState, useRef, useEffect, ChangeEvent, DragEvent } from 'react';
import { Link } from 'react-router-dom';
import {
  UploadCloud,
  RefreshCw,
  AlertCircle,
  Sparkles,
  ZoomIn,
  ZoomOut,
  Maximize2,
  UserCheck,
  Stethoscope,
  BookOpen,
  HelpCircle,
  SplitSquareVertical,
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { xrayApi } from '@/services/api/xrayApi';
import { useHistoryStore } from '@/store/useHistoryStore';
import { useAppStore } from '@/store/useAppStore';
import {
  DiseaseName,
  XRayBinaryResponse,
  XRayPredictionResponse,
} from '@/types/xray';
import { PatientFriendlyXRayView } from './components/PatientFriendlyXRayView';
import { DoctorResearcherXRayView } from './components/DoctorResearcherXRayView';
import { DiseaseExplainabilityModal } from './components/DiseaseExplainabilityModal';
import { MedicalGlossaryModal } from './components/MedicalGlossaryModal';

export const XrayPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [activeModelType, setActiveModelType] = useState<'multilabel' | 'binary'>('multilabel');
  const [multilabelModelId, setMultilabelModelId] = useState<string>(
    'lungvision_phase7_msd121_224_direct'
  );
  const [binaryModelId, setBinaryModelId] = useState<'original_resnet18' | 'blur_augmented_resnet18'>(
    'original_resnet18'
  );
  
  // Image Viewer Zoom / Inspection States
  const [zoomScale, setZoomScale] = useState<number>(1.0);

  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [multiLabelResult, setMultiLabelResult] = useState<XRayPredictionResponse | null>(null);
  const [binaryResult, setBinaryResult] = useState<XRayBinaryResponse | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // Explainability & Glossary Modals
  const [isExplainModalOpen, setIsExplainModalOpen] = useState(false);
  const [isGlossaryModalOpen, setIsGlossaryModalOpen] = useState(false);
  const [selectedExplainDisease, setSelectedExplainDisease] = useState<DiseaseName>('Cardiomegaly');

  // App Store user experience mode
  const { userExperienceMode, setUserExperienceMode } = useAppStore();
  const [loadingSample, setLoadingSample] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const addHistoryItem = useHistoryStore((state) => state.addHistoryItem);

  const loadSampleImage = async () => {
    try {
      setLoadingSample(true);
      const res = await fetch('/samples/sample_frontal.png');
      const blob = await res.blob();
      const file = new File([blob], 'demo_frontal_chest_xray.png', { type: 'image/png' });
      handleFileSelection(file);
    } catch (err) {
      console.error('Failed to load sample X-ray:', err);
    } finally {
      setLoadingSample(false);
    }
  };

  // Revoke object URL on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileSelection = (file: File) => {
    // Client-side file validation
    const validTypes = ['image/png', 'image/jpeg', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setErrorMessage(`Unsupported format '${file.type}'. Allowed: PNG, JPEG, WebP.`);
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setErrorMessage('File exceeds maximum size limit of 20 MB.');
      return;
    }

    setErrorMessage(null);
    setSelectedFile(file);
    setMultiLabelResult(null);
    setBinaryResult(null);
    setZoomScale(1.0);

    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const onFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const runInference = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setErrorMessage(null);

    try {
      if (activeModelType === 'multilabel') {
        const result = await xrayApi.predictMultiLabel(selectedFile, multilabelModelId);
        setMultiLabelResult(result);
        addHistoryItem({
          type: 'xray_multilabel',
          filename: selectedFile.name,
          resultSummary: `${result.summary.flagged_conditions_count} of 6 conditions above threshold (${result.summary.highest_probability_disease}: ${(result.summary.highest_probability_value * 100).toFixed(1)}%)`,
          data: result,
        });
      } else {
        const result = await xrayApi.predictBinary(selectedFile, binaryModelId);
        setBinaryResult(result);
        addHistoryItem({
          type: 'xray_binary',
          filename: selectedFile.name,
          resultSummary: `${result.model.id}: ${result.prediction.classification} (${(result.prediction.abnormal_probability * 100).toFixed(1)}% vs threshold ${(result.prediction.calibrated_threshold * 100).toFixed(1)}%)`,
          data: result,
        });
      }
    } catch (err: any) {
      setErrorMessage(
        err.message || 'Inference execution failed. Ensure backend gateway is running on port 8000.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const resetAll = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setMultiLabelResult(null);
    setBinaryResult(null);
    setErrorMessage(null);
    setZoomScale(1.0);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const zoomIn = () => setZoomScale((prev) => Math.min(prev + 0.25, 3.0));
  const zoomOut = () => setZoomScale((prev) => Math.max(prev - 0.25, 0.75));
  const zoomReset = () => setZoomScale(1.0);

  const handleXrayWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      setZoomScale((prev) => Math.min(prev + 0.15, 3.5));
    } else {
      setZoomScale((prev) => Math.max(prev - 0.15, 0.75));
    }
  };

  const openExplainModal = (disease?: DiseaseName) => {
    if (disease) setSelectedExplainDisease(disease);
    setIsExplainModalOpen(true);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-[#e2e8f0]">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-amber-900 bg-amber-100 px-2 py-0.5 rounded border border-amber-300 font-semibold">
              Workstation Module
            </span>
            <span className="text-xs text-slate-300">·</span>
            <Badge variant="amber" size="sm">
              DenseNet121 & ResNet18
            </Badge>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-[#1a082e] flex items-center gap-2.5">
            2D Chest X-Ray Screening
          </h2>
          <p className="text-sm text-[#5e4b75] mt-1">
            Multi-label pathology screening (DenseNet121) and binary normal/abnormal triage (ResNet18) with locked empirical thresholds.
          </p>
        </div>

        {/* Global Toolbar: Model Selector & Quick Module Links */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Screening Mode Tabs */}
          <div className="flex items-center gap-1.5 p-1 bg-white rounded-xl border border-[#e2e8f0] shadow-xs">
            <button
              onClick={() => {
                setActiveModelType('multilabel');
                setErrorMessage(null);
              }}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeModelType === 'multilabel'
                  ? 'bg-gradient-to-r from-amber-500 via-amber-500 to-amber-600 text-white font-semibold shadow-xs'
                  : 'text-[#5e4b75] hover:text-[#1a082e]'
              }`}
            >
              6-Disease Multi-Label
            </button>
            <button
              onClick={() => {
                setActiveModelType('binary');
                setErrorMessage(null);
              }}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeModelType === 'binary'
                  ? 'bg-gradient-to-r from-amber-500 via-amber-500 to-amber-600 text-white font-semibold shadow-xs'
                  : 'text-[#5e4b75] hover:text-[#1a082e]'
              }`}
            >
              Binary (Normal / Abnormal)
            </button>
          </div>

          {/* Quick Dual-View Lab Link */}
          <Link
            to="/dual-view"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white border border-[#e2e8f0] hover:border-amber-400 text-xs text-[#1a082e] hover:text-amber-800 transition-all shadow-xs"
            title="Open Dual-View X-Ray Research Lab"
          >
            <SplitSquareVertical className="w-3.5 h-3.5 text-amber-600" />
            <span className="hidden sm:inline">Dual-View Lab</span>
          </Link>
        </div>
      </div>

      <DisclaimerBanner />

      {/* 3-Step Guided Workflow Banner */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 p-1.5 bg-white rounded-xl border border-[#e2e8f0] text-xs shadow-xs">
        <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all ${selectedFile ? 'bg-amber-50 text-amber-900 font-semibold border border-amber-300 shadow-xs' : 'text-[#5e4b75]'}`}>
          <span className="w-5 h-5 rounded-full bg-amber-100 text-amber-900 flex items-center justify-center font-mono text-[10px] font-bold">1</span>
          <span>Upload or Select Radiograph</span>
        </div>
        <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all ${selectedFile && (multiLabelResult || binaryResult || isLoading) ? 'bg-amber-50 text-amber-900 font-semibold border border-amber-300 shadow-xs' : 'text-[#5e4b75]'}`}>
          <span className="w-5 h-5 rounded-full bg-amber-100 text-amber-900 flex items-center justify-center font-mono text-[10px] font-bold">2</span>
          <span>Choose Screening Model</span>
        </div>
        <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all ${(multiLabelResult || binaryResult) ? 'bg-emerald-50 text-emerald-900 font-semibold border border-emerald-300 shadow-xs' : 'text-[#5e4b75]'}`}>
          <span className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-900 flex items-center justify-center font-mono text-[10px] font-bold">3</span>
          <span>Review Verified Findings</span>
        </div>
      </div>

      {/* Main Grid: Upload & Controls (5 Cols) + Results (7 Cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (5 Cols): Radiograph Ingestion & Inspection */}
        <div className="lg:col-span-5 space-y-4">
          <Card
            title="Radiograph Upload"
            subtitle="Frontal thoracic projection & inspection (DICOM-derived PNG, JPEG, or WebP)"
            headerAction={
              selectedFile ? (
                <Badge variant="amber" size="sm">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </Badge>
              ) : (
                <button
                  type="button"
                  onClick={loadSampleImage}
                  disabled={loadingSample}
                  className="px-2.5 py-1 rounded-lg bg-amber-100 hover:bg-amber-200 text-amber-900 border border-amber-300 text-[11px] font-semibold transition-all flex items-center gap-1.5 shadow-xs"
                  title="Load demo frontal radiograph"
                >
                  <Sparkles className="w-3 h-3 text-amber-700" />
                  <span>{loadingSample ? 'Loading...' : 'Try Demo X-Ray'}</span>
                </button>
              )
            }
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={onFileInputChange}
              accept="image/png,image/jpeg,image/webp"
              className="hidden"
            />

            {!previewUrl ? (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`h-72 border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 text-center cursor-pointer transition-all relative overflow-hidden group ${
                  isDragOver
                    ? 'border-amber-500 bg-amber-50/60'
                    : 'border-[#cbd5e1] hover:border-amber-500 hover:bg-amber-50/20'
                }`}
              >
                <div className="w-14 h-14 rounded-2xl bg-amber-50 border border-amber-200 flex items-center justify-center text-amber-700 mb-3 shadow-sm group-hover:scale-105 group-hover:border-amber-400 transition-all z-10">
                  <UploadCloud className="w-7 h-7" />
                </div>
                <p className="text-sm font-semibold text-[#1a082e] z-10 group-hover:text-amber-800 transition-colors">
                  Click or drag X-ray image here
                </p>
                <p className="text-xs text-[#5e4b75] mt-1.5 max-w-xs z-10">
                  Native 224×224 bilinear interpolation · RGB normalized · Max 20 MB
                </p>

                {/* Instant Try Demo Button inside dropzone */}
                <div className="mt-4 pt-3 border-t border-[#e2e8f0] w-full max-w-xs flex items-center justify-center gap-2 z-10">
                  <span className="text-[11px] text-[#5e4b75]">No image on hand?</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      loadSampleImage();
                    }}
                    disabled={loadingSample}
                    className="px-3 py-1 rounded-lg bg-amber-100 hover:bg-amber-200 text-amber-900 border border-amber-300 text-xs font-semibold transition-all flex items-center gap-1.5 shadow-xs hover:scale-105 active:scale-95"
                  >
                    <Sparkles className="w-3.5 h-3.5 text-amber-700" />
                    <span>{loadingSample ? 'Loading Demo...' : 'Load Sample X-Ray'}</span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                {/* Image Inspection Viewport with Zoom Controls */}
                <div
                  onWheel={handleXrayWheel}
                  className="relative rounded-xl overflow-hidden border border-[#e2e8f0] bg-[#120720] h-80 flex items-center justify-center group select-none shadow-inner cursor-ns-resize"
                  title="Scroll mouse wheel to zoom image"
                >
                  <div className="absolute inset-0 medical-grid-fine opacity-20 pointer-events-none" />
                  <div
                    className="w-full h-full flex items-center justify-center overflow-hidden cursor-grab active:cursor-grabbing"
                    style={{
                      transform: `scale(${zoomScale})`,
                      transition: 'transform 0.15s ease-out',
                    }}
                  >
                    <img
                      src={previewUrl}
                      alt="Uploaded X-ray preview"
                      className="max-h-full max-w-full object-contain pointer-events-none"
                    />
                  </div>

                  {/* Zoom Toolbar Overlay */}
                  <div className="absolute bottom-2.5 right-2.5 flex items-center gap-1 bg-white/95 backdrop-blur-md px-2.5 py-1 rounded-lg border border-[#e2e8f0] shadow-md z-20">
                    <button
                      onClick={zoomOut}
                      disabled={zoomScale <= 0.75}
                      className="p-1 text-[#5e4b75] hover:text-[#1a082e] disabled:opacity-30 transition-colors"
                      title="Zoom Out"
                    >
                      <ZoomOut className="w-3.5 h-3.5" />
                    </button>
                    <span className="text-[10px] font-mono text-amber-900 px-1 min-w-[32px] text-center font-bold">
                      {Math.round(zoomScale * 100)}%
                    </span>
                    <button
                      onClick={zoomIn}
                      disabled={zoomScale >= 3.0}
                      className="p-1 text-[#5e4b75] hover:text-[#1a082e] disabled:opacity-30 transition-colors"
                      title="Zoom In"
                    >
                      <ZoomIn className="w-3.5 h-3.5" />
                    </button>
                    <div className="h-3 w-px bg-slate-200 mx-0.5" />
                    <button
                      onClick={zoomReset}
                      className="p-1 text-[#5e4b75] hover:text-[#1a082e] transition-colors"
                      title="Fit to Screen"
                    >
                      <Maximize2 className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <div className="absolute top-2.5 left-2.5 bg-white/95 backdrop-blur-md px-2.5 py-1 rounded-md text-[10px] font-mono text-amber-900 border border-amber-300 flex items-center gap-1.5 z-20 font-semibold shadow-xs">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                    Frontal Projection
                  </div>

                  {/* Luxury Animated Laser Scanning Overlay */}
                  {isLoading && (
                    <div className="absolute inset-0 z-30 pointer-events-none overflow-hidden rounded-xl">
                      <div className="absolute inset-0 bg-[#1a082e]/40 backdrop-blur-[1px]" />
                      <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-amber-400 to-transparent shadow-[0_0_15px_#f59e0b] animate-laser-sweep" />
                      <div className="absolute inset-x-0 bottom-12 flex justify-center px-4">
                        <div className="bg-white/95 backdrop-blur-md px-3.5 py-1.5 rounded-lg border border-amber-300 text-xs font-mono text-amber-900 flex items-center gap-2 shadow-xl">
                          <RefreshCw className="w-3.5 h-3.5 text-amber-600 animate-spin" />
                          <span>Screening Radiograph with Deep Neural Network...</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* File Metadata & Replacement Action */}
                <div className="flex items-center justify-between text-xs text-[#5e4b75] px-1 font-mono">
                  <span className="truncate max-w-[220px] text-[11px] font-medium text-[#1a082e]" title={selectedFile?.name}>
                    {selectedFile?.name}
                  </span>
                  <button
                    onClick={resetAll}
                    className="text-rose-700 hover:text-rose-900 hover:underline font-semibold flex items-center gap-1 transition-colors text-xs"
                  >
                    <RefreshCw className="w-3 h-3" />
                    <span>Replace Image</span>
                  </button>
                </div>
              </div>
            )}

            {/* Multi-Label Model Selector if in Multi-Label Mode */}
            {activeModelType === 'multilabel' && (
              <div className="mt-4 pt-4 border-t border-[#e2e8f0] space-y-2">
                <label className="text-[11px] font-semibold text-[#5e4b75] uppercase tracking-wider block">
                  Select 6-Disease Classifier:
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setMultilabelModelId('lungvision_phase7_msd121_224_direct')}
                    className={`p-2.5 rounded-lg border text-left text-xs transition-all ${
                      multilabelModelId === 'lungvision_phase7_msd121_224_direct'
                        ? 'border-amber-500 bg-amber-50 text-amber-950 font-semibold ring-1 ring-amber-400'
                        : 'border-[#e2e8f0] text-[#5e4b75] hover:border-amber-400 hover:text-[#1a082e] hover:bg-slate-50'
                    }`}
                  >
                    <div className="font-semibold text-amber-900 flex items-center justify-between">
                      <span>Phase 7 MultiScale</span>
                      <span className="text-[9px] bg-amber-200 text-amber-950 px-1 py-0.5 rounded font-mono font-bold">SOTA</span>
                    </div>
                    <div className="text-[10px] text-[#5e4b75] mt-0.5">8.07M params · AUROC 0.8197</div>
                  </button>
                  <button
                    type="button"
                    onClick={() => setMultilabelModelId('densenet121_nih_6disease')}
                    className={`p-2.5 rounded-lg border text-left text-xs transition-all ${
                      multilabelModelId === 'densenet121_nih_6disease'
                        ? 'border-amber-500 bg-amber-50 text-amber-950 font-semibold ring-1 ring-amber-400'
                        : 'border-[#e2e8f0] text-[#5e4b75] hover:border-amber-400 hover:text-[#1a082e] hover:bg-slate-50'
                    }`}
                  >
                    <div className="font-semibold text-amber-900">DenseNet121 NIH</div>
                    <div className="text-[10px] text-[#5e4b75] mt-0.5">6.96M params · Frozen NIH</div>
                  </button>
                </div>
              </div>
            )}

            {/* Binary Model Selector if in Binary Mode */}
            {activeModelType === 'binary' && (
              <div className="mt-4 pt-4 border-t border-[#e2e8f0] space-y-2">
                <label className="text-[11px] font-semibold text-[#5e4b75] uppercase tracking-wider block">
                  Select ResNet18 Variant:
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setBinaryModelId('original_resnet18')}
                    className={`p-2.5 rounded-lg border text-left text-xs transition-all ${
                      binaryModelId === 'original_resnet18'
                        ? 'border-amber-500 bg-amber-50 text-amber-950 font-semibold ring-1 ring-amber-400'
                        : 'border-[#e2e8f0] text-[#5e4b75] hover:border-amber-400 hover:text-[#1a082e] hover:bg-slate-50'
                    }`}
                  >
                    <div className="font-semibold text-amber-900">Original ResNet18</div>
                    <div className="text-[10px] text-[#5e4b75] mt-0.5">Threshold: 0.1765 (90% Sens)</div>
                  </button>

                  <button
                    onClick={() => setBinaryModelId('blur_augmented_resnet18')}
                    className={`p-2.5 rounded-lg border text-left text-xs transition-all ${
                      binaryModelId === 'blur_augmented_resnet18'
                        ? 'border-purple-400 bg-purple-50 text-purple-950 font-semibold ring-1 ring-purple-300'
                        : 'border-[#e2e8f0] text-[#5e4b75] hover:border-amber-400 hover:text-[#1a082e] hover:bg-slate-50'
                    }`}
                  >
                    <div className="font-semibold text-[#1a082e]">Blur-Augmented</div>
                    <div className="text-[10px] text-[#5e4b75] mt-0.5">Threshold: 0.1879 (Robust)</div>
                  </button>
                </div>
              </div>
            )}

            {/* Execution Trigger Button */}
            <div className="mt-5">
              <Button
                variant="primary"
                size="md"
                className="w-full bg-gradient-to-r from-amber-500 via-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 text-white font-semibold shadow-md shadow-amber-500/25 transition-all border border-amber-400/40"
                onClick={runInference}
                disabled={!selectedFile || isLoading}
                isLoading={isLoading}
              >
                <Sparkles className="w-4 h-4 mr-1.5 text-white" />
                <span>
                  {isLoading
                    ? 'Computing Neural Probabilities...'
                    : `Analyze with ${activeModelType === 'multilabel' ? 'DenseNet121' : 'ResNet18'}`}
                </span>
              </Button>
            </div>

            {errorMessage && (
              <div className="mt-4 p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <span>{errorMessage}</span>
              </div>
            )}
          </Card>

          {/* Quick Help Links */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <button
              onClick={() => openExplainModal()}
              className="p-3 rounded-xl bg-white border border-[#e2e8f0] hover:border-amber-400 text-left transition-all shadow-xs"
            >
              <div className="flex items-center gap-1.5 text-amber-700 font-semibold">
                <BookOpen className="w-3.5 h-3.5" />
                <span>Clinical Guide</span>
              </div>
              <p className="text-[11px] text-[#5e4b75] mt-1">
                Explore the 6 diseases & threshold definitions
              </p>
            </button>

            <button
              onClick={() => setIsGlossaryModalOpen(true)}
              className="p-3 rounded-xl bg-white border border-[#e2e8f0] hover:border-amber-400 text-left transition-all shadow-xs"
            >
              <div className="flex items-center gap-1.5 text-[#1a082e] font-semibold">
                <HelpCircle className="w-3.5 h-3.5 text-amber-600" />
                <span>Medical Glossary</span>
              </div>
              <p className="text-[11px] text-[#5e4b75] mt-1">
                Definitions for AUROC, Youden J & Projections
              </p>
            </button>
          </div>
        </div>

        {/* Right Column (7 Cols): Dual User Experience Results (Patient vs Doctor) */}
        <div className="lg:col-span-7 space-y-4">
          <Card
            title={
              activeModelType === 'multilabel'
                ? 'Multi-Label Pathology Screening'
                : 'Binary Normal vs. Abnormal Classification'
            }
            subtitle={
              activeModelType === 'multilabel'
                ? 'DenseNet121 (NIH ChestX-ray14 Trained — Best Epoch 4 Checkpoint)'
                : `ResNet18 (${binaryModelId === 'original_resnet18' ? 'Original' : 'Blur-Augmented'} — IN-CXR Trained)`
            }
            headerAction={
              <div className="flex items-center gap-2">
                {/* INTERFACE MODE SELECTOR (Patient vs Doctor) */}
                <div className="flex items-center gap-1 p-0.5 bg-slate-100 rounded-lg border border-[#e2e8f0] text-xs">
                  <button
                    onClick={() => setUserExperienceMode('patient')}
                    className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                      userExperienceMode === 'patient'
                        ? 'bg-white text-[#1a082e] font-semibold shadow-xs border border-[#e2e8f0]'
                        : 'text-[#5e4b75] hover:text-[#1a082e]'
                    }`}
                    title="Switch to Patient-Friendly View"
                  >
                    <UserCheck className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Patient View</span>
                  </button>

                  <button
                    onClick={() => setUserExperienceMode('doctor')}
                    className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                      userExperienceMode === 'doctor'
                        ? 'bg-gradient-to-r from-amber-500 via-amber-500 to-amber-600 text-white font-semibold shadow-xs'
                        : 'text-[#5e4b75] hover:text-[#1a082e]'
                    }`}
                    title="Switch to Doctor / Researcher View"
                  >
                    <Stethoscope className="w-3.5 h-3.5" />
                    <span>Doctor View</span>
                  </button>
                </div>

                {/* Reset Action */}
                {(multiLabelResult || binaryResult) && (
                  <button
                    onClick={resetAll}
                    className="p-1.5 rounded-lg bg-white border border-[#e2e8f0] text-[#5e4b75] hover:text-[#1a082e] hover:bg-slate-50 shadow-xs"
                    title="Screen Another Image"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            }
          >
            {userExperienceMode === 'patient' ? (
              <PatientFriendlyXRayView
                fileName={selectedFile?.name}
                fileSize={selectedFile?.size}
                multiLabelResult={multiLabelResult}
                binaryResult={binaryResult}
                activeModelType={activeModelType}
                onOpenExplainability={openExplainModal}
                onOpenGlossary={() => setIsGlossaryModalOpen(true)}
              />
            ) : (
              <DoctorResearcherXRayView
                fileName={selectedFile?.name}
                fileSize={selectedFile?.size}
                multiLabelResult={multiLabelResult}
                binaryResult={binaryResult}
                activeModelType={activeModelType}
                onOpenExplainability={openExplainModal}
                onOpenGlossary={() => setIsGlossaryModalOpen(true)}
              />
            )}
          </Card>
        </div>
      </div>

      {/* Disease Explainability Modal */}
      <DiseaseExplainabilityModal
        isOpen={isExplainModalOpen}
        onClose={() => setIsExplainModalOpen(false)}
        initialDisease={selectedExplainDisease}
        currentProbabilities={multiLabelResult?.predictions}
      />

      {/* Medical Glossary Modal */}
      <MedicalGlossaryModal
        isOpen={isGlossaryModalOpen}
        onClose={() => setIsGlossaryModalOpen(false)}
      />
    </div>
  );
};
