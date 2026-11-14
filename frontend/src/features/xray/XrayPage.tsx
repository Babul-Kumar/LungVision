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
        const result = await xrayApi.predictMultiLabel(selectedFile);
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

  const openExplainModal = (disease?: DiseaseName) => {
    if (disease) setSelectedExplainDisease(disease);
    setIsExplainModalOpen(true);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-medical-border">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-medical-cyan bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/40">
              Workstation Module
            </span>
            <span className="text-xs text-medical-subtle">·</span>
            <Badge variant="cyan" size="sm">
              DenseNet121 & ResNet18
            </Badge>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            2D Chest X-Ray Screening
          </h2>
          <p className="text-sm text-medical-muted mt-1">
            Multi-label pathology screening (DenseNet121) and binary normal/abnormal triage (ResNet18) with locked empirical thresholds.
          </p>
        </div>

        {/* Global Toolbar: Model Selector & Quick Module Links */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Screening Mode Tabs */}
          <div className="flex items-center gap-1.5 p-1 bg-medical-surface rounded-xl border border-medical-border">
            <button
              onClick={() => {
                setActiveModelType('multilabel');
                setErrorMessage(null);
              }}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeModelType === 'multilabel'
                  ? 'bg-medical-cyan text-slate-950 font-semibold shadow-sm'
                  : 'text-medical-muted hover:text-white'
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
                  ? 'bg-medical-cyan text-slate-950 font-semibold shadow-sm'
                  : 'text-medical-muted hover:text-white'
              }`}
            >
              Binary (Normal / Abnormal)
            </button>
          </div>

          {/* Quick Dual-View Lab Link */}
          <Link
            to="/dual-view"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-medical-surface border border-medical-border hover:border-medical-cyan text-xs text-medical-muted hover:text-medical-cyan transition-all"
            title="Open Dual-View X-Ray Research Lab"
          >
            <SplitSquareVertical className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Dual-View Lab</span>
          </Link>
        </div>
      </div>

      <DisclaimerBanner />

      {/* 3-Step Guided Workflow Banner */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 p-1.5 bg-slate-900/60 rounded-xl border border-slate-800/80 text-xs shadow-inner">
        <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all ${selectedFile ? 'bg-cyan-950/40 text-cyan-300 font-semibold border border-cyan-800/40 shadow-xs' : 'text-slate-400'}`}>
          <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-mono text-[10px] font-bold">1</span>
          <span>Upload or Select Radiograph</span>
        </div>
        <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all ${selectedFile && (multiLabelResult || binaryResult || isLoading) ? 'bg-cyan-950/40 text-cyan-300 font-semibold border border-cyan-800/40 shadow-xs' : 'text-slate-400'}`}>
          <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-mono text-[10px] font-bold">2</span>
          <span>Choose Screening Model</span>
        </div>
        <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all ${(multiLabelResult || binaryResult) ? 'bg-emerald-950/40 text-emerald-300 font-semibold border border-emerald-800/40 shadow-xs' : 'text-slate-400'}`}>
          <span className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-mono text-[10px] font-bold">3</span>
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
                <Badge variant="cyan" size="sm">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </Badge>
              ) : (
                <button
                  type="button"
                  onClick={loadSampleImage}
                  disabled={loadingSample}
                  className="px-2.5 py-1 rounded-lg bg-cyan-950/70 hover:bg-cyan-900/80 text-cyan-300 border border-cyan-500/40 text-[11px] font-medium transition-all flex items-center gap-1.5 shadow-sm"
                  title="Load demo frontal radiograph"
                >
                  <Sparkles className="w-3 h-3 text-cyan-400" />
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
                className={`h-72 border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 text-center cursor-pointer transition-all hud-corner-bracket relative overflow-hidden group ${
                  isDragOver
                    ? 'border-cyan-400 bg-cyan-950/30 glow-cyan-sm'
                    : 'border-slate-700/80 hover:border-cyan-400/60 hover:bg-slate-900/40'
                }`}
              >
                <div className="absolute inset-0 medical-grid opacity-30 pointer-events-none" />
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-cyan-950 to-slate-900 border border-cyan-500/30 flex items-center justify-center text-cyan-400 mb-3 shadow-lg shadow-cyan-950/50 group-hover:scale-110 group-hover:border-cyan-400 group-hover:shadow-cyan-500/20 transition-all z-10">
                  <UploadCloud className="w-7 h-7" />
                </div>
                <p className="text-sm font-semibold text-white z-10 group-hover:text-cyan-300 transition-colors">
                  Click or drag X-ray image here
                </p>
                <p className="text-xs text-medical-muted mt-1.5 max-w-xs z-10">
                  Native 224×224 bilinear interpolation · RGB normalized · Max 20 MB
                </p>

                {/* Instant Try Demo Button inside dropzone */}
                <div className="mt-4 pt-3 border-t border-slate-800/80 w-full max-w-xs flex items-center justify-center gap-2 z-10">
                  <span className="text-[11px] text-slate-400">No image on hand?</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      loadSampleImage();
                    }}
                    disabled={loadingSample}
                    className="px-3 py-1 rounded-lg bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 border border-cyan-500/40 text-xs font-medium transition-all flex items-center gap-1.5 shadow-sm hover:scale-105 active:scale-95"
                  >
                    <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                    <span>{loadingSample ? 'Loading Demo...' : 'Load Sample X-Ray'}</span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                {/* Image Inspection Viewport with Zoom Controls */}
                <div className="relative rounded-xl overflow-hidden border border-slate-700/80 bg-black h-80 flex items-center justify-center group select-none hud-corner-bracket shadow-inner">
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
                  <div className="absolute bottom-2.5 right-2.5 flex items-center gap-1 bg-slate-950/85 backdrop-blur-md px-2.5 py-1 rounded-lg border border-slate-700/80 shadow-xl z-20">
                    <button
                      onClick={zoomOut}
                      disabled={zoomScale <= 0.75}
                      className="p-1 text-slate-400 hover:text-white disabled:opacity-30 transition-colors"
                      title="Zoom Out"
                    >
                      <ZoomOut className="w-3.5 h-3.5" />
                    </button>
                    <span className="text-[10px] font-mono text-cyan-400 px-1 min-w-[32px] text-center font-bold">
                      {Math.round(zoomScale * 100)}%
                    </span>
                    <button
                      onClick={zoomIn}
                      disabled={zoomScale >= 3.0}
                      className="p-1 text-slate-400 hover:text-white disabled:opacity-30 transition-colors"
                      title="Zoom In"
                    >
                      <ZoomIn className="w-3.5 h-3.5" />
                    </button>
                    <div className="h-3 w-px bg-slate-700 mx-0.5" />
                    <button
                      onClick={zoomReset}
                      className="p-1 text-slate-400 hover:text-white transition-colors"
                      title="Fit to Screen"
                    >
                      <Maximize2 className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <div className="absolute top-2.5 left-2.5 bg-black/80 backdrop-blur-md px-2.5 py-1 rounded-md text-[10px] font-mono text-cyan-300 border border-cyan-500/30 flex items-center gap-1.5 z-20">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                    Frontal Projection
                  </div>

                  {/* High-Tech Animated Laser Scanning Overlay */}
                  {isLoading && (
                    <div className="absolute inset-0 z-30 pointer-events-none overflow-hidden rounded-xl">
                      <div className="absolute inset-0 bg-cyan-950/25 backdrop-blur-[1px]" />
                      <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_15px_#06b6d4] animate-laser-sweep" />
                      <div className="absolute inset-x-0 bottom-12 flex justify-center px-4">
                        <div className="bg-black/90 backdrop-blur-md px-3.5 py-1.5 rounded-lg border border-cyan-500/50 text-xs font-mono text-cyan-300 flex items-center gap-2 shadow-2xl">
                          <RefreshCw className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
                          <span>Screening Radiograph with Deep Neural Network...</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* File Metadata & Replacement Action */}
                <div className="flex items-center justify-between text-xs text-medical-muted px-1">
                  <span className="truncate max-w-[220px] font-mono text-[11px]" title={selectedFile?.name}>
                    {selectedFile?.name}
                  </span>
                  <button
                    onClick={resetAll}
                    className="text-rose-400 hover:text-rose-300 hover:underline flex items-center gap-1 transition-colors text-xs"
                  >
                    <RefreshCw className="w-3 h-3" />
                    <span>Replace Image</span>
                  </button>
                </div>
              </div>
            )}

            {/* Binary Model Selector if in Binary Mode */}
            {activeModelType === 'binary' && (
              <div className="mt-4 pt-4 border-t border-medical-border/60 space-y-2">
                <label className="text-[11px] font-semibold text-medical-subtle uppercase tracking-wider block">
                  Select ResNet18 Variant:
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setBinaryModelId('original_resnet18')}
                    className={`p-2.5 rounded-lg border text-left text-xs transition-all ${
                      binaryModelId === 'original_resnet18'
                        ? 'border-medical-cyan bg-cyan-950/30 text-white font-medium ring-1 ring-medical-cyan/30'
                        : 'border-medical-border text-medical-muted hover:border-slate-700 hover:text-slate-200'
                    }`}
                  >
                    <div className="font-semibold text-medical-cyan">Original ResNet18</div>
                    <div className="text-[10px] text-medical-muted mt-0.5">Threshold: 0.1765 (90% Sens)</div>
                  </button>

                  <button
                    onClick={() => setBinaryModelId('blur_augmented_resnet18')}
                    className={`p-2.5 rounded-lg border text-left text-xs transition-all ${
                      binaryModelId === 'blur_augmented_resnet18'
                        ? 'border-medical-teal bg-teal-950/30 text-white font-medium ring-1 ring-medical-teal/30'
                        : 'border-medical-border text-medical-muted hover:border-slate-700 hover:text-slate-200'
                    }`}
                  >
                    <div className="font-semibold text-medical-teal">Blur-Augmented</div>
                    <div className="text-[10px] text-medical-muted mt-0.5">Threshold: 0.1879 (Robust)</div>
                  </button>
                </div>
              </div>
            )}

            {/* Execution Trigger Button */}
            <div className="mt-5">
              <Button
                variant="primary"
                size="md"
                className="w-full bg-gradient-to-r from-cyan-600 via-teal-600 to-cyan-500 hover:from-cyan-500 hover:to-teal-400 font-semibold shadow-lg shadow-cyan-950/50 hover:shadow-cyan-500/30 hover:scale-[1.01] transition-all"
                onClick={runInference}
                disabled={!selectedFile || isLoading}
                isLoading={isLoading}
              >
                <Sparkles className="w-4 h-4 mr-1.5 text-cyan-200" />
                <span>
                  {isLoading
                    ? 'Computing Neural Probabilities...'
                    : `Analyze with ${activeModelType === 'multilabel' ? 'DenseNet121' : 'ResNet18'}`}
                </span>
              </Button>
            </div>

            {errorMessage && (
              <div className="mt-4 p-3 bg-rose-950/50 border border-rose-800/80 rounded-lg text-xs text-rose-300 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <span>{errorMessage}</span>
              </div>
            )}
          </Card>

          {/* Quick Help Links */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <button
              onClick={() => openExplainModal()}
              className="p-3 rounded-xl bg-medical-card border border-medical-border hover:border-medical-cyan/50 text-left transition-all"
            >
              <div className="flex items-center gap-1.5 text-medical-cyan font-semibold">
                <BookOpen className="w-3.5 h-3.5" />
                <span>Clinical Guide</span>
              </div>
              <p className="text-[11px] text-medical-muted mt-1">
                Explore the 6 diseases & threshold definitions
              </p>
            </button>

            <button
              onClick={() => setIsGlossaryModalOpen(true)}
              className="p-3 rounded-xl bg-medical-card border border-medical-border hover:border-medical-teal/50 text-left transition-all"
            >
              <div className="flex items-center gap-1.5 text-medical-teal font-semibold">
                <HelpCircle className="w-3.5 h-3.5" />
                <span>Medical Glossary</span>
              </div>
              <p className="text-[11px] text-medical-muted mt-1">
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
                <div className="flex items-center gap-1 p-0.5 bg-medical-bg rounded-lg border border-medical-border text-xs">
                  <button
                    onClick={() => setUserExperienceMode('patient')}
                    className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                      userExperienceMode === 'patient'
                        ? 'bg-medical-teal text-slate-950 font-semibold shadow-sm'
                        : 'text-medical-muted hover:text-white'
                    }`}
                    title="Switch to Patient-Friendly View"
                  >
                    <UserCheck className="w-3.5 h-3.5" />
                    <span>Patient View</span>
                  </button>

                  <button
                    onClick={() => setUserExperienceMode('doctor')}
                    className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                      userExperienceMode === 'doctor'
                        ? 'bg-medical-cyan text-slate-950 font-semibold shadow-sm'
                        : 'text-medical-muted hover:text-white'
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
                    className="p-1.5 rounded-lg bg-medical-surface border border-medical-border text-medical-muted hover:text-white"
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
