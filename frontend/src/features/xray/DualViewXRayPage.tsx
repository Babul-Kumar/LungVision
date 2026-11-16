import React, { useState, useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import {
  UploadCloud,
  AlertCircle,
  Sparkles,
  ZoomIn,
  ZoomOut,
  Maximize2,
  ShieldAlert,
  FileImage,
  Sliders,
  Box,
  Activity,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { xrayApi } from '@/services/api/xrayApi';
import { reconstructionApi } from '@/services/api/reconstructionApi';
import {
  EXPECTED_6_DISEASES,
  XRayPredictionResponse,
} from '@/types/xray';
import {
  SSMModelInfo,
  SSMReconstructionResponse,
} from '@/types/reconstruction';
import { ProbabilityBar } from '@/components/common/ProbabilityBar';
import { SSM3DReconstructionViewer } from './SSM3DReconstructionViewer';

interface ViewImageState {
  file: File | null;
  previewUrl: string | null;
  width: number;
  height: number;
  isDragOver: boolean;
  error: string | null;
}

const initialViewState: ViewImageState = {
  file: null,
  previewUrl: null,
  width: 0,
  height: 0,
  isDragOver: false,
  error: null,
};

export const DualViewXRayPage: React.FC = () => {
  const [frontal, setFrontal] = useState<ViewImageState>(initialViewState);
  const [lateral, setLateral] = useState<ViewImageState>(initialViewState);

  // Active Workflow Tab
  const [activeTab, setActiveTab] = useState<'reconstruction' | 'screening' | 'roadmap'>('reconstruction');

  // Viewport inspection controls
  const [frontalZoom, setFrontalZoom] = useState<number>(1.0);
  const [lateralZoom, setLateralZoom] = useState<number>(1.0);
  const [syncZoom, setSyncZoom] = useState<boolean>(true);
  const [isInverted, setIsInverted] = useState<boolean>(false);

  // Inference state
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [frontalResult, setFrontalResult] = useState<XRayPredictionResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // 3D SSM Reconstruction state
  const [isReconstructing, setIsReconstructing] = useState<boolean>(false);
  const [reconstructionResult, setReconstructionResult] = useState<SSMReconstructionResponse | null>(null);
  const [reconstructionError, setReconstructionError] = useState<string | null>(null);
  const [ssmModelInfo, setSsmModelInfo] = useState<SSMModelInfo | null>(null);
  const [isExportingObj, setIsExportingObj] = useState<boolean>(false);

  // SSM Hyperparameters
  const [inspiratoryEffort, setInspiratoryEffort] = useState<number>(0.0);
  const [silhouetteScale, setSilhouetteScale] = useState<number>(1.0);
  const [optimizationIterations, setOptimizationIterations] = useState<number>(40);
  const [includeLateralConstraint, setIncludeLateralConstraint] = useState<boolean>(true);

  const frontalInputRef = useRef<HTMLInputElement>(null);
  const lateralInputRef = useRef<HTMLInputElement>(null);
  const location = useLocation();
  const [loadingSamplePair, setLoadingSamplePair] = useState(false);
  const [loadingSampleSlot, setLoadingSampleSlot] = useState<'frontal' | 'lateral' | null>(null);

  const loadSamplePair = async () => {
    try {
      setLoadingSamplePair(true);
      const [resF, resL] = await Promise.all([
        fetch('/samples/sample_frontal.png'),
        fetch('/samples/sample_lateral.png'),
      ]);
      const [blobF, blobL] = await Promise.all([resF.blob(), resL.blob()]);
      const fileF = new File([blobF], 'demo_frontal_chest_xray.png', { type: 'image/png' });
      const fileL = new File([blobL], 'demo_lateral_chest_xray.png', { type: 'image/png' });
      processImageFile(fileF, setFrontal);
      processImageFile(fileL, setLateral);
    } catch (err) {
      console.error('Failed to load sample pair:', err);
    } finally {
      setLoadingSamplePair(false);
    }
  };

  const loadSampleSlot = async (slot: 'frontal' | 'lateral') => {
    try {
      setLoadingSampleSlot(slot);
      const url = slot === 'frontal' ? '/samples/sample_frontal.png' : '/samples/sample_lateral.png';
      const filename = slot === 'frontal' ? 'demo_frontal_projection.png' : 'demo_lateral_projection.png';
      const setter = slot === 'frontal' ? setFrontal : setLateral;
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'image/png' });
      processImageFile(file, setter);
    } catch (err) {
      console.error(`Failed to load sample ${slot}:`, err);
    } finally {
      setLoadingSampleSlot(null);
    }
  };

  // Load exported DRRs if passed via navigation state from CT Explorer
  useEffect(() => {
    const state = location.state as { frontalUrl?: string; lateralUrl?: string } | null;
    if (!state) return;

    let isMounted = true;

    const loadExportedImages = async () => {
      if (state.frontalUrl) {
        try {
          const res = await fetch(state.frontalUrl);
          const blob = await res.blob();
          const file = new File([blob], 'frontal_drr.png', { type: 'image/png' });
          const img = new Image();
          img.onload = () => {
            if (isMounted) {
              setFrontal({
                file,
                previewUrl: state.frontalUrl!,
                width: img.naturalWidth,
                height: img.naturalHeight,
                isDragOver: false,
                error: null,
              });
            }
          };
          img.src = state.frontalUrl;
        } catch (err) {
          console.error('Error loading exported frontal DRR:', err);
        }
      }

      if (state.lateralUrl) {
        try {
          const res = await fetch(state.lateralUrl);
          const blob = await res.blob();
          const file = new File([blob], 'lateral_drr.png', { type: 'image/png' });
          const img = new Image();
          img.onload = () => {
            if (isMounted) {
              setLateral({
                file,
                previewUrl: state.lateralUrl!,
                width: img.naturalWidth,
                height: img.naturalHeight,
                isDragOver: false,
                error: null,
              });
            }
          };
          img.src = state.lateralUrl;
        } catch (err) {
          console.error('Error loading exported lateral DRR:', err);
        }
      }
    };

    loadExportedImages();

    return () => {
      isMounted = false;
    };
  }, [location.state]);

  // Fetch SSM Model Metadata on mount
  useEffect(() => {
    let isCancelled = false;
    const fetchInfo = async () => {
      try {
        const info = await reconstructionApi.getModelInfo();
        if (!isCancelled) setSsmModelInfo(info);
      } catch (err) {
        console.warn('SSM model metadata not available yet:', err);
      }
    };
    fetchInfo();
    return () => {
      isCancelled = true;
    };
  }, []);

  // File loading helper
  const processImageFile = (
    file: File,
    setView: React.Dispatch<React.SetStateAction<ViewImageState>>
  ) => {
    const validTypes = ['image/png', 'image/jpeg', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setView((prev) => ({
        ...prev,
        error: `Invalid format '${file.type}'. Allowed: PNG, JPEG, WebP.`,
      }));
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setView((prev) => ({
        ...prev,
        error: 'File exceeds 25 MB size limit.',
      }));
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setView((prev) => {
        if (prev.previewUrl && prev.previewUrl.startsWith('blob:')) {
          URL.revokeObjectURL(prev.previewUrl);
        }
        return {
          file,
          previewUrl: objectUrl,
          width: img.naturalWidth,
          height: img.naturalHeight,
          isDragOver: false,
          error: null,
        };
      });
    };
    img.onerror = () => {
      setView((prev) => ({
        ...prev,
        error: 'Failed to decode image data.',
      }));
    };
    img.src = objectUrl;
  };

  // Zoom helpers
  const handleZoomIn = (target: 'frontal' | 'lateral' | 'both') => {
    if (syncZoom || target === 'both') {
      setFrontalZoom((z) => Math.min(z + 0.25, 3.5));
      setLateralZoom((z) => Math.min(z + 0.25, 3.5));
    } else if (target === 'frontal') {
      setFrontalZoom((z) => Math.min(z + 0.25, 3.5));
    } else {
      setLateralZoom((z) => Math.min(z + 0.25, 3.5));
    }
  };

  const handleZoomOut = (target: 'frontal' | 'lateral' | 'both') => {
    if (syncZoom || target === 'both') {
      setFrontalZoom((z) => Math.max(z - 0.25, 0.75));
      setLateralZoom((z) => Math.max(z - 0.25, 0.75));
    } else if (target === 'frontal') {
      setFrontalZoom((z) => Math.max(z - 0.25, 0.75));
    } else {
      setLateralZoom((z) => Math.max(z - 0.25, 0.75));
    }
  };

  const handleZoomReset = () => {
    setFrontalZoom(1.0);
    setLateralZoom(1.0);
  };

  const handleZoomWheel = (e: React.WheelEvent<HTMLDivElement>, target: 'frontal' | 'lateral') => {
    e.preventDefault();
    if (e.deltaY < 0) {
      handleZoomIn(target);
    } else {
      handleZoomOut(target);
    }
  };

  // Run frontal inference
  const executeAnalysis = async () => {
    if (!frontal.file) return;

    setIsAnalyzing(true);
    setAnalysisError(null);
    setActiveTab('screening');

    try {
      const res = await xrayApi.predictMultiLabel(frontal.file);
      setFrontalResult(res);
    } catch (err: any) {
      setAnalysisError(err.message || 'Dual-view analysis failed. Verify backend gateway on port 8000.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Execute 3D SSM Reconstruction
  const executeReconstruction = async () => {
    setIsReconstructing(true);
    setReconstructionError(null);
    setActiveTab('reconstruction');

    try {
      // Calculate aspect ratios if files are loaded, otherwise use default 1.05
      const frontalAR = frontal.width > 0 && frontal.height > 0
        ? Math.min(2.0, Math.max(0.6, frontal.height / frontal.width))
        : 1.05;

      const lateralAR = includeLateralConstraint && lateral.width > 0 && lateral.height > 0
        ? Math.min(2.0, Math.max(0.6, lateral.height / lateral.width))
        : (includeLateralConstraint ? 1.0 : null);

      const res = await reconstructionApi.fitReconstruction({
        frontal_aspect_ratio: frontalAR,
        lateral_aspect_ratio: lateralAR,
        estimated_inspiratory_effort: inspiratoryEffort,
        silhouette_scale: silhouetteScale,
        n_iterations: optimizationIterations,
        include_mesh_geometry: true,
      });

      setReconstructionResult(res);
    } catch (err: any) {
      setReconstructionError(err.message || '3D Reconstruction failed. Ensure backend service is running.');
    } finally {
      setIsReconstructing(false);
    }
  };

  // Export patient-specific OBJ file
  const handleExportObj = async () => {
    if (!reconstructionResult) return;
    setIsExportingObj(true);
    try {
      const objText = await reconstructionApi.exportReconstructedObj(
        reconstructionResult.optimal_coefficients
      );
      const blob = new Blob([objText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `lungvision_patient_3d_lungs_${Date.now()}.obj`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(`Failed to export OBJ: ${err.message}`);
    } finally {
      setIsExportingObj(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Module Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-white/[0.08]">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-amber-400 bg-amber-950/40 px-2 py-0.5 rounded border border-amber-800/40">
              Research Workstation
            </span>
            <span className="text-xs text-white/20">·</span>
            <Badge variant="amber" size="sm">
              Dual-View CXR Lab
            </Badge>
            <Badge variant="silver" size="sm">
              Bi-Planar Ingestion
            </Badge>
            <Badge variant="amber" size="sm">
              Phase 2B: 3D SSM Engine
            </Badge>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-100 flex items-center gap-2.5">
            Dual-View X-Ray Research Lab & 3D Reconstruction
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Bi-planar thoracic inspection comparing paired Frontal (PA/AP) and Lateral chest radiographs with patient-specific 3D lung surface reconstruction and organ volumetry.
          </p>
        </div>

        {/* Global Action Bar */}
        <div className="flex flex-wrap items-center gap-2.5">
          {(!frontal.file || !lateral.file) && (
            <Button
              variant="outline"
              size="md"
              onClick={loadSamplePair}
              disabled={loadingSamplePair}
              className="text-amber-300 border-amber-500/40 bg-amber-950/40 hover:bg-amber-900/60 transition-all"
              title="Load demo bi-planar paired radiograph"
            >
              <Sparkles className="w-4 h-4 mr-1.5 text-amber-400" />
              <span>{loadingSamplePair ? 'Loading Pair...' : 'Try Demo Pair'}</span>
            </Button>
          )}

          <Button
            variant="outline"
            size="md"
            onClick={executeAnalysis}
            disabled={!frontal.file || isAnalyzing}
            isLoading={isAnalyzing}
            className="text-slate-200 border-white/20 hover:border-amber-400/50"
          >
            <Sparkles className="w-4 h-4 mr-1.5 text-amber-400" />
            <span>{isAnalyzing ? 'Analyzing...' : 'Screen Pathology'}</span>
          </Button>

          <Button
            variant="primary"
            size="md"
            onClick={executeReconstruction}
            disabled={isReconstructing}
            isLoading={isReconstructing}
          >
            <Box className="w-4 h-4 mr-1.5" />
            <span>{isReconstructing ? 'Optimizing 3D Surface...' : 'Reconstruct 3D Lung Surface'}</span>
          </Button>
        </div>
      </div>

      <DisclaimerBanner />

      {/* Synchronized Viewport Controls Header */}
      <div className="p-3.5 rounded-xl bg-[#150a22]/80 border border-white/10 flex flex-wrap items-center justify-between gap-3 text-xs shadow-inner">
        <div className="flex items-center gap-2 text-slate-400">
          <Sliders className="w-4 h-4 text-amber-400" />
          <span className="font-semibold text-slate-100">Comparative Viewport Controls</span>
          <span className="text-white/20">|</span>
          <span>Zoom: Frontal <strong className="text-slate-100">{Math.round(frontalZoom * 100)}%</strong> · Lateral <strong className="text-slate-100">{Math.round(lateralZoom * 100)}%</strong></span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setSyncZoom(!syncZoom)}
            className={`px-3 py-1.5 rounded-lg border font-medium transition-all ${
              syncZoom
                ? 'bg-amber-950/60 border-amber-500/50 text-amber-300 font-semibold'
                : 'bg-medical-card border-white/10 text-slate-400 hover:text-white'
            }`}
          >
            {syncZoom ? '✓ Synchronized Zoom' : 'Independent Zoom'}
          </button>

          <button
            onClick={() => setIsInverted(!isInverted)}
            className={`px-3 py-1.5 rounded-lg border font-medium transition-all ${
              isInverted
                ? 'bg-white/10 border-white/30 text-slate-100 font-semibold'
                : 'bg-medical-card border-white/10 text-slate-400 hover:text-white'
            }`}
          >
            {isInverted ? 'Inverted (Bone White)' : 'Standard X-Ray'}
          </button>

          <button
            onClick={handleZoomReset}
            className="px-2.5 py-1.5 rounded-lg bg-medical-card border border-white/10 text-slate-400 hover:text-white"
            title="Reset Zoom to 100%"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Dual-View Grid: 2 Side-by-Side Radiograph Viewports */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* VIEWPORT 1: Frontal PA/AP Projection */}
        <Card
          title="Projection 1: Frontal Radiograph (PA / AP)"
          subtitle="Primary thoracic projection evaluating bilateral lung fields, cardiac silhouette, and costophrenic sulci"
          headerAction={
            frontal.file && (
              <div className="flex items-center gap-2">
                <Badge variant="amber" size="sm">
                  {frontal.width} × {frontal.height} px
                </Badge>
                <button
                  onClick={() => setFrontal(initialViewState)}
                  className="text-xs text-rose-400 hover:underline"
                >
                  Clear
                </button>
              </div>
            )
          }
        >
          <input
            type="file"
            ref={frontalInputRef}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                processImageFile(e.target.files[0], setFrontal);
              }
            }}
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
          />

          {!frontal.previewUrl ? (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setFrontal((prev) => ({ ...prev, isDragOver: true }));
              }}
              onDragLeave={() => setFrontal((prev) => ({ ...prev, isDragOver: false }))}
              onDrop={(e) => {
                e.preventDefault();
                setFrontal((prev) => ({ ...prev, isDragOver: false }));
                if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                  processImageFile(e.dataTransfer.files[0], setFrontal);
                }
              }}
              onClick={() => frontalInputRef.current?.click()}
              className={`h-96 border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 text-center cursor-pointer transition-all relative overflow-hidden group ${
                frontal.isDragOver
                  ? 'border-amber-400 bg-amber-950/30'
                  : 'border-white/10 hover:border-amber-400/50 hover:bg-[#180e28]/50'
              }`}
            >
              <div className="absolute inset-0 medical-grid opacity-20 pointer-events-none" />
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-[#28133d] to-[#150a22] border border-amber-500/30 flex items-center justify-center text-amber-400 mb-3 shadow-xl shadow-amber-950/40 group-hover:scale-110 group-hover:border-amber-400 transition-all z-10">
                <UploadCloud className="w-7 h-7" />
              </div>
              <p className="text-sm font-semibold text-white z-10 group-hover:text-amber-300 transition-colors">Upload Frontal (PA / AP) View</p>
              <p className="text-xs text-slate-400 mt-1.5 max-w-xs z-10">
                Drag and drop posteroanterior or anteroposterior chest radiograph
              </p>
              <span className="mt-3 text-[10px] font-mono px-2 py-0.5 rounded bg-[#0e0719]/90 border border-amber-500/30 text-amber-400 font-semibold z-10 shadow-sm">
                Slot A: Frontal View
              </span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  loadSampleSlot('frontal');
                }}
                disabled={loadingSampleSlot === 'frontal'}
                className="mt-3 px-3 py-1.5 text-[11px] font-mono rounded-lg bg-amber-950/80 hover:bg-amber-900 border border-amber-500/50 text-amber-300 flex items-center gap-1.5 shadow-md z-20 hover:scale-105 transition-all"
              >
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                <span>{loadingSampleSlot === 'frontal' ? 'Loading Frontal...' : 'Try Demo Frontal'}</span>
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              <div
                onWheel={(e) => handleZoomWheel(e, 'frontal')}
                className="relative rounded-xl overflow-hidden border border-white/10 bg-[#0a0512] h-96 flex items-center justify-center select-none cursor-ns-resize"
                title="Scroll mouse wheel to zoom frontal radiograph"
              >
                <div
                  className="w-full h-full flex items-center justify-center overflow-hidden"
                  style={{
                    transform: `scale(${frontalZoom})`,
                    filter: isInverted ? 'invert(1)' : 'none',
                    transition: 'transform 0.15s ease-out',
                  }}
                >
                  <img
                    src={frontal.previewUrl}
                    alt="Frontal X-ray"
                    className="max-h-full max-w-full object-contain pointer-events-none"
                  />
                </div>

                {/* Orientation & Anatomical Markers */}
                <div className="absolute top-3 left-3 bg-[#0e0719]/90 backdrop-blur-md px-2.5 py-1 rounded text-[11px] font-mono text-amber-300 border border-amber-500/30">
                  PA / AP Frontal Projection
                </div>

                {/* Laser Scanning Overlay during Reconstruction / Screening */}
                {(isAnalyzing || isReconstructing) && (
                  <div className="absolute inset-0 z-30 pointer-events-none overflow-hidden rounded-xl">
                    <div className="absolute inset-0 bg-[#0c0614]/40 backdrop-blur-[1px]" />
                    <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-amber-400 to-transparent shadow-[0_0_15px_#f59e0b] animate-laser-sweep" />
                    <div className="absolute inset-x-0 bottom-12 flex justify-center px-4">
                      <div className="bg-[#0e0719]/95 backdrop-blur-md px-3 py-1 rounded-lg border border-amber-500/50 text-[11px] font-mono text-amber-300 flex items-center gap-2 shadow-2xl">
                        <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                        <span>{isReconstructing ? 'Calibrating Frontal Contour Silhouette...' : 'Analyzing NIH Pathologies...'}</span>
                      </div>
                    </div>
                  </div>
                )}
                <div className="absolute top-3 right-3 flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-black/70 font-mono text-xs text-white border border-white/20">
                    L
                  </span>
                </div>
                <div className="absolute bottom-3 left-3 flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-black/70 font-mono text-xs text-white border border-white/20">
                    R
                  </span>
                </div>

                {/* Local Viewport Zoom Controls */}
                <div className="absolute bottom-3 right-3 flex items-center gap-1 bg-[#0e0719]/90 backdrop-blur-md px-2 py-1 rounded-lg border border-white/10">
                  <button
                    onClick={() => handleZoomOut('frontal')}
                    className="p-1 text-slate-400 hover:text-white"
                  >
                    <ZoomOut className="w-3.5 h-3.5" />
                  </button>
                  <span className="text-[10px] font-mono text-amber-400 px-1 font-bold">
                    {Math.round(frontalZoom * 100)}%
                  </span>
                  <button
                    onClick={() => handleZoomIn('frontal')}
                    className="p-1 text-slate-400 hover:text-white"
                  >
                    <ZoomIn className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Frontal File Metadata */}
              <div className="flex items-center justify-between text-xs text-medical-muted px-1 font-mono">
                <span className="truncate max-w-[240px]" title={frontal.file?.name}>
                  {frontal.file?.name}
                </span>
                <span>
                  {frontal.file ? `${(frontal.file.size / (1024 * 1024)).toFixed(2)} MB` : ''} ·{' '}
                  {frontal.width}×{frontal.height}
                </span>
              </div>
            </div>
          )}
          {frontal.error && (
            <div className="mt-2 text-xs text-rose-400 flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              <span>{frontal.error}</span>
            </div>
          )}
        </Card>

        {/* VIEWPORT 2: Lateral Projection */}
        <Card
          title="Projection 2: Lateral Radiograph"
          subtitle="Orthogonal sagittal projection visualizing retrosternal clear space, spine sign, and retrocardiac recess"
          headerAction={
            lateral.file && (
              <div className="flex items-center gap-2">
                <Badge variant="silver" size="sm">
                  {lateral.width} × {lateral.height} px
                </Badge>
                <button
                  onClick={() => setLateral(initialViewState)}
                  className="text-xs text-rose-400 hover:underline"
                >
                  Clear
                </button>
              </div>
            )
          }
        >
          <input
            type="file"
            ref={lateralInputRef}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                processImageFile(e.target.files[0], setLateral);
              }
            }}
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
          />

          {!lateral.previewUrl ? (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setLateral((prev) => ({ ...prev, isDragOver: true }));
              }}
              onDragLeave={() => setLateral((prev) => ({ ...prev, isDragOver: false }))}
              onDrop={(e) => {
                e.preventDefault();
                setLateral((prev) => ({ ...prev, isDragOver: false }));
                if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                  processImageFile(e.dataTransfer.files[0], setLateral);
                }
              }}
              onClick={() => lateralInputRef.current?.click()}
              className={`h-96 border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 text-center cursor-pointer transition-all relative overflow-hidden group ${
                lateral.isDragOver
                  ? 'border-slate-300 bg-white/10'
                  : 'border-white/10 hover:border-white/30 hover:bg-[#180e28]/50'
              }`}
            >
              <div className="absolute inset-0 medical-grid opacity-20 pointer-events-none" />
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-[#201235] to-[#12081d] border border-white/20 flex items-center justify-center text-slate-200 mb-3 shadow-xl shadow-black/40 group-hover:scale-110 group-hover:border-white/40 transition-all z-10">
                <UploadCloud className="w-7 h-7" />
              </div>
              <p className="text-sm font-semibold text-white z-10 group-hover:text-slate-100 transition-colors">Upload Lateral View</p>
              <p className="text-xs text-slate-400 mt-1.5 max-w-xs z-10">
                Drag and drop left or right lateral thoracic radiograph
              </p>
              <span className="mt-3 text-[10px] font-mono px-2 py-0.5 rounded bg-[#0e0719]/90 border border-white/20 text-slate-200 font-semibold z-10 shadow-sm">
                Slot B: Lateral View
              </span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  loadSampleSlot('lateral');
                }}
                disabled={loadingSampleSlot === 'lateral'}
                className="mt-3 px-3 py-1.5 text-[11px] font-mono rounded-lg bg-[#24133b] hover:bg-[#2d184a] border border-white/20 text-slate-200 flex items-center gap-1.5 shadow-md z-20 hover:scale-105 transition-all"
              >
                <Sparkles className="w-3.5 h-3.5 text-slate-300" />
                <span>{loadingSampleSlot === 'lateral' ? 'Loading Lateral...' : 'Try Demo Lateral'}</span>
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              <div
                onWheel={(e) => handleZoomWheel(e, 'lateral')}
                className="relative rounded-xl overflow-hidden border border-white/10 bg-[#0a0512] h-96 flex items-center justify-center select-none cursor-ns-resize"
                title="Scroll mouse wheel to zoom lateral radiograph"
              >
                <div
                  className="w-full h-full flex items-center justify-center overflow-hidden"
                  style={{
                    transform: `scale(${lateralZoom})`,
                    filter: isInverted ? 'invert(1)' : 'none',
                    transition: 'transform 0.15s ease-out',
                  }}
                >
                  <img
                    src={lateral.previewUrl}
                    alt="Lateral X-ray"
                    className="max-h-full max-w-full object-contain pointer-events-none"
                  />
                </div>

                {/* Lateral Orientation Labels */}
                <div className="absolute top-3 left-3 bg-[#0e0719]/90 backdrop-blur-md px-2.5 py-1 rounded text-[11px] font-mono text-slate-200 border border-white/20">
                  Lateral Projection (Sagittal Plane)
                </div>

                {/* Laser Scanning Overlay during Reconstruction / Screening */}
                {isReconstructing && (
                  <div className="absolute inset-0 z-30 pointer-events-none overflow-hidden rounded-xl">
                    <div className="absolute inset-0 bg-[#0c0614]/40 backdrop-blur-[1px]" />
                    <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-amber-400 to-transparent shadow-[0_0_15px_#f59e0b] animate-laser-sweep" />
                    <div className="absolute inset-x-0 bottom-12 flex justify-center px-4">
                      <div className="bg-[#0e0719]/95 backdrop-blur-md px-3 py-1 rounded-lg border border-amber-500/50 text-[11px] font-mono text-amber-300 flex items-center gap-2 shadow-2xl">
                        <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                        <span>Optimizing Lateral Depth Silhouette...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div className="absolute top-3 right-3 flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-black/70 font-mono text-xs text-white border border-white/20">
                    A (Anterior)
                  </span>
                </div>
                <div className="absolute bottom-3 left-3 flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-black/70 font-mono text-xs text-white border border-white/20">
                    P (Posterior)
                  </span>
                </div>

                {/* Local Zoom Controls */}
                <div className="absolute bottom-3 right-3 flex items-center gap-1 bg-[#0e0719]/90 backdrop-blur-md px-2 py-1 rounded-lg border border-white/10">
                  <button
                    onClick={() => handleZoomOut('lateral')}
                    className="p-1 text-slate-400 hover:text-white"
                  >
                    <ZoomOut className="w-3.5 h-3.5" />
                  </button>
                  <span className="text-[10px] font-mono text-slate-200 px-1 font-bold">
                    {Math.round(lateralZoom * 100)}%
                  </span>
                  <button
                    onClick={() => handleZoomIn('lateral')}
                    className="p-1 text-slate-400 hover:text-white"
                  >
                    <ZoomIn className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Lateral File Metadata */}
              <div className="flex items-center justify-between text-xs text-medical-muted px-1 font-mono">
                <span className="truncate max-w-[240px]" title={lateral.file?.name}>
                  {lateral.file?.name}
                </span>
                <span>
                  {lateral.file ? `${(lateral.file.size / (1024 * 1024)).toFixed(2)} MB` : ''} ·{' '}
                  {lateral.width}×{lateral.height}
                </span>
              </div>
            </div>
          )}
          {lateral.error && (
            <div className="mt-2 text-xs text-rose-400 flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              <span>{lateral.error}</span>
            </div>
          )}
        </Card>
      </div>

      {/* Workflow Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-white/[0.08] pb-2 text-sm font-medium">
        <button
          onClick={() => setActiveTab('reconstruction')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === 'reconstruction'
              ? 'bg-gradient-to-r from-amber-400 via-amber-500 to-amber-600 text-slate-950 font-bold shadow-md shadow-amber-950/40 border border-amber-300/40'
              : 'text-slate-400 hover:text-white hover:bg-white/5'
          }`}
        >
          <Box className="w-4 h-4 text-inherit" />
          <span>3D Lung Reconstruction (Phase 2B SSM)</span>
          {reconstructionResult && (
            <span className="w-2 h-2 rounded-full bg-slate-950" />
          )}
        </button>

        <button
          onClick={() => setActiveTab('screening')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === 'screening'
              ? 'bg-gradient-to-r from-amber-400 via-amber-500 to-amber-600 text-slate-950 font-bold shadow-md shadow-amber-950/40 border border-amber-300/40'
              : 'text-slate-400 hover:text-white hover:bg-white/5'
          }`}
        >
          <Activity className="w-4 h-4 text-inherit" />
          <span>2D Pathology Screening (DenseNet-121)</span>
          {frontalResult && (
            <span className="w-2 h-2 rounded-full bg-slate-950" />
          )}
        </button>

        <button
          onClick={() => setActiveTab('roadmap')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === 'roadmap'
              ? 'bg-gradient-to-r from-amber-400 via-amber-500 to-amber-600 text-slate-950 font-bold shadow-md shadow-amber-950/40 border border-amber-300/40'
              : 'text-slate-400 hover:text-white hover:bg-white/5'
          }`}
        >
          <Info className="w-4 h-4 text-inherit" />
          <span>Scientific Architecture & Roadmap</span>
        </button>
      </div>

      {/* TAB 1: 3D SSM RECONSTRUCTION */}
      {activeTab === 'reconstruction' && (
        <div className="space-y-6">
          {reconstructionError && (
            <div className="p-3 bg-rose-950/40 border border-rose-800/80 rounded-xl text-xs text-rose-300 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{reconstructionError}</span>
            </div>
          )}

          {/* SSM Parameter Control Bar */}
          <div className="p-4 rounded-xl bg-[#150a22]/80 border border-white/10 grid grid-cols-1 md:grid-cols-4 gap-4 text-xs shadow-inner">
            <div>
              <label className="text-slate-400 block font-medium mb-1">
                Inspiratory Effort: <strong className="text-white font-mono">{inspiratoryEffort > 0 ? `+${inspiratoryEffort}` : inspiratoryEffort}</strong>
              </label>
              <input
                type="range"
                min="-2.0"
                max="2.0"
                step="0.1"
                value={inspiratoryEffort}
                onChange={(e) => setInspiratoryEffort(parseFloat(e.target.value))}
                className="w-full accent-amber-400 cursor-pointer"
              />
              <span className="text-[10px] text-slate-400 flex justify-between">
                <span>Expiration (-2.0)</span>
                <span>Deep Insp (+2.0)</span>
              </span>
            </div>

            <div>
              <label className="text-slate-400 block font-medium mb-1">
                Anatomical Scale Factor: <strong className="text-white font-mono">{silhouetteScale.toFixed(2)}×</strong>
              </label>
              <input
                type="range"
                min="0.75"
                max="1.35"
                step="0.05"
                value={silhouetteScale}
                onChange={(e) => setSilhouetteScale(parseFloat(e.target.value))}
                className="w-full accent-amber-400 cursor-pointer"
              />
              <span className="text-[10px] text-slate-400 flex justify-between">
                <span>0.75×</span>
                <span>1.35×</span>
              </span>
            </div>

            <div>
              <label className="text-slate-400 block font-medium mb-1">
                Optimization Iterations: <strong className="text-white font-mono">{optimizationIterations}</strong>
              </label>
              <input
                type="range"
                min="10"
                max="80"
                step="5"
                value={optimizationIterations}
                onChange={(e) => setOptimizationIterations(parseInt(e.target.value))}
                className="w-full accent-amber-400 cursor-pointer"
              />
              <span className="text-[10px] text-slate-400 flex justify-between">
                <span>Fast (10)</span>
                <span>Refined (80)</span>
              </span>
            </div>

            <div className="flex flex-col justify-between">
              <label className="text-slate-400 block font-medium mb-1">
                Dual-View Constraints:
              </label>
              <label className="flex items-center gap-2 cursor-pointer mt-1">
                <input
                  type="checkbox"
                  checked={includeLateralConstraint}
                  onChange={(e) => setIncludeLateralConstraint(e.target.checked)}
                  className="rounded border-white/20 text-amber-500 focus:ring-0"
                />
                <span className="text-white font-medium">Use Lateral Projection (Mode 5 AP)</span>
              </label>
              <span className="text-[10px] text-slate-400 mt-1">
                {lateral.file ? 'Paired lateral study loaded' : 'Synthesized lateral prior active'}
              </span>
            </div>
          </div>

          {/* 3D Reconstruction Interactive Section */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left (7 Cols): 3D Surface Viewport */}
            <div className="lg:col-span-7 space-y-4">
              <Card
                title="Patient-Specific 3D Lung Surface Mesh"
                subtitle="Continuous 3D organ boundary synthesized via Statistical Shape Model (SSM) deformation"
                headerAction={
                  reconstructionResult && (
                    <Badge variant="amber" size="sm">
                      {reconstructionResult.surface_mesh.vertex_count.toLocaleString()} Vertices · {reconstructionResult.surface_mesh.face_count.toLocaleString()} Faces
                    </Badge>
                  )
                }
              >
                {!reconstructionResult ? (
                  <div className="h-[460px] border border-dashed border-white/10 rounded-xl flex flex-col items-center justify-center p-8 text-center bg-[#160c26]/40">
                    <Box className="w-12 h-12 text-slate-500 mb-3 animate-pulse" />
                    <p className="font-semibold text-white text-sm">3D Surface Reconstruction Ready</p>
                    <p className="text-xs text-slate-400 mt-1.5 max-w-sm">
                      Click the "Reconstruct 3D Lung Surface" button above to optimize the Statistical Shape Model against frontal and lateral radiographic silhouettes.
                    </p>
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={executeReconstruction}
                      disabled={isReconstructing}
                      isLoading={isReconstructing}
                      className="mt-4"
                    >
                      <Sparkles className="w-3.5 h-3.5 mr-1" />
                      <span>Execute 3D Reconstruction</span>
                    </Button>
                  </div>
                ) : (
                  <SSM3DReconstructionViewer
                    vertices={reconstructionResult.surface_mesh.vertices}
                    faces={reconstructionResult.surface_mesh.faces}
                    landmarks={reconstructionResult.surface_mesh.landmarks}
                    onExportObj={handleExportObj}
                    isExporting={isExportingObj}
                  />
                )}
              </Card>
            </div>

            {/* Right (5 Cols): Clinical Volumetry, Metrics & Landmark Decomposition */}
            <div className="lg:col-span-5 space-y-4">
              {/* Volumetry Card */}
              <Card
                title="Thoracic Organ Volumetry"
                subtitle="Continuous surface integration via Discrete Divergence Theorem"
              >
                {reconstructionResult ? (
                  <div className="space-y-3.5">
                    <div className="p-3.5 bg-medical-surface rounded-xl border border-white/10 flex items-center justify-between">
                      <div>
                        <span className="text-xs text-slate-400 block">Total Lung Volume (TLV):</span>
                        <span className="text-xl font-bold font-mono text-amber-400">
                          {reconstructionResult.volumetry.total_lung_volume_ml.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} mL
                        </span>
                      </div>
                      <Badge variant="amber" size="sm">
                        {reconstructionResult.volumetry.total_lung_volume_ml > 800 ? 'Normal Capacity' : 'Restricted'}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded-xl bg-medical-card border border-white/10">
                        <span className="text-[11px] text-slate-400 block">Right Lung (3 Lobes)</span>
                        <strong className="text-base font-mono text-white">
                          {reconstructionResult.volumetry.right_lung_ml.toFixed(1)} mL
                        </strong>
                        <span className="text-[10px] text-amber-300 block mt-0.5">
                          ~{((reconstructionResult.volumetry.right_lung_ml / reconstructionResult.volumetry.total_lung_volume_ml) * 100).toFixed(0)}% volume
                        </span>
                      </div>

                      <div className="p-3 rounded-xl bg-medical-card border border-white/10">
                        <span className="text-[11px] text-slate-400 block">Left Lung (2 Lobes)</span>
                        <strong className="text-base font-mono text-white">
                          {reconstructionResult.volumetry.left_lung_ml.toFixed(1)} mL
                        </strong>
                        <span className="text-[10px] text-slate-300 block mt-0.5">
                          ~{((reconstructionResult.volumetry.left_lung_ml / reconstructionResult.volumetry.total_lung_volume_ml) * 100).toFixed(0)}% volume
                        </span>
                      </div>
                    </div>

                    <div className="p-3 rounded-xl bg-medical-surface border border-white/10 flex items-center justify-between text-xs">
                      <span className="text-slate-400">Bilateral Symmetry Ratio (R/L):</span>
                      <span className="font-mono font-bold text-white">
                        {reconstructionResult.volumetry.symmetry_ratio.toFixed(2)}
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="py-8 text-center text-xs text-slate-400">
                    No volume computed yet. Run 3D reconstruction to calculate organ volumetry.
                  </div>
                )}
              </Card>

              {/* Quality & Confidence Card */}
              <Card
                title="Surface Reconstruction Quality & Confidence"
                subtitle="Non-linear L-BFGS-B optimization convergence and spatial uncertainty"
              >
                {reconstructionResult ? (
                  <div className="space-y-3 text-xs">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-2.5 rounded-lg bg-medical-surface border border-white/10">
                        <span className="text-slate-400 block text-[11px]">Estimated Dice Score:</span>
                        <strong className="text-sm font-mono text-amber-400">
                          {(reconstructionResult.metrics.estimated_dice_score * 100).toFixed(1)}%
                        </strong>
                      </div>

                      <div className="p-2.5 rounded-lg bg-medical-surface border border-white/10">
                        <span className="text-slate-400 block text-[11px]">Surface Proximity (ASSD):</span>
                        <strong className="text-sm font-mono text-white">
                          {reconstructionResult.metrics.estimated_assd_mm.toFixed(2)} mm
                        </strong>
                      </div>

                      <div className="p-2.5 rounded-lg bg-medical-surface border border-white/10">
                        <span className="text-slate-400 block text-[11px]">Mean Vertex Uncertainty:</span>
                        <strong className="text-sm font-mono text-white">
                          ±{reconstructionResult.metrics.mean_vertex_uncertainty_mm.toFixed(2)} mm
                        </strong>
                      </div>

                      <div className="p-2.5 rounded-lg bg-medical-surface border border-white/10">
                        <span className="text-slate-400 block text-[11px]">Convergence Latency:</span>
                        <strong className="text-sm font-mono text-slate-200">
                          {reconstructionResult.optimization_latency_ms.toFixed(1)} ms
                        </strong>
                      </div>
                    </div>

                    <div className="p-2.5 rounded-lg bg-[#0e0719]/90 border border-white/10 text-[11px] text-slate-400 flex items-center justify-between">
                      <span>Status: <strong className="text-emerald-400 uppercase">{reconstructionResult.status}</strong></span>
                      <span>Iterations: <strong className="text-white font-mono">{reconstructionResult.iterations_computed}</strong></span>
                      <span>Confidence Margin: <strong className="text-amber-300">{reconstructionResult.metrics.confidence_margin_pct}%</strong></span>
                    </div>
                  </div>
                ) : (
                  <div className="py-8 text-center text-xs text-slate-400">
                    Execute reconstruction to inspect mathematical convergence metrics.
                  </div>
                )}
              </Card>

              {/* SSM Principal Modes Decomposition */}
              {ssmModelInfo && (
                <Card
                  title="Statistical Shape Modes (PCA Decomposition)"
                  subtitle="Top orthogonal deformation eigenvectors explaining 100% cohort variance"
                >
                  <div className="space-y-2 text-xs">
                    {ssmModelInfo.modes.slice(0, 5).map((mode) => (
                      <div
                        key={mode.mode_index}
                        className="p-2 rounded-lg bg-medical-surface border border-white/10 flex items-center justify-between"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-amber-400 font-bold">M{mode.mode_index}</span>
                          <span className="text-white text-[11px]">{mode.description}</span>
                        </div>
                        <span className="font-mono text-slate-400 text-[11px]">
                          {mode.explained_variance_pct.toFixed(1)}% var (±{mode.standard_deviation_mm} mm)
                        </span>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: 2D PATHOLOGY SCREENING */}
      {activeTab === 'screening' && (
        <Card
          title="Frontal Radiograph Neural Inference (Synchronized Dual-View Review)"
          subtitle="DenseNet-121 multi-label inference on frontal projection; Lateral radiograph provided for synchronized visual assessment"
        >
          {analysisError && (
            <div className="p-3 bg-rose-950/40 border border-rose-800/80 rounded-xl text-xs text-rose-300 flex items-center gap-2 mb-3">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{analysisError}</span>
            </div>
          )}

          {/* Workflow Transparency Disclosure */}
          <div className="p-3 bg-amber-950/20 border border-amber-800/40 rounded-xl text-xs text-slate-300 mb-3 flex items-start gap-2.5">
            <Sliders className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-white block">Bi-Planar Clinical Workflow Transparency:</span>
              <p className="text-[11px] text-slate-300 leading-relaxed mt-0.5">
                The DenseNet-121 multi-label classifier is validated exclusively on PA/AP frontal chest radiographs. The lateral view is loaded into the synchronized inspection viewport for radiologist anatomical review of the retrocardiac space and spine.
              </p>
            </div>
          </div>

          {!frontalResult ? (
            <div className="py-12 text-center text-slate-400 text-xs border border-dashed border-white/10 rounded-xl p-6">
              <FileImage className="w-10 h-10 text-slate-500 mx-auto mb-2" />
              <p className="font-semibold text-white">No Dual-View Analysis Executed</p>
              <p className="mt-1 max-w-sm mx-auto">
                Upload at least the Frontal radiograph above and click "Screen Pathology" to screen for 6 thoracic conditions.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="p-3.5 bg-medical-surface rounded-xl border border-white/10 flex items-center justify-between text-xs">
                <div>
                  <span className="text-slate-400 block">Frontal Projection Findings:</span>
                  <strong className="text-white text-sm font-mono">
                    {frontalResult.summary.flagged_conditions_count} of 6 Conditions Above Threshold
                  </strong>
                </div>
                <div className="text-right">
                  <span className="text-slate-400 block">Inference Latency:</span>
                  <span className="font-mono text-amber-400 font-bold">
                    {frontalResult.inference_time_ms.toFixed(1)} ms
                  </span>
                </div>
              </div>

              <div className="space-y-2">
                {EXPECTED_6_DISEASES.map((disease) => {
                  const pred = frontalResult.predictions[disease];
                  return (
                    <ProbabilityBar
                      key={disease}
                      label={disease}
                      probability={pred.probability}
                      threshold={pred.threshold}
                      experimentalFlag={pred.experimental_flag}
                    />
                  );
                })}
              </div>
            </div>
          )}
        </Card>
      )}

      {/* TAB 3: SCIENTIFIC ARCHITECTURE & ROADMAP */}
      {activeTab === 'roadmap' && (
        <Card
          title="3D Reconstruction Scientific Roadmap & Methodology"
          subtitle="Advancing from 2D dual projections to patient-specific volumetric anatomical synthesis"
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="p-3.5 rounded-xl bg-[#1c102f] border border-white/10 space-y-2">
              <div className="flex items-center gap-1.5 text-slate-200 font-semibold">
                <CheckCircle2 className="w-4 h-4 text-amber-400" />
                <span>Phase 1: Dual-View Review</span>
              </div>
              <p className="text-[11px] text-slate-300 leading-relaxed">
                Simultaneous paired review of frontal and lateral radiographs with independent pan, zoom, and orientation labeling.
              </p>
              <Badge variant="silver" size="sm">Completed</Badge>
            </div>

            <div className="p-3.5 rounded-xl bg-amber-950/20 border border-amber-500/40 space-y-2">
              <div className="flex items-center gap-1.5 text-amber-300 font-semibold">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                <span>Phase 2: Morphable Shape Models</span>
              </div>
              <p className="text-[11px] text-slate-200 leading-relaxed">
                Point Distribution Model with 15 PCA eigenvectors deforming 7,652 canonical vertices against frontal & lateral silhouettes.
              </p>
              <Badge variant="amber" size="sm">Implemented & Active</Badge>
            </div>

            <div className="p-3.5 rounded-xl bg-[#150a22] border border-white/10 space-y-2">
              <div className="flex items-center gap-1.5 text-slate-400 font-semibold">
                <span className="w-2 h-2 rounded-full bg-white/20" />
                <span>Phase 3: Differentiable DRR & NeRF</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Continuous 3D attenuation volume synthesis directly from orthogonal 2D rays without CT radiation dose.
              </p>
              <Badge variant="slate" size="sm">Future Milestone</Badge>
            </div>
          </div>

          <div className="mt-4 p-3 rounded-xl bg-amber-950/25 border border-amber-800/40 text-[11px] text-amber-200 flex items-start gap-2">
            <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <p className="leading-relaxed">
              <strong>Medical Disclaimer:</strong> Statistical Shape Model (SSM) 3D surface reconstruction estimates macroscopic organ boundary geometry based on prior shape distributions and radiographic silhouettes. It does not synthesize internal Hounsfield Unit densities or diagnostic parenchyma. Not for primary diagnostic use.
            </p>
          </div>
        </Card>
      )}
    </div>
  );
};
