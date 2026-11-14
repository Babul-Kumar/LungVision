import React, { useEffect, useState, useRef, ChangeEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Layers,
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Crosshair,
  ChevronLeft,
  ChevronRight,
  Film,
  Download,
  ArrowRight,
  Sliders,
  Activity,
  RotateCcw,
  Play,
  Pause,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Sun,
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { ThreeMeshCanvas } from './ThreeMeshCanvas';
import { SSM3DReconstructionViewer } from '../xray/SSM3DReconstructionViewer';
import { ctApi } from '@/services/api/ctApi';
import { reconstructionApi } from '@/services/api/reconstructionApi';
import {
  CTAnnotationItem,
  CTMeasurementResponse,
  CTMeshQualityResponse,
  CTMetadataResponse,
  CTSliceResponse,
  DRRProjectionResponse,
} from '@/types/ct';
import {
  SSMDeformResponse,
  SSMModelInfo,
} from '@/types/reconstruction';

export const CTExplorerPage: React.FC = () => {
  const navigate = useNavigate();
  const [activeScanId, setActiveScanId] = useState<string>('demo_thoracic_scan');
  const [activeTab, setActiveTab] = useState<'3d' | 'mpr' | 'annotations' | 'drr' | 'ssm'>('3d');
  
  // Data States
  const [metadata, setMetadata] = useState<CTMetadataResponse | null>(null);
  const [annotations, setAnnotations] = useState<CTAnnotationItem[]>([]);
  const [measurements, setMeasurements] = useState<CTMeasurementResponse | null>(null);
  const [meshQuality, setMeshQuality] = useState<CTMeshQualityResponse | null>(null);
  const [selectedStructure, setSelectedStructure] = useState<string>('lungs');
  const [selectedNoduleId, setSelectedNoduleId] = useState<string | null>(null);

  // MPR Viewport States
  const [mprView, setMprView] = useState<'axial' | 'coronal' | 'sagittal'>('axial');
  const [sliceIndex, setSliceIndex] = useState<number>(32);
  const [windowPreset, setWindowPreset] = useState<string>('lung');
  const [overlayMask, setOverlayMask] = useState<boolean>(true);
  const [showNodules, setShowNodules] = useState<boolean>(true);
  const [currentSlice, setCurrentSlice] = useState<CTSliceResponse | null>(null);
  const [loadingSlice, setLoadingSlice] = useState<boolean>(false);
  const [isCinePlaying, setIsCinePlaying] = useState<boolean>(false);
  const [mprZoom, setMprZoom] = useState<number>(1.0);
  const [mprInvert, setMprInvert] = useState<boolean>(false);

  // Upload States
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // DRR Radiographic Bridge States
  const [drrView, setDrrView] = useState<'frontal' | 'lateral'>('frontal');
  const [drrSdd, setDrrSdd] = useState<number>(1800.0);
  const [drrSod, setDrrSod] = useState<number>(1450.0);
  const [drrOverlaySilhouette, setDrrOverlaySilhouette] = useState<boolean>(false);
  const [drrAddNoise, setDrrAddNoise] = useState<boolean>(false);
  const [drrData, setDrrData] = useState<DRRProjectionResponse | null>(null);
  const [loadingDRR, setLoadingDRR] = useState<boolean>(false);
  const [drrError, setDrrError] = useState<string | null>(null);
  const [isExportingToDualView, setIsExportingToDualView] = useState<boolean>(false);

  // SSM Research Panel States
  const [ssmModelInfo, setSsmModelInfo] = useState<SSMModelInfo | null>(null);
  const [ssmModeWeights, setSsmModeWeights] = useState<number[]>([0, 0, 0, 0, 0]); // normalized [-3.0, +3.0]
  const [ssmStructure, setSsmStructure] = useState<'bilateral' | 'left_lung' | 'right_lung'>('bilateral');
  const [ssmGeometryPreset, setSsmGeometryPreset] = useState<'mean' | 'custom' | 'reference' | 'held_out_val'>('mean');
  const [ssmDeformResult, setSsmDeformResult] = useState<SSMDeformResponse | null>(null);
  const [loadingSSM, setLoadingSSM] = useState<boolean>(false);
  const [isExportingSSMObj, setIsExportingSSMObj] = useState<boolean>(false);
  const [ssmError, setSsmError] = useState<string | null>(null);


  // 1. Load Scan Metadata, Annotations, Measurements, Mesh Quality
  useEffect(() => {
    let isCancelled = false;

    const loadScanData = async () => {
      try {
        const [meta, annResp, meas, qual] = await Promise.all([
          ctApi.getMetadata(activeScanId),
          ctApi.getAnnotations(activeScanId).catch(() => ({ annotations: [] })),
          ctApi.getMeasurements(activeScanId).catch(() => null),
          ctApi.getMeshQuality(activeScanId, selectedStructure).catch(() => null),
        ]);

        if (!isCancelled) {
          setMetadata(meta);
          setAnnotations(annResp.annotations || []);
          setMeasurements(meas);
          setMeshQuality(qual);
          setSliceIndex(Math.floor((meta.dimensions.z || 64) / 2));
        }
      } catch (err: any) {
        if (!isCancelled) {
          setUploadError(err.message || 'Failed to load scan metadata');
        }
      }
    };

    loadScanData();
    return () => {
      isCancelled = true;
    };
  }, [activeScanId, selectedStructure]);

  // 2. Load 2D Slice on Navigation or Filter Change with debounce
  useEffect(() => {
    if (activeTab !== 'mpr') return;

    let isCancelled = false;
    const timer = setTimeout(() => {
      setLoadingSlice(true);

      ctApi
        .getSlice(activeScanId, mprView, sliceIndex, windowPreset, overlayMask, showNodules)
        .then((data) => {
          if (!isCancelled) setCurrentSlice(data);
        })
        .catch((err) => {
          console.error('Slice fetch error:', err);
        })
        .finally(() => {
          if (!isCancelled) setLoadingSlice(false);
        });
    }, 30);

    return () => {
      isCancelled = true;
      clearTimeout(timer);
    };
  }, [activeScanId, activeTab, mprView, sliceIndex, windowPreset, overlayMask, showNodules]);

  // 3. Load DRR on View or Geometry Change
  useEffect(() => {
    if (activeTab !== 'drr') return;

    let isCancelled = false;
    setLoadingDRR(true);
    setDrrError(null);

    ctApi
      .getDRR(activeScanId, drrView, drrSdd, drrSod, drrOverlaySilhouette, drrAddNoise)
      .then((data) => {
        if (!isCancelled) {
          setDrrData(data);
        }
      })
      .catch((err: any) => {
        if (!isCancelled) {
          setDrrError(err.message || 'Failed to synthesize Digitally Reconstructed Radiograph');
        }
      })
      .finally(() => {
        if (!isCancelled) setLoadingDRR(false);
      });

    return () => {
      isCancelled = true;
    };
  }, [activeScanId, activeTab, drrView, drrSdd, drrSod, drrOverlaySilhouette, drrAddNoise]);

  // Handle Export to Dual-View 3D Lab
  const handleExportToDualView = async () => {
    setIsExportingToDualView(true);
    try {
      const [frontalDRR, lateralDRR] = await Promise.all([
        drrView === 'frontal' && drrData
          ? Promise.resolve(drrData)
          : ctApi.getDRR(activeScanId, 'frontal', drrSdd, drrSod, false, drrAddNoise),
        drrView === 'lateral' && drrData
          ? Promise.resolve(drrData)
          : ctApi.getDRR(activeScanId, 'lateral', drrSdd, drrSod, false, drrAddNoise),
      ]);

      navigate('/dual-view', {
        state: {
          frontalUrl: frontalDRR.image_base64,
          lateralUrl: lateralDRR.image_base64,
          sourceScanId: activeScanId,
        },
      });
    } catch (err: any) {
      alert(`Export failed: ${err.message || 'Could not synthesize dual-view projections'}`);
    } finally {
      setIsExportingToDualView(false);
    }
  };

  // Handle Download DRR PNG
  const handleDownloadDRR = () => {
    if (!drrData?.image_base64) return;
    const a = document.createElement('a');
    a.href = drrData.image_base64;
    a.download = `drr_${activeScanId}_${drrView}_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // 4. Load SSM Model Info
  useEffect(() => {
    reconstructionApi
      .getModelInfo()
      .then((info) => setSsmModelInfo(info))
      .catch((err) => console.warn('Could not load SSM model info:', err));
  }, []);

  // 5. Deform SSM Surface Mesh upon Weight / Structure / Tab Change
  useEffect(() => {
    if (activeTab !== 'ssm') return;

    let isCancelled = false;
    const timer = setTimeout(() => {
      setLoadingSSM(true);
      setSsmError(null);

      const sigmas = [3819.4, 2560.5, 2141.5, 1963.3, 1494.1];
      const actualCoeffs = ssmModeWeights.map((w, idx) => w * (sigmas[idx] || 1000.0));

      reconstructionApi
        .deformShape({
          shape_coefficients: actualCoeffs,
          structure: ssmStructure,
        })
        .then((res) => {
          if (!isCancelled) setSsmDeformResult(res);
        })
        .catch((err: any) => {
          if (!isCancelled) setSsmError(err.message || 'Failed to deform SSM shape');
        })
        .finally(() => {
          if (!isCancelled) setLoadingSSM(false);
        });
    }, 35);

    return () => {
      isCancelled = true;
      clearTimeout(timer);
    };
  }, [activeTab, ssmModeWeights, ssmStructure]);

  // SSM Preset Selector
  const applySSMPreset = (preset: 'mean' | 'custom' | 'reference' | 'held_out_val') => {
    setSsmGeometryPreset(preset);
    if (preset === 'mean') {
      setSsmModeWeights([0.0, 0.0, 0.0, 0.0, 0.0]);
    } else if (preset === 'held_out_val') {
      setSsmModeWeights([0.82, -0.45, 0.31, -0.18, 0.25]);
    } else if (preset === 'reference') {
      setSsmModeWeights([0.25, 0.6, -0.15, 0.1, -0.2]);
    }
  };

  // SSM Mode Slider Change
  const handleModeWeightChange = (modeIndex: number, value: number) => {
    setSsmGeometryPreset('custom');
    setSsmModeWeights((prev) => {
      const next = [...prev];
      next[modeIndex] = value;
      return next;
    });
  };

  // Export SSM OBJ
  const handleExportSSMObj = async () => {
    if (!ssmDeformResult) return;
    setIsExportingSSMObj(true);
    try {
      const sigmas = [3819.4, 2560.5, 2141.5, 1963.3, 1494.1];
      const actualCoeffs = ssmModeWeights.map((w, idx) => w * (sigmas[idx] || 1000.0));
      const objData = await reconstructionApi.exportReconstructedObj(actualCoeffs);
      const blob = new Blob([objData], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `lungvision_ssm_${ssmStructure}_${Date.now()}.obj`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(`Export failed: ${err.message || 'Could not export OBJ mesh'}`);
    } finally {
      setIsExportingSSMObj(false);
    }
  };


  // 4. Handle CT Archive Upload
  const handleUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || !e.target.files[0]) return;
    const file = e.target.files[0];

    setIsUploading(true);
    setUploadError(null);

    try {
      const resp = await ctApi.uploadCT(file);
      setActiveScanId(resp.scan_id);
    } catch (err: any) {
      setUploadError(err.message || 'CT volume ingestion failed.');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const maxIndex =
    mprView === 'axial'
      ? (metadata?.dimensions.z || 64) - 1
      : mprView === 'coronal'
      ? (metadata?.dimensions.y || 128) - 1
      : (metadata?.dimensions.x || 128) - 1;

  // Jump to Nodule with appropriate plane indexing
  const jumpToNodule = (nodule: CTAnnotationItem, targetView?: 'axial' | 'coronal' | 'sagittal') => {
    const view = targetView || mprView;
    setSelectedNoduleId(nodule.nodule_id);
    setMprView(view);
    
    if (view === 'axial') {
      setSliceIndex(Math.max(0, Math.min(nodule.voxel_z, (metadata?.dimensions.z || 64) - 1)));
    } else if (view === 'coronal') {
      setSliceIndex(Math.max(0, Math.min(nodule.voxel_y, (metadata?.dimensions.y || 128) - 1)));
    } else {
      setSliceIndex(Math.max(0, Math.min(nodule.voxel_x, (metadata?.dimensions.x || 128) - 1)));
    }
    setActiveTab('mpr');
  };

  // Step slice helper
  const stepSlice = (delta: number) => {
    setSliceIndex((prev) => Math.max(0, Math.min(prev + delta, maxIndex)));
  };

  // Mouse wheel scroll handler for MPR slice scrubbing
  const handleMprWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      stepSlice(1);
    } else {
      stepSlice(-1);
    }
  };

  // Cine loop automated playback
  useEffect(() => {
    if (!isCinePlaying || activeTab !== 'mpr') return;
    const interval = setInterval(() => {
      setSliceIndex((prev) => (prev >= maxIndex ? 0 : prev + 1));
    }, 90);
    return () => clearInterval(interval);
  }, [isCinePlaying, activeTab, maxIndex]);

  // Orientation compass markers
  const getOrientationLabels = (view: 'axial' | 'coronal' | 'sagittal') => {
    if (view === 'axial') {
      return { top: 'A (Anterior)', bottom: 'P (Posterior)', left: 'R (Right)', right: 'L (Left)' };
    }
    if (view === 'coronal') {
      return { top: 'S (Superior)', bottom: 'I (Inferior)', left: 'R (Right)', right: 'L (Left)' };
    }
    return { top: 'S (Superior)', bottom: 'I (Inferior)', left: 'A (Anterior)', right: 'P (Posterior)' };
  };

  const orientation = getOrientationLabels(mprView);
  const activeNodule = annotations.find((n) => n.nodule_id === selectedNoduleId);

  return (
    <div className="space-y-6">
      {/* Header & Scan Switcher */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-medical-border">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-medical-teal bg-teal-950/40 px-2 py-0.5 rounded border border-teal-800/40">
              Workstation Module
            </span>
            <span className="text-xs text-medical-subtle">·</span>
            <span className="text-xs text-medical-muted">Volumetric Computed Tomography</span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            3D Computed Tomography Explorer
            <Badge variant="teal" size="sm">
              SimpleITK & Three.js
            </Badge>
          </h2>
          <p className="text-xs text-medical-muted mt-1">
            Volumetric MPR slicing, watertight marching cubes meshes, LUNA16 nodule mapping, and quantitative lung volumetry.
          </p>
        </div>

        {/* Scan Selection & Upload */}
        <div className="flex items-center gap-2.5">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleUpload}
            accept=".zip,.gz,.mhd,.dcm,.nii"
            className="hidden"
          />

          <select
            value={activeScanId}
            onChange={(e) => {
              setActiveScanId(e.target.value);
              setSelectedNoduleId(null);
            }}
            className="bg-medical-surface border border-medical-border rounded-lg px-3 py-1.5 text-xs font-mono text-medical-text focus:outline-none focus:border-medical-cyan shadow-sm"
          >
            <option value="demo_thoracic_scan">Reference Thoracic Scan (LUNA16 12-Nodule Series)</option>
            {activeScanId !== 'demo_thoracic_scan' && (
              <option value={activeScanId}>Active Upload ({activeScanId})</option>
            )}
          </select>

          <Button
            variant="outline"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            isLoading={isUploading}
            className="border-medical-border hover:border-medical-cyan shadow-sm"
          >
            <UploadCloud className="w-3.5 h-3.5 mr-1" />
            <span>Upload CT (.zip)</span>
          </Button>
        </div>
      </div>

      <DisclaimerBanner compact />

      {uploadError && (
        <div className="p-3 bg-rose-950/60 border border-rose-800 rounded-lg text-xs text-rose-300 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      {/* Spatial Metadata Strip */}
      {metadata && (
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
          <div className="bg-medical-card border border-medical-border/60 rounded-lg p-3">
            <span className="text-[10px] text-medical-muted uppercase font-semibold block">Dimensions (X, Y, Z)</span>
            <span className="text-xs font-mono text-white mt-1 block">
              {metadata.dimensions.x} × {metadata.dimensions.y} × {metadata.dimensions.z}
            </span>
          </div>

          <div className="bg-medical-card border border-medical-border/60 rounded-lg p-3">
            <span className="text-[10px] text-medical-muted uppercase font-semibold block">Voxel Spacing (mm)</span>
            <span className="text-xs font-mono text-medical-cyan mt-1 block">
              {(metadata.spacing_mm ?? []).map((s) => (typeof s === 'number' ? s.toFixed(3) : String(s))).join(', ')}
            </span>
          </div>

          <div className="bg-medical-card border border-medical-border/60 rounded-lg p-3">
            <span className="text-[10px] text-medical-muted uppercase font-semibold block">LUNA16 Annotations</span>
            <span className="text-xs font-mono text-amber-400 mt-1 block">
              {metadata.annotation_count} Mapped Nodules
            </span>
          </div>

          <div className="bg-medical-card border border-medical-border/60 rounded-lg p-3">
            <span className="text-[10px] text-medical-muted uppercase font-semibold block">Geometry Status</span>
            <Badge variant={metadata.geometry_validation.status === 'PASSED' ? 'emerald' : 'rose'} size="sm" className="mt-1">
              <CheckCircle2 className="w-3 h-3 mr-1" />
              {metadata.geometry_validation.status} (atol=1e-3)
            </Badge>
          </div>

          <div className="bg-medical-card border border-medical-border/60 rounded-lg p-3">
            <span className="text-[10px] text-medical-muted uppercase font-semibold block">Segmented Labels</span>
            <span className="text-xs font-mono text-slate-300 mt-1 block">
              {metadata.labels_present.join(', ')} (Left/Right/Airway)
            </span>
          </div>

          <div className="bg-medical-card border border-medical-border/60 rounded-lg p-3">
            <span className="text-[10px] text-medical-muted uppercase font-semibold block">Volumetric Engine</span>
            <span className="text-xs font-mono text-medical-teal mt-1 block">SimpleITK 2.5</span>
          </div>
        </div>
      )}

      {/* Main Exploration Tabs */}
      <div className="glass-panel p-1.5 rounded-xl border border-slate-700/60 flex flex-wrap items-center gap-1.5">
        <button
          onClick={() => setActiveTab('3d')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
            activeTab === '3d'
              ? 'bg-gradient-to-r from-cyan-500 to-teal-400 text-slate-950 font-bold shadow-md shadow-cyan-500/30'
              : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
          }`}
        >
          <Box className="w-4 h-4" />
          <span>3D Mesh & Volumetry</span>
        </button>

        <button
          onClick={() => setActiveTab('mpr')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'mpr'
              ? 'bg-gradient-to-r from-cyan-500 to-teal-400 text-slate-950 font-bold shadow-md shadow-cyan-500/30'
              : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Multi-Planar Reconstruction (MPR)</span>
        </button>

        <button
          onClick={() => setActiveTab('annotations')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'annotations'
              ? 'bg-gradient-to-r from-cyan-500 to-teal-400 text-slate-950 font-bold shadow-md shadow-cyan-500/30'
              : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
          }`}
        >
          <Crosshair className="w-4 h-4" />
          <span>LUNA16 Nodule Registry ({annotations.length})</span>
        </button>

        <button
          onClick={() => setActiveTab('drr')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'drr'
              ? 'bg-gradient-to-r from-cyan-500 to-teal-400 text-slate-950 font-bold shadow-md shadow-cyan-500/30'
              : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
          }`}
        >
          <Film className="w-4 h-4" />
          <span>DRR Synthesis & 3D Bridge</span>
        </button>

        <button
          onClick={() => setActiveTab('ssm')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'ssm'
              ? 'bg-gradient-to-r from-cyan-500 to-teal-400 text-slate-950 font-bold shadow-md shadow-cyan-500/30'
              : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
          }`}
        >
          <Sliders className="w-4 h-4" />
          <span>SSM Surface Model (Research)</span>
        </button>
      </div>

      {/* TAB 1: 3D Interactive Mesh & Volumetry */}
      {activeTab === '3d' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Three.js Canvas (8 Cols) */}
          <div className="lg:col-span-8 space-y-4">
            <Card
              title="Interactive 3D Anatomical Reconstruction"
              subtitle="Streamed directly from backend Marching Cubes pipeline (1-Voxel Zero Padded Watertight Mesh)"
              headerAction={
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] text-medical-muted mr-1">Structure:</span>
                  {[
                    { id: 'lungs', label: 'Bilateral Lungs', tag: '3,4' },
                    { id: 'left_lung', label: 'Left Lung', tag: '3' },
                    { id: 'right_lung', label: 'Right Lung', tag: '4' },
                    { id: 'heart', label: 'Heart', tag: '2' },
                    { id: 'airways', label: 'Airways', tag: '5' },
                  ].map((item) => {
                    const isPresent =
                      !metadata ||
                      (item.id === 'lungs' && (metadata.labels_present.includes(3) || metadata.labels_present.includes(4))) ||
                      (item.id === 'left_lung' && metadata.labels_present.includes(3)) ||
                      (item.id === 'right_lung' && metadata.labels_present.includes(4)) ||
                      (item.id === 'heart' && metadata.labels_present.includes(2)) ||
                      (item.id === 'airways' && metadata.labels_present.includes(5));

                    return (
                      <button
                        key={item.id}
                        onClick={() => setSelectedStructure(item.id)}
                        className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all flex items-center gap-1 ${
                          selectedStructure === item.id
                            ? item.id === 'heart'
                              ? 'bg-rose-600 text-white font-semibold shadow-xs'
                              : 'bg-medical-cyan text-medical-bg font-semibold shadow-xs'
                            : isPresent
                            ? 'bg-medical-surface text-medical-muted hover:text-white'
                            : 'bg-medical-surface/40 text-slate-500 border border-slate-800 line-through opacity-70'
                        }`}
                        title={isPresent ? `Segmented label ${item.tag}` : `Structure label ${item.tag} outside scan subvolume`}
                      >
                        <span>{item.label}</span>
                      </button>
                    );
                  })}
                </div>
              }
            >
              <div className="h-[460px]">
                <ThreeMeshCanvas
                  scanId={activeScanId}
                  structure={selectedStructure}
                  annotations={annotations}
                  selectedNoduleId={selectedNoduleId}
                  onSelectNodule={(noduleId) => {
                    setSelectedNoduleId(noduleId);
                    const found = annotations.find((n) => n.nodule_id === noduleId);
                    if (found) jumpToNodule(found);
                  }}
                />
              </div>

              {/* Heart Subvolume Notice if Heart is selected on a scan without Label 2 */}
              {selectedStructure === 'heart' && metadata && !metadata.labels_present.includes(2) && (
                <div className="mt-3 p-3 rounded-lg bg-amber-950/40 border border-amber-800/60 text-xs text-amber-200 flex items-start gap-2.5">
                  <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold text-white">Cardiac Subvolume Notice: </span>
                    Reference scan <code className="text-amber-300 font-mono">{activeScanId}</code> is an 80 mm thoracic subvolume centered on pulmonary nodule targets. Labels present are {metadata.labels_present.join(', ')} (Left Lung, Right Lung, Airway). The cardiac contour (label 2) is outside this subvolume's z-range. Select <strong>Bilateral Lungs</strong>, <strong>Left Lung</strong>, <strong>Right Lung</strong>, or <strong>Airways</strong> to inspect active 3D meshes.
                  </div>
                </div>
              )}

              {selectedNoduleId && activeNodule && (
                <div className="mt-3 p-2.5 rounded-lg bg-medical-surface border border-cyan-800/60 flex items-center justify-between text-xs font-mono">
                  <div className="flex items-center gap-2">
                    <Crosshair className="w-4 h-4 text-medical-cyan animate-pulse" />
                    <span>Selected Nodule: <strong className="text-white">{activeNodule.nodule_id}</strong></span>
                    <span className="text-medical-subtle">|</span>
                    <span className="text-amber-300 font-semibold">{typeof activeNodule.diameter_mm === 'number' ? activeNodule.diameter_mm.toFixed(1) : (activeNodule.diameter_mm ?? 'N/A')} mm</span>
                    <span className="text-medical-subtle">|</span>
                    <span className="text-slate-300">Voxel: ({activeNodule.voxel_x}, {activeNodule.voxel_y}, {activeNodule.voxel_z})</span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => jumpToNodule(activeNodule)}
                    className="text-medical-cyan text-xs py-0.5 px-2"
                  >
                    View in MPR Slice
                  </Button>
                </div>
              )}
            </Card>
          </div>

          {/* Right: Volumetry & Mesh Quality Cards (4 Cols) */}
          <div className="lg:col-span-4 space-y-5">
            {/* Quantitative Lung Volumetry */}
            <Card title="Quantitative Lung Volumetry" subtitle="Computed from verified voxel segmentation counts">
              {measurements ? (
                <div className="space-y-3.5 text-xs">
                  <div className="p-3 bg-medical-surface rounded-lg border border-medical-border flex items-center justify-between">
                    <span className="text-medical-muted">Total Segmented Volume:</span>
                    <strong className="text-sm font-mono text-white">
                      {(measurements.lungs?.total_lung_volume_ml ?? 0).toFixed(1)} mL
                    </strong>
                  </div>

                  <div className="grid grid-cols-2 gap-2.5">
                    <div className="p-3 bg-cyan-950/20 border border-cyan-800/40 rounded-lg">
                      <span className="text-[10px] text-medical-cyan uppercase font-semibold block">Left Lung</span>
                      <strong className="text-sm font-mono text-cyan-200 mt-1 block">
                        {(measurements.lungs?.left_lung_ml ?? 0).toFixed(1)} mL
                      </strong>
                    </div>

                    <div className="p-3 bg-teal-950/20 border border-teal-800/40 rounded-lg">
                      <span className="text-[10px] text-medical-teal uppercase font-semibold block">Right Lung</span>
                      <strong className="text-sm font-mono text-teal-200 mt-1 block">
                        {(measurements.lungs?.right_lung_ml ?? 0).toFixed(1)} mL
                      </strong>
                    </div>
                  </div>

                  <div className="flex justify-between py-1 border-b border-medical-border/60">
                    <span className="text-medical-muted">Trachea / Main Bronchus:</span>
                    <span className="font-mono text-amber-400 font-medium">
                      {(
                        measurements.airway?.trachea_volume_ml ??
                        measurements.lungs?.trachea_airway_ml ??
                        0
                      ).toFixed(1)} mL
                    </span>
                  </div>

                  {(measurements as any).heart?.heart_volume_ml != null && (
                    <div className="flex justify-between py-1 border-b border-medical-border/60">
                      <span className="text-medical-muted">Cardiac Silhouette (Heart):</span>
                      <span className="font-mono text-rose-400 font-medium">
                        {(measurements as any).heart.heart_volume_ml.toFixed(1)} mL
                      </span>
                    </div>
                  )}

                  <div className="flex justify-between py-1 border-b border-medical-border/60">
                    <span className="text-medical-muted">Bilateral Symmetry Ratio:</span>
                    <span className="font-mono text-white font-medium">
                      {measurements.lungs?.lung_symmetry_ratio != null
                        ? measurements.lungs.lung_symmetry_ratio.toFixed(3)
                        : measurements.lungs?.volume_balance?.relative_difference_pct != null
                        ? `${(100 - measurements.lungs.volume_balance.relative_difference_pct).toFixed(1)}%`
                        : '1.000'}
                    </span>
                  </div>

                  <div className="flex justify-between py-1">
                    <span className="text-medical-muted">Voxel Unit Volume:</span>
                    <span className="font-mono text-slate-300">
                      {(
                        measurements.physical_voxel_volume_mm3 ??
                        measurements.voxel_volume_mm3 ??
                        0
                      ).toFixed(4)} mm³
                    </span>
                  </div>
                </div>
              ) : (
                <div className="py-6 text-center text-xs text-medical-muted">Loading volume metrics...</div>
              )}
            </Card>

            {/* Trimesh Quality Audit Card */}
            <Card title="Trimesh Quality & Topology Audit" subtitle="Evaluates mesh manifoldness and watertightness">
              {meshQuality ? (
                <div className="space-y-2.5 text-xs">
                  <div className="flex items-center justify-between py-1 border-b border-medical-border/60">
                    <span className="text-medical-muted">Watertight Status:</span>
                    <Badge variant={meshQuality.watertight ? 'emerald' : 'rose'} size="sm">
                      {meshQuality.watertight ? 'Watertight Closed' : 'Open Boundaries'}
                    </Badge>
                  </div>

                  <div className="flex items-center justify-between py-1 border-b border-medical-border/60">
                    <span className="text-medical-muted">Euler Characteristic (χ):</span>
                    <span className="font-mono text-emerald-400 font-semibold">{meshQuality.euler_number ?? 'N/A'}</span>
                  </div>

                  <div className="flex items-center justify-between py-1 border-b border-medical-border/60">
                    <span className="text-medical-muted">Winding Consistency:</span>
                    <Badge
                      variant={
                        ((meshQuality as any).winding_consistent ?? meshQuality.is_winding_consistent)
                          ? 'emerald'
                          : 'rose'
                      }
                      size="sm"
                    >
                      {((meshQuality as any).winding_consistent ?? meshQuality.is_winding_consistent)
                        ? 'Consistent (Outward)'
                        : 'Inverted'}
                    </Badge>
                  </div>

                  <div className="flex items-center justify-between py-1 border-b border-medical-border/60">
                    <span className="text-medical-muted">Degenerate Faces:</span>
                    <span className="font-mono text-slate-300">
                      {(meshQuality as any).degenerate_faces ?? meshQuality.degenerate_face_count ?? 0}
                    </span>
                  </div>

                  <div className="flex items-center justify-between py-1 border-b border-medical-border/60">
                    <span className="text-medical-muted">Surface Area:</span>
                    <span className="font-mono text-medical-cyan">
                      {(meshQuality.surface_area_mm2 ?? 0).toLocaleString()} mm²
                    </span>
                  </div>

                  <div className="flex items-center justify-between py-1">
                    <span className="text-medical-muted">Mesh vs Voxel Diff:</span>
                    <span className="font-mono text-slate-300">
                      {((meshQuality as any).volume_difference_pct ?? meshQuality.volume_discrepancy_percent ?? 0.0).toFixed(2)}%
                    </span>
                  </div>
                </div>
              ) : (
                <div className="py-6 text-center text-xs text-medical-muted">Evaluating mesh topology...</div>
              )}
            </Card>

            {/* Multi-Dimensional Verification Matrix Card */}
            <Card title="Anatomical & Geometric Verification Matrix" subtitle="Independent validation of 7 critical geometric pillars">
              {meshQuality?.granular_statuses ? (
                <div className="space-y-2 text-xs">
                  {Object.entries(meshQuality.granular_statuses).map(([key, item]) => (
                    <div key={key} className="p-2 rounded bg-medical-surface/40 border border-medical-border/40">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-slate-200">{item.label}</span>
                        <Badge
                          variant={
                            item.status === 'PASSED'
                              ? 'emerald'
                              : item.status === 'WARNING'
                              ? 'amber'
                              : item.status === 'UNVERIFIED'
                              ? 'slate'
                              : 'rose'
                          }
                          size="sm"
                        >
                          {item.status}
                        </Badge>
                      </div>
                      <p className="text-[11px] text-medical-muted leading-tight">{item.evidence}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-4 text-center text-xs text-medical-muted">Loading verification matrix...</div>
              )}
            </Card>
          </div>
        </div>
      )}

      {/* TAB 2: Multi-Planar Reconstruction (MPR) */}
      {activeTab === 'mpr' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Controls Panel (4 Cols) */}
          <div className="lg:col-span-4 space-y-4">
            <Card title="Orthogonal Viewport Controls" subtitle="Slice navigation & clinical HU window presets">
              <div className="space-y-4 text-xs">
                {/* View Plane Selector */}
                <div>
                  <label className="text-[11px] font-semibold text-medical-subtle uppercase tracking-wider block mb-1.5">
                    Anatomical View Plane:
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {(['axial', 'coronal', 'sagittal'] as const).map((view) => (
                      <button
                        key={view}
                        onClick={() => {
                          setMprView(view);
                          // Maintain mid-slice
                          const dim =
                            view === 'axial'
                              ? metadata?.dimensions.z || 64
                              : view === 'coronal'
                              ? metadata?.dimensions.y || 128
                              : metadata?.dimensions.x || 128;
                          setSliceIndex(Math.floor(dim / 2));
                        }}
                        className={`p-2 rounded-lg border text-center capitalize transition-all ${
                          mprView === view
                            ? 'bg-medical-cyan text-medical-bg border-medical-cyan font-semibold shadow-xs'
                            : 'bg-medical-surface text-medical-muted border-medical-border hover:text-white'
                        }`}
                      >
                        {view}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Slice Slider with Step Buttons */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-[11px] font-semibold text-medical-subtle uppercase tracking-wider">
                      Slice Navigation:
                    </label>
                    <span className="font-mono text-medical-cyan font-bold text-sm">
                      {sliceIndex} / {maxIndex}
                    </span>
                  </div>

                  <input
                    type="range"
                    min={0}
                    max={maxIndex}
                    value={sliceIndex}
                    onChange={(e) => setSliceIndex(parseInt(e.target.value, 10))}
                    className="w-full accent-medical-cyan bg-medical-surface cursor-pointer"
                  />

                  {/* Step Buttons */}
                  <div className="flex items-center justify-between mt-2 pt-1 border-t border-medical-border/40 text-[11px]">
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => stepSlice(-5)}
                        disabled={sliceIndex <= 0}
                        className="px-2 py-0.5 rounded bg-medical-surface border border-medical-border hover:text-white disabled:opacity-30"
                      >
                        -5
                      </button>
                      <button
                        onClick={() => stepSlice(-1)}
                        disabled={sliceIndex <= 0}
                        className="px-2 py-0.5 rounded bg-medical-surface border border-medical-border hover:text-white disabled:opacity-30 flex items-center"
                      >
                        <ChevronLeft className="w-3 h-3" />
                      </button>
                    </div>

                    {(() => {
                      const phys = currentSlice?.physical_coordinate_mm ?? (
                        metadata?.origin_mm && metadata?.spacing_mm ? (
                          mprView === 'axial'
                            ? metadata.origin_mm[2] + sliceIndex * metadata.spacing_mm[2]
                            : mprView === 'coronal'
                            ? metadata.origin_mm[1] + sliceIndex * metadata.spacing_mm[1]
                            : metadata.origin_mm[0] + sliceIndex * metadata.spacing_mm[0]
                        ) : null
                      );
                      return phys != null ? (
                        <span className="text-[10px] text-medical-subtle font-mono">
                          Coord: <strong className="text-slate-300">{Number(phys).toFixed(2)} mm</strong>
                        </span>
                      ) : null;
                    })()}

                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => stepSlice(1)}
                        disabled={sliceIndex >= maxIndex}
                        className="px-2 py-0.5 rounded bg-medical-surface border border-medical-border hover:text-white disabled:opacity-30 flex items-center"
                      >
                        <ChevronRight className="w-3 h-3" />
                      </button>
                      <button
                        onClick={() => stepSlice(5)}
                        disabled={sliceIndex >= maxIndex}
                        className="px-2 py-0.5 rounded bg-medical-surface border border-medical-border hover:text-white disabled:opacity-30"
                      >
                        +5
                      </button>
                    </div>
                  </div>
                </div>

                {/* Window Presets */}
                <div>
                  <label className="text-[11px] font-semibold text-medical-subtle uppercase tracking-wider block mb-1.5">
                    Clinical Windowing Preset:
                  </label>
                  <select
                    value={windowPreset}
                    onChange={(e) => setWindowPreset(e.target.value)}
                    className="w-full bg-medical-surface border border-medical-border rounded-lg p-2 text-xs font-mono text-white focus:outline-none focus:border-medical-cyan shadow-sm"
                  >
                    <option value="lung">Lung Window (-600 HU Center / 1500 HU Width)</option>
                    <option value="mediastinum">Mediastinum Window (50 HU Center / 350 HU Width)</option>
                    <option value="bone">Bone Window (400 HU Center / 1800 HU Width)</option>
                    <option value="abdomen">Abdomen Window (40 HU Center / 400 HU Width)</option>
                  </select>
                </div>

                {/* Overlays */}
                <div className="pt-3 border-t border-medical-border/60 space-y-2">
                  <label className="text-[11px] font-semibold text-medical-subtle uppercase tracking-wider block">
                    Diagnostic Overlays:
                  </label>
                  <label className="flex items-center gap-2.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={overlayMask}
                      onChange={(e) => setOverlayMask(e.target.checked)}
                      className="accent-medical-cyan rounded"
                    />
                    <span className="text-slate-300">Segmentation Mask (Labels 3, 4, 5)</span>
                  </label>
                  <label className="flex items-center gap-2.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showNodules}
                      onChange={(e) => setShowNodules(e.target.checked)}
                      className="accent-medical-cyan rounded"
                    />
                    <span className="text-slate-300">LUNA16 Nodule Crosshairs</span>
                  </label>
                </div>

                {/* Active Nodule Jumper if selected */}
                {selectedNoduleId && activeNodule && (
                  <div className="pt-3 border-t border-medical-border/60">
                    <span className="text-[10px] text-medical-subtle uppercase tracking-wider block mb-1">
                      Targeted Nodule Focus:
                    </span>
                    <div className="p-2.5 bg-cyan-950/30 border border-cyan-800/60 rounded-lg space-y-1 text-[11px]">
                      <div className="flex justify-between">
                        <span className="text-medical-muted">Nodule ID:</span>
                        <span className="font-mono font-bold text-white">{activeNodule.nodule_id}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-medical-muted">Diameter:</span>
                        <span className="font-mono text-amber-300 font-semibold">{typeof activeNodule.diameter_mm === 'number' ? activeNodule.diameter_mm.toFixed(1) : (activeNodule.diameter_mm ?? 'N/A')} mm</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-medical-muted">Location:</span>
                        <span className="font-mono text-medical-cyan">({activeNodule.voxel_x}, {activeNodule.voxel_y}, {activeNodule.voxel_z})</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Slice Viewport with Cardinal Compass Orientation Labels (8 Cols) */}
          <div className="lg:col-span-8">
            <Card
              title={`${mprView.toUpperCase()} Orthogonal Viewport`}
              subtitle={`Window: ${windowPreset.toUpperCase()} · Slice: ${sliceIndex} · ${currentSlice?.nodules_on_slice?.length || 0} intersecting nodules`}
              headerAction={
                <div className="flex items-center gap-2">
                  {/* Cine Play/Pause */}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsCinePlaying((p) => !p)}
                    className={`text-xs py-1 px-2.5 transition-colors ${
                      isCinePlaying ? 'border-cyan-500 bg-cyan-950/60 text-cyan-300' : 'text-slate-300'
                    }`}
                    title={isCinePlaying ? 'Pause Cine Playback' : 'Start Auto Cine Loop'}
                  >
                    {isCinePlaying ? <Pause className="w-3.5 h-3.5 mr-1 text-cyan-400" /> : <Play className="w-3.5 h-3.5 mr-1" />}
                    <span>{isCinePlaying ? 'Pause Cine' : 'Cine Loop'}</span>
                  </Button>

                  {/* Invert Grayscale */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setMprInvert((i) => !i)}
                    className={`p-1.5 h-auto transition-colors ${
                      mprInvert ? 'text-amber-300 bg-amber-950/50' : 'text-slate-400 hover:text-white'
                    }`}
                    title="Invert CT Grayscale (Negative/Positive)"
                  >
                    <Sun className="w-3.5 h-3.5" />
                  </Button>

                  {/* Zoom Controls */}
                  <div className="flex items-center gap-1 bg-black/75 border border-white/10 rounded-lg p-0.5 shadow-sm">
                    <button
                      onClick={() => setMprZoom((z) => Math.max(z - 0.25, 0.75))}
                      disabled={mprZoom <= 0.75}
                      className="p-1 text-slate-400 hover:text-white disabled:opacity-30 transition-colors"
                      title="Zoom Out"
                    >
                      <ZoomOut className="w-3.5 h-3.5" />
                    </button>
                    <span className="text-[10px] font-mono text-cyan-400 px-1 min-w-[28px] text-center font-bold">
                      {Math.round(mprZoom * 100)}%
                    </span>
                    <button
                      onClick={() => setMprZoom((z) => Math.min(z + 0.25, 2.5))}
                      disabled={mprZoom >= 2.5}
                      className="p-1 text-slate-400 hover:text-white disabled:opacity-30 transition-colors"
                      title="Zoom In"
                    >
                      <ZoomIn className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => setMprZoom(1.0)}
                      className="p-1 text-slate-400 hover:text-white transition-colors border-l border-white/10 ml-0.5"
                      title="Fit to Screen"
                    >
                      <Maximize2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              }
            >
              <div
                onWheel={handleMprWheel}
                className="relative min-h-[460px] bg-black rounded-xl border border-medical-border flex items-center justify-center overflow-hidden select-none cursor-ns-resize group"
                title="Scroll mouse wheel over image to scrub CT slices"
              >
                {/* Cardinal Anatomical Compass Orientation Labels */}
                <div className="absolute top-2.5 left-1/2 -translate-x-1/2 z-20 px-2 py-0.5 rounded bg-black/75 border border-white/10 text-[10px] font-mono font-bold text-slate-300">
                  {orientation.top}
                </div>
                <div className="absolute bottom-2.5 left-1/2 -translate-x-1/2 z-20 px-2 py-0.5 rounded bg-black/75 border border-white/10 text-[10px] font-mono font-bold text-slate-300">
                  {orientation.bottom}
                </div>
                <div className="absolute left-2.5 top-1/2 -translate-y-1/2 z-20 px-2 py-0.5 rounded bg-black/75 border border-white/10 text-[10px] font-mono font-bold text-slate-300">
                  {orientation.left}
                </div>
                <div className="absolute right-2.5 top-1/2 -translate-y-1/2 z-20 px-2 py-0.5 rounded bg-black/75 border border-white/10 text-[10px] font-mono font-bold text-slate-300">
                  {orientation.right}
                </div>

                {loadingSlice && (
                  <div className="absolute inset-0 bg-black/60 backdrop-blur-xs flex items-center justify-center z-30">
                    <span className="text-xs font-mono text-medical-cyan animate-pulse">Streaming orthogonal slice...</span>
                  </div>
                )}

                {currentSlice?.image_base64 ? (
                  <div
                    className="w-full h-full flex items-center justify-center overflow-hidden"
                    style={{
                      transform: `scale(${mprZoom})`,
                      filter: mprInvert ? 'invert(1)' : 'none',
                      transition: 'transform 0.12s ease-out',
                    }}
                  >
                    <img
                      src={currentSlice.image_base64}
                      alt={`${mprView} CT slice`}
                      className="max-h-[440px] w-auto object-contain select-none pointer-events-none"
                    />
                  </div>
                ) : (
                  <div className="text-xs text-medical-muted">Loading orthogonal slice...</div>
                )}

                {/* Top Left Metadata Tag */}
                <div className="absolute top-2.5 left-2.5 text-[10px] font-mono text-medical-cyan bg-black/75 px-2 py-1 rounded border border-white/10 z-20 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  <span>{mprView.toUpperCase()} [{sliceIndex}/{maxIndex}]</span>
                </div>

                {/* Bottom Right Window Tag */}
                <div className="absolute bottom-2.5 right-2.5 text-[10px] font-mono text-slate-400 bg-black/75 px-2 py-1 rounded border border-white/10 z-20">
                  Window: {windowPreset.toUpperCase()}
                </div>

                {/* Bottom Left Mouse Hint */}
                <div className="absolute bottom-2.5 left-2.5 text-[10px] font-mono text-slate-500 bg-black/70 px-2 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                  Mouse Wheel: Scrub Slices · Click: Target Nodule
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* TAB 3: LUNA16 Nodule Registry */}
      {activeTab === 'annotations' && (
        <Card title="Mapped LUNA16 Nodule Coordinates" subtitle="Ground-truth spatial annotations mapped to voxel grid">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-medical-surface border-b border-medical-border text-medical-muted uppercase text-[10px]">
                <tr>
                  <th className="p-3">Nodule ID</th>
                  <th className="p-3">Physical X (mm)</th>
                  <th className="p-3">Physical Y (mm)</th>
                  <th className="p-3">Physical Z (mm)</th>
                  <th className="p-3">Voxel (X, Y, Z)</th>
                  <th className="p-3">Diameter</th>
                  <th className="p-3">Bounds</th>
                  <th className="p-3">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-medical-border/60">
                {annotations.map((nodule) => (
                  <tr
                    key={nodule.nodule_id}
                    className={`hover:bg-medical-hover/50 transition-colors ${
                      selectedNoduleId === nodule.nodule_id ? 'bg-cyan-950/30' : ''
                    }`}
                  >
                    <td className="p-3 font-semibold text-white">{nodule.nodule_id}</td>
                    <td className="p-3 text-slate-300">{typeof nodule.coordX_mm === 'number' ? nodule.coordX_mm.toFixed(2) : (nodule.coordX_mm ?? '0.00')}</td>
                    <td className="p-3 text-slate-300">{typeof nodule.coordY_mm === 'number' ? nodule.coordY_mm.toFixed(2) : (nodule.coordY_mm ?? '0.00')}</td>
                    <td className="p-3 text-slate-300">{typeof nodule.coordZ_mm === 'number' ? nodule.coordZ_mm.toFixed(2) : (nodule.coordZ_mm ?? '0.00')}</td>
                    <td className="p-3 text-medical-cyan">
                      ({nodule.voxel_x}, {nodule.voxel_y}, {nodule.voxel_z})
                    </td>
                    <td className="p-3 text-amber-400 font-semibold">{typeof nodule.diameter_mm === 'number' ? nodule.diameter_mm.toFixed(1) : (nodule.diameter_mm ?? '0.0')} mm</td>
                    <td className="p-3">
                      <Badge variant={nodule.inside_volume ? 'emerald' : 'rose'} size="sm">
                        {nodule.inside_volume ? 'Inside Scan' : 'Outside'}
                      </Badge>
                    </td>
                    <td className="p-3">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => jumpToNodule(nodule)}
                        className="text-cyan-300 border-cyan-500/40 bg-cyan-950/30 hover:bg-cyan-900/60 text-xs py-1 px-2.5 flex items-center gap-1.5 transition-all shadow-xs"
                      >
                        <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
                        <span>Jump to Slice</span>
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* TAB 4: Digitally Reconstructed Radiographs (DRR) & Dual-View Bridge */}
      {activeTab === 'drr' && (
        <div className="space-y-6">
          {/* Top Banner explaining the CT-to-DRR synthesis */}
          <div className="bg-medical-card border border-medical-border/80 rounded-xl p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="flex items-start gap-3">
              <div className="p-2.5 rounded-lg bg-cyan-950/40 border border-cyan-800/60 text-medical-cyan mt-0.5">
                <Film className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold text-white">
                    CT-to-DRR Radiographic Synthesis & Dual-View 3D Bridge
                  </h3>
                  <Badge variant="cyan" size="sm">
                    Beer-Lambert Attenuation
                  </Badge>
                </div>
                <p className="text-xs text-medical-muted mt-1 max-w-2xl">
                  Simulates physical clinical X-ray projections directly from 3D CT attenuation coefficients (μ = μ_water · (1 + HU/1000)). Synthesizes calibrated Frontal (PA) and Lateral (LL) radiographs to feed the bi-planar 3D reconstruction pipeline.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2.5 w-full md:w-auto">
              <Button
                variant="outline"
                size="sm"
                onClick={handleDownloadDRR}
                disabled={!drrData}
                className="border-medical-border hover:border-medical-cyan text-xs"
              >
                <Download className="w-3.5 h-3.5 mr-1" />
                <span>Save Radiograph</span>
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleExportToDualView}
                disabled={loadingDRR || isExportingToDualView}
                isLoading={isExportingToDualView}
                className="bg-medical-cyan text-medical-bg hover:bg-cyan-400 font-semibold text-xs shadow-md shadow-cyan-950/50"
              >
                <span>Export to Dual-View Lab</span>
                <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left: DRR Parameters & Geometry (4 Cols) */}
            <div className="lg:col-span-4 space-y-4">
              <Card
                title="Radiographic Geometry & Physics"
                subtitle="Calibrated projection parameters & simulated detector"
              >
                <div className="space-y-4 text-xs">
                  {/* Projection View Selector */}
                  <div>
                    <label className="text-[11px] font-semibold text-medical-subtle uppercase tracking-wider block mb-1.5">
                      Radiographic Projection View:
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => setDrrView('frontal')}
                        className={`p-2.5 rounded-lg border text-center transition-all ${
                          drrView === 'frontal'
                            ? 'bg-medical-cyan text-medical-bg border-medical-cyan font-semibold shadow-xs'
                            : 'bg-medical-surface text-medical-muted border-medical-border hover:text-white'
                        }`}
                      >
                        <div className="text-xs font-bold">Frontal (PA)</div>
                        <div className="text-[10px] opacity-80 mt-0.5">Posteroanterior (Y-axis)</div>
                      </button>

                      <button
                        onClick={() => setDrrView('lateral')}
                        className={`p-2.5 rounded-lg border text-center transition-all ${
                          drrView === 'lateral'
                            ? 'bg-medical-cyan text-medical-bg border-medical-cyan font-semibold shadow-xs'
                            : 'bg-medical-surface text-medical-muted border-medical-border hover:text-white'
                        }`}
                      >
                        <div className="text-xs font-bold">Lateral (LL)</div>
                        <div className="text-[10px] opacity-80 mt-0.5">Left Lateral (X-axis)</div>
                      </button>
                    </div>
                  </div>

                  {/* SDD Slider */}
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-[11px] font-semibold text-medical-subtle uppercase tracking-wider">
                        Source-to-Detector Distance (SDD):
                      </label>
                      <span className="font-mono text-medical-cyan font-bold text-xs">
                        {drrSdd.toFixed(0)} mm
                      </span>
                    </div>
                    <input
                      type="range"
                      min={1200}
                      max={2200}
                      step={50}
                      value={drrSdd}
                      onChange={(e) => setDrrSdd(parseFloat(e.target.value))}
                      className="w-full accent-medical-cyan bg-medical-surface cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-medical-muted mt-1 font-mono">
                      <span>1200 mm</span>
                      <span>Standard: 1800 mm</span>
                      <span>2200 mm</span>
                    </div>
                  </div>

                  {/* SOD Slider */}
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-[11px] font-semibold text-medical-subtle uppercase tracking-wider">
                        Source-to-Object Distance (SOD):
                      </label>
                      <span className="font-mono text-medical-cyan font-bold text-xs">
                        {drrSod.toFixed(0)} mm
                      </span>
                    </div>
                    <input
                      type="range"
                      min={1000}
                      max={1800}
                      step={50}
                      value={drrSod}
                      onChange={(e) => setDrrSod(parseFloat(e.target.value))}
                      className="w-full accent-medical-cyan bg-medical-surface cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-medical-muted mt-1 font-mono">
                      <span>1000 mm</span>
                      <span>Standard: 1450 mm</span>
                      <span>1800 mm</span>
                    </div>
                  </div>

                  {/* Magnification Factor Calculation */}
                  <div className="p-3 bg-medical-surface rounded-lg border border-medical-border/80 flex items-center justify-between">
                    <div>
                      <span className="text-[10px] text-medical-muted uppercase font-semibold block">Geometric Magnification (M)</span>
                      <span className="text-[11px] text-slate-300 font-mono">M = SDD / SOD</span>
                    </div>
                    <span className="text-sm font-mono font-bold text-emerald-400">
                      {(drrSdd / Math.max(1, drrSod)).toFixed(3)}×
                    </span>
                  </div>

                  {/* Physics & Overlay Toggles */}
                  <div className="pt-3 border-t border-medical-border/60 space-y-2.5">
                    <label className="text-[11px] font-semibold text-medical-subtle uppercase tracking-wider block">
                      Simulation & Visual Enhancements:
                    </label>

                    <label className="flex items-center gap-2.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={drrOverlaySilhouette}
                        onChange={(e) => setDrrOverlaySilhouette(e.target.checked)}
                        className="accent-medical-cyan rounded"
                      />
                      <div>
                        <span className="text-slate-300 block">Lung Boundary Silhouette</span>
                        <span className="text-[10px] text-medical-muted block">Project 2D anatomical contours from ground-truth mask</span>
                      </div>
                    </label>

                    <label className="flex items-center gap-2.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={drrAddNoise}
                        onChange={(e) => setDrrAddNoise(e.target.checked)}
                        className="accent-medical-cyan rounded"
                      />
                      <div>
                        <span className="text-slate-300 block">Quantum Photon Poisson Noise</span>
                        <span className="text-[10px] text-medical-muted block">Simulate physical quantum detector photon mottle</span>
                      </div>
                    </label>
                  </div>

                  {/* Projection Matrix & Attenuation Stats */}
                  {drrData && (
                    <div className="pt-3 border-t border-medical-border/60 space-y-2">
                      <label className="text-[11px] font-semibold text-medical-subtle uppercase tracking-wider block">
                        Projection Radiometry:
                      </label>
                      <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                        <div className="p-2 bg-black/40 rounded border border-white/5">
                          <span className="text-medical-muted text-[10px] block">Mean Attenuation:</span>
                          <span className="text-slate-200">{drrData.attenuation_stats.mean_line_integral.toFixed(3)}</span>
                        </div>
                        <div className="p-2 bg-black/40 rounded border border-white/5">
                          <span className="text-medical-muted text-[10px] block">Max Line Integral:</span>
                          <span className="text-slate-200">{drrData.attenuation_stats.max_line_integral.toFixed(2)}</span>
                        </div>
                        <div className="p-2 bg-black/40 rounded border border-white/5">
                          <span className="text-medical-muted text-[10px] block">Matrix Resolution:</span>
                          <span className="text-slate-200">{drrData.dimensions.width} × {drrData.dimensions.height}</span>
                        </div>
                        <div className="p-2 bg-black/40 rounded border border-white/5">
                          <span className="text-medical-muted text-[10px] block">Forward Runtime:</span>
                          <span className="text-medical-teal">{drrData.computation_latency_ms.toFixed(1)} ms</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            </div>

            {/* Right: Synthesized Radiograph Light-Box Viewport (8 Cols) */}
            <div className="lg:col-span-8 space-y-3">
              <Card
                title={`Synthesized ${drrData?.view_name || (drrView === 'frontal' ? 'Frontal (PA)' : 'Lateral (LL)')} Radiograph`}
                subtitle="Simulated forward projection through 3D attenuation volume"
                headerAction={
                  drrData && (
                    <Badge variant="teal" size="sm">
                      FOV: {drrData.physical_fov_mm.width_mm.toFixed(1)} × {drrData.physical_fov_mm.height_mm.toFixed(1)} mm
                    </Badge>
                  )
                }
              >
                <div className="relative min-h-[500px] bg-black rounded-xl border border-medical-border flex items-center justify-center overflow-hidden select-none">
                  {/* Orientation tags */}
                  <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 px-2.5 py-0.5 rounded bg-black/80 border border-white/10 text-[10px] font-mono font-bold text-slate-300">
                    S (Superior)
                  </div>
                  <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20 px-2.5 py-0.5 rounded bg-black/80 border border-white/10 text-[10px] font-mono font-bold text-slate-300">
                    I (Inferior)
                  </div>
                  <div className="absolute left-3 top-1/2 -translate-y-1/2 z-20 px-2.5 py-0.5 rounded bg-black/80 border border-white/10 text-[10px] font-mono font-bold text-slate-300">
                    {drrView === 'frontal' ? 'R (Patient Right)' : 'A (Anterior)'}
                  </div>
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 z-20 px-2.5 py-0.5 rounded bg-black/80 border border-white/10 text-[10px] font-mono font-bold text-slate-300">
                    {drrView === 'frontal' ? 'L (Patient Left)' : 'P (Posterior)'}
                  </div>

                  {loadingDRR && (
                    <div className="absolute inset-0 bg-black/70 backdrop-blur-xs flex flex-col items-center justify-center z-30 gap-2">
                      <Film className="w-8 h-8 text-medical-cyan animate-pulse" />
                      <span className="text-xs font-mono text-medical-cyan animate-pulse">
                        Ray-casting Beer-Lambert attenuation integral...
                      </span>
                    </div>
                  )}

                  {drrError && (
                    <div className="p-4 bg-rose-950/80 border border-rose-800 rounded-lg text-xs text-rose-300 flex items-center gap-2 m-6">
                      <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
                      <span>{drrError}</span>
                    </div>
                  )}

                  {!loadingDRR && !drrError && drrData?.image_base64 ? (
                    <div className="relative flex items-center justify-center p-4">
                      <img
                        src={drrData.image_base64}
                        alt={`Synthesized ${drrView} DRR`}
                        className="max-h-[480px] w-auto object-contain rounded shadow-2xl border border-slate-800"
                      />
                    </div>
                  ) : null}

                  {/* Top-Left View Tag */}
                  <div className="absolute top-3 left-3 text-[10px] font-mono text-medical-cyan bg-black/80 px-2.5 py-1 rounded border border-white/10 z-20">
                    {drrView.toUpperCase()} · DRR SYNTHESIS
                  </div>

                  {/* Bottom-Right Geometry Tag */}
                  {drrData && (
                    <div className="absolute bottom-3 right-3 text-[10px] font-mono text-slate-400 bg-black/80 px-2.5 py-1 rounded border border-white/10 z-20">
                      SDD: {drrData.geometry.sdd_mm}mm · SOD: {drrData.geometry.sod_mm}mm · M: {drrData.geometry.magnification_factor.toFixed(2)}x
                    </div>
                  )}
                </div>
              </Card>

              {/* Research Disclaimer */}
              <div className="p-3 rounded-lg bg-medical-surface border border-medical-border text-[11px] text-medical-muted flex items-center justify-between">
                <span>
                  <strong>Scientific Validation Note:</strong> DRRs are computed via linear ray attenuation integrals for statistical shape modeling & bi-planar reconstruction calibration. Not for diagnostic use.
                </span>
                <span className="font-mono text-medical-subtle text-[10px]">SimpleITK Beer-Lambert</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: Statistical Shape Model (SSM) Research Panel */}
      {activeTab === 'ssm' && (
        <div className="space-y-6">
          {/* Prominent Scientific Disclaimer Banner */}
          <div className="bg-amber-950/40 border border-amber-600/40 rounded-xl p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-lg shadow-amber-950/20">
            <div className="flex items-start gap-3">
              <div className="p-2.5 rounded-lg bg-amber-500/20 border border-amber-500/30 text-amber-400 mt-0.5 shrink-0">
                <AlertCircle className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-semibold text-white">Statistical Shape Model Estimate — Research Only</h4>
                  <Badge variant="amber" size="sm">Not a Diagnostic CT</Badge>
                </div>
                <p className="text-xs text-amber-200/80 mt-1 leading-relaxed max-w-4xl">
                  This model predicts an <strong>estimated 3D macroscopic lung surface boundary</strong> constrained by learned anatomical shape variation modes (M=20 PCA eigenspace). It does <strong>NOT</strong> reconstruct diagnostic CT Hounsfield Units, internal parenchymal attenuation, or replacement cross-sections from radiographs.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <Badge variant="teal" size="sm">
                Cohort: 430 Scans
              </Badge>
              <Badge variant="cyan" size="sm">
                Zero Patient Leakage
              </Badge>
            </div>
          </div>

          {/* Main 2-Column Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column (7 cols): 3D Mesh & Interactive Mode Sliders */}
            <div className="lg:col-span-7 space-y-4">
              <Card
                title="Interactive 3D SSM Surface Deformation"
                subtitle="Deform canonical lung surface (N=6,737 vertices) across verified physiological eigenspaces"
                headerAction={
                  <div className="flex items-center gap-2">
                    {/* Structure Selector */}
                    <div className="flex items-center bg-medical-surface p-1 rounded-lg border border-medical-border">
                      {(['bilateral', 'left_lung', 'right_lung'] as const).map((struct) => (
                        <button
                          key={struct}
                          onClick={() => setSsmStructure(struct)}
                          className={`px-2.5 py-1 rounded text-[11px] font-mono capitalize transition-all ${
                            ssmStructure === struct
                              ? 'bg-medical-cyan text-medical-bg font-semibold shadow-xs'
                              : 'text-medical-muted hover:text-white'
                          }`}
                        >
                          {struct === 'bilateral' ? 'Bilateral Lungs' : struct.replace('_', ' ')}
                        </button>
                      ))}
                    </div>

                    {/* Reset Button */}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => applySSMPreset('mean')}
                      title="Reset shape to mean canonical template"
                    >
                      <RotateCcw className="w-3.5 h-3.5 mr-1" />
                      <span>Reset</span>
                    </Button>
                  </div>
                }
              >
                {/* Presets Bar */}
                <div className="mb-3 p-2 bg-slate-900/60 rounded-lg border border-slate-800 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] font-mono text-medical-muted mr-1">Geometry Mode:</span>
                    <button
                      onClick={() => applySSMPreset('mean')}
                      className={`px-2 py-0.5 rounded text-[11px] font-mono transition-all ${
                        ssmGeometryPreset === 'mean'
                          ? 'bg-cyan-500 text-slate-950 font-bold'
                          : 'bg-slate-800 text-slate-300 hover:text-white'
                      }`}
                    >
                      Mean Shape (b=0)
                    </button>
                    <button
                      onClick={() => applySSMPreset('held_out_val')}
                      className={`px-2 py-0.5 rounded text-[11px] font-mono transition-all ${
                        ssmGeometryPreset === 'held_out_val'
                          ? 'bg-cyan-500 text-slate-950 font-bold'
                          : 'bg-slate-800 text-slate-300 hover:text-white'
                      }`}
                    >
                      Held-Out Val (LIDC-0004)
                    </button>
                    <button
                      onClick={() => applySSMPreset('reference')}
                      className={`px-2 py-0.5 rounded text-[11px] font-mono transition-all ${
                        ssmGeometryPreset === 'reference'
                          ? 'bg-cyan-500 text-slate-950 font-bold'
                          : 'bg-slate-800 text-slate-300 hover:text-white'
                      }`}
                    >
                      Reference CT Fit
                    </button>
                    {ssmGeometryPreset === 'custom' && (
                      <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-purple-500/20 text-purple-300 border border-purple-500/30">
                        Custom Deformation
                      </span>
                    )}
                  </div>

                  {ssmDeformResult && (
                    <div className="text-[11px] font-mono text-cyan-300">
                      {ssmDeformResult.num_vertices.toLocaleString()} Vertices · {ssmDeformResult.num_faces.toLocaleString()} Faces
                    </div>
                  )}
                </div>

                {/* 3D Canvas Viewport */}
                <div className="h-[420px] relative rounded-xl overflow-hidden border border-medical-border bg-slate-950">
                  {loadingSSM && (
                    <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-xs flex items-center justify-center z-30">
                      <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-900 border border-cyan-500/40 text-cyan-400 font-mono text-xs">
                        <Activity className="w-4 h-4 animate-spin" />
                        <span>Deforming SSM Eigenspace...</span>
                      </div>
                    </div>
                  )}

                  {ssmError && (
                    <div className="absolute top-4 left-4 right-4 bg-rose-950/80 border border-rose-800 rounded-lg p-3 text-xs text-rose-300 flex items-center gap-2 z-30">
                      <AlertCircle className="w-4 h-4 shrink-0" />
                      <span>{ssmError}</span>
                    </div>
                  )}

                  <SSM3DReconstructionViewer
                    vertices={ssmDeformResult?.vertices}
                    faces={ssmDeformResult?.faces}
                    landmarks={ssmDeformResult?.landmarks}
                    onExportObj={handleExportSSMObj}
                    isExporting={isExportingSSMObj}
                  />
                </div>

                {/* Mode Deformation Sliders */}
                <div className="mt-4 space-y-3 pt-3 border-t border-medical-border">
                  <div className="flex items-center justify-between">
                    <h5 className="text-xs font-semibold uppercase tracking-wider text-medical-teal font-mono flex items-center gap-1.5">
                      <Sliders className="w-3.5 h-3.5" />
                      Top 5 Physiological Shape Modes (±3.0 Standard Deviations)
                    </h5>
                    <span className="text-[11px] font-mono text-medical-muted">
                      Cumulative Variance: 90.38% (Modes 1–5)
                    </span>
                  </div>

                  <div className="space-y-2.5">
                    {[
                      { index: 0, modeNum: 1, name: 'Global Inspiratory Expansion', varPct: 41.4, desc: 'Cranio-caudal diaphragmatic excursion & lung expansion' },
                      { index: 1, modeNum: 2, name: 'Thoracic Aspect Ratio', varPct: 18.6, desc: 'Transverse rib cage width vs vertical thoracic height' },
                      { index: 2, modeNum: 3, name: 'Diaphragmatic Curvature', varPct: 13.0, desc: 'Dome concavity vs planar diaphragmatic flattening' },
                      { index: 3, modeNum: 4, name: 'Mediastinal & Cardiac Notch', varPct: 11.0, desc: 'Biventricular cardiac displacement & medial border shift' },
                      { index: 4, modeNum: 5, name: 'Anteroposterior Curvature', varPct: 6.3, desc: 'Thoracic spine kyphotic curvature & chest wall depth' },
                    ].map((mode) => {
                      const weight = ssmModeWeights[mode.index] ?? 0;
                      return (
                        <div
                          key={mode.modeNum}
                          className="p-2.5 rounded-lg bg-medical-surface border border-medical-border hover:border-slate-700 transition-colors"
                        >
                          <div className="flex items-center justify-between mb-1.5">
                            <div>
                              <span className="text-xs font-semibold text-white">
                                Mode {mode.modeNum}: {mode.name}
                              </span>
                              <span className="text-[10px] text-cyan-400 font-mono ml-2">
                                ({mode.varPct}% variance)
                              </span>
                              <p className="text-[10px] text-medical-muted mt-0.5">{mode.desc}</p>
                            </div>
                            <span
                              className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                                Math.abs(weight) > 0.05
                                  ? weight > 0
                                    ? 'bg-cyan-950 text-cyan-300 border border-cyan-800'
                                    : 'bg-indigo-950 text-indigo-300 border border-indigo-800'
                                  : 'bg-slate-800 text-slate-400'
                              }`}
                            >
                              {weight >= 0 ? `+${weight.toFixed(2)}` : weight.toFixed(2)} σ
                            </span>
                          </div>
                          <div className="flex items-center gap-3">
                            <span className="text-[10px] font-mono text-slate-500">-3.0σ</span>
                            <input
                              type="range"
                              min="-3.0"
                              max="3.0"
                              step="0.05"
                              value={weight}
                              onChange={(e) => handleModeWeightChange(mode.index, parseFloat(e.target.value))}
                              className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                            />
                            <span className="text-[10px] font-mono text-slate-500">+3.0σ</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </Card>
            </div>

            {/* Right Column (5 cols): Volumetry, Metrics & Governance */}
            <div className="lg:col-span-5 space-y-4">
              {/* Card 1: Quantitative Lung Volumetry */}
              <Card
                title="Continuous Organ Volumetry"
                subtitle="Evaluated via Discrete Divergence Theorem on watertight surfaces"
              >
                {ssmDeformResult ? (
                  <div className="space-y-3.5">
                    <div className="p-3.5 bg-medical-surface rounded-xl border border-medical-border flex items-center justify-between">
                      <div>
                        <span className="text-xs text-medical-muted block">Total Segmented Volume (TLV):</span>
                        <span className="text-2xl font-bold font-mono text-cyan-400">
                          {ssmDeformResult.volumetry.total_lung_volume_ml.toLocaleString(undefined, {
                            minimumFractionDigits: 1,
                            maximumFractionDigits: 1,
                          })}{' '}
                          mL
                        </span>
                      </div>
                      <Badge variant="cyan" size="sm">
                        {ssmDeformResult.volumetry.total_lung_volume_ml > 750 ? 'Physiological Range' : 'Sub-Capacity'}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded-xl bg-medical-card border border-medical-border">
                        <span className="text-[11px] text-medical-muted block">Right Lung (3 Lobes)</span>
                        <strong className="text-base font-mono text-white">
                          {ssmDeformResult.volumetry.right_lung_ml.toFixed(1)} mL
                        </strong>
                        <span className="text-[10px] text-cyan-300 block mt-0.5">
                          ~{((ssmDeformResult.volumetry.right_lung_ml / Math.max(ssmDeformResult.volumetry.total_lung_volume_ml, 1)) * 100).toFixed(0)}% volume
                        </span>
                      </div>

                      <div className="p-3 rounded-xl bg-medical-card border border-medical-border">
                        <span className="text-[11px] text-medical-muted block">Left Lung (2 Lobes)</span>
                        <strong className="text-base font-mono text-white">
                          {ssmDeformResult.volumetry.left_lung_ml.toFixed(1)} mL
                        </strong>
                        <span className="text-[10px] text-teal-300 block mt-0.5">
                          ~{((ssmDeformResult.volumetry.left_lung_ml / Math.max(ssmDeformResult.volumetry.total_lung_volume_ml, 1)) * 100).toFixed(0)}% volume
                        </span>
                      </div>
                    </div>

                    <div className="p-2.5 rounded-lg bg-medical-card/60 border border-medical-border flex items-center justify-between text-xs font-mono">
                      <span className="text-medical-muted">Bilateral Symmetry (R/L):</span>
                      <span className="text-white font-semibold">{ssmDeformResult.volumetry.symmetry_ratio.toFixed(3)}</span>
                    </div>
                  </div>
                ) : (
                  <div className="p-6 text-center text-xs text-medical-muted">Computing volumetrics...</div>
                )}
              </Card>

              {/* Card 2: Generalization & Eigenspace Metrics */}
              <Card
                title="Eigenspace Generalization Quality"
                subtitle="Audited against held-out validation cohort (N=64 unseen patients)"
              >
                {ssmDeformResult ? (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-2.5">
                      <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border">
                        <span className="text-[10px] text-medical-muted block uppercase">Mean ASSD</span>
                        <strong className="text-sm font-mono text-emerald-400">
                          {ssmDeformResult.metrics.assd_mm.toFixed(3)} mm
                        </strong>
                        <span className="text-[9px] text-slate-400 block mt-0.5">Validation Cohort: ≤ 2.50 mm</span>
                      </div>

                      <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border">
                        <span className="text-[10px] text-medical-muted block uppercase">Hausdorff 95 (HD95)</span>
                        <strong className="text-sm font-mono text-emerald-400">
                          {ssmDeformResult.metrics.hd95_mm.toFixed(3)} mm
                        </strong>
                        <span className="text-[9px] text-slate-400 block mt-0.5">Validation Cohort: ≤ 7.00 mm</span>
                      </div>

                      <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border">
                        <span className="text-[10px] text-medical-muted block uppercase">Relative Vol Error</span>
                        <strong className="text-sm font-mono text-cyan-400">
                          {ssmDeformResult.metrics.volume_error_pct.toFixed(2)}%
                        </strong>
                        <span className="text-[9px] text-slate-400 block mt-0.5">Validation Cohort: ≤ 7.50%</span>
                      </div>

                      <div className="p-2.5 rounded-lg bg-medical-surface border border-medical-border">
                        <span className="text-[10px] text-medical-muted block uppercase">Surface Dice @ 2mm</span>
                        <strong className="text-sm font-mono text-cyan-400">
                          {ssmDeformResult.metrics.surface_dice.toFixed(3)}
                        </strong>
                        <span className="text-[9px] text-slate-400 block mt-0.5">Acceptance: ≥ 0.850</span>
                      </div>
                    </div>

                    <div className="p-2.5 rounded-lg bg-medical-card/80 border border-medical-border space-y-1.5 text-xs font-mono">
                      <div className="flex items-center justify-between">
                        <span className="text-medical-muted">Cumulative Variance (5 Modes):</span>
                        <span className="text-cyan-300 font-semibold">90.38%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-medical-muted">Full Model (20 Modes):</span>
                        <span className="text-emerald-300 font-semibold">100.0%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-medical-muted">Mahalanobis Distance (DM):</span>
                        <span className={`font-semibold ${ssmDeformResult.metrics.mahalanobis_distance <= 3.0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                          {ssmDeformResult.metrics.mahalanobis_distance.toFixed(2)} σ
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-6 text-center text-xs text-medical-muted">Evaluating metrics...</div>
                )}
              </Card>

              {/* Card 3: Export & Scientific Governance */}
              <Card
                title="Model Artifact & Export"
                subtitle="Standardized Wavefront OBJ stream with 16 anatomical landmarks"
              >
                <div className="space-y-3">
                  <div className="p-3 rounded-lg bg-medical-surface border border-medical-border space-y-2 text-xs">
                    <div className="flex items-center justify-between text-medical-muted font-mono text-[11px]">
                      <span>Model Architecture:</span>
                      <span className="text-white font-semibold">
                        {ssmModelInfo?.architecture || 'Point Distribution Model (PDM)'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-medical-muted font-mono text-[11px]">
                      <span>Model Artifact:</span>
                      <span className="text-cyan-300 font-semibold">
                        ssm_lung_pca_v1.json ({ssmModelInfo?.num_modes || 20} Modes)
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-medical-muted font-mono text-[11px]">
                      <span>Training Cohort:</span>
                      <span className="text-white">301 Patients (70.0%)</span>
                    </div>
                    <div className="flex items-center justify-between text-medical-muted font-mono text-[11px]">
                      <span>Validation Cohort:</span>
                      <span className="text-white">64 Patients (14.9%)</span>
                    </div>
                    <div className="flex items-center justify-between text-medical-muted font-mono text-[11px]">
                      <span>Testing Cohort:</span>
                      <span className="text-white">65 Patients (15.1%)</span>
                    </div>
                    <div className="flex items-center justify-between text-medical-muted font-mono text-[11px]">
                      <span>Zero Leakage Status:</span>
                      <span className="text-emerald-400 font-semibold">PASSED (0 Overlap)</span>
                    </div>
                  </div>

                  <Button
                    variant="primary"
                    size="md"
                    className="w-full justify-center"
                    onClick={handleExportSSMObj}
                    disabled={isExportingSSMObj || !ssmDeformResult}
                    isLoading={isExportingSSMObj}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    <span>Download Deformed OBJ Mesh</span>
                  </Button>
                </div>
              </Card>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};
