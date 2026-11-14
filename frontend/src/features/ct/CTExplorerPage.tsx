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
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { ThreeMeshCanvas } from './ThreeMeshCanvas';
import { ctApi } from '@/services/api/ctApi';
import {
  CTAnnotationItem,
  CTMeasurementResponse,
  CTMeshQualityResponse,
  CTMetadataResponse,
  CTSliceResponse,
  DRRProjectionResponse,
} from '@/types/ct';

export const CTExplorerPage: React.FC = () => {
  const navigate = useNavigate();
  const [activeScanId, setActiveScanId] = useState<string>('demo_thoracic_scan');
  const [activeTab, setActiveTab] = useState<'3d' | 'mpr' | 'annotations' | 'drr'>('3d');
  
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

  // 2. Load 2D Slice on Navigation or Filter Change
  useEffect(() => {
    if (activeTab !== 'mpr') return;

    let isCancelled = false;
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

    return () => {
      isCancelled = true;
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
      <div className="flex items-center gap-2 border-b border-medical-border pb-2">
        <button
          onClick={() => setActiveTab('3d')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
            activeTab === '3d'
              ? 'bg-medical-cyan text-medical-bg font-semibold shadow-sm'
              : 'text-medical-muted hover:text-white hover:bg-medical-surface'
          }`}
        >
          <Box className="w-4 h-4" />
          <span>3D Mesh & Volumetry</span>
        </button>

        <button
          onClick={() => setActiveTab('mpr')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'mpr'
              ? 'bg-medical-cyan text-medical-bg font-semibold shadow-sm'
              : 'text-medical-muted hover:text-white hover:bg-medical-surface'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Multi-Planar Reconstruction (MPR)</span>
        </button>

        <button
          onClick={() => setActiveTab('annotations')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'annotations'
              ? 'bg-medical-cyan text-medical-bg font-semibold shadow-sm'
              : 'text-medical-muted hover:text-white hover:bg-medical-surface'
          }`}
        >
          <Crosshair className="w-4 h-4" />
          <span>LUNA16 Nodule Registry ({annotations.length})</span>
        </button>

        <button
          onClick={() => setActiveTab('drr')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'drr'
              ? 'bg-medical-cyan text-medical-bg font-semibold shadow-sm'
              : 'text-medical-muted hover:text-white hover:bg-medical-surface'
          }`}
        >
          <Film className="w-4 h-4" />
          <span>DRR Synthesis & 3D Bridge</span>
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
                  {['lungs', 'left_lung', 'right_lung', 'airways'].map((struct) => (
                    <button
                      key={struct}
                      onClick={() => setSelectedStructure(struct)}
                      className={`px-2.5 py-1 rounded text-[11px] font-mono capitalize transition-all ${
                        selectedStructure === struct
                          ? 'bg-medical-cyan text-medical-bg font-semibold shadow-xs'
                          : 'bg-medical-surface text-medical-muted hover:text-white'
                      }`}
                    >
                      {struct.replace('_', ' ')}
                    </button>
                  ))}
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
                        ? 'Consistent'
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
            >
              <div className="relative min-h-[460px] bg-black rounded-xl border border-medical-border flex items-center justify-center overflow-hidden select-none">
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
                  <img
                    src={currentSlice.image_base64}
                    alt={`${mprView} CT slice`}
                    className="max-h-[440px] w-auto object-contain select-none"
                  />
                ) : (
                  <div className="text-xs text-medical-muted">Loading orthogonal slice...</div>
                )}

                {/* Top Left Metadata Tag */}
                <div className="absolute top-2.5 left-2.5 text-[10px] font-mono text-medical-cyan bg-black/75 px-2 py-1 rounded border border-white/10 z-20">
                  {mprView.toUpperCase()} [{sliceIndex}/{maxIndex}]
                </div>

                {/* Bottom Right Window Tag */}
                <div className="absolute bottom-2.5 right-2.5 text-[10px] font-mono text-slate-400 bg-black/75 px-2 py-1 rounded border border-white/10 z-20">
                  Window: {windowPreset.toUpperCase()}
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
                        variant="ghost"
                        size="sm"
                        onClick={() => jumpToNodule(nodule)}
                        className="text-medical-cyan text-xs p-1 hover:text-cyan-300"
                      >
                        View in Slice
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
    </div>
  );
};
