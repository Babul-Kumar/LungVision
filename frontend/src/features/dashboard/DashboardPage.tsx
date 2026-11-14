import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Scan,
  Box,
  LineChart,
  Cpu,
  ArrowRight,
  History,
  CheckCircle2,
  AlertCircle,
  FileText,
  SplitSquareVertical,
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { useAppStore } from '@/store/useAppStore';
import { useHistoryStore } from '@/store/useHistoryStore';
import { modelApi } from '@/services/api/modelApi';
import { ModelInfo } from '@/types/models';

export const DashboardPage: React.FC = () => {
  const { isBackendOnline, readiness, fetchHealth } = useAppStore();
  const { history } = useHistoryStore();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);

  useEffect(() => {
    fetchHealth();
    modelApi
      .listModels()
      .then((data) => setModels(data?.models || []))
      .catch(() => setModels([]))
      .finally(() => setLoadingModels(false));
  }, [fetchHealth]);

  const recentHistory = history.slice(0, 4);

  return (
    <div className="space-y-6">
      {/* Top Welcome & System Status Hero */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-medical-border">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-medical-cyan bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/40">
              Station Overview
            </span>
            <span className="text-xs text-medical-subtle">·</span>
            <span className="text-xs text-medical-muted">Full-Stack Intelligence Gateway</span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            Thoracic Intelligence Station
            <Badge variant="cyan" size="sm">
              Research Edition
            </Badge>
          </h2>
          <p className="text-sm text-medical-muted mt-1">
            Automated multi-label chest radiograph screening & 3D computed tomography volumetric exploration.
          </p>
        </div>

        {/* Live System Badges */}
        <div className="flex items-center gap-2.5">
          <Badge variant={isBackendOnline ? 'emerald' : 'rose'} size="md">
            <span className="w-2 h-2 rounded-full bg-current animate-pulse mr-1" />
            {isBackendOnline ? 'Live Backend: Online' : 'Live Backend: Offline'}
          </Badge>
          {readiness && (
            <Badge variant={readiness.status === 'ready' ? 'cyan' : 'amber'} size="md">
              <Cpu className="w-3.5 h-3.5 mr-1" />
              {readiness.status === 'ready'
                ? `Models Ready: ${
                    typeof readiness.models_loaded === 'object' && readiness.models_loaded !== null
                      ? Object.values(readiness.models_loaded).filter(Boolean).length
                      : (readiness as any).models_loaded_count ?? 3
                  }/3`
                : 'Models Degraded'}
            </Badge>
          )}
        </div>
      </div>

      <DisclaimerBanner compact />

      {/* Futuristic Clinical Telemetry Bar */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-panel p-3.5 rounded-xl border border-cyan-500/20 hud-corner-bracket relative overflow-hidden group hover:border-cyan-400/50 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/5 rounded-full blur-2xl group-hover:bg-cyan-500/10 transition-colors pointer-events-none" />
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-mono uppercase tracking-wider text-cyan-400 font-semibold">NIH Macro AUROC</span>
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
          </div>
          <div className="text-2xl font-bold font-mono tracking-tight text-white flex items-baseline gap-1.5">
            0.843
            <span className="text-[11px] font-normal text-cyan-300/80 font-sans">val peak</span>
          </div>
          <div className="text-[10px] text-medical-subtle mt-0.5">DenseNet121 6-Condition Screening</div>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-teal-500/20 hud-corner-bracket relative overflow-hidden group hover:border-teal-400/50 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-teal-500/5 rounded-full blur-2xl group-hover:bg-teal-500/10 transition-colors pointer-events-none" />
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-mono uppercase tracking-wider text-teal-400 font-semibold">SSM 3D Variance</span>
            <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
          </div>
          <div className="text-2xl font-bold font-mono tracking-tight text-white flex items-baseline gap-1.5">
            99.2%
            <span className="text-[11px] font-normal text-teal-300/80 font-sans">15 modes</span>
          </div>
          <div className="text-[10px] text-medical-subtle mt-0.5">Bi-Planar PDM Eigenmodes</div>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-emerald-500/20 hud-corner-bracket relative overflow-hidden group hover:border-emerald-400/50 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-full blur-2xl group-hover:bg-emerald-500/10 transition-colors pointer-events-none" />
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-mono uppercase tracking-wider text-emerald-400 font-semibold">Mesh Watertightness</span>
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono tracking-tight text-white flex items-baseline gap-1.5">
            100%
            <span className="text-[11px] font-normal text-emerald-300/80 font-sans">Euler=2</span>
          </div>
          <div className="text-[10px] text-medical-subtle mt-0.5">Divergence Theorem Organ Volumes</div>
        </div>

        <div className="glass-panel p-3.5 rounded-xl border border-blue-500/20 hud-corner-bracket relative overflow-hidden group hover:border-blue-400/50 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/5 rounded-full blur-2xl group-hover:bg-blue-500/10 transition-colors pointer-events-none" />
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-mono uppercase tracking-wider text-blue-400 font-semibold">FastAPI Event Loop</span>
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
          </div>
          <div className="text-2xl font-bold font-mono tracking-tight text-white flex items-baseline gap-1.5">
            ~85 ms
            <span className="text-[11px] font-normal text-blue-300/80 font-sans">async threadpool</span>
          </div>
          <div className="text-[10px] text-medical-subtle mt-0.5">Non-blocking Concurrency Safe</div>
        </div>
      </div>

      {/* Quick Action Diagnostic Modules */}
      <div>
        <h3 className="text-xs font-semibold text-medical-subtle tracking-wider uppercase mb-3 px-1">
          Core Diagnostic Workspaces
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {/* Module 1: 2D X-Ray */}
          <Card className="hover:border-cyan-400/60 hover:shadow-lg hover:shadow-cyan-950/40 transition-all duration-300 flex flex-col justify-between group glass-panel hud-corner-bracket relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/5 rounded-full blur-2xl group-hover:bg-cyan-500/15 transition-all pointer-events-none" />
            <div className="space-y-3 relative z-10">
              <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-cyan-950 to-cyan-900/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-105 group-hover:shadow-md group-hover:shadow-cyan-500/30 group-hover:text-cyan-300 transition-all">
                <Scan className="w-5 h-5" />
              </div>
              <h4 className="text-lg font-bold text-white group-hover:text-cyan-300 transition-colors flex items-center justify-between">
                <span>2D X-Ray Screening</span>
              </h4>
              <p className="text-xs text-medical-muted leading-relaxed">
                Multi-label DenseNet121 pathology detection with dual Patient and Doctor clinical views.
              </p>
            </div>
            <div className="pt-5 relative z-10">
              <Link to="/xray">
                <Button variant="outline" size="sm" className="w-full justify-between group-hover:border-cyan-500/50 group-hover:bg-cyan-950/40 transition-all">
                  <span>Open X-Ray Workspace</span>
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
            </div>
          </Card>

          {/* Module 2: Dual-View CXR Lab */}
          <Card className="hover:border-cyan-400/60 hover:shadow-lg hover:shadow-cyan-950/40 transition-all duration-300 flex flex-col justify-between group glass-panel hud-corner-bracket relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/5 rounded-full blur-2xl group-hover:bg-cyan-500/15 transition-all pointer-events-none" />
            <div className="space-y-3 relative z-10">
              <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-cyan-950 to-cyan-900/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-105 group-hover:shadow-md group-hover:shadow-cyan-500/30 group-hover:text-cyan-300 transition-all">
                <SplitSquareVertical className="w-5 h-5" />
              </div>
              <h4 className="text-lg font-bold text-white group-hover:text-cyan-300 transition-colors flex items-center justify-between">
                <span>Dual-View CXR Lab</span>
              </h4>
              <p className="text-xs text-medical-muted leading-relaxed">
                Bi-planar Frontal (PA/AP) and Lateral radiograph comparison, zoom inspection, and 3D reconstruction staging.
              </p>
            </div>
            <div className="pt-5 relative z-10">
              <Link to="/dual-view">
                <Button variant="outline" size="sm" className="w-full justify-between group-hover:border-cyan-500/50 group-hover:bg-cyan-950/40 transition-all">
                  <span>Open Dual-View Lab</span>
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
            </div>
          </Card>

          {/* Module 3: 3D CT Explorer */}
          <Card className="hover:border-teal-400/60 hover:shadow-lg hover:shadow-teal-950/40 transition-all duration-300 flex flex-col justify-between group glass-panel hud-corner-bracket relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-teal-500/5 rounded-full blur-2xl group-hover:bg-teal-500/15 transition-all pointer-events-none" />
            <div className="space-y-3 relative z-10">
              <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-teal-950 to-teal-900/80 border border-teal-500/30 flex items-center justify-center text-teal-400 group-hover:scale-105 group-hover:shadow-md group-hover:shadow-teal-500/30 group-hover:text-teal-300 transition-all">
                <Box className="w-5 h-5" />
              </div>
              <h4 className="text-lg font-bold text-white group-hover:text-teal-300 transition-colors flex items-center justify-between">
                <span>3D CT Explorer</span>
              </h4>
              <p className="text-xs text-medical-muted leading-relaxed">
                Multi-planar orthogonal slicing with clinical HU windowing, watertight 3D meshes, and LUNA16 nodule mapping.
              </p>
            </div>
            <div className="pt-5 relative z-10">
              <Link to="/ct-explorer">
                <Button variant="outline" size="sm" className="w-full justify-between group-hover:border-teal-500/50 group-hover:bg-teal-950/40 transition-all">
                  <span>Open CT Workspace</span>
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
            </div>
          </Card>

          {/* Module 4: Research Benchmarks */}
          <Card className="hover:border-blue-400/60 hover:shadow-lg hover:shadow-blue-950/40 transition-all duration-300 flex flex-col justify-between group glass-panel hud-corner-bracket relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-2xl group-hover:bg-blue-500/15 transition-all pointer-events-none" />
            <div className="space-y-3 relative z-10">
              <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-blue-950 to-blue-900/80 border border-blue-500/30 flex items-center justify-center text-blue-400 group-hover:scale-105 group-hover:shadow-md group-hover:shadow-blue-500/30 group-hover:text-blue-300 transition-all">
                <LineChart className="w-5 h-5" />
              </div>
              <h4 className="text-lg font-bold text-white group-hover:text-blue-300 transition-colors flex items-center justify-between">
                <span>Research Benchmarks</span>
              </h4>
              <p className="text-xs text-medical-muted leading-relaxed">
                Audited clinical research metrics across 40,798 test cases from NIH, CheXpert, and VinDr cohorts.
              </p>
            </div>
            <div className="pt-5 relative z-10">
              <Link to="/research">
                <Button variant="outline" size="sm" className="w-full justify-between group-hover:border-blue-500/50 group-hover:bg-blue-950/40 transition-all">
                  <span>View Benchmark Evidence</span>
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
            </div>
          </Card>
        </div>
      </div>

      {/* Bottom Grid: Model Inventory & Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Model Inventory (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <Card
            title="Active Model Inventory"
            subtitle="Frozen PyTorch weights initialized in FastAPI backend runtime"
            headerAction={
              <Link to="/models">
                <Button variant="ghost" size="sm" className="text-medical-cyan text-xs">
                  <span>Full Registry</span>
                  <ArrowRight className="w-3.5 h-3.5 ml-1" />
                </Button>
              </Link>
            }
          >
            {loadingModels ? (
              <div className="h-44 flex items-center justify-center text-xs text-medical-muted font-mono">
                Querying backend models...
              </div>
            ) : models.length === 0 ? (
              <div className="h-44 flex flex-col items-center justify-center text-center text-xs text-medical-muted p-4">
                <AlertCircle className="w-8 h-8 text-medical-amber mb-2" />
                <p>No models returned from gateway.</p>
                <p className="text-[11px] text-medical-subtle mt-0.5">Verify backend is active on port 8000.</p>
              </div>
            ) : (
              <div className="divide-y divide-medical-border/60">
                {models.map((m) => {
                  const isLoaded = (readiness?.models_loaded as Record<string, boolean> | undefined)?.[m.id] ?? false;
                  return (
                    <div key={m.id} className="py-3 flex items-center justify-between text-xs">
                      <div className="space-y-0.5">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-white">{m.name}</span>
                          <Badge variant="cyan" size="sm">
                            {m.architecture}
                          </Badge>
                        </div>
                        <p className="text-medical-muted text-[11px] font-mono">
                          Dataset: {m.metrics?.training_dataset ?? (m as any).dataset ?? 'Clinical Cohort'} · Epoch: {m.metrics?.checkpoint_epoch ?? 'Frozen'}
                        </p>
                      </div>

                      <div className="text-right font-mono text-[11px] space-y-0.5">
                        <span className={`font-semibold flex items-center gap-1 justify-end ${isLoaded ? 'text-emerald-400' : 'text-amber-400'}`}>
                          {isLoaded ? (
                            <>
                              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                              Ready in Memory
                            </>
                          ) : (
                            <>
                              <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
                              On Disk (Frozen)
                            </>
                          )}
                        </span>
                        <div className="flex items-center gap-1.5 text-medical-subtle text-[10px] justify-end">
                          {m.metrics?.parameters && (
                            <span>{(m.metrics.parameters / 1000000).toFixed(2)}M</span>
                          )}
                          {m.metrics?.parameters && <span>·</span>}
                          <span>{m.labels ? `${m.labels.length} Classes` : 'Binary'}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>
        </div>

        {/* Recent Session Activity (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <Card
            title="Recent Client Activity"
            subtitle="Ephemeral session audit trail"
            headerAction={
              <Link to="/history">
                <Button variant="ghost" size="sm" className="text-medical-cyan text-xs">
                  <span>View All</span>
                  <ArrowRight className="w-3.5 h-3.5 ml-1" />
                </Button>
              </Link>
            }
          >
            {recentHistory.length === 0 ? (
              <div className="h-44 flex flex-col items-center justify-center text-center text-xs text-medical-muted p-4">
                <History className="w-8 h-8 text-medical-subtle mb-2" />
                <p>No inferences recorded in this session.</p>
                <p className="text-[11px] text-medical-subtle mt-0.5">Screen an image to record an audit entry.</p>
              </div>
            ) : (
              <div className="divide-y divide-medical-border/60">
                {recentHistory.map((item) => (
                  <div key={item.id} className="py-2.5 flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2.5 truncate max-w-[200px]">
                      <FileText className="w-3.5 h-3.5 text-medical-cyan shrink-0" />
                      <span className="font-mono text-white truncate text-[11px]">{item.filename}</span>
                    </div>
                    <span className="text-medical-subtle text-[10px] font-mono">
                      {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};
