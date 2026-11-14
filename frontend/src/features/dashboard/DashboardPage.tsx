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

      {/* Quick Action Diagnostic Modules */}
      <div>
        <h3 className="text-xs font-semibold text-medical-subtle tracking-wider uppercase mb-3 px-1">
          Core Diagnostic Workspaces
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {/* Module 1: 2D X-Ray */}
          <Card className="hover:border-medical-cyan/50 transition-all flex flex-col justify-between group">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-cyan-950/60 border border-cyan-800/60 flex items-center justify-center text-medical-cyan group-hover:bg-medical-cyan group-hover:text-medical-bg transition-colors shadow-sm">
                <Scan className="w-5 h-5" />
              </div>
              <h4 className="text-lg font-bold text-white group-hover:text-medical-cyan transition-colors">
                2D X-Ray Screening
              </h4>
              <p className="text-xs text-medical-muted leading-relaxed">
                Multi-label DenseNet121 pathology detection with dual Patient and Doctor clinical views.
              </p>
            </div>
            <div className="pt-5">
              <Link to="/xray">
                <Button variant="outline" size="sm" className="w-full justify-between">
                  <span>Open X-Ray Workspace</span>
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
            </div>
          </Card>

          {/* Module 2: Dual-View CXR Lab */}
          <Card className="hover:border-medical-cyan/50 transition-all flex flex-col justify-between group">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-cyan-950/60 border border-cyan-800/60 flex items-center justify-center text-medical-cyan group-hover:bg-medical-cyan group-hover:text-medical-bg transition-colors shadow-sm">
                <SplitSquareVertical className="w-5 h-5" />
              </div>
              <h4 className="text-lg font-bold text-white group-hover:text-medical-cyan transition-colors">
                Dual-View CXR Lab
              </h4>
              <p className="text-xs text-medical-muted leading-relaxed">
                Bi-planar Frontal (PA/AP) and Lateral radiograph comparison, zoom inspection, and 3D reconstruction staging.
              </p>
            </div>
            <div className="pt-5">
              <Link to="/dual-view">
                <Button variant="outline" size="sm" className="w-full justify-between">
                  <span>Open Dual-View Lab</span>
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
            </div>
          </Card>

          {/* Module 3: 3D CT Explorer */}
          <Card className="hover:border-medical-teal/50 transition-all flex flex-col justify-between group">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-teal-950/60 border border-teal-800/60 flex items-center justify-center text-medical-teal group-hover:bg-medical-teal group-hover:text-medical-bg transition-colors shadow-sm">
                <Box className="w-5 h-5" />
              </div>
              <h4 className="text-lg font-bold text-white group-hover:text-medical-teal transition-colors">
                3D CT Explorer
              </h4>
              <p className="text-xs text-medical-muted leading-relaxed">
                Multi-planar orthogonal slicing with clinical HU windowing, watertight 3D meshes, and LUNA16 nodule mapping.
              </p>
            </div>
            <div className="pt-5">
              <Link to="/ct-explorer">
                <Button variant="outline" size="sm" className="w-full justify-between">
                  <span>Open CT Workspace</span>
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
            </div>
          </Card>

          {/* Module 4: Research Benchmarks */}
          <Card className="hover:border-medical-border transition-all flex flex-col justify-between group">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-medical-card border border-medical-border flex items-center justify-center text-medical-muted group-hover:text-white transition-colors shadow-sm">
                <LineChart className="w-5 h-5" />
              </div>
              <h4 className="text-lg font-bold text-white group-hover:text-medical-cyan transition-colors">
                Research Benchmarks
              </h4>
              <p className="text-xs text-medical-muted leading-relaxed">
                Audited clinical research metrics across 40,798 test cases from NIH, CheXpert, and VinDr cohorts.
              </p>
            </div>
            <div className="pt-5">
              <Link to="/research">
                <Button variant="outline" size="sm" className="w-full justify-between">
                  <span>View Benchmark Evidence</span>
                  <ArrowRight className="w-4 h-4" />
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
