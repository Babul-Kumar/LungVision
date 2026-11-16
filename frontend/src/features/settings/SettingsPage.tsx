import React, { useState, useEffect } from 'react';
import {
  RefreshCw,
  Monitor,
  ExternalLink,
  Activity,
  ShieldCheck,
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { useAppStore } from '@/store/useAppStore';
import { healthApi } from '@/services/api/healthApi';
import { HealthResponse, ReadinessResponse } from '@/types/api';

export const SettingsPage: React.FC = () => {
  const { isBackendOnline, fetchHealth } = useAppStore();
  const [pingLatency, setPingLatency] = useState<number | null>(null);
  const [testingConnection, setTestingConnection] = useState(false);
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [readinessData, setReadinessData] = useState<ReadinessResponse | null>(null);
  const [pingError, setPingError] = useState<string | null>(null);

  // WebGL capabilities detection
  const [webglInfo, setWebglInfo] = useState<{ supported: boolean; renderer?: string; vendor?: string }>({
    supported: false,
  });

  useEffect(() => {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        setWebglInfo({
          supported: true,
          renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'Generic WebGL',
          vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'Generic Vendor',
        });
      }
    } catch {
      setWebglInfo({ supported: false });
    }

    testConnection();
  }, []);

  const testConnection = async () => {
    setTestingConnection(true);
    setPingError(null);
    const start = performance.now();
    try {
      const [h, r] = await Promise.all([healthApi.getHealth(), healthApi.getReady()]);
      const end = performance.now();
      setPingLatency(Math.round(end - start));
      setHealthData(h);
      setReadinessData(r);
      fetchHealth();
    } catch (err: any) {
      setPingError(err.message || 'Connection to backend gateway failed.');
      setHealthData(null);
      setReadinessData(null);
    } finally {
      setTestingConnection(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-white/[0.08]">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-amber-400 bg-amber-950/40 px-2 py-0.5 rounded border border-amber-800/40">
              System Control
            </span>
            <span className="text-xs text-white/20">·</span>
            <span className="text-xs text-slate-400">Infrastructure & Hardware</span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            System Configuration & Diagnostics
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Backend gateway telemetry, GPU/CPU runtime verification, client WebGL capabilities & research parameters.
          </p>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={testConnection}
          isLoading={testingConnection}
          className="gap-2 border-white/10 hover:border-amber-400 text-slate-200 hover:text-amber-300"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Test Connection</span>
        </Button>
      </div>

      <DisclaimerBanner compact />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Backend Connectivity & Telemetry */}
        <Card title="Backend API Gateway Status" subtitle="FastAPI Uvicorn REST Service (Port 8000)">
          <div className="space-y-4 text-xs">
            <div className="flex items-center justify-between p-3.5 rounded-xl bg-medical-surface border border-white/10">
              <div className="flex items-center gap-3">
                <div
                  className={`w-3 h-3 rounded-full ${
                    isBackendOnline ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'
                  }`}
                />
                <div>
                  <span className="font-semibold text-white text-sm block">
                    {isBackendOnline ? 'Live Gateway Connected' : 'Gateway Offline'}
                  </span>
                  <span className="text-medical-muted font-mono text-[11px]">
                    http://127.0.0.1:8000/api/v1
                  </span>
                </div>
              </div>

              {pingLatency !== null && (
                <Badge variant={pingLatency < 100 ? 'emerald' : 'amber'} size="sm">
                  <Activity className="w-3 h-3 mr-1" />
                  {pingLatency} ms RTT
                </Badge>
              )}
            </div>

            {pingError ? (
              <div className="p-3 bg-rose-950/50 border border-rose-800 rounded-lg text-rose-300 space-y-1">
                <p className="font-semibold">Connection Check Failed</p>
                <p className="text-[11px]">{pingError}</p>
                <p className="text-[10px] text-rose-400/80 pt-1">
                  Ensure the Uvicorn backend process is active on port 8000 via:
                  <code> uvicorn backend.app.main:app --port 8000</code>
                </p>
              </div>
            ) : healthData ? (
              <div className="space-y-2 border-t border-white/10 pt-3 font-mono">
                <div className="flex justify-between py-1">
                  <span className="text-medical-muted">Service Title:</span>
                  <span className="text-white font-medium">{(healthData as any).service || 'LungVision FastAPI Gateway'}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-medical-muted">Gateway API Version:</span>
                  <span className="text-amber-400">v{healthData.version}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-medical-muted">Liveness Probe:</span>
                  <span className="text-emerald-400 font-semibold">{healthData.status.toUpperCase()}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-medical-muted">Inference Ready:</span>
                  <span className="text-emerald-400 font-semibold">
                    {readinessData?.status?.toUpperCase() || 'READY'}
                  </span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-medical-muted">Compute Device:</span>
                  <span className="text-amber-300 font-semibold uppercase">
                    {readinessData?.device || 'CPU'}
                  </span>
                </div>
              </div>
            ) : null}

            <div className="pt-2 flex items-center justify-between">
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="text-xs text-amber-400 hover:text-amber-300 hover:underline flex items-center gap-1 font-medium"
              >
                <span>Interactive Swagger UI Documentation</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>

              <a
                href="http://localhost:8000/openapi.json"
                target="_blank"
                rel="noreferrer"
                className="text-xs text-medical-subtle hover:text-white flex items-center gap-1 font-mono"
              >
                <span>OpenAPI 3.1.0 JSON</span>
              </a>
            </div>
          </div>
        </Card>

        {/* Client Hardware & WebGL Telemetry */}
        <Card title="Client Hardware & 3D Acceleration" subtitle="Browser GPU graphics driver status for Three.js volumetric rendering">
          <div className="space-y-4 text-xs font-mono">
            <div className="flex items-center justify-between p-3.5 rounded-xl bg-medical-surface border border-white/10">
              <div className="flex items-center gap-3">
                <Monitor className="w-5 h-5 text-amber-400 shrink-0" />
                <div>
                  <span className="font-semibold text-white text-sm block">WebGL 2.0 Engine</span>
                  <span className="text-medical-muted text-[11px]">Hardware Acceleration</span>
                </div>
              </div>

              <Badge variant={webglInfo.supported ? 'emerald' : 'rose'} size="sm">
                {webglInfo.supported ? 'Hardware Accelerated' : 'Software Fallback'}
              </Badge>
            </div>

            <div className="space-y-2 border-t border-white/10 pt-3">
              <div className="flex justify-between py-1">
                <span className="text-medical-muted font-sans">GPU Renderer:</span>
                <span className="text-slate-200 text-right truncate max-w-xs" title={webglInfo.renderer}>
                  {webglInfo.renderer || 'Standard Canvas'}
                </span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-medical-muted font-sans">Driver Vendor:</span>
                <span className="text-slate-200">{webglInfo.vendor || 'Generic'}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-medical-muted font-sans">Three.js Version:</span>
                <span className="text-slate-200 font-semibold">r170.0</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-medical-muted font-sans">Recharts Engine:</span>
                <span className="text-amber-400 font-semibold">v2.15.0</span>
              </div>
            </div>

            {/* Privacy & Governance Notice */}
            <div className="mt-4 p-3 rounded-xl bg-medical-card/60 border border-white/10 font-sans text-[11px] text-medical-muted space-y-1">
              <div className="flex items-center gap-1.5 font-semibold text-slate-200">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                <span>Zero Server-Side PHI Retention Policy</span>
              </div>
              <p className="leading-relaxed">
                Images uploaded for inference are held in volatile RAM only for the duration of inference execution and are never written to permanent disk or sent to external cloud APIs.
              </p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
