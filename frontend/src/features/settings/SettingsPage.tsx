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
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-[#e2e8f0]">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-amber-900 bg-amber-100 px-2 py-0.5 rounded border border-amber-300 font-semibold">
              System Control
            </span>
            <span className="text-xs text-slate-300">·</span>
            <span className="text-xs text-[#5e4b75]">Infrastructure & Hardware</span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-[#1a082e] flex items-center gap-2.5">
            System Configuration & Diagnostics
          </h2>
          <p className="text-sm text-[#5e4b75] mt-1">
            Backend gateway telemetry, GPU/CPU runtime verification, client WebGL capabilities & research parameters.
          </p>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={testConnection}
          isLoading={testingConnection}
          className="gap-2 border-[#e2e8f0] hover:border-amber-400 text-[#1a082e] hover:text-amber-900 bg-white shadow-xs"
        >
          <RefreshCw className="w-4 h-4 text-amber-600" />
          <span>Test Connection</span>
        </Button>
      </div>

      <DisclaimerBanner compact />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Backend Connectivity & Telemetry */}
        <Card title="Backend API Gateway Status" subtitle="FastAPI Uvicorn REST Service (Port 8000)">
          <div className="space-y-4 text-xs">
            <div className="flex items-center justify-between p-3.5 rounded-xl bg-[#f8fafc] border border-[#e2e8f0]">
              <div className="flex items-center gap-3">
                <div
                  className={`w-3 h-3 rounded-full ${
                    isBackendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
                  }`}
                />
                <div>
                  <span className="font-semibold text-[#1a082e] text-sm block">
                    {isBackendOnline ? 'Live Gateway Connected' : 'Gateway Offline'}
                  </span>
                  <span className="text-[#5e4b75] font-mono text-[11px]">
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
              <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-rose-800 space-y-1">
                <p className="font-semibold">Connection Check Failed</p>
                <p className="text-[11px]">{pingError}</p>
                <p className="text-[10px] text-rose-600 pt-1">
                  Ensure the Uvicorn backend process is active on port 8000 via:
                  <code> uvicorn backend.app.main:app --port 8000</code>
                </p>
              </div>
            ) : healthData ? (
              <div className="space-y-2 border-t border-[#e2e8f0] pt-3 font-mono">
                <div className="flex justify-between py-1">
                  <span className="text-[#5e4b75]">Service Title:</span>
                  <span className="text-[#1a082e] font-medium">{(healthData as any).service || 'LungVision FastAPI Gateway'}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-[#5e4b75]">Gateway API Version:</span>
                  <span className="text-amber-800 font-semibold">v{healthData.version}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-[#5e4b75]">Liveness Probe:</span>
                  <span className="text-emerald-700 font-semibold">{healthData.status.toUpperCase()}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-[#5e4b75]">Inference Ready:</span>
                  <span className="text-emerald-700 font-semibold">
                    {readinessData?.status?.toUpperCase() || 'READY'}
                  </span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-[#5e4b75]">Compute Device:</span>
                  <span className="text-amber-800 font-semibold uppercase">
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
                className="text-xs text-amber-700 hover:text-amber-900 hover:underline flex items-center gap-1 font-medium"
              >
                <span>Interactive Swagger UI Documentation</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>

              <a
                href="http://localhost:8000/openapi.json"
                target="_blank"
                rel="noreferrer"
                className="text-xs text-[#7e6c96] hover:text-[#1a082e] flex items-center gap-1 font-mono"
              >
                <span>OpenAPI 3.1.0 JSON</span>
              </a>
            </div>
          </div>
        </Card>

        {/* Client Hardware & WebGL Telemetry */}
        <Card title="Client Hardware & 3D Acceleration" subtitle="Browser GPU graphics driver status for Three.js volumetric rendering">
          <div className="space-y-4 text-xs font-mono">
            <div className="flex items-center justify-between p-3.5 rounded-xl bg-[#f8fafc] border border-[#e2e8f0]">
              <div className="flex items-center gap-3">
                <Monitor className="w-5 h-5 text-amber-600 shrink-0" />
                <div>
                  <span className="font-semibold text-[#1a082e] text-sm block">WebGL 2.0 Engine</span>
                  <span className="text-[#5e4b75] text-[11px]">Hardware Acceleration</span>
                </div>
              </div>

              <Badge variant={webglInfo.supported ? 'emerald' : 'rose'} size="sm">
                {webglInfo.supported ? 'Hardware Accelerated' : 'Software Fallback'}
              </Badge>
            </div>

            <div className="space-y-2 border-t border-[#e2e8f0] pt-3">
              <div className="flex justify-between py-1">
                <span className="text-[#5e4b75] font-sans">GPU Renderer:</span>
                <span className="text-[#1a082e] text-right truncate max-w-xs" title={webglInfo.renderer}>
                  {webglInfo.renderer || 'Standard Canvas'}
                </span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-[#5e4b75] font-sans">Driver Vendor:</span>
                <span className="text-[#1a082e]">{webglInfo.vendor || 'Generic'}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-[#5e4b75] font-sans">Three.js Version:</span>
                <span className="text-[#1a082e] font-semibold">r170.0</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-[#5e4b75] font-sans">Recharts Engine:</span>
                <span className="text-amber-800 font-semibold">v2.15.0</span>
              </div>
            </div>

            {/* Privacy & Governance Notice */}
            <div className="mt-4 p-3 rounded-xl bg-white border border-[#e2e8f0] font-sans text-[11px] text-[#5e4b75] space-y-1 shadow-xs">
              <div className="flex items-center gap-1.5 font-semibold text-[#1a082e]">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
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
