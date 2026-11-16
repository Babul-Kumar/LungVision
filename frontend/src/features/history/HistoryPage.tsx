import React, { useState } from 'react';
import {
  History,
  Trash2,
  Scan,
  Box,
  Clock,
  ShieldCheck,
  Download,
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { useHistoryStore } from '@/store/useHistoryStore';

export const HistoryPage: React.FC = () => {
  const { history, isPersistent, setPersistence, clearHistory } = useHistoryStore();
  const [filter, setFilter] = useState<'all' | 'xray' | 'ct'>('all');
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  const filteredItems = history.filter((item) => {
    if (filter === 'xray') return item.type === 'xray_multilabel' || item.type === 'xray_binary';
    if (filter === 'ct') return item.type === 'ct_session';
    return true;
  });

  const xrayCount = history.filter((i) => i.type.startsWith('xray')).length;
  const ctCount = history.filter((i) => i.type === 'ct_session').length;

  const exportHistoryJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `lungvision_session_audit_${new Date().toISOString().slice(0, 10)}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-white/[0.08]">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-amber-400 bg-amber-950/40 px-2 py-0.5 rounded border border-amber-800/40">
              Audit Ledger
            </span>
            <span className="text-xs text-white/20">·</span>
            <span className="text-xs text-slate-400">Client Telemetry</span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            Session History & Inference Ledger
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Audit log of radiograph screenings & CT explorations conducted in this client workstation.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {history.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={exportHistoryJson}
              className="gap-1.5 border-white/10 hover:border-amber-400 text-slate-200 hover:text-amber-300 text-xs"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Audit Log</span>
            </Button>
          )}

          {history.length > 0 && (
            <>
              {showClearConfirm ? (
                <div className="flex items-center gap-2 bg-rose-950/40 p-1.5 rounded-lg border border-rose-800">
                  <span className="text-xs text-rose-300 font-medium px-2">Clear all entries?</span>
                  <Button
                    variant="danger"
                    size="sm"
                    onClick={() => {
                      clearHistory();
                      setShowClearConfirm(false);
                    }}
                  >
                    Confirm Clear
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowClearConfirm(false)}
                  >
                    Cancel
                  </Button>
                </div>
              ) : (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowClearConfirm(true)}
                  className="gap-2 text-rose-400 hover:text-rose-300 hover:border-rose-700/60 text-xs"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Clear Ledger</span>
                </Button>
              )}
            </>
          )}
        </div>
      </div>

      <DisclaimerBanner compact />

      {/* HIPAA Privacy & Storage Policy Strip */}
      <div className="p-4 rounded-xl bg-medical-card/80 border border-white/10 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-start gap-3">
          <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
          <div className="text-xs text-slate-400 leading-relaxed">
            <strong className="text-white block font-semibold mb-0.5">
              Client Privacy & PHI Boundary Mode: {isPersistent ? 'Persistent Local Storage' : 'Session-Only (Recommended)'}
            </strong>
            {isPersistent
              ? 'Scan records are saved across browser restarts in unencrypted browser LocalStorage. Only use on private dedicated workstations.'
              : 'Scan records are stored ephemerally in browser SessionStorage and are automatically wiped upon closing the browser tab.'}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <label className="text-xs text-slate-400 cursor-pointer flex items-center gap-2">
            <input
              type="checkbox"
              checked={isPersistent}
              onChange={(e) => setPersistence(e.target.checked)}
              className="accent-amber-400 rounded"
            />
            <span className="text-xs font-mono text-slate-300">Persist across restarts</span>
          </label>
        </div>
      </div>

      {/* Summary Metrics Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="bg-medical-surface/60">
          <span className="text-xs text-slate-400 uppercase font-semibold">Total Session Events</span>
          <p className="text-2xl font-bold font-mono text-white mt-1">{history.length}</p>
        </Card>
        <Card className="bg-medical-surface/60">
          <span className="text-xs text-slate-400 uppercase font-semibold">Chest X-Ray Inferences</span>
          <p className="text-2xl font-bold font-mono text-amber-400 mt-1">{xrayCount}</p>
        </Card>
        <Card className="bg-medical-surface/60">
          <span className="text-xs text-slate-400 uppercase font-semibold">3D CT Exploration Sessions</span>
          <p className="text-2xl font-bold font-mono text-emerald-400 mt-1">{ctCount}</p>
        </Card>
      </div>

      {/* Filter Tabs & History List */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          {(['all', 'xray', 'ct'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setFilter(tab)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium capitalize transition-all ${
                filter === tab
                  ? 'bg-gradient-to-r from-amber-400 via-amber-500 to-amber-600 text-slate-950 font-bold shadow-md shadow-amber-950/40 border border-amber-300/40'
                  : 'bg-medical-surface text-slate-400 border border-white/10 hover:text-white'
              }`}
            >
              {tab === 'all' ? 'All Activity' : tab === 'xray' ? '2D X-Ray Inferences' : '3D CT Sessions'}
            </button>
          ))}
        </div>

        {filteredItems.length === 0 ? (
          <div className="h-64 flex flex-col items-center justify-center border border-dashed border-white/10 rounded-xl text-center p-6 text-slate-400">
            <History className="w-10 h-10 text-slate-500 mb-2" />
            <p className="text-sm font-semibold text-white">No Activity Logged Yet</p>
            <p className="text-xs text-slate-400 max-w-sm mt-1">
              Screen chest radiographs or explore 3D CT volumes to generate an ephemeral audit trail.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredItems.map((item) => (
              <div
                key={item.id}
                className="p-4 rounded-xl bg-medical-surface/80 border border-white/10 hover:border-amber-400/40 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
              >
                <div className="flex items-start gap-3">
                  <div className="w-9 h-9 rounded-lg bg-medical-card border border-white/10 flex items-center justify-center text-amber-400 shrink-0 mt-0.5">
                    {item.type.startsWith('xray') ? (
                      <Scan className="w-4 h-4" />
                    ) : (
                      <Box className="w-4 h-4 text-emerald-400" />
                    )}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-white font-mono text-sm">{item.filename}</span>
                      <Badge variant={item.type.startsWith('xray') ? 'amber' : 'emerald'} size="sm">
                        {item.type === 'xray_multilabel'
                          ? 'DenseNet121'
                          : item.type === 'xray_binary'
                          ? 'ResNet18'
                          : 'CT Volume'}
                      </Badge>
                    </div>
                    <p className="text-slate-400 mt-1 text-[11px] font-mono">{item.resultSummary}</p>
                  </div>
                </div>

                <div className="flex items-center gap-4 text-slate-400 font-mono text-[11px] self-end sm:self-center">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5" />
                    {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                  <span>{new Date(item.timestamp).toLocaleDateString()}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
