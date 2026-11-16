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
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-[#e2e8f0]">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-amber-900 bg-amber-100 px-2 py-0.5 rounded border border-amber-300 font-semibold">
              Audit Ledger
            </span>
            <span className="text-xs text-slate-300">·</span>
            <span className="text-xs text-[#5e4b75]">Client Telemetry</span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-[#1a082e] flex items-center gap-2.5">
            Session History & Inference Ledger
          </h2>
          <p className="text-sm text-[#5e4b75] mt-1">
            Audit log of radiograph screenings & CT explorations conducted in this client workstation.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {history.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={exportHistoryJson}
              className="gap-1.5 border-[#e2e8f0] hover:border-amber-400 text-[#1a082e] hover:text-amber-900 bg-white shadow-xs text-xs"
            >
              <Download className="w-3.5 h-3.5 text-amber-600" />
              <span>Export Audit Log</span>
            </Button>
          )}

          {history.length > 0 && (
            <>
              {showClearConfirm ? (
                <div className="flex items-center gap-2 bg-rose-50 p-1.5 rounded-lg border border-rose-200">
                  <span className="text-xs text-rose-800 font-medium px-2">Clear all entries?</span>
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
                  className="gap-2 text-rose-600 hover:text-rose-800 hover:border-rose-300 border-[#e2e8f0] bg-white shadow-xs text-xs"
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
      <div className="p-4 rounded-xl bg-white border border-[#e2e8f0] flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-xs">
        <div className="flex items-start gap-3">
          <ShieldCheck className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
          <div className="text-xs text-[#5e4b75] leading-relaxed">
            <strong className="text-[#1a082e] block font-semibold mb-0.5">
              Client Privacy & PHI Boundary Mode: {isPersistent ? 'Persistent Local Storage' : 'Session-Only (Recommended)'}
            </strong>
            {isPersistent
              ? 'Scan records are saved across browser restarts in unencrypted browser LocalStorage. Only use on private dedicated workstations.'
              : 'Scan records are stored ephemerally in browser SessionStorage and are automatically wiped upon closing the browser tab.'}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <label className="text-xs text-[#5e4b75] cursor-pointer flex items-center gap-2">
            <input
              type="checkbox"
              checked={isPersistent}
              onChange={(e) => setPersistence(e.target.checked)}
              className="accent-amber-500 rounded"
            />
            <span className="text-xs font-mono text-[#1a082e]">Persist across restarts</span>
          </label>
        </div>
      </div>

      {/* Summary Metrics Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="bg-white border border-[#e2e8f0] shadow-xs">
          <span className="text-xs text-[#5e4b75] uppercase font-semibold">Total Session Events</span>
          <p className="text-2xl font-bold font-mono text-[#1a082e] mt-1">{history.length}</p>
        </Card>
        <Card className="bg-white border border-[#e2e8f0] shadow-xs">
          <span className="text-xs text-[#5e4b75] uppercase font-semibold">Chest X-Ray Inferences</span>
          <p className="text-2xl font-bold font-mono text-amber-700 mt-1">{xrayCount}</p>
        </Card>
        <Card className="bg-white border border-[#e2e8f0] shadow-xs">
          <span className="text-xs text-[#5e4b75] uppercase font-semibold">3D CT Exploration Sessions</span>
          <p className="text-2xl font-bold font-mono text-emerald-700 mt-1">{ctCount}</p>
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
                  ? 'bg-gradient-to-r from-amber-500 via-amber-500 to-amber-600 text-white font-semibold shadow-xs'
                  : 'bg-white text-[#5e4b75] border border-[#e2e8f0] hover:text-[#1a082e] hover:bg-slate-50 shadow-xs'
              }`}
            >
              {tab === 'all' ? 'All Activity' : tab === 'xray' ? '2D X-Ray Inferences' : '3D CT Sessions'}
            </button>
          ))}
        </div>

        {filteredItems.length === 0 ? (
          <div className="h-64 flex flex-col items-center justify-center border border-dashed border-[#e2e8f0] rounded-xl text-center p-6 text-[#5e4b75] bg-white shadow-xs">
            <History className="w-10 h-10 text-slate-400 mb-2" />
            <p className="text-sm font-semibold text-[#1a082e]">No Activity Logged Yet</p>
            <p className="text-xs text-[#5e4b75] max-w-sm mt-1">
              Screen chest radiographs or explore 3D CT volumes to generate an ephemeral audit trail.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredItems.map((item) => (
              <div
                key={item.id}
                className="p-4 rounded-xl bg-white border border-[#e2e8f0] hover:border-amber-400 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs shadow-xs"
              >
                <div className="flex items-start gap-3">
                  <div className="w-9 h-9 rounded-lg bg-[#f8fafc] border border-[#e2e8f0] flex items-center justify-center text-amber-600 shrink-0 mt-0.5">
                    {item.type.startsWith('xray') ? (
                      <Scan className="w-4 h-4" />
                    ) : (
                      <Box className="w-4 h-4 text-emerald-600" />
                    )}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-[#1a082e] font-mono text-sm">{item.filename}</span>
                      <Badge variant={item.type.startsWith('xray') ? 'amber' : 'emerald'} size="sm">
                        {item.type === 'xray_multilabel'
                          ? 'DenseNet121'
                          : item.type === 'xray_binary'
                          ? 'ResNet18'
                          : 'CT Volume'}
                      </Badge>
                    </div>
                    <p className="text-[#5e4b75] mt-1 text-[11px] font-mono">{item.resultSummary}</p>
                  </div>
                </div>

                <div className="flex items-center gap-4 text-[#7e6c96] font-mono text-[11px] self-end sm:self-center">
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
