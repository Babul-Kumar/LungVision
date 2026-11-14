import React, { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import {
  LineChart,
  Download,
  Info,
  ShieldAlert,
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { researchApi } from '@/services/api/researchApi';
import { ResearchMetricsResponse } from '@/types/models';

type BenchmarkTab = 'nih' | 'chexpert' | 'vindr';

export const ResearchPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<BenchmarkTab>('nih');
  const [data, setData] = useState<ResearchMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    researchApi
      .getResearchMetrics()
      .then((res) => {
        setData(res);
        setError(null);
      })
      .catch((err) => {
        console.error('Failed to load research metrics:', err);
        setError('Failed to connect to backend research API. Verify backend server is running on port 8000.');
      })
      .finally(() => setLoading(false));
  }, []);

  const exportCsv = (filename: string, rows: Record<string, any>[]) => {
    if (!rows || rows.length === 0) return;
    const headers = Object.keys(rows[0]);
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [
        headers.join(','),
        ...rows.map((row) =>
          headers
            .map((h) => {
              const val = row[h];
              return typeof val === 'string' && val.includes(',') ? `"${val}"` : val;
            })
            .join(',')
        ),
      ].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `${filename}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-medical-border">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            <LineChart className="w-6 h-6 text-medical-cyan" />
            Research Evidence & Benchmark Registry
          </h2>
          <p className="text-sm text-medical-muted mt-1">
            Audited retrospective benchmarks across NIH ChestX-ray14, Stanford CheXpert, and VinDr-CXR cohorts.
          </p>
        </div>

        {/* Tab Controls */}
        <div className="flex items-center gap-2 bg-medical-surface p-1 rounded-lg border border-medical-border">
          <button
            onClick={() => setActiveTab('nih')}
            className={`px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeTab === 'nih'
                ? 'bg-medical-cyan text-slate-950 font-semibold shadow'
                : 'text-medical-muted hover:text-white'
            }`}
          >
            NIH ChestX-ray14
          </button>
          <button
            onClick={() => setActiveTab('chexpert')}
            className={`px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeTab === 'chexpert'
                ? 'bg-medical-cyan text-slate-950 font-semibold shadow'
                : 'text-medical-muted hover:text-white'
            }`}
          >
            CheXpert External
          </button>
          <button
            onClick={() => setActiveTab('vindr')}
            className={`px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
              activeTab === 'vindr'
                ? 'bg-medical-cyan text-slate-950 font-semibold shadow'
                : 'text-medical-muted hover:text-white'
            }`}
          >
            VinDr-CXR (15k)
          </button>
        </div>
      </div>

      <DisclaimerBanner />

      {loading ? (
        <div className="h-64 flex flex-col items-center justify-center border border-medical-border rounded-xl bg-medical-surface/40 space-y-3">
          <div className="w-8 h-8 border-2 border-medical-cyan border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-medical-muted">Loading research metrics from backend...</span>
        </div>
      ) : error ? (
        <div className="p-6 border border-rose-500/40 bg-rose-950/20 rounded-xl text-rose-300 text-sm flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-rose-200">Benchmark Data Unavailable</h4>
            <p className="mt-1 text-xs text-rose-300/80">{error}</p>
          </div>
        </div>
      ) : !data ? null : (
        <>
          {/* TAB 1: NIH ChestX-ray14 */}
          {activeTab === 'nih' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    NIH ChestX-ray14 Held-Out Test Set (DenseNet121)
                    <Badge variant="emerald" size="sm">Calibrated Epoch 4</Badge>
                  </h3>
                  <p className="text-xs text-medical-muted mt-0.5">
                    Multi-label evaluation on held-out test split. Thresholds calibrated for high-sensitivity triage.
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => exportCsv('nih_chestxray14_metrics', data.nih_test_metrics)}
                  className="gap-2"
                >
                  <Download className="w-3.5 h-3.5" />
                  Export CSV
                </Button>
              </div>

              {/* Chart: AUROC vs AUPRC by Disease */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <Card title="Diagnostic Discrimination (AUROC & AUPRC)" className="lg:col-span-2">
                  <div className="h-72 w-full pt-4">
                    <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                      <BarChart
                        data={data.nih_test_metrics}
                        margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" vertical={false} />
                        <XAxis
                          dataKey="Disease"
                          stroke="#64748b"
                          tick={{ fill: '#94a3b8', fontSize: 11 }}
                        />
                        <YAxis
                          domain={[0, 1]}
                          stroke="#64748b"
                          tick={{ fill: '#94a3b8', fontSize: 11 }}
                          tickFormatter={(v) => v.toFixed(2)}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#070b14',
                            borderColor: '#1e293b',
                            borderRadius: '8px',
                            fontSize: '12px',
                          }}
                        />
                        <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                        <Bar dataKey="AUROC" fill="#06b6d4" name="AUROC" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="AUPRC" fill="#14b8a6" name="AUPRC" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="Sensitivity" fill="#3b82f6" name="Sensitivity" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                {/* Radar Performance Distribution */}
                <Card title="Pathology Sensitivity Profiling">
                  <div className="h-72 w-full pt-2">
                    <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                      <RadarChart
                        outerRadius={90}
                        data={data.nih_test_metrics.map((m) => ({
                          disease: m.Disease.slice(0, 5),
                          Sensitivity: m.Sensitivity,
                          Specificity: m.Specificity,
                        }))}
                      >
                        <PolarGrid stroke="#1f293d" />
                        <PolarAngleAxis dataKey="disease" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                        <PolarRadiusAxis angle={30} domain={[0, 1]} stroke="#475569" tick={{ fontSize: 10 }} />
                        <Radar
                          name="Sensitivity"
                          dataKey="Sensitivity"
                          stroke="#06b6d4"
                          fill="#06b6d4"
                          fillOpacity={0.4}
                        />
                        <Radar
                          name="Specificity"
                          dataKey="Specificity"
                          stroke="#3b82f6"
                          fill="#3b82f6"
                          fillOpacity={0.2}
                        />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </div>

              {/* Data Table */}
              <Card title="Detailed NIH Benchmark Metrics Table">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-medical-border text-medical-subtle font-semibold uppercase tracking-wider">
                        <th className="py-2.5 px-3">Disease Label</th>
                        <th className="py-2.5 px-3">Threshold</th>
                        <th className="py-2.5 px-3">AUROC</th>
                        <th className="py-2.5 px-3">AUPRC</th>
                        <th className="py-2.5 px-3">Sensitivity</th>
                        <th className="py-2.5 px-3">Specificity</th>
                        <th className="py-2.5 px-3">Precision</th>
                        <th className="py-2.5 px-3">F1 Score</th>
                        <th className="py-2.5 px-3">TP / FP / FN</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-medical-border/50 text-slate-200">
                      {data.nih_test_metrics.map((row) => (
                        <tr key={row.Disease} className="hover:bg-medical-surface/40 transition-colors">
                          <td className="py-2.5 px-3 font-semibold text-white">{row.Disease}</td>
                          <td className="py-2.5 px-3 font-mono text-medical-cyan">{row.Threshold.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-emerald-400 font-bold">{row.AUROC.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-teal-400">{row.AUPRC.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-blue-400">{(row.Sensitivity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-slate-300">{(row.Specificity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-slate-300">{(row.Precision * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-amber-400">{row.F1.toFixed(3)}</td>
                          <td className="py-2.5 px-3 font-mono text-medical-muted text-[11px]">
                            {row.TP ?? '-'} / {row.FP ?? '-'} / {row.FN ?? '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          )}

          {/* TAB 2: CheXpert External Generalization */}
          {activeTab === 'chexpert' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    Stanford CheXpert External Generalization
                    <Badge variant="blue" size="sm">Cross-Hospital Validation</Badge>
                  </h3>
                  <p className="text-xs text-medical-muted mt-0.5">
                    Testing domain transferability of NIH-calibrated thresholds against the Stanford CheXpert validation cohort.
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => exportCsv('chexpert_external_metrics', data.chexpert_external_metrics)}
                  className="gap-2"
                >
                  <Download className="w-3.5 h-3.5" />
                  Export CSV
                </Button>
              </div>

              {/* Chart: CheXpert AUROC & AUPRC */}
              <Card title="CheXpert Discrimination & Positive Case Prevalence">
                <div className="h-72 w-full pt-4">
                  <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                    <BarChart
                      data={data.chexpert_external_metrics}
                      margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" vertical={false} />
                      <XAxis
                        dataKey="Disease"
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                      />
                      <YAxis
                        domain={[0, 1]}
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#070b14',
                          borderColor: '#1e293b',
                          borderRadius: '8px',
                          fontSize: '12px',
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                      <Bar dataKey="AUROC" fill="#3b82f6" name="CheXpert AUROC" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="AUPRC" fill="#06b6d4" name="CheXpert AUPRC" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Sensitivity" fill="#10b981" name="Sensitivity" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              {/* Data Table */}
              <Card title="CheXpert External Cohort Performance Table">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-medical-border text-medical-subtle font-semibold uppercase tracking-wider">
                        <th className="py-2.5 px-3">Disease</th>
                        <th className="py-2.5 px-3">Threshold</th>
                        <th className="py-2.5 px-3">Cases (+ / -)</th>
                        <th className="py-2.5 px-3">AUROC</th>
                        <th className="py-2.5 px-3">AUPRC</th>
                        <th className="py-2.5 px-3">Sensitivity</th>
                        <th className="py-2.5 px-3">Specificity</th>
                        <th className="py-2.5 px-3">F1 Score</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-medical-border/50 text-slate-200">
                      {data.chexpert_external_metrics.map((row) => (
                        <tr key={row.Disease} className="hover:bg-medical-surface/40 transition-colors">
                          <td className="py-2.5 px-3 font-semibold text-white">{row.Disease}</td>
                          <td className="py-2.5 px-3 font-mono text-medical-cyan">{row.Threshold.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-medical-muted">
                            {row.Positive} / {row.Negative} ({row.Evaluable} total)
                          </td>
                          <td className="py-2.5 px-3 font-mono text-emerald-400 font-bold">{row.AUROC.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-teal-400">{row.AUPRC.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-blue-400">{(row.Sensitivity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-slate-300">{(row.Specificity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-amber-400">{row.F1.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          )}

          {/* TAB 3: VinDr-CXR (15,000 images) */}
          {activeTab === 'vindr' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    VinDr-CXR 15,000-Image External Validation (ResNet18)
                    <Badge variant="cyan" size="sm">Binary Normal/Abnormal</Badge>
                  </h3>
                  <p className="text-xs text-medical-muted mt-0.5">
                    Rigorous external verification comparing Baseline vs Blur-Augmented ResNet18 across real Vietnamese hospital radiographs.
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => exportCsv('vindr_cxr_metrics', data.vindr_external_metrics)}
                  className="gap-2"
                >
                  <Download className="w-3.5 h-3.5" />
                  Export CSV
                </Button>
              </div>

              {/* Highlight Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {data.vindr_external_metrics.slice(0, 2).map((m, idx) => (
                  <Card key={idx} className="bg-medical-surface/70">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-medical-cyan uppercase">
                        {m.model || (idx === 0 ? 'Original ResNet18' : 'Blur-Augmented ResNet18')}
                      </span>
                      <Badge variant={idx === 1 ? 'emerald' : 'blue'} size="sm">
                        {m.cohort || 'Full Cohort'}
                      </Badge>
                    </div>
                    <div className="mt-3 space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-medical-muted">ROC AUC:</span>
                        <span className="font-mono font-bold text-emerald-400 text-sm">
                          {m.ROC_AUC.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-medical-muted">Avg Precision:</span>
                        <span className="font-mono text-cyan-300">
                          {m.Average_Precision.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-medical-muted">Sensitivity / Spec:</span>
                        <span className="font-mono text-slate-300">
                          {(m.Sensitivity * 100).toFixed(1)}% / {(m.Specificity * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-medical-muted">Brier / ECE:</span>
                        <span className="font-mono text-amber-400">
                          {m.Brier_Score?.toFixed(4) ?? 'N/A'} / {m.ECE_10_bins?.toFixed(4) ?? 'N/A'}
                        </span>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>

              {/* Data Table */}
              <Card title="VinDr-CXR Comprehensive External Validation Table">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-medical-border text-medical-subtle font-semibold uppercase tracking-wider">
                        <th className="py-2.5 px-3">Model</th>
                        <th className="py-2.5 px-3">Cohort</th>
                        <th className="py-2.5 px-3">ROC AUC</th>
                        <th className="py-2.5 px-3">Avg Precision</th>
                        <th className="py-2.5 px-3">Accuracy</th>
                        <th className="py-2.5 px-3">Sensitivity</th>
                        <th className="py-2.5 px-3">Specificity</th>
                        <th className="py-2.5 px-3">Brier Score</th>
                        <th className="py-2.5 px-3">ECE (10 bins)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-medical-border/50 text-slate-200">
                      {data.vindr_external_metrics.map((row, idx) => (
                        <tr key={idx} className="hover:bg-medical-surface/40 transition-colors">
                          <td className="py-2.5 px-3 font-semibold text-white">{row.model}</td>
                          <td className="py-2.5 px-3 text-medical-muted">{row.cohort}</td>
                          <td className="py-2.5 px-3 font-mono text-emerald-400 font-bold">{row.ROC_AUC.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-teal-400">{row.Average_Precision.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-slate-300">{(row.Accuracy * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-blue-400">{(row.Sensitivity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-slate-300">{(row.Specificity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-amber-300">{row.Brier_Score?.toFixed(4) ?? 'N/A'}</td>
                          <td className="py-2.5 px-3 font-mono text-amber-400">{row.ECE_10_bins?.toFixed(4) ?? 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          )}

          {/* Research Notes Section */}
          <Card className="bg-medical-surface/50 border-medical-border/70">
            <h4 className="text-xs font-semibold text-medical-subtle tracking-wider uppercase mb-3 flex items-center gap-2">
              <Info className="w-4 h-4 text-medical-cyan" />
              Audited Benchmark Notes & Methodology
            </h4>
            <ul className="space-y-2 text-xs text-medical-muted list-disc list-inside">
              {data.notes.map((note, index) => (
                <li key={index} className="leading-relaxed">
                  {note}
                </li>
              ))}
            </ul>
          </Card>
        </>
      )}
    </div>
  );
};
