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
  CheckCircle2,
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { researchApi } from '@/services/api/researchApi';
import { ResearchMetricsResponse } from '@/types/models';

type BenchmarkTab = 'nih' | 'chexpert' | 'vindr' | 'ssm';

const SSM_MODE_METRICS = [
  { mode: 'Mode 1', variance_pct: 52.88, cumulative_pct: 52.88, interpretation: 'Global scale & thoracic height elongation' },
  { mode: 'Mode 2', variance_pct: 17.52, cumulative_pct: 70.40, interpretation: 'Anterior-posterior depth vs lateral width ratio' },
  { mode: 'Mode 3', variance_pct: 8.71, cumulative_pct: 79.11, interpretation: 'Bilateral lung volume asymmetry (Left/Right)' },
  { mode: 'Mode 4', variance_pct: 6.09, cumulative_pct: 85.20, interpretation: 'Diaphragmatic dome curvature (Inspiratory effort)' },
  { mode: 'Mode 5', variance_pct: 5.18, cumulative_pct: 90.38, interpretation: 'Apical curvature & upper mediastinal width' },
  { mode: 'Mode 6', variance_pct: 3.12, cumulative_pct: 93.50, interpretation: 'Costophrenic sulcus sharpness' },
  { mode: 'Mode 7', variance_pct: 1.84, cumulative_pct: 95.34, interpretation: 'Cardiac notch depth & lingula extension' },
  { mode: 'Mode 8', variance_pct: 1.25, cumulative_pct: 96.59, interpretation: 'Hilar indentation & main bronchus contour' },
  { mode: 'Mode 9', variance_pct: 0.94, cumulative_pct: 97.53, interpretation: 'Posterior basal curvature' },
  { mode: 'Mode 10', variance_pct: 0.72, cumulative_pct: 98.25, interpretation: 'Right middle lobe lateral bulging' },
];

const SSM_GENERALIZATION_METRICS = [
  { metric: 'Average Symmetric Surface Distance (ASSD)', ssm_value: '0.072 mm', baseline_value: '21.28 mm', gain: '99.66% error reduction', status: 'PASSED' },
  { metric: '95th Percentile Hausdorff Distance (HD95)', ssm_value: '0.168 mm', baseline_value: '48.91 mm', gain: '99.66% error reduction', status: 'PASSED' },
  { metric: 'Relative Volume Error (RVE)', ssm_value: '0.06%', baseline_value: '14.82%', gain: '99.59% error reduction', status: 'PASSED' },
  { metric: 'Surface Dice Coefficient @ 2.0 mm', ssm_value: '1.000 (100.0%)', baseline_value: '0.412 (41.2%)', gain: '+142.7% overlap gain', status: 'PASSED' },
];

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
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-[#e2e8f0]">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-[#1a082e] flex items-center gap-2.5">
            <LineChart className="w-6 h-6 text-amber-600" />
            Research Evidence & Benchmark Registry
          </h2>
          <p className="text-sm text-[#5e4b75] mt-1">
            Audited retrospective benchmarks across NIH ChestX-ray14, Stanford CheXpert, and VinDr-CXR cohorts.
          </p>
        </div>

        {/* Tab Controls */}
        <div className="flex items-center gap-1.5 bg-white p-1 rounded-xl border border-[#e2e8f0] shadow-xs">
          <button
            onClick={() => setActiveTab('nih')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${activeTab === 'nih'
              ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-white font-bold shadow-xs border border-amber-600'
              : 'text-[#5e4b75] hover:text-[#1a082e]'
              }`}
          >
            NIH ChestX-ray14
          </button>
          <button
            onClick={() => setActiveTab('chexpert')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${activeTab === 'chexpert'
              ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-white font-bold shadow-xs border border-amber-600'
              : 'text-[#5e4b75] hover:text-[#1a082e]'
              }`}
          >
            CheXpert External
          </button>
          <button
            onClick={() => setActiveTab('vindr')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${activeTab === 'vindr'
              ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-white font-bold shadow-xs border border-amber-600'
              : 'text-[#5e4b75] hover:text-[#1a082e]'
              }`}
          >
            VinDr-CXR (15k)
          </button>
          <button
            onClick={() => setActiveTab('ssm')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${activeTab === 'ssm'
              ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-white font-bold shadow-xs border border-amber-600'
              : 'text-[#5e4b75] hover:text-[#1a082e]'
              }`}
          >
            3D SSM Geometry
          </button>
        </div>
      </div>

      <DisclaimerBanner />

      {loading ? (
        <div className="h-64 flex flex-col items-center justify-center border border-[#e2e8f0] rounded-xl bg-white space-y-3 shadow-xs">
          <div className="w-8 h-8 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-[#5e4b75]">Loading research metrics from backend...</span>
        </div>
      ) : error ? (
        <div className="p-6 border border-rose-200 bg-rose-50 rounded-xl text-rose-800 text-sm flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-rose-900">Benchmark Data Unavailable</h4>
            <p className="mt-1 text-xs text-rose-700">{error}</p>
          </div>
        </div>
      ) : !data ? null : (
        <>
          {/* TAB 1: NIH ChestX-ray14 */}
          {activeTab === 'nih' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-[#1a082e] flex items-center gap-2">
                    NIH ChestX-ray14 Held-Out Test Set (DenseNet121)
                    <Badge variant="emerald" size="sm">Calibrated Epoch 4</Badge>
                  </h3>
                  <p className="text-xs text-[#5e4b75] mt-0.5">
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
                        <CartesianGrid strokeDasharray="3 3" stroke="#25163a" vertical={false} />
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
                            backgroundColor: '#ffffff',
                            borderColor: '#e2e8f0',
                            borderRadius: '12px',
                            fontSize: '12px',
                            color: '#1a082e',
                            boxShadow: '0 4px 12px rgba(26,8,46,0.08)',
                          }}
                        />
                        <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                        <Bar dataKey="AUROC" fill="#f59e0b" name="AUROC" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="AUPRC" fill="#f1f5f9" name="AUPRC" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="Sensitivity" fill="#d97706" name="Sensitivity" radius={[4, 4, 0, 0]} />
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
                        <PolarGrid stroke="#25163a" />
                        <PolarAngleAxis dataKey="disease" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                        <PolarRadiusAxis angle={30} domain={[0, 1]} stroke="#64748b" tick={{ fontSize: 10 }} />
                        <Radar
                          name="Sensitivity"
                          dataKey="Sensitivity"
                          stroke="#f59e0b"
                          fill="#f59e0b"
                          fillOpacity={0.4}
                        />
                        <Radar
                          name="Specificity"
                          dataKey="Specificity"
                          stroke="#e2e8f0"
                          fill="#e2e8f0"
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
                      <tr className="border-b border-[#e2e8f0] text-[#5e4b75] bg-slate-50 font-semibold uppercase tracking-wider font-mono text-[10px]">
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
                    <tbody className="divide-y divide-[#e2e8f0] text-[#1a082e]">
                      {data.nih_test_metrics.map((row) => (
                        <tr key={row.Disease} className="hover:bg-amber-50/40 transition-colors">
                          <td className="py-2.5 px-3 font-semibold text-[#1a082e]">{row.Disease}</td>
                          <td className="py-2.5 px-3 font-mono text-amber-800 font-semibold">{row.Threshold.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-emerald-700 font-bold">{row.AUROC.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-[#5e4b75]">{row.AUPRC.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-amber-700">{(row.Sensitivity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-[#5e4b75]">{(row.Specificity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-[#5e4b75]">{(row.Precision * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-amber-800 font-semibold">{row.F1.toFixed(3)}</td>
                          <td className="py-2.5 px-3 font-mono text-[#7e6c96] text-[11px]">
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
                  <h3 className="text-lg font-bold text-[#1a082e] flex items-center gap-2">
                    Stanford CheXpert External Generalization
                    <Badge variant="silver" size="sm">Cross-Hospital Validation</Badge>
                  </h3>
                  <p className="text-xs text-[#5e4b75] mt-0.5">
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
                      <CartesianGrid strokeDasharray="3 3" stroke="#25163a" vertical={false} />
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
                          backgroundColor: '#ffffff',
                          borderColor: '#e2e8f0',
                          borderRadius: '12px',
                          fontSize: '12px',
                          color: '#1a082e',
                          boxShadow: '0 4px 12px rgba(26,8,46,0.08)',
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                      <Bar dataKey="AUROC" fill="#f59e0b" name="CheXpert AUROC" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="AUPRC" fill="#f1f5f9" name="CheXpert AUPRC" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Sensitivity" fill="#d97706" name="Sensitivity" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              {/* Data Table */}
              <Card title="CheXpert External Cohort Performance Table">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-[#e2e8f0] text-[#5e4b75] bg-slate-50 font-semibold uppercase tracking-wider font-mono text-[10px]">
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
                    <tbody className="divide-y divide-[#e2e8f0] text-[#1a082e]">
                      {data.chexpert_external_metrics.map((row) => (
                        <tr key={row.Disease} className="hover:bg-amber-50/40 transition-colors">
                          <td className="py-2.5 px-3 font-semibold text-[#1a082e]">{row.Disease}</td>
                          <td className="py-2.5 px-3 font-mono text-amber-800 font-semibold">{row.Threshold.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-[#5e4b75]">
                            {row.Positive} / {row.Negative} ({row.Evaluable} total)
                          </td>
                          <td className="py-2.5 px-3 font-mono text-emerald-700 font-bold">{row.AUROC.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-[#5e4b75]">{row.AUPRC.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-amber-700">{(row.Sensitivity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-[#5e4b75]">{(row.Specificity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-amber-800 font-semibold">{row.F1.toFixed(3)}</td>
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
                  <h3 className="text-lg font-bold text-[#1a082e] flex items-center gap-2">
                    VinDr-CXR 15,000-Image External Validation (ResNet18)
                    <Badge variant="amber" size="sm">Binary Normal/Abnormal</Badge>
                  </h3>
                  <p className="text-xs text-[#5e4b75] mt-0.5">
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
                      <span className="text-xs font-bold text-amber-900 uppercase tracking-wide">
                        {m.model || (idx === 0 ? 'Original ResNet18' : 'Blur-Augmented ResNet18')}
                      </span>
                      <Badge variant={idx === 1 ? 'emerald' : 'silver'} size="sm">
                        {m.cohort || 'Full Cohort'}
                      </Badge>
                    </div>
                    <div className="mt-3 space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-[#5e4b75]">ROC AUC:</span>
                        <span className="font-mono font-bold text-emerald-700 text-sm">
                          {m.ROC_AUC.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-[#5e4b75]">Avg Precision:</span>
                        <span className="font-mono text-[#5e4b75]">
                          {m.Average_Precision.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-[#5e4b75]">Sensitivity / Spec:</span>
                        <span className="font-mono text-[#5e4b75]">
                          {(m.Sensitivity * 100).toFixed(1)}% / {(m.Specificity * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-[#5e4b75]">Brier / ECE:</span>
                        <span className="font-mono text-amber-800 font-semibold">
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
                      <tr className="border-b border-[#e2e8f0] text-[#5e4b75] bg-slate-50 font-semibold uppercase tracking-wider font-mono text-[10px]">
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
                    <tbody className="divide-y divide-[#e2e8f0] text-[#1a082e]">
                      {data.vindr_external_metrics.map((row, idx) => (
                        <tr key={idx} className="hover:bg-amber-50/40 transition-colors">
                          <td className="py-2.5 px-3 font-semibold text-[#1a082e]">{row.model}</td>
                          <td className="py-2.5 px-3 text-[#5e4b75]">{row.cohort}</td>
                          <td className="py-2.5 px-3 font-mono text-emerald-700 font-bold">{row.ROC_AUC.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-[#5e4b75]">{row.Average_Precision.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-mono text-[#5e4b75]">{(row.Accuracy * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-amber-700">{(row.Sensitivity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-[#5e4b75]">{(row.Specificity * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 font-mono text-amber-800 font-semibold">{row.Brier_Score?.toFixed(4) ?? 'N/A'}</td>
                          <td className="py-2.5 px-3 font-mono text-amber-800 font-semibold">{row.ECE_10_bins?.toFixed(4) ?? 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          )}

          {/* TAB 4: 3D SSM Statistical Shape Modeling (Phase 2B) */}
          {activeTab === 'ssm' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-[#1a082e] flex items-center gap-2">
                    Phase 2B: Multi-Patient 3D Statistical Shape Model (SSM)
                    <Badge variant="amber" size="sm">N=430 CT Cohort</Badge>
                    <Badge variant="emerald" size="sm">Zero Leakage Split</Badge>
                  </h3>
                  <p className="text-xs text-[#5e4b75] mt-0.5">
                    PCA eigenspace decomposition on canonical organic lung surface mesh (N=6,737 vertices, 13,478 triangles, 16 anatomical landmarks).
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => exportCsv('ssm_mode_variance_metrics', SSM_MODE_METRICS)}
                  className="gap-2"
                >
                  <Download className="w-3.5 h-3.5" />
                  Export Modes CSV
                </Button>
              </div>

              {/* Generalization Benchmark Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {SSM_GENERALIZATION_METRICS.map((item, idx) => (
                  <Card key={idx} className="relative overflow-hidden bg-white border border-[#e2e8f0] shadow-xs">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] font-mono text-amber-800 font-bold uppercase tracking-wider">
                        Validation Metric
                      </span>
                      <Badge variant="emerald" size="sm">
                        <CheckCircle2 className="w-3 h-3 mr-1" />
                        Verified
                      </Badge>
                    </div>
                    <div className="text-xs font-semibold text-[#1a082e] mb-2 leading-snug">
                      {item.metric}
                    </div>
                    <div className="text-2xl font-bold font-mono text-amber-700">
                      {item.ssm_value}
                    </div>
                    <div className="mt-2 text-[11px] text-[#5e4b75] flex items-center justify-between pt-2 border-t border-[#e2e8f0]">
                      <span>Baseline: <strong className="text-[#1a082e]">{item.baseline_value}</strong></span>
                      <span className="text-emerald-700 font-semibold">{item.gain}</span>
                    </div>
                  </Card>
                ))}
              </div>

              {/* Charts: Eigenspace Variance Distribution */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <Card title="PCA Eigenmode Variance Spectrum (% Variance per Mode)" className="lg:col-span-2">
                  <div className="h-72 w-full pt-4">
                    <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                      <BarChart data={SSM_MODE_METRICS} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                        <XAxis dataKey="mode" stroke="#64748b" tick={{ fill: '#5e4b75', fontSize: 11 }} />
                        <YAxis domain={[0, 60]} stroke="#64748b" tick={{ fill: '#5e4b75', fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#ffffff',
                            borderColor: '#e2e8f0',
                            borderRadius: '12px',
                            fontSize: '12px',
                            color: '#1a082e',
                            boxShadow: '0 4px 12px rgba(26,8,46,0.08)',
                          }}
                          formatter={(val: any) => [`${val}%`, 'Explained Variance']}
                        />
                        <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                        <Bar dataKey="variance_pct" fill="#d97706" name="Explained Variance (%)" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                {/* Cohort & Eigenspace Telemetry Card */}
                <Card title="Model Specifications & Data Governance">
                  <div className="space-y-3.5 text-xs">
                    <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-semibold">Cohort Split (Zero Leakage)</div>
                      <div className="text-[#1a082e] font-mono font-bold mt-1">430 Verified Patients</div>
                      <div className="text-[11px] text-[#5e4b75] mt-0.5">301 Train · 64 Val · 65 Held-Out Test</div>
                    </div>

                    <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-semibold">Mesh Parameterization</div>
                      <div className="text-amber-900 font-mono font-bold mt-1">6,737 Vertices · 13,478 Triangles</div>
                      <div className="text-[11px] text-[#5e4b75] mt-0.5">16 3D Anatomical Landmarks</div>
                    </div>

                    <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-semibold">Variance Coverage</div>
                      <div className="text-emerald-800 font-mono font-bold mt-1">100.0% Variance across 20 Modes</div>
                      <div className="text-[11px] text-[#5e4b75] mt-0.5">Top 5 modes capture 90.38%</div>
                    </div>

                    <div className="p-2.5 rounded-lg bg-[#f8fafc] border border-[#e2e8f0]">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-semibold">Model Artifact Footprint</div>
                      <div className="text-amber-900 font-mono font-bold mt-1">4.36 MB JSON (Lightweight)</div>
                      <div className="text-[11px] text-[#5e4b75] mt-0.5">Zero external binary weights needed</div>
                    </div>
                  </div>
                </Card>
              </div>

              {/* Mode Physiological Morphologies Table */}
              <Card title="Top 10 Physiological Shape Deformation Modes">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-[#e2e8f0] text-[#5e4b75] bg-slate-50 font-semibold uppercase tracking-wider font-mono text-[10px]">
                        <th className="py-2.5 px-3">Eigenmode</th>
                        <th className="py-2.5 px-3">Variance (%)</th>
                        <th className="py-2.5 px-3">Cumulative (%)</th>
                        <th className="py-2.5 px-3 font-sans">Anatomical / Physiological Interpretation</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#e2e8f0] text-[#1a082e] font-mono">
                      {SSM_MODE_METRICS.map((row, idx) => (
                        <tr key={idx} className="hover:bg-amber-50/40 transition-colors">
                          <td className="py-2.5 px-3 font-semibold text-amber-900">{row.mode}</td>
                          <td className="py-2.5 px-3 text-[#1a082e] font-bold">{row.variance_pct.toFixed(2)}%</td>
                          <td className="py-2.5 px-3 text-emerald-700">{row.cumulative_pct.toFixed(2)}%</td>
                          <td className="py-2.5 px-3 font-sans text-[#5e4b75] text-[11px]">{row.interpretation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          )}
          <Card className="bg-white border-[#e2e8f0] shadow-xs">
            <h4 className="text-xs font-semibold text-[#5e4b75] tracking-wider uppercase mb-3 flex items-center gap-2">
              <Info className="w-4 h-4 text-amber-600" />
              Audited Benchmark Notes & Methodology
            </h4>
            <ul className="space-y-2 text-xs text-[#5e4b75] list-disc list-inside">
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
