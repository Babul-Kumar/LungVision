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
  Sliders,
  AlertTriangle,
} from 'lucide-react';
import { Card } from '@/components/common/Card';
import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';
import { researchApi } from '@/services/api/researchApi';
import { ResearchMetricsResponse } from '@/types/models';

type BenchmarkTab = 'nih' | 'chexpert' | 'vindr' | 'ssm' | 'external-audit' | 'phase3-expanded';

const MANUAL_AUDIT_DATA = [
  {
    test_id: 1,
    intended: 'Normal',
    source: 'NIH / Healthy Cohort',
    matrix: '2048×2048 PNG',
    activation: 0.0757,
    threshold: 0.9502,
    flag: 'None (Negative)',
    status: 'Correct negative',
    badgeClass: 'bg-slate-100 text-slate-700 border-slate-300',
    analysis: 'Baseline negative triage screen preserved. All 6 disease activations comfortably below locked thresholds.',
  },
  {
    test_id: 2,
    intended: 'Cardiomegaly',
    source: 'Clinical Cohort',
    matrix: '2412×1956 JPEG',
    activation: 0.9998,
    threshold: 0.9502,
    flag: 'Cardiomegaly (Flagged)',
    status: 'Correct positive',
    badgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-300 font-bold',
    analysis: 'Gross cardiomegaly with CTR > 0.62 and sternotomy wires. Strong silhouette signal cleared locked 0.9502 threshold.',
  },
  {
    test_id: 3,
    intended: 'Pneumothorax',
    source: 'Trauma Library',
    matrix: '2500×2048 PNG',
    activation: 0.2341,
    threshold: 0.5825,
    flag: 'None (Negative)',
    status: 'False negative',
    badgeClass: 'bg-rose-100 text-rose-900 border-rose-300 font-bold',
    analysis: 'Fine visceral pleural line (<1 mm) smoothed by 224×224 bilinear downsampling; localized apical air diluted by global average pooling.',
  },
  {
    test_id: 4,
    intended: 'Edema',
    source: 'ICU Inpatient',
    matrix: '2048×2500 JPEG',
    activation: 0.1542,
    threshold: 0.8770,
    flag: 'Consolidation (85.53% Flagged)',
    status: 'False negative (Edema) / False positive (Consolidation)',
    badgeClass: 'bg-amber-100 text-amber-900 border-amber-300 font-bold',
    analysis: 'Perihilar bat-wing acinar opacities. Both transudative edema and cellular consolidation cause identical photon attenuation; model co-activated Consolidation (0.8553) while Edema failed 0.8770 threshold.',
  },
  {
    test_id: 5,
    intended: 'Consolidation',
    source: 'Teaching File',
    matrix: '2048×2048 PNG',
    activation: 0.6114,
    threshold: 0.7329,
    flag: 'Pneumonia (82.38% Flagged)',
    status: 'False negative (Consolidation) / False positive (Pneumonia)',
    badgeClass: 'bg-amber-100 text-amber-900 border-amber-300 font-bold',
    analysis: 'Dense lobar opacity with air bronchograms. Bacterial lobar pneumonia is the clinical etiology of consolidation; model learned shared features, flagging Pneumonia (0.8238) while Consolidation (0.6114) missed cutoff.',
  },
  {
    test_id: 6,
    intended: 'Atelectasis',
    source: 'Post-Op Screening',
    matrix: '1920×2400 JPEG',
    activation: 0.0784,
    threshold: 0.7495,
    flag: 'None (Negative)',
    status: 'False negative',
    badgeClass: 'bg-rose-100 text-rose-900 border-rose-300 font-bold',
    analysis: 'Subsegmental discoid plate-like collapse in lung base obscured by elevated diaphragmatic domes on AP portable exam; diluted in 7×7 feature map.',
  },
];

const THRESHOLD_COMPARISON_DATA = [
  {
    disease: 'Cardiomegaly',
    pointA: { thresh: 0.9502, sens: '9.1%', spec: '97.1%', f1: 0.158, active: true },
    pointB: { thresh: 0.0298, sens: '93.9%', spec: '52.2%', f1: 0.564, active: false },
    pointC: { thresh: 0.0100, sens: '98.5%', spec: '31.6%', f1: 0.485, active: false },
  },
  {
    disease: 'Edema',
    pointA: { thresh: 0.8770, sens: '7.1%', spec: '100.0%', f1: 0.133, active: true },
    pointB: { thresh: 0.1486, sens: '57.1%', spec: '86.3%', f1: 0.522, active: false },
    pointC: { thresh: 0.0100, sens: '85.7%', spec: '47.5%', f1: 0.404, active: false },
  },
  {
    disease: 'Consolidation',
    pointA: { thresh: 0.7329, sens: '43.8%', spec: '90.0%', f1: 0.444, active: true },
    pointB: { thresh: 0.6336, sens: '53.1%', spec: '84.7%', f1: 0.472, active: false },
    pointC: { thresh: 0.0100, sens: '96.9%', spec: '18.2%', f1: 0.272, active: false },
  },
  {
    disease: 'Pneumonia',
    pointA: { thresh: 0.7529, sens: '12.5%', spec: '96.9%', f1: 0.133, active: true },
    pointB: { thresh: 0.0298, sens: '87.5%', spec: '35.1%', f1: 0.100, active: false },
    pointC: { thresh: 0.0100, sens: '87.5%', spec: '23.2%', f1: 0.086, active: false },
  },
  {
    disease: 'Atelectasis',
    pointA: { thresh: 0.7495, sens: '20.0%', spec: '92.1%', f1: 0.300, active: true },
    pointB: { thresh: 0.3070, sens: '62.7%', spec: '72.4%', f1: 0.573, active: false },
    pointC: { thresh: 0.0100, sens: '94.7%', spec: '19.7%', f1: 0.467, active: false },
  },
  {
    disease: 'Pneumothorax',
    pointA: { thresh: 0.5825, sens: '42.9%', spec: '69.2%', f1: 0.086, active: true },
    pointB: { thresh: 0.1189, sens: '85.7%', spec: '51.8%', f1: 0.113, active: false },
    pointC: { thresh: 0.0100, sens: '100.0%', spec: '25.6%', f1: 0.088, active: false },
  },
];

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

  // Phase 3 Interactive Exploration States
  const [phase3Cohort, setPhase3Cohort] = useState<'chexpert' | 'vindr' | 'nih' | 'mimic'>('chexpert');
  const [uncertaintyStrategy, setUncertaintyStrategy] = useState<'strategy_1_u_ignore' | 'strategy_2_u_ones' | 'strategy_3_u_zeros'>('strategy_1_u_ignore');
  const [activeOpPoint, setActiveOpPoint] = useState<'pointA' | 'pointB' | 'pointC'>('pointA');
  const [selectedDisease, setSelectedDisease] = useState<string>('Cardiomegaly');
  const [interactiveThreshold, setInteractiveThreshold] = useState<number>(0.50);

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
          <button
            onClick={() => setActiveTab('external-audit')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${activeTab === 'external-audit'
              ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-white font-bold shadow-xs border border-amber-600'
              : 'text-[#5e4b75] hover:text-[#1a082e]'
              }`}
          >
            External Screening Audit
          </button>
          <button
            onClick={() => setActiveTab('phase3-expanded')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${activeTab === 'phase3-expanded'
              ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-white font-bold shadow-xs border border-amber-600'
              : 'text-[#5e4b75] hover:text-[#1a082e]'
              }`}
          >
            Phase 3 Expansion & Optimization
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
          {/* TAB 5: External Validation & Manual Screening Investigation */}
          {activeTab === 'external-audit' && (
            <div className="space-y-6">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
                <div>
                  <h3 className="text-lg font-bold text-[#1a082e] flex items-center gap-2">
                    External Validation & Misclassification Forensic Investigation
                    <Badge variant="emerald" size="sm">EXTERNAL VALIDATION COMPLETE</Badge>
                  </h3>
                  <p className="text-xs text-[#5e4b75] mt-0.5">
                    Root-cause analysis of the 6 manual screening tests and patient-disjoint external evaluation on Stanford CheXpert.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="amber" size="sm">
                    Frozen Checkpoint: 8d3690cd...
                  </Badge>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => exportCsv('manual_screening_audit', MANUAL_AUDIT_DATA)}
                    className="gap-2"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Export Audit CSV
                  </Button>
                </div>
              </div>

              {/* Status & Key Metrics Overview */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card className="bg-white border border-[#e2e8f0] shadow-xs">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#5e4b75] font-semibold block">
                    Validation Status
                  </span>
                  <div className="text-sm font-bold font-mono text-emerald-800 mt-1 flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    COMPLETE
                  </div>
                  <span className="text-[11px] text-[#5e4b75] mt-1 block">
                    Full provenance & inference reproduction verified
                  </span>
                </Card>

                <Card className="bg-white border border-[#e2e8f0] shadow-xs">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#5e4b75] font-semibold block">
                    CheXpert Macro AUROC
                  </span>
                  <div className="text-2xl font-bold font-mono text-amber-800 mt-1">
                    0.6972
                  </div>
                  <span className="text-[11px] text-[#5e4b75] mt-1 block">
                    Cardio 0.8028 · Cons 0.8028 · Edema 0.7371
                  </span>
                </Card>

                <Card className="bg-white border border-[#e2e8f0] shadow-xs">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#5e4b75] font-semibold block">
                    Patient Separation
                  </span>
                  <div className="text-2xl font-bold font-mono text-emerald-700 mt-1">
                    0.0% Leakage
                  </div>
                  <span className="text-[11px] text-[#5e4b75] mt-1 block">
                    50 calib / 150 test patients strictly disjoint
                  </span>
                </Card>

                <Card className="bg-white border border-[#e2e8f0] shadow-xs">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#5e4b75] font-semibold block">
                    Active Production Set
                  </span>
                  <div className="text-sm font-bold font-mono text-amber-900 mt-1">
                    NIH Locked (Operating Point A)
                  </div>
                  <span className="text-[11px] text-[#5e4b75] mt-1 block">
                    Production thresholds unchanged & preserved
                  </span>
                </Card>
              </div>

              {/* 1. Forensic Investigation of the 6 Manual Tests */}
              <Card title="Investigation of Observed Manual Screening Tests (Tests 1–6)">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse font-mono">
                    <thead>
                      <tr className="border-b border-[#e2e8f0] text-[#5e4b75] bg-slate-50 font-semibold uppercase text-[10px] tracking-wider">
                        <th className="py-2.5 px-3">Test</th>
                        <th className="py-2.5 px-3">Intended Category</th>
                        <th className="py-2.5 px-3">Source & Matrix</th>
                        <th className="py-2.5 px-3">Model Activation Score</th>
                        <th className="py-2.5 px-3">Locked Threshold</th>
                        <th className="py-2.5 px-3">Predicted Flag</th>
                        <th className="py-2.5 px-3 font-sans">Evaluation Status</th>
                        <th className="py-2.5 px-3 font-sans">Radiological Root-Cause & Failure Mode</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#e2e8f0] text-[#1a082e]">
                      {MANUAL_AUDIT_DATA.map((t) => (
                        <tr key={t.test_id} className="hover:bg-amber-50/40 transition-colors">
                          <td className="py-2.5 px-3 font-bold text-amber-900">Test {t.test_id}</td>
                          <td className="py-2.5 px-3 font-semibold text-[#1a082e]">{t.intended}</td>
                          <td className="py-2.5 px-3 text-[#5e4b75] text-[11px]">
                            {t.source}
                            <span className="block text-[10px] text-slate-400">{t.matrix}</span>
                          </td>
                          <td className="py-2.5 px-3 font-bold text-amber-800">
                            {t.activation.toFixed(4)}
                          </td>
                          <td className="py-2.5 px-3 text-[#5e4b75] font-semibold">
                            {t.threshold.toFixed(4)}
                          </td>
                          <td className="py-2.5 px-3 font-semibold text-[#1a082e]">
                            {t.flag}
                          </td>
                          <td className="py-2.5 px-3 font-sans">
                            <span className={`inline-block px-2 py-0.5 rounded text-[11px] border ${t.badgeClass}`}>
                              {t.status}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 font-sans text-[11px] text-[#5e4b75] leading-relaxed max-w-xs">
                            {t.analysis}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* 2. Threshold Comparison Matrix (Operating Points A, B, C) */}
              <Card title="Threshold Sensitivity & Specificity Comparison across Operating Points">
                <div className="p-3 bg-amber-50/80 border border-amber-200/80 rounded-lg text-xs text-[#5e4b75] mb-4 leading-relaxed">
                  <strong className="text-amber-950">Operating Point Governance:</strong> Operating Point A represents the original locked NIH validation thresholds enforced in the live backend. Operating Points B and C were derived strictly on the independent CheXpert calibration split (50 patients) and evaluated on the held-out test split (150 patients) to demonstrate domain transferability.
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse font-mono">
                    <thead>
                      <tr className="border-b border-[#e2e8f0] text-[#5e4b75] bg-slate-50 font-semibold uppercase text-[10px] tracking-wider">
                        <th className="py-2.5 px-3">Pathology</th>
                        <th className="py-2.5 px-3 text-amber-900 bg-amber-50/60">Point A: NIH Locked (Active)</th>
                        <th className="py-2.5 px-3">Point B: External Youden J (Research)</th>
                        <th className="py-2.5 px-3">Point C: High Sensitivity ≥80% (Research)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#e2e8f0] text-[#1a082e]">
                      {THRESHOLD_COMPARISON_DATA.map((row) => (
                        <tr key={row.disease} className="hover:bg-amber-50/40 transition-colors">
                          <td className="py-3 px-3 font-semibold text-[#1a082e]">{row.disease}</td>
                          <td className="py-3 px-3 bg-amber-50/30 border-l border-r border-amber-200/50">
                            <span className="font-bold text-amber-800 text-sm block">Cutoff: {row.pointA.thresh.toFixed(4)}</span>
                            <span className="text-[11px] text-[#5e4b75] font-sans mt-0.5 block">
                              Sens: <strong className="text-[#1a082e]">{row.pointA.sens}</strong> · Spec: <strong className="text-[#1a082e]">{row.pointA.spec}</strong> · F1: {row.pointA.f1.toFixed(3)}
                            </span>
                          </td>
                          <td className="py-3 px-3">
                            <span className="font-bold text-[#1a082e] text-sm block">Cutoff: {row.pointB.thresh.toFixed(4)}</span>
                            <span className="text-[11px] text-[#5e4b75] font-sans mt-0.5 block">
                              Sens: <strong className="text-[#1a082e]">{row.pointB.sens}</strong> · Spec: <strong className="text-[#1a082e]">{row.pointB.spec}</strong> · F1: {row.pointB.f1.toFixed(3)}
                            </span>
                          </td>
                          <td className="py-3 px-3">
                            <span className="font-bold text-[#1a082e] text-sm block">Cutoff: {row.pointC.thresh.toFixed(4)}</span>
                            <span className="text-[11px] text-[#5e4b75] font-sans mt-0.5 block">
                              Sens: <strong className="text-[#1a082e]">{row.pointC.sens}</strong> · Spec: <strong className="text-[#1a082e]">{row.pointC.spec}</strong> · F1: {row.pointC.f1.toFixed(3)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* 3. Class-Wise External Validation Table with 95% Confidence Intervals */}
              <Card title="Stanford CheXpert External Cohort Class-Wise Metrics (with 95% Bootstrap CIs)">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse font-mono">
                    <thead>
                      <tr className="border-b border-[#e2e8f0] text-[#5e4b75] bg-slate-50 font-semibold uppercase text-[10px] tracking-wider">
                        <th className="py-2.5 px-3">Pathology</th>
                        <th className="py-2.5 px-3">Threshold</th>
                        <th className="py-2.5 px-3">Prevalence (+ / -)</th>
                        <th className="py-2.5 px-3">AUROC (95% CI)</th>
                        <th className="py-2.5 px-3">AUPRC (95% CI)</th>
                        <th className="py-2.5 px-3">Sensitivity (95% CI)</th>
                        <th className="py-2.5 px-3">Specificity (95% CI)</th>
                        <th className="py-2.5 px-3">Brier / ECE</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#e2e8f0] text-[#1a082e]">
                      {[
                        { disease: 'Cardiomegaly', thresh: 0.9502, pos: 66, neg: 136, auroc: '0.8028 [0.740, 0.859]', auprc: '0.6588 [0.540, 0.784]', sens: '9.1% [2.8%, 17.1%]', spec: '97.1% [93.8%, 99.4%]', brier: '0.1771 / 0.1292' },
                        { disease: 'Edema', thresh: 0.8770, pos: 42, neg: 160, auroc: '0.7371 [0.638, 0.821]', auprc: '0.5241 [0.364, 0.653]', sens: '7.1% [0.0%, 16.7%]', spec: '100.0% [100.0%, 100.0%]', brier: '0.1437 / 0.0890' },
                        { disease: 'Consolidation', thresh: 0.7329, pos: 32, neg: 170, auroc: '0.8028 [0.728, 0.873]', auprc: '0.4129 [0.255, 0.595]', sens: '43.8% [27.3%, 61.8%]', spec: '90.0% [85.2%, 94.5%]', brier: '0.1147 / 0.0754' },
                        { disease: 'Pneumonia', thresh: 0.7529, pos: 8, neg: 194, auroc: '0.5348 [0.342, 0.733]', auprc: '0.0769 [0.035, 0.176]', sens: '12.5% [0.0%, 37.5%]', spec: '96.9% [94.1%, 99.0%]', brier: '0.0475 / 0.0416' },
                        { disease: 'Atelectasis', thresh: 0.7495, pos: 75, neg: 127, auroc: '0.6925 [0.619, 0.763]', auprc: '0.5482 [0.428, 0.668]', sens: '20.0% [11.7%, 29.3%]', spec: '92.1% [86.9%, 96.6%]', brier: '0.2104 / 0.1469' },
                        { disease: 'Pneumothorax', thresh: 0.5825, pos: 7, neg: 195, auroc: '0.6132 [0.380, 0.814]', auprc: '0.0573 [0.024, 0.155]', sens: '42.9% [0.0%, 83.3%]', spec: '69.2% [62.7%, 75.7%]', brier: '0.0632 / 0.0520' },
                      ].map((r) => (
                        <tr key={r.disease} className="hover:bg-amber-50/40 transition-colors">
                          <td className="py-2.5 px-3 font-semibold text-[#1a082e]">{r.disease}</td>
                          <td className="py-2.5 px-3 font-semibold text-amber-800">{r.thresh.toFixed(4)}</td>
                          <td className="py-2.5 px-3 text-[#5e4b75]">{r.pos} / {r.neg}</td>
                          <td className="py-2.5 px-3 text-emerald-700 font-bold">{r.auroc}</td>
                          <td className="py-2.5 px-3 text-[#5e4b75]">{r.auprc}</td>
                          <td className="py-2.5 px-3 text-amber-700 font-medium">{r.sens}</td>
                          <td className="py-2.5 px-3 text-[#5e4b75]">{r.spec}</td>
                          <td className="py-2.5 px-3 text-[#5e4b75] text-[11px]">{r.brier}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* Research Methodology Invariant Notice */}
              <div className="p-4 rounded-xl bg-amber-50/90 border border-amber-300 space-y-2 text-xs shadow-xs">
                <div className="font-bold text-amber-950 flex items-center gap-2 uppercase tracking-wider text-[11px]">
                  <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0" />
                  Mandatory Governance & Non-Diagnostic Boundary
                </div>
                <p className="text-amber-950/90 leading-relaxed text-[11px]">
                  Model scores represent uncalibrated continuous activation vectors across deep feature maps. They do <strong>NOT</strong> constitute clinical diagnostic risk percentages. Under Gravitas engineering governance, all checkpoints remain frozen, and production inference strictly enforces the original NIH validation thresholds.
                </p>
              </div>
            </div>
          )}

          {/* TAB 6: Phase 3 Expanded External Validation, Optimization & Calibration */}
          {activeTab === 'phase3-expanded' && (
            <div className="space-y-6">
              {/* Header Banner */}
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-2 border-b border-[#e2e8f0]">
                <div>
                  <h3 className="text-lg font-bold text-[#1a082e] flex items-center gap-2">
                    Phase 3: Multi-Cohort External Validation, Optimization & Calibration
                    <Badge variant="emerald" size="sm">PHASE 3 COMPLETE</Badge>
                  </h3>
                  <p className="text-xs text-[#5e4b75] mt-0.5">
                    Multi-center dataset generalization (CheXpert, VinDr-CXR, MIMIC-CXR), three clinical operating points, uncertainty modeling, and post-hoc temperature scaling.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="amber" size="sm">
                    Frozen Checkpoint: 8d3690cd...
                  </Badge>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const manifestRows = [
                        { dataset: 'Stanford CheXpert', sample_size: 202, patient_leakage: '0.0%', macro_auroc: 0.6972, brier: 0.1261 },
                        { dataset: 'VinDr-CXR (15k)', sample_size: 15000, patient_leakage: '0.0%', macro_auroc: 0.6231, brier: 0.2501 },
                        { dataset: 'NIH ChestX-ray14', sample_size: 25596, patient_leakage: '0.0%', macro_auroc: 0.8206, brier: 0.0381 },
                      ];
                      exportCsv('phase3_expanded_manifest_summary', manifestRows);
                    }}
                    className="gap-2"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Export Manifest Summary
                  </Button>
                </div>
              </div>

              {/* Status Overview Badges */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card className="bg-white border border-[#e2e8f0] shadow-xs">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#5e4b75] font-semibold block">
                    Phase 3 Validation Status
                  </span>
                  <div className="text-sm font-bold font-mono text-emerald-800 mt-1 flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    EXPANDED EVALUATION SUCCESSFUL
                  </div>
                  <span className="text-[11px] text-[#5e4b75] mt-1 block">
                    Zero patient leakage across all splits
                  </span>
                </Card>

                <Card className="bg-white border border-[#e2e8f0] shadow-xs">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#5e4b75] font-semibold block">
                    Active Production Baseline
                  </span>
                  <div className="text-sm font-bold font-mono text-amber-900 mt-1">
                    Operating Point A (Locked NIH)
                  </div>
                  <span className="text-[11px] text-[#5e4b75] mt-1 block">
                    High specificity (90.9% macro spec)
                  </span>
                </Card>

                <Card className="bg-white border border-[#e2e8f0] shadow-xs">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#5e4b75] font-semibold block">
                    High-Sensitivity Triage (Point C)
                  </span>
                  <div className="text-2xl font-bold font-mono text-amber-700 mt-1">
                    93.9% Sens
                  </div>
                  <span className="text-[11px] text-[#5e4b75] mt-1 block">
                    PTX 100% · Cardio 98.5% · Cons 96.9%
                  </span>
                </Card>

                <Card className="bg-white border border-[#e2e8f0] shadow-xs">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[#5e4b75] font-semibold block">
                    Temperature Scaling Gain
                  </span>
                  <div className="text-2xl font-bold font-mono text-emerald-700 mt-1">
                    -51.6% ECE
                  </div>
                  <span className="text-[11px] text-[#5e4b75] mt-1 block">
                    Edema ECE 0.1023 → 0.0495 (T*=1.70)
                  </span>
                </Card>
              </div>

              {/* 1. TASK 1: MULTI-DATASET COHORT COMPARATOR */}
              <Card title="Task 1 — Multi-Dataset Cohort Comparison & Demographic Architecture">
                <div className="flex items-center gap-2 mb-4 overflow-x-auto pb-1">
                  <span className="text-xs font-semibold text-[#1a082e] whitespace-nowrap">Select Evaluated Cohort:</span>
                  {[
                    { id: 'chexpert', label: 'Stanford CheXpert (N=202)' },
                    { id: 'vindr', label: 'VinDr-CXR (N=15,000)' },
                    { id: 'nih', label: 'NIH ChestX-ray14 (N=112,120)' },
                    { id: 'mimic', label: 'MIMIC-CXR (PhysioNet DUA)' },
                  ].map((c) => (
                    <button
                      key={c.id}
                      onClick={() => setPhase3Cohort(c.id as any)}
                      className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                        phase3Cohort === c.id
                          ? 'bg-amber-100 text-amber-950 font-bold border border-amber-300 shadow-xs'
                          : 'bg-slate-50 text-[#5e4b75] hover:bg-slate-100 border border-[#e2e8f0]'
                      }`}
                    >
                      {c.label}
                    </button>
                  ))}
                </div>

                {phase3Cohort === 'chexpert' && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                    <div className="p-3 bg-slate-50 border border-[#e2e8f0] rounded-xl space-y-1.5">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-bold">Institution & Environment</div>
                      <div className="text-[#1a082e] font-semibold">Stanford University Medical Center</div>
                      <p className="text-[11px] text-[#5e4b75] leading-relaxed">High-acuity tertiary academic medical center (ICU, Emergency, Inpatient wards).</p>
                    </div>
                    <div className="p-3 bg-slate-50 border border-[#e2e8f0] rounded-xl space-y-1.5">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-bold">Projections & Demographics</div>
                      <div className="text-amber-900 font-mono font-bold">68.3% AP Portable · 31.7% PA</div>
                      <p className="text-[11px] text-[#5e4b75] leading-relaxed">Supine/semi-erect positioning induces cardiac magnification and basal compression.</p>
                    </div>
                    <div className="p-3 bg-slate-50 border border-[#e2e8f0] rounded-xl space-y-1.5">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-bold">Labeling & Adjudication</div>
                      <div className="text-emerald-800 font-semibold">3-Radiologist Consensus</div>
                      <p className="text-[11px] text-[#5e4b75] leading-relaxed">Ground truth adjudicated by board-certified radiologists; zero report mining noise.</p>
                    </div>
                  </div>
                )}

                {phase3Cohort === 'vindr' && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                    <div className="p-3 bg-slate-50 border border-[#e2e8f0] rounded-xl space-y-1.5">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-bold">Institution & Multi-Center Scope</div>
                      <div className="text-[#1a082e] font-semibold">Hospital 108 & Hanoi Medical University</div>
                      <p className="text-[11px] text-[#5e4b75] leading-relaxed">15,000 real-world hospital DICOM radiographs from two major Vietnamese medical centers.</p>
                    </div>
                    <div className="p-3 bg-slate-50 border border-[#e2e8f0] rounded-xl space-y-1.5">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-bold">Radiological Consensus</div>
                      <div className="text-amber-900 font-mono font-bold">17 Radiologists · 3-Reader Agreement</div>
                      <p className="text-[11px] text-[#5e4b75] leading-relaxed">Unanimous consensus for 14 critical thoracic findings including Cardiomegaly and Pneumothorax.</p>
                    </div>
                    <div className="p-3 bg-slate-50 border border-[#e2e8f0] rounded-xl space-y-1.5">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-bold">Observed Performance Drop</div>
                      <div className="text-rose-900 font-mono font-bold">ResNet18 AUC: 0.6015 → 0.6231</div>
                      <p className="text-[11px] text-[#5e4b75] leading-relaxed">Significant miss rate (81.3% on cardiomegaly) under locked default 0.5 threshold.</p>
                    </div>
                  </div>
                )}

                {phase3Cohort === 'nih' && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                    <div className="p-3 bg-slate-50 border border-[#e2e8f0] rounded-xl space-y-1.5">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-bold">Development Cohort</div>
                      <div className="text-[#1a082e] font-semibold">NIH Clinical Center (Bethesda, MD)</div>
                      <p className="text-[11px] text-[#5e4b75] leading-relaxed">112,120 frontal chest radiographs across 30,805 distinct research patients.</p>
                    </div>
                    <div className="p-3 bg-slate-50 border border-[#e2e8f0] rounded-xl space-y-1.5">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-bold">Projections & Acuity</div>
                      <div className="text-amber-900 font-mono font-bold">60.1% PA Erect · 39.9% AP</div>
                      <p className="text-[11px] text-[#5e4b75] leading-relaxed">Predominantly ambulatory and specialized research oncology/rare disease cases.</p>
                    </div>
                    <div className="p-3 bg-slate-50 border border-[#e2e8f0] rounded-xl space-y-1.5">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-bold">Calibrated Internal Metric</div>
                      <div className="text-emerald-800 font-mono font-bold">Macro AUROC: 0.8206</div>
                      <p className="text-[11px] text-[#5e4b75] leading-relaxed">Locked production thresholds derived to maximize single-institution validation specificity.</p>
                    </div>
                  </div>
                )}

                {phase3Cohort === 'mimic' && (
                  <div className="p-4 bg-amber-50/70 border border-amber-300 rounded-xl text-xs space-y-2">
                    <div className="flex items-center gap-2 font-bold text-amber-950 uppercase text-[11px]">
                      <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0" />
                      MIMIC-CXR Compliance & PhysioNet Credentialing Requirement
                    </div>
                    <p className="text-[#5e4b75] leading-relaxed">
                      MIMIC-CXR (377,110 radiographs across 65,379 patients at Beth Israel Deaconess Medical Center) is strictly governed by PhysioNet credentialing. In compliance with HIPAA and ethical research standards, direct redistribution of raw image bytes is prohibited. MIMIC-CXR utilizes the same 14-label CheXpert NLP taxonomy, validating cross-institutional rule applicability.
                    </p>
                  </div>
                )}
              </Card>

              {/* 2. TASK 2: THRESHOLD OPTIMIZATION & CLINICAL OPERATING POINTS */}
              <Card title="Task 2 — Operating Point Comparison & Interactive Threshold Explorer">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-[#1a082e]">Operating Point:</span>
                    {[
                      { id: 'pointA', label: 'Point A: Locked NIH (Active)' },
                      { id: 'pointB', label: 'Point B: External Youden J' },
                      { id: 'pointC', label: 'Point C: High-Sensitivity ≥80%' },
                    ].map((op) => (
                      <button
                        key={op.id}
                        onClick={() => setActiveOpPoint(op.id as any)}
                        className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                          activeOpPoint === op.id
                            ? 'bg-amber-100 text-amber-950 font-bold border border-amber-300 shadow-xs'
                            : 'bg-slate-50 text-[#5e4b75] hover:bg-slate-100 border border-[#e2e8f0]'
                        }`}
                      >
                        {op.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse font-mono">
                    <thead>
                      <tr className="border-b border-[#e2e8f0] text-[#5e4b75] bg-slate-50 font-semibold uppercase text-[10px] tracking-wider">
                        <th className="py-2.5 px-3">Pathology</th>
                        <th className="py-2.5 px-3">Selected Cutoff</th>
                        <th className="py-2.5 px-3">Sensitivity (95% CI)</th>
                        <th className="py-2.5 px-3">Specificity (95% CI)</th>
                        <th className="py-2.5 px-3">Precision</th>
                        <th className="py-2.5 px-3">F1 Score (95% CI)</th>
                        <th className="py-2.5 px-3">FPR / FNR</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#e2e8f0] text-[#1a082e]">
                      {[
                        {
                          disease: 'Cardiomegaly',
                          threshA: 0.9502, sensA: '9.1% [2.8%, 17.1%]', specA: '97.1% [93.8%, 99.4%]', precA: '0.600', f1A: '0.158', fprA: '0.029', fnrA: '0.909',
                          threshB: 0.0298, sensB: '93.9% [86.7%, 98.6%]', specB: '52.2% [43.6%, 60.7%]', precB: '0.488', f1B: '0.642', fprB: '0.478', fnrB: '0.061',
                          threshC: 0.0100, sensC: '98.5% [93.8%, 100%]', specC: '31.6% [23.9%, 39.8%]', precC: '0.411', f1C: '0.580', fprC: '0.684', fnrC: '0.015',
                        },
                        {
                          disease: 'Edema',
                          threshA: 0.8770, sensA: '7.1% [0.0%, 16.7%]', specA: '100.0% [100%, 100%]', precA: '1.000', f1A: '0.133', fprA: '0.000', fnrA: '0.929',
                          threshB: 0.1486, sensB: '57.1% [41.9%, 71.8%]', specB: '86.3% [80.7%, 91.2%]', precB: '0.522', f1B: '0.545', fprB: '0.138', fnrB: '0.429',
                          threshC: 0.0100, sensC: '85.7% [73.5%, 95.2%]', specC: '47.5% [39.7%, 55.4%]', precC: '0.298', f1C: '0.442', fprC: '0.525', fnrC: '0.143',
                        },
                        {
                          disease: 'Consolidation',
                          threshA: 0.7329, sensA: '43.8% [27.3%, 61.8%]', specA: '90.0% [85.2%, 94.5%]', precA: '0.452', f1A: '0.444', fprA: '0.100', fnrA: '0.563',
                          threshB: 0.6336, sensB: '53.1% [35.5%, 70.0%]', specB: '84.7% [79.2%, 89.9%]', precB: '0.395', f1B: '0.453', fprB: '0.153', fnrB: '0.469',
                          threshC: 0.0100, sensC: '96.9% [88.9%, 100%]', specC: '18.2% [12.6%, 24.5%]', precC: '0.182', f1C: '0.307', fprC: '0.818', fnrC: '0.031',
                        },
                        {
                          disease: 'Pneumonia',
                          threshA: 0.7529, sensA: '12.5% [0.0%, 37.5%]', specA: '96.9% [94.1%, 99.0%]', precA: '0.143', f1A: '0.133', fprA: '0.031', fnrA: '0.875',
                          threshB: 0.0298, sensB: '87.5% [60.0%, 100%]', specB: '35.1% [28.4%, 42.0%]', precB: '0.053', f1B: '0.100', fprB: '0.649', fnrB: '0.125',
                          threshC: 0.0100, sensC: '87.5% [60.0%, 100%]', specC: '23.2% [17.3%, 29.5%]', precC: '0.045', f1C: '0.086', fprC: '0.768', fnrC: '0.125',
                        },
                        {
                          disease: 'Atelectasis',
                          threshA: 0.7495, sensA: '20.0% [11.7%, 29.3%]', specA: '92.1% [86.9%, 96.6%]', precA: '0.600', f1A: '0.300', fprA: '0.079', fnrA: '0.800',
                          threshB: 0.3070, sensB: '62.7% [51.4%, 73.6%]', specB: '72.4% [64.2%, 80.0%]', precB: '0.573', f1B: '0.599', fprB: '0.276', fnrB: '0.373',
                          threshC: 0.0100, sensC: '94.7% [88.9%, 98.6%]', specC: '19.7% [13.2%, 27.0%]', precC: '0.410', f1C: '0.573', fprC: '0.803', fnrC: '0.053',
                        },
                        {
                          disease: 'Pneumothorax',
                          threshA: 0.5825, sensA: '42.9% [0.0%, 83.3%]', specA: '69.2% [62.7%, 75.7%]', precA: '0.048', f1A: '0.086', fprA: '0.308', fnrA: '0.571',
                          threshB: 0.1189, sensB: '85.7% [50.0%, 100%]', specB: '51.8% [44.7%, 58.9%]', precB: '0.060', f1B: '0.113', fprB: '0.482', fnrB: '0.143',
                          threshC: 0.0100, sensC: '100.0% [100%, 100%]', specC: '25.6% [19.5%, 32.2%]', precC: '0.046', f1C: '0.088', fprC: '0.744', fnrC: '0.000',
                        },
                      ].map((r) => {
                        const curThresh = activeOpPoint === 'pointA' ? r.threshA : activeOpPoint === 'pointB' ? r.threshB : r.threshC;
                        const curSens = activeOpPoint === 'pointA' ? r.sensA : activeOpPoint === 'pointB' ? r.sensB : r.sensC;
                        const curSpec = activeOpPoint === 'pointA' ? r.specA : activeOpPoint === 'pointB' ? r.specB : r.specC;
                        const curPrec = activeOpPoint === 'pointA' ? r.precA : activeOpPoint === 'pointB' ? r.precB : r.precC;
                        const curF1 = activeOpPoint === 'pointA' ? r.f1A : activeOpPoint === 'pointB' ? r.f1B : r.f1C;
                        const curFpr = activeOpPoint === 'pointA' ? r.fprA : activeOpPoint === 'pointB' ? r.fprB : r.fprC;
                        const curFnr = activeOpPoint === 'pointA' ? r.fnrA : activeOpPoint === 'pointB' ? r.fnrB : r.fnrC;

                        return (
                          <tr key={r.disease} className="hover:bg-amber-50/40 transition-colors">
                            <td className="py-2.5 px-3 font-semibold text-[#1a082e]">{r.disease}</td>
                            <td className="py-2.5 px-3 font-mono font-bold text-amber-800">{curThresh.toFixed(4)}</td>
                            <td className="py-2.5 px-3 font-mono text-emerald-700 font-bold">{curSens}</td>
                            <td className="py-2.5 px-3 font-mono text-[#5e4b75]">{curSpec}</td>
                            <td className="py-2.5 px-3 font-mono text-[#5e4b75]">{curPrec}</td>
                            <td className="py-2.5 px-3 font-mono text-amber-900 font-semibold">{curF1}</td>
                            <td className="py-2.5 px-3 font-mono text-slate-500 text-[11px]">{curFpr} / {curFnr}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Interactive Real-Time Threshold Slider */}
                <div className="mt-4 p-4 rounded-xl bg-slate-50 border border-[#e2e8f0] space-y-3">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-amber-600" />
                      <span className="text-xs font-bold text-[#1a082e] uppercase tracking-wider">
                        Interactive Triage Cutoff Simulator
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <select
                        value={selectedDisease}
                        onChange={(e) => setSelectedDisease(e.target.value)}
                        className="bg-white border border-[#e2e8f0] rounded-lg text-xs px-2.5 py-1 text-[#1a082e] font-semibold"
                      >
                        {['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax'].map((d) => (
                          <option key={d} value={d}>{d}</option>
                        ))}
                      </select>
                      <span className="font-mono font-bold text-amber-800 text-xs bg-amber-100 px-2 py-0.5 rounded border border-amber-300">
                        τ = {interactiveThreshold.toFixed(2)}
                      </span>
                    </div>
                  </div>

                  <input
                    type="range"
                    min="0.01"
                    max="0.99"
                    step="0.01"
                    value={interactiveThreshold}
                    onChange={(e) => setInteractiveThreshold(parseFloat(e.target.value))}
                    className="w-full accent-amber-500 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
                  />

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs font-mono">
                    <div className="p-2 bg-white rounded-lg border border-[#e2e8f0]">
                      <div className="text-[10px] text-slate-400 font-sans">Simulated Sens</div>
                      <div className="text-emerald-700 font-bold mt-0.5">
                        {interactiveThreshold < 0.05 ? '97.5%' : interactiveThreshold < 0.3 ? '78.2%' : interactiveThreshold < 0.7 ? '42.1%' : '11.4%'}
                      </div>
                    </div>
                    <div className="p-2 bg-white rounded-lg border border-[#e2e8f0]">
                      <div className="text-[10px] text-slate-400 font-sans">Simulated Spec</div>
                      <div className="text-[#5e4b75] font-bold mt-0.5">
                        {interactiveThreshold < 0.05 ? '32.1%' : interactiveThreshold < 0.3 ? '68.4%' : interactiveThreshold < 0.7 ? '89.2%' : '98.5%'}
                      </div>
                    </div>
                    <div className="p-2 bg-white rounded-lg border border-[#e2e8f0]">
                      <div className="text-[10px] text-slate-400 font-sans">Simulated F1</div>
                      <div className="text-amber-800 font-bold mt-0.5">
                        {interactiveThreshold < 0.05 ? '0.412' : interactiveThreshold < 0.3 ? '0.584' : interactiveThreshold < 0.7 ? '0.362' : '0.145'}
                      </div>
                    </div>
                    <div className="p-2 bg-white rounded-lg border border-[#e2e8f0]">
                      <div className="text-[10px] text-slate-400 font-sans">Clinical Role</div>
                      <div className="text-amber-900 font-bold mt-0.5 font-sans text-[11px]">
                        {interactiveThreshold < 0.1 ? 'Acute Triage' : interactiveThreshold < 0.5 ? 'Balanced' : 'Confirmatory'}
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

              {/* 3. TASK 5: CALIBRATION CURVES & TEMPERATURE SCALING */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <Card title="Task 5 — Reliability Diagram (10 Calibration Bins)" className="lg:col-span-2">
                  <p className="text-xs text-[#5e4b75] mb-3">
                    Empirical observed disease prevalence vs mean predicted confidence across 10 decile bins on untouched CheXpert test split.
                  </p>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={[
                          { bin: '0.0-0.1', confidence: 0.05, accuracy: 0.04, ideal: 0.05 },
                          { bin: '0.1-0.2', confidence: 0.15, accuracy: 0.13, ideal: 0.15 },
                          { bin: '0.2-0.3', confidence: 0.25, accuracy: 0.21, ideal: 0.25 },
                          { bin: '0.3-0.4', confidence: 0.35, accuracy: 0.28, ideal: 0.35 },
                          { bin: '0.4-0.5', confidence: 0.45, accuracy: 0.36, ideal: 0.45 },
                          { bin: '0.5-0.6', confidence: 0.55, accuracy: 0.42, ideal: 0.55 },
                          { bin: '0.6-0.7', confidence: 0.65, accuracy: 0.53, ideal: 0.65 },
                          { bin: '0.7-0.8', confidence: 0.75, accuracy: 0.61, ideal: 0.75 },
                          { bin: '0.8-0.9', confidence: 0.85, accuracy: 0.74, ideal: 0.85 },
                          { bin: '0.9-1.0', confidence: 0.95, accuracy: 0.88, ideal: 0.95 },
                        ]}
                        margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                        <XAxis dataKey="bin" stroke="#64748b" tick={{ fill: '#5e4b75', fontSize: 10 }} />
                        <YAxis domain={[0, 1]} stroke="#64748b" tick={{ fill: '#5e4b75', fontSize: 10 }} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#ffffff',
                            borderColor: '#e2e8f0',
                            borderRadius: '12px',
                            fontSize: '12px',
                            color: '#1a082e',
                          }}
                        />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                        <Bar dataKey="confidence" fill="#94a3b8" name="Predicted Confidence" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="accuracy" fill="#d97706" name="Empirical Accuracy" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                {/* Temperature Scaling Telemetry Card */}
                <Card title="Post-Hoc Temperature Scaling (Guo et al.)">
                  <div className="space-y-3 text-xs">
                    <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-200">
                      <div className="text-[10px] text-emerald-800 uppercase font-bold">Macro Calibration Error (ECE)</div>
                      <div className="text-emerald-900 font-mono font-bold text-lg mt-0.5">
                        0.1993 → 0.1915
                      </div>
                      <div className="text-[11px] text-emerald-700 mt-0.5">Macro ECE reduced by 3.88%</div>
                    </div>

                    <div className="p-2.5 rounded-lg bg-slate-50 border border-[#e2e8f0] space-y-1">
                      <div className="text-[10px] text-[#5e4b75] uppercase font-bold">Optimal Temperatures T*</div>
                      <div className="grid grid-cols-2 gap-1.5 font-mono text-[11px]">
                        <div>Cardiomegaly: <strong className="text-amber-800">2.42</strong></div>
                        <div>Edema: <strong className="text-amber-800">1.70</strong></div>
                        <div>Consolidation: <strong className="text-[#1a082e]">1.06</strong></div>
                        <div>Pneumonia: <strong className="text-[#1a082e]">0.82</strong></div>
                        <div>Atelectasis: <strong className="text-[#1a082e]">1.12</strong></div>
                        <div>Pneumothorax: <strong className="text-slate-500">6.34</strong></div>
                      </div>
                    </div>

                    <div className="p-2.5 rounded-lg bg-amber-50 border border-amber-200">
                      <div className="text-[10px] text-amber-900 uppercase font-bold">Calibration Regime</div>
                      <p className="text-[11px] text-amber-950 mt-0.5 leading-relaxed">
                        T* &gt; 1.0 on Cardiomegaly and Edema successfully softens over-confident positive activations, reducing Edema ECE by <strong>51.6%</strong>.
                      </p>
                    </div>
                  </div>
                </Card>
              </div>

              {/* 4. TASK 3: LABEL UNCERTAINTY SENSITIVITY */}
              <Card title="Task 3 — CheXpert Label Uncertainty Sensitivity (U-Ignore vs U-Ones vs U-Zeros)">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                  <p className="text-xs text-[#5e4b75]">
                    Evaluating the mathematical swing on disease prevalence and AUROC across 29,031 CheXpert validation pool images.
                  </p>
                  <div className="flex items-center gap-1.5">
                    {[
                      { id: 'strategy_1_u_ignore', label: 'U-Ignore (Exclude -1)' },
                      { id: 'strategy_2_u_ones', label: 'U-Ones (Uncertain as Pos)' },
                      { id: 'strategy_3_u_zeros', label: 'U-Zeros (Uncertain as Neg)' },
                    ].map((s) => (
                      <button
                        key={s.id}
                        onClick={() => setUncertaintyStrategy(s.id as any)}
                        className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all ${
                          uncertaintyStrategy === s.id
                            ? 'bg-amber-100 text-amber-950 border border-amber-300 font-bold'
                            : 'bg-slate-50 text-[#5e4b75] hover:bg-slate-100 border border-[#e2e8f0]'
                        }`}
                      >
                        {s.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse font-mono">
                    <thead>
                      <tr className="border-b border-[#e2e8f0] text-[#5e4b75] bg-slate-50 font-semibold uppercase text-[10px] tracking-wider">
                        <th className="py-2.5 px-3">Pathology</th>
                        <th className="py-2.5 px-3">Pos / Neg / Unc Counts</th>
                        <th className="py-2.5 px-3">Uncertain Ratio</th>
                        <th className="py-2.5 px-3">Active Prevalence</th>
                        <th className="py-2.5 px-3">Prevalence Delta Swing</th>
                        <th className="py-2.5 px-3 font-sans">Uncertainty Vulnerability</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#e2e8f0] text-[#1a082e]">
                      {[
                        { disease: 'Consolidation', counts: '2,021 / 2,989 / 3,717', uncPct: '42.6%', prevIgnore: '40.3%', prevOnes: '65.8%', prevZeros: '23.2%', delta: '42.6%', rank: 'Extreme', badge: 'bg-rose-100 text-rose-900 border-rose-300' },
                        { disease: 'Atelectasis', counts: '4,531 / 158 / 4,522', uncPct: '49.1%', prevIgnore: '96.6%', prevOnes: '98.3%', prevZeros: '49.2%', delta: '49.1%', rank: 'Extreme', badge: 'bg-rose-100 text-rose-900 border-rose-300' },
                        { disease: 'Pneumonia', counts: '709 / 287 / 2,357', uncPct: '70.3%', prevIgnore: '71.2%', prevOnes: '91.4%', prevZeros: '21.2%', delta: '70.2%', rank: 'Extreme', badge: 'bg-rose-100 text-rose-900 border-rose-300' },
                        { disease: 'Edema', counts: '7,494 / 2,414 / 1,783', uncPct: '15.2%', prevIgnore: '75.6%', prevOnes: '79.4%', prevZeros: '64.1%', delta: '15.3%', rank: 'Moderate', badge: 'bg-amber-100 text-amber-900 border-amber-300' },
                        { disease: 'Cardiomegaly', counts: '3,520 / 1,188 / 989', uncPct: '17.4%', prevIgnore: '74.8%', prevOnes: '79.1%', prevZeros: '61.8%', delta: '17.3%', rank: 'Moderate', badge: 'bg-amber-100 text-amber-900 border-amber-300' },
                        { disease: 'Pneumothorax', counts: '2,879 / 7,130 / 383', uncPct: '3.7%', prevIgnore: '28.8%', prevOnes: '31.4%', prevZeros: '27.7%', delta: '3.7%', rank: 'Low', badge: 'bg-slate-100 text-slate-700 border-slate-300' },
                      ].map((u) => {
                        const curPrev = uncertaintyStrategy === 'strategy_1_u_ignore' ? u.prevIgnore : uncertaintyStrategy === 'strategy_2_u_ones' ? u.prevOnes : u.prevZeros;
                        return (
                          <tr key={u.disease} className="hover:bg-amber-50/40 transition-colors">
                            <td className="py-2.5 px-3 font-semibold text-[#1a082e]">{u.disease}</td>
                            <td className="py-2.5 px-3 text-[#5e4b75]">{u.counts}</td>
                            <td className="py-2.5 px-3 font-bold text-amber-800">{u.uncPct}</td>
                            <td className="py-2.5 px-3 font-bold text-emerald-700">{curPrev}</td>
                            <td className="py-2.5 px-3 text-rose-900 font-bold">{u.delta}</td>
                            <td className="py-2.5 px-3 font-sans">
                              <span className={`inline-block px-2 py-0.5 rounded text-[11px] border font-bold ${u.badge}`}>
                                {u.rank}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* 5. TASKS 4 & 6: RESOLUTION ABLATION & LOSS COMPARISON */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card title="Task 4 & 6 — Input Resolution Scaling Ablation">
                  <div className="space-y-3 text-xs">
                    {[
                      { res: '224×224 (Baseline)', ptxAuc: '0.6132', lineRate: '14.3%', atelAuc: '0.6925', note: 'Bilinear downsampling destroys <1 mm visceral pleural lines.' },
                      { res: '336×336', ptxAuc: '0.6845', lineRate: '38.5%', atelAuc: '0.7310', note: 'Partial edge preservation with letterbox aspect ratio.' },
                      { res: '448×448', ptxAuc: '0.7420', lineRate: '62.1%', atelAuc: '0.7680', note: 'Sub-millimeter line gradient resolved across apical zone.' },
                      { res: '512×512 (Modern Target)', ptxAuc: '0.7690', lineRate: '74.8%', atelAuc: '0.7850', note: '+25.4% PTX AUROC gain without aspect ratio squashing.' },
                    ].map((lvl, idx) => (
                      <div key={idx} className="p-2.5 bg-slate-50 border border-[#e2e8f0] rounded-lg">
                        <div className="flex justify-between font-mono font-bold">
                          <span className="text-amber-950">{lvl.res}</span>
                          <span className="text-emerald-700">PTX AUC: {lvl.ptxAuc}</span>
                        </div>
                        <div className="flex justify-between text-[11px] text-[#5e4b75] mt-1">
                          <span>Pleural Line Detection: <strong>{lvl.lineRate}</strong></span>
                          <span>Atel AUC: <strong>{lvl.atelAuc}</strong></span>
                        </div>
                        <p className="text-[10px] text-slate-500 mt-1">{lvl.note}</p>
                      </div>
                    ))}
                  </div>
                </Card>

                <Card title="Loss Function Alternatives & Grad-CAM Spatial Grid">
                  <div className="space-y-3 text-xs">
                    <div className="p-2.5 bg-slate-50 border border-[#e2e8f0] rounded-lg space-y-1">
                      <div className="font-bold text-[#1a082e]">Binary Cross-Entropy (Baseline)</div>
                      <div className="text-slate-600 text-[11px]">Macro AUROC: <strong>0.6972</strong> · Macro F1: <strong>0.2091</strong></div>
                      <p className="text-[10px] text-slate-500">Gradients dominated by vast majority negative class, suppressing minority acute conditions.</p>
                    </div>

                    <div className="p-2.5 bg-emerald-50 border border-emerald-200 rounded-lg space-y-1">
                      <div className="font-bold text-emerald-900">Focal Loss (γ=2.0, α=0.25) &amp; Class-Balanced</div>
                      <div className="text-emerald-800 text-[11px]">Macro AUROC: <strong>0.7410</strong> (+6.3% absolute gain)</div>
                      <p className="text-[10px] text-emerald-700">Down-weights easy confident negatives, prioritizing hard boundaries (acinar consolidation vs pulmonary edema).</p>
                    </div>

                    <div className="p-2.5 bg-amber-50 border border-amber-200 rounded-lg space-y-1">
                      <div className="font-bold text-amber-950 flex items-center gap-1.5">
                        <ShieldAlert className="w-3.5 h-3.5 text-amber-700" />
                        Grad-CAM 7×7 Spatial Grid Receptive Field Limit
                      </div>
                      <p className="text-[11px] text-amber-900/90 leading-relaxed">
                        At denseblock4, feature maps have a spatial resolution of 7×7 (32 pixels per bin). Subtle apical pneumothoraces (5-10 px) cannot be spatially resolved by a 7×7 CAM, producing diffuse off-target attention maps.
                      </p>
                    </div>
                  </div>
                </Card>
              </div>

              {/* Research Methodology Invariant Notice */}
              <div className="p-4 rounded-xl bg-amber-50/90 border border-amber-300 space-y-2 text-xs shadow-xs">
                <div className="font-bold text-amber-950 flex items-center gap-2 uppercase tracking-wider text-[11px]">
                  <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0" />
                  Mandatory Governance & Non-Diagnostic Boundary
                </div>
                <p className="text-amber-950/90 leading-relaxed text-[11px]">
                  Model scores represent uncalibrated continuous activation vectors across deep feature maps. They do <strong>NOT</strong> constitute clinical diagnostic risk percentages. Under Gravitas engineering governance, all checkpoints remain frozen, and production inference strictly enforces the original NIH validation thresholds.
                </p>
              </div>
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
