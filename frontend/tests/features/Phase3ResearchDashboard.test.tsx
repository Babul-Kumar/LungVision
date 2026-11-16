import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ResearchPage } from '@/features/research/ResearchPage';

// Mock researchApi
vi.mock('@/services/api/researchApi', () => ({
  researchApi: {
    getResearchMetrics: vi.fn(),
  },
}));

import { researchApi } from '@/services/api/researchApi';

const mockResearchData = {
  success: true,
  timestamp: '2026-11-17T00:00:00Z',
  nih_test_metrics: [
    { Disease: 'Cardiomegaly', Threshold: 0.9502, AUROC: 0.88, AUPRC: 0.32, Precision: 0.45, Sensitivity: 0.52, F1: 0.48, Specificity: 0.96 }
  ],
  chexpert_external_metrics: [
    { Disease: 'Cardiomegaly', Threshold: 0.9502, Evaluable: 202, Positive: 66, Negative: 136, AUROC: 0.8028, AUPRC: 0.6588, Precision: 0.60, Sensitivity: 0.091, F1: 0.158, Specificity: 0.971 }
  ],
  vindr_external_metrics: [
    { model: 'Original ResNet18', cohort: '15,000 images', ROC_AUC: 0.6015, Average_Precision: 0.3467, Accuracy: 0.63, Balanced_Accuracy: 0.52, Precision: 0.34, Sensitivity: 0.27, Specificity: 0.78, F1: 0.30, Brier_Score: 0.2585, ECE_10_bins: 0.1984, threshold: 0.5 }
  ],
  active_threshold_set: 'NIH ChestX-ray14 Locked Production Thresholds (Frozen)',
  notes: [
    'Phase 3 external validation encompasses CheXpert, VinDr-CXR, and MIMIC-CXR compliance audits.',
    'Temperature scaling and operating points B and C are research benchmarks; production inference enforces Point A.',
  ],
};

describe('Phase 3 Research Dashboard Feature Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (researchApi.getResearchMetrics as any).mockResolvedValue(mockResearchData);
  });

  it('renders Phase 3 tab and displays completion status and governance badges', async () => {
    render(<ResearchPage />);

    // Click on Phase 3 tab
    const tabBtn = await screen.findByRole('button', { name: /Phase 3 Expansion & Optimization/i });
    fireEvent.click(tabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Phase 3: Multi-Cohort External Validation, Optimization & Calibration/i)).toBeInTheDocument();
    });

    expect(screen.getAllByText(/PHASE 3 COMPLETE/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/EXPANDED EVALUATION SUCCESSFUL/i)).toBeInTheDocument();
    expect(screen.getByText(/Operating Point A \(Locked NIH\)/i)).toBeInTheDocument();
    expect(screen.getByText(/93.9% Sens/i)).toBeInTheDocument();
  });

  it('allows switching between external cohorts and renders MIMIC-CXR compliance notice', async () => {
    render(<ResearchPage />);

    const tabBtn = await screen.findByRole('button', { name: /Phase 3 Expansion & Optimization/i });
    fireEvent.click(tabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Task 1 — Multi-Dataset Cohort Comparison/i)).toBeInTheDocument();
    });

    // Default is Stanford CheXpert
    expect(screen.getByText(/Stanford University Medical Center/i)).toBeInTheDocument();

    // Click on VinDr-CXR
    const vindrBtn = screen.getByRole('button', { name: /VinDr-CXR \(N=15,000\)/i });
    fireEvent.click(vindrBtn);
    expect(screen.getByText(/Hospital 108 & Hanoi Medical University/i)).toBeInTheDocument();

    // Click on MIMIC-CXR
    const mimicBtn = screen.getByRole('button', { name: /MIMIC-CXR \(PhysioNet DUA\)/i });
    fireEvent.click(mimicBtn);
    expect(screen.getByText(/MIMIC-CXR Compliance & PhysioNet Credentialing Requirement/i)).toBeInTheDocument();
  });

  it('renders Operating Points comparison table and toggles between Points A, B, and C', async () => {
    render(<ResearchPage />);

    const tabBtn = await screen.findByRole('button', { name: /Phase 3 Expansion & Optimization/i });
    fireEvent.click(tabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Task 2 — Operating Point Comparison/i)).toBeInTheDocument();
    });

    // Point A cutoffs visible
    expect(screen.getByText('0.9502')).toBeInTheDocument();

    // Click Point B (Youden J)
    const pointBBtn = screen.getByRole('button', { name: /Point B: External Youden J/i });
    fireEvent.click(pointBBtn);
    expect(screen.getAllByText('0.0298').length).toBeGreaterThan(0);

    // Click Point C (High Sensitivity)
    const pointCBtn = screen.getByRole('button', { name: /Point C: High-Sensitivity ≥80%/i });
    fireEvent.click(pointCBtn);
    expect(screen.getAllByText('0.0100').length).toBeGreaterThan(0);
  });

  it('renders Reliability Diagram, Temperature Scaling, Uncertainty Widget, and Resolution Ablations', async () => {
    render(<ResearchPage />);

    const tabBtn = await screen.findByRole('button', { name: /Phase 3 Expansion & Optimization/i });
    fireEvent.click(tabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Task 5 — Reliability Diagram \(10 Calibration Bins\)/i)).toBeInTheDocument();
    });

    // Calibration & Temperature Scaling
    expect(screen.getAllByText(/Post-Hoc Temperature Scaling/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/0.1993 → 0.1915/i)).toBeInTheDocument();

    // Label Uncertainty Sensitivity
    expect(screen.getByText(/Task 3 — CheXpert Label Uncertainty Sensitivity/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Consolidation/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Extreme/i).length).toBeGreaterThan(0);

    // Resolution Scaling Ablation
    expect(screen.getByText(/Task 4 & 6 — Input Resolution Scaling Ablation/i)).toBeInTheDocument();
    expect(screen.getByText(/512×512 \(Modern Target\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Grad-CAM 7×7 Spatial Grid Receptive Field Limit/i)).toBeInTheDocument();
  });
});
