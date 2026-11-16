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
    { Disease: 'Cardiomegaly', Threshold: 0.9502, Evaluable: 202, Positive: 66, Negative: 136, AUROC: 0.8028, AUPRC: 0.6588, Precision: 0.45, Sensitivity: 0.50, F1: 0.47, Specificity: 0.85 }
  ],
  vindr_external_metrics: [],
  notes: [
    'Phase 4 high-resolution multi-scale FPN achieves 0.7690 Pneumothorax AUROC (+25.4% gain over 224x224 baseline).'
  ]
};

describe('Phase 4 Research Comparison Feature Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (researchApi.getResearchMetrics as any).mockResolvedValue(mockResearchData);
  });

  it('renders Phase 4 tab button and navigates to Multi-Scale Model comparison view', async () => {
    render(<ResearchPage />);

    await waitFor(() => {
      expect(screen.getByText('Phase 4: Multi-Scale Model')).toBeInTheDocument();
    });

    const phase4TabBtn = screen.getByText('Phase 4: Multi-Scale Model');
    fireEvent.click(phase4TabBtn);

    expect(screen.getByText('High-Resolution Multi-Scale Architectural Improvement')).toBeInTheDocument();
    expect(screen.getByText(/PHASE 4 COMPLETE — RESEARCH MODEL IMPROVEMENT REPRODUCED/i)).toBeInTheDocument();
  });

  it('displays Pneumothorax AUROC reproduction from 0.6132 to 0.7690', async () => {
    render(<ResearchPage />);

    await waitFor(() => {
      expect(screen.getByText('Phase 4: Multi-Scale Model')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Phase 4: Multi-Scale Model'));

    expect(screen.getByText('Pneumothorax AUROC Reproduction')).toBeInTheDocument();
    expect(screen.getAllByText('0.6132').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('0.7690').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/\+25\.4%/i).length).toBeGreaterThanOrEqual(1);
  });

  it('allows switching resolution ablation levels and renders receptive field steps', async () => {
    render(<ResearchPage />);

    await waitFor(() => {
      expect(screen.getByText('Phase 4: Multi-Scale Model')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Phase 4: Multi-Scale Model'));

    const res224Btn = screen.getByText('224 × 224 (Baseline)');
    const res384Btn = screen.getByText('384 × 384 (Mid-Res)');
    const res512Btn = screen.getByText('512 × 512 (Multi-Scale FPN)');

    expect(res224Btn).toBeInTheDocument();
    expect(res384Btn).toBeInTheDocument();
    expect(res512Btn).toBeInTheDocument();

    // Default 512x512
    expect(screen.getByText('8.0 px (P2)')).toBeInTheDocument();

    // Click 224x224
    fireEvent.click(res224Btn);
    expect(screen.getByText('32.0 px / bin')).toBeInTheDocument();

    // Click 384x384
    fireEvent.click(res384Btn);
    expect(screen.getByText('21.3 px / bin')).toBeInTheDocument();
  });

  it('renders research-only disclaimer and model decoupling architecture cards', async () => {
    render(<ResearchPage />);

    await waitFor(() => {
      expect(screen.getByText('Phase 4: Multi-Scale Model')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Phase 4: Multi-Scale Model'));

    expect(screen.getByText('Baseline Production Model')).toBeInTheDocument();
    expect(screen.getByText('Multi-Scale Experimental Model')).toBeInTheDocument();
    expect(screen.getByText('Research Only — Non-Diagnostic Warning')).toBeInTheDocument();
  });
});
