import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ResearchPage } from '@/features/research/ResearchPage';
import { BrowserRouter } from 'react-router-dom';
import { researchApi } from '@/services/api/researchApi';

vi.mock('@/services/api/researchApi', () => ({
  researchApi: {
    getResearchMetrics: vi.fn(),
  },
}));

describe('ResearchPage Feature', () => {
  const mockMetrics = {
    success: true,
    nih_test_metrics: [
      {
        Disease: 'Cardiomegaly',
        Threshold: 0.950195,
        AUROC: 0.887,
        AUPRC: 0.324,
        Precision: 0.45,
        Sensitivity: 0.82,
        Specificity: 0.94,
        F1: 0.58,
      },
    ],
    chexpert_external_metrics: [
      {
        Disease: 'Cardiomegaly',
        Threshold: 0.950195,
        Evaluable: 200,
        Positive: 80,
        Negative: 120,
        AUROC: 0.812,
        AUPRC: 0.45,
        Precision: 0.55,
        Sensitivity: 0.75,
        Specificity: 0.88,
        F1: 0.63,
      },
    ],
    vindr_external_metrics: [
      {
        model: 'Original ResNet18',
        cohort: 'Full VinDr',
        ROC_AUC: 0.845,
        Average_Precision: 0.612,
        Accuracy: 0.79,
        Balanced_Accuracy: 0.81,
        Precision: 0.65,
        Sensitivity: 0.88,
        Specificity: 0.76,
        F1: 0.74,
        Brier_Score: 0.142,
        ECE_10_bins: 0.045,
        threshold: 0.176463,
      },
    ],
    notes: [
      'NIH metrics reflect calibrated test set performance across 6 multi-label diseases.',
    ],
  };

  it('renders benchmark tabs and loads data from backend', async () => {
    vi.mocked(researchApi.getResearchMetrics).mockResolvedValueOnce(mockMetrics);

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <ResearchPage />
      </BrowserRouter>
    );

    expect(screen.getByText('Research Evidence & Benchmark Registry')).toBeInTheDocument();
    expect(screen.getByText('NIH ChestX-ray14')).toBeInTheDocument();
    expect(screen.getByText('CheXpert External')).toBeInTheDocument();
    expect(screen.getByText('VinDr-CXR (15k)')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('NIH ChestX-ray14 Held-Out Test Set (DenseNet121)')).toBeInTheDocument();
      expect(screen.getByText('Detailed NIH Benchmark Metrics Table')).toBeInTheDocument();
      expect(screen.getByText('0.8870')).toBeInTheDocument();
    });
  });

  it('switches to CheXpert and VinDr tabs', async () => {
    vi.mocked(researchApi.getResearchMetrics).mockResolvedValueOnce(mockMetrics);

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <ResearchPage />
      </BrowserRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('NIH ChestX-ray14 Held-Out Test Set (DenseNet121)')).toBeInTheDocument();
    });

    const chexpertTab = screen.getByText('CheXpert External');
    fireEvent.click(chexpertTab);

    expect(screen.getByText('Stanford CheXpert External Generalization')).toBeInTheDocument();

    const vindrTab = screen.getByText('VinDr-CXR (15k)');
    fireEvent.click(vindrTab);

    expect(screen.getByText('VinDr-CXR 15,000-Image External Validation (ResNet18)')).toBeInTheDocument();
  });
});

