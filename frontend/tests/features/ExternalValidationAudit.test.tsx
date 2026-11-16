import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import { DoctorResearcherXRayView } from '@/features/xray/components/DoctorResearcherXRayView';
import { ModelsPage } from '@/features/models/ModelsPage';
import { ResearchPage } from '@/features/research/ResearchPage';
import { modelApi } from '@/services/api/modelApi';
import { researchApi } from '@/services/api/researchApi';
import { XRayPredictionResponse } from '@/types/xray';

vi.mock('@/services/api/modelApi', () => ({
  modelApi: {
    listModels: vi.fn(),
  },
}));

vi.mock('@/services/api/researchApi', () => ({
  researchApi: {
    getResearchMetrics: vi.fn(),
  },
}));

const mockMultiLabelResult: XRayPredictionResponse = {
  success: true,
  timestamp: '2026-11-16T00:00:00Z',
  disclaimer: 'For research and educational use only.',
  model: {
    id: 'densenet121_nih_6disease',
    name: 'DenseNet121 NIH 6-Disease Multi-Label Classifier',
    architecture: 'DenseNet121',
    task: 'Multi-label chest X-ray disease classification',
    checkpoint_epoch: 4,
    training_dataset: 'NIH ChestX-ray14',
    input_dimensions: [224, 224, 3],
  },
  predictions: {
    Cardiomegaly: { probability: 0.9998, threshold: 0.9502, experimental_flag: true },
    Edema: { probability: 0.0412, threshold: 0.8770, experimental_flag: false },
    Consolidation: { probability: 0.1285, threshold: 0.7329, experimental_flag: false },
    Pneumonia: { probability: 0.0934, threshold: 0.7529, experimental_flag: false },
    Atelectasis: { probability: 0.1872, threshold: 0.7495, experimental_flag: false },
    Pneumothorax: { probability: 0.0156, threshold: 0.5825, experimental_flag: false },
  },
  summary: {
    flagged_conditions_count: 1,
    highest_probability_disease: 'Cardiomegaly',
    highest_probability_value: 0.9998,
  },
  inference_time_ms: 82.5,
};

describe('External Validation & Screening Audit Tests', () => {
  it('renders Research Evaluation Panel separating model activation scores from locked thresholds and reference labels', () => {
    render(
      <DoctorResearcherXRayView
        fileName="test2_cardiomegaly.jpg"
        fileSize={2412000}
        multiLabelResult={mockMultiLabelResult}
        binaryResult={null}
        activeModelType="multilabel"
        onOpenExplainability={vi.fn()}
        onOpenGlossary={vi.fn()}
      />
    );

    // Verify Research Evaluation Panel Header
    expect(screen.getByText('Research Evaluation & Clinical Concordance Panel')).toBeInTheDocument();
    expect(screen.getByText('Model Activation Score')).toBeInTheDocument();
    expect(screen.getByText('Locked Threshold')).toBeInTheDocument();
    expect(screen.getByText('Predicted Flag')).toBeInTheDocument();
    expect(screen.getByText('Reference Label')).toBeInTheDocument();
    expect(screen.getByText('Evaluation Status')).toBeInTheDocument();

    // Verify Unverified Reference initial state
    expect(screen.getAllByText('Unverified reference').length).toBeGreaterThan(0);

    // Verify Non-Diagnostic Disclaimer
    expect(screen.getByText(/Model activation scores are raw continuous mathematical values/i)).toBeInTheDocument();

    // Switch Reference Case to Test 2 (Cardiomegaly)
    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'test2_cardiomegaly' } });

    // Status should now be Correct positive for Cardiomegaly
    expect(screen.getByText('Correct positive')).toBeInTheDocument();
    expect(screen.getAllByText('Correct negative').length).toBe(5);
  });

  it('displays active operating threshold set and allows toggling operating points in ModelsPage', async () => {
    vi.mocked(modelApi.listModels).mockResolvedValueOnce({
      success: true,
      total_models: 1,
      models: [
        {
          id: 'densenet121_nih_6disease',
          name: 'DenseNet121 NIH 6-Disease Multi-Label Classifier',
          architecture: 'DenseNet121',
          task: 'Multi-label chest X-ray disease classification',
          labels: ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax'],
          status: 'production_frozen',
          active_threshold_set: 'NIH ChestX-ray14 Locked Production Thresholds (Frozen)',
        },
      ],
    });

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <ModelsPage />
      </BrowserRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('Active: NIH Locked Production Set')).toBeInTheDocument();
      expect(screen.getByText('Active (Enforced by Backend)')).toBeInTheDocument();
      expect(screen.getByText(/Point A \(Active\)/i)).toBeInTheDocument();
      expect(screen.getByText(/Point B \(Research\)/i)).toBeInTheDocument();
    });

    // Switch to Operating Point B
    const pointBButton = screen.getByText(/Point B \(Research\)/i);
    fireEvent.click(pointBButton);

    expect(screen.getByText('Research Comparison Only')).toBeInTheDocument();
    expect(screen.getAllByText('0.0298').length).toBeGreaterThan(0); // CheXpert Youden threshold
  });

  it('renders External Screening Audit tab with COMPLETE status in ResearchPage', async () => {
    vi.mocked(researchApi.getResearchMetrics).mockResolvedValueOnce({
      success: true,
      nih_test_metrics: [],
      chexpert_external_metrics: [],
      vindr_external_metrics: [],
      notes: ['Historical verification benchmark.'],
    });

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <ResearchPage />
      </BrowserRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('External Screening Audit')).toBeInTheDocument();
    });

    // Click tab
    const auditTab = screen.getByText('External Screening Audit');
    fireEvent.click(auditTab);

    // Verify COMPLETE status and audit tables
    expect(screen.getByText('EXTERNAL VALIDATION COMPLETE')).toBeInTheDocument();
    expect(screen.getByText(/Investigation of Observed Manual Screening Tests/i)).toBeInTheDocument();
    expect(screen.getByText(/Threshold Sensitivity & Specificity Comparison across Operating Points/i)).toBeInTheDocument();
  });
});
