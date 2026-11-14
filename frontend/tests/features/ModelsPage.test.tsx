import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ModelsPage } from '@/features/models/ModelsPage';
import { BrowserRouter } from 'react-router-dom';
import { modelApi } from '@/services/api/modelApi';

vi.mock('@/services/api/modelApi', () => ({
  modelApi: {
    listModels: vi.fn(),
  },
}));

describe('ModelsPage Feature', () => {
  it('renders model list and specifications', async () => {
    vi.mocked(modelApi.listModels).mockResolvedValueOnce({
      success: true,
      total_models: 2,
      models: [
        {
          id: 'densenet121_nih_6disease',
          name: 'DenseNet121 NIH 6-Disease Multi-Label Classifier',
          architecture: 'DenseNet121',
          task: 'Multi-label chest X-ray disease classification',
          checkpoint_path: '/checkpoints/densenet121.pt',
          sha256: 'a1b2c3d4e5f6',
          input_size: [224, 224, 3],
          labels: ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax'],
          status: 'production_frozen',
          metrics: {
            parameters: 7000000,
            training_dataset: 'NIH ChestX-ray14',
          },
          calibration: {
            Cardiomegaly: 0.950195,
          },
        },
        {
          id: 'original_resnet18',
          name: 'Original ResNet18 Baseline Classifier',
          architecture: 'ResNet18',
          task: 'Binary Normal vs Abnormal classification',
          checkpoint_path: '/checkpoints/resnet18.pt',
          sha256: 'f6e5d4c3b2a1',
          input_size: [224, 224, 3],
          labels: ['Normal', 'Abnormal'],
          status: 'production_frozen',
          metrics: {
            parameters: 11177025,
          },
          calibration: {
            'Normal vs Abnormal': 0.176463,
          },
        },
      ],
    });

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <ModelsPage />
      </BrowserRouter>
    );

    expect(screen.getByText('Audited Model Registry & Checkpoints')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Available Models (2)')).toBeInTheDocument();
      expect(screen.getByText('Original ResNet18 Baseline Classifier')).toBeInTheDocument();
      expect(screen.getByText('224 × 224 × 3')).toBeInTheDocument();
    });

    // Switch model
    const resnetButton = screen.getByText('Original ResNet18 Baseline Classifier');
    fireEvent.click(resnetButton);

    expect(screen.getAllByText('Binary Normal vs Abnormal classification').length).toBeGreaterThan(0);
  });
});
