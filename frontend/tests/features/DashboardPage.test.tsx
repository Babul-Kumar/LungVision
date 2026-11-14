import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DashboardPage } from '@/features/dashboard/DashboardPage';
import { BrowserRouter } from 'react-router-dom';
import { modelApi } from '@/services/api/modelApi';
import { healthApi } from '@/services/api/healthApi';

vi.mock('@/services/api/modelApi', () => ({
  modelApi: {
    listModels: vi.fn(),
  },
}));

vi.mock('@/services/api/healthApi', () => ({
  healthApi: {
    getHealth: vi.fn(),
    getReady: vi.fn(),
  },
}));

describe('DashboardPage Feature', () => {
  it('renders station title, quick action cards, and model inventory', async () => {
    vi.mocked(healthApi.getHealth).mockResolvedValueOnce({
      status: 'healthy',
      version: '1.0.0',
      timestamp: new Date().toISOString(),
    });
    vi.mocked(healthApi.getReady).mockResolvedValueOnce({
      status: 'ready',
      device: 'cpu',
      models_loaded: {
        densenet121_6disease: true,
        resnet18_original: true,
        resnet18_augmented: true,
      },
      timestamp: new Date().toISOString(),
    });
    vi.mocked(modelApi.listModels).mockResolvedValueOnce({
      success: true,
      total_models: 1,
      models: [
        {
          id: 'densenet121_nih_6disease',
          name: 'DenseNet121 NIH 6-Disease Multi-Label Classifier',
          architecture: 'DenseNet121',
          task: 'Multi-label chest X-ray disease classification',
          checkpoint_path: '/checkpoints/densenet121.pt',
          labels: ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax'],
          status: 'production_frozen',
          metrics: {
            training_dataset: 'NIH ChestX-ray14',
            parameters: 7000000,
            macro_val_auc: 0.820582,
          },
          calibration: {
            Cardiomegaly: 0.950195,
          },
        },
      ],
    });

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <DashboardPage />
      </BrowserRouter>
    );

    expect(screen.getByText('Thoracic Intelligence Station')).toBeInTheDocument();
    expect(screen.getByText('2D X-Ray Screening')).toBeInTheDocument();
    expect(screen.getByText('3D CT Explorer')).toBeInTheDocument();
    expect(screen.getByText('Research Benchmarks')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('DenseNet121 NIH 6-Disease Multi-Label Classifier')).toBeInTheDocument();
      expect(screen.getByText('6 Classes')).toBeInTheDocument();
      expect(screen.getByText('7.00M')).toBeInTheDocument();
    });
  });
});
