import { render, screen, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { App } from '@/App';

vi.mock('@/services/api/healthApi', () => ({
  healthApi: {
    getHealth: vi.fn().mockResolvedValue({ status: 'healthy', version: '1.0.0', timestamp: '2026-01-01' }),
    getReady: vi.fn().mockResolvedValue({
      status: 'ready',
      device: 'cpu',
      models_loaded: {
        densenet121_6disease: true,
        resnet18_original: true,
        resnet18_augmented: true,
      },
      timestamp: '2026-01-01',
    }),
  },
}));

vi.mock('@/services/api/modelApi', () => ({
  modelApi: {
    listModels: vi.fn().mockResolvedValue({ success: true, total_models: 0, models: [] }),
  },
}));

describe('App Root Integration', () => {
  it('renders application brand header and navigation items', async () => {
    await act(async () => {
      render(<App />);
    });

    expect(screen.getByText('LUNGVISION')).toBeInTheDocument();
    expect(screen.getByText('Thoracic Imaging Intelligence')).toBeInTheDocument();

    // Check navigation items (found in desktop sidebar & mobile bar)
    expect(screen.getAllByText('Dashboard').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('2D X-Ray Analysis').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('3D CT Explorer').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Research Benchmarks').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Model Registry').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Session History').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Settings').length).toBeGreaterThanOrEqual(1);
  });
});
