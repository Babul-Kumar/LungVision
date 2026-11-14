import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { XrayPage } from '@/features/xray/XrayPage';
import { BrowserRouter } from 'react-router-dom';
import { EXPECTED_6_DISEASES } from '@/types/xray';

vi.mock('@/services/api/xrayApi', () => ({
  xrayApi: {
    predictMultiLabel: vi.fn(),
    predictBinary: vi.fn(),
  },
}));

describe('XrayPage Feature', () => {
  it('renders page title and upload dropzone', () => {
    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <XrayPage />
      </BrowserRouter>
    );

    expect(screen.getByText('2D Chest X-Ray Screening')).toBeInTheDocument();
    expect(screen.getByText('Radiograph Upload')).toBeInTheDocument();
    expect(screen.getByText(/Click or drag X-ray image here/i)).toBeInTheDocument();
  });

  it('verifies exact 6-disease NIH locked label ordering specification', () => {
    expect(EXPECTED_6_DISEASES).toEqual([
      'Cardiomegaly',
      'Edema',
      'Consolidation',
      'Pneumonia',
      'Atelectasis',
      'Pneumothorax',
    ]);
  });

  it('switches between Multi-Label and Binary screening modes', () => {
    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <XrayPage />
      </BrowserRouter>
    );

    const binaryTab = screen.getByText('Binary (Normal / Abnormal)');
    fireEvent.click(binaryTab);

    expect(screen.getByText('Select ResNet18 Variant:')).toBeInTheDocument();
    expect(screen.getByText('Original ResNet18')).toBeInTheDocument();
    expect(screen.getByText('Blur-Augmented')).toBeInTheDocument();

    const multiTab = screen.getByText('6-Disease Multi-Label');
    fireEvent.click(multiTab);

    expect(screen.getByText('DenseNet121 & ResNet18')).toBeInTheDocument();
  });
});
