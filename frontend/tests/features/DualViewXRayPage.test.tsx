import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DualViewXRayPage } from '@/features/xray/DualViewXRayPage';
import { BrowserRouter } from 'react-router-dom';
import { reconstructionApi } from '@/services/api/reconstructionApi';

vi.mock('@/services/api/xrayApi', () => ({
  xrayApi: {
    predictMultiLabel: vi.fn(),
  },
}));

vi.mock('@/services/api/reconstructionApi', () => ({
  reconstructionApi: {
    getModelInfo: vi.fn().mockResolvedValue({
      success: true,
      loaded: true,
      architecture: 'Statistical Shape Model (Point Distribution Model)',
      num_vertices: 7652,
      num_faces: 15296,
      num_modes: 15,
      total_explained_variance_pct: 100.0,
      modes: [
        { mode_index: 1, explained_variance_pct: 48.14, standard_deviation_mm: 18.25, description: 'Volume / Inspiratory Effort' },
        { mode_index: 2, explained_variance_pct: 12.90, standard_deviation_mm: 9.42, description: 'Cranio-Caudal Aspect Ratio' },
      ],
      landmarks: [
        { landmark_index: 1, name: 'Right Lung Apex', vertex_index: 105 },
        { landmark_index: 2, name: 'Left Lung Apex', vertex_index: 3950 },
      ],
      reference_volume_ml: 975.45,
    }),
    fitReconstruction: vi.fn().mockResolvedValue({
      success: true,
      status: 'converged',
      optimization_latency_ms: 14.2,
      iterations_computed: 18,
      optimal_coefficients: [0.35, -0.12, 0.05],
      metrics: {
        estimated_dice_score: 0.915,
        estimated_assd_mm: 1.75,
        mean_vertex_uncertainty_mm: 1.25,
        max_vertex_uncertainty_mm: 2.45,
        confidence_margin_pct: 95.0,
      },
      volumetry: {
        total_lung_volume_ml: 1042.5,
        left_lung_ml: 479.5,
        right_lung_ml: 563.0,
        symmetry_ratio: 1.17,
      },
      surface_mesh: {
        vertex_count: 7652,
        face_count: 15296,
        landmarks: {
          'Right Lung Apex': [35.2, 50.1, 140.2],
          'Left Lung Apex': [-36.1, 48.9, 139.8],
        },
        vertices: [[0, 0, 0], [1, 1, 1]],
        faces: [[0, 1, 0]],
      },
      disclaimer: 'SSM surface reconstruction represents macroscopic organ boundary geometry.',
    }),
    exportReconstructedObj: vi.fn().mockResolvedValue('# Wavefront OBJ - Reconstructed Lungs\nv 0 0 0\nf 1 1 1'),
    getTemplateMeshObj: vi.fn().mockResolvedValue('# Wavefront OBJ - Template\nv 0 0 0\nf 1 1 1'),
  },
}));

// Mock Three.js WebGL canvas for jsdom
vi.mock('@/features/xray/SSM3DReconstructionViewer', () => ({
  SSM3DReconstructionViewer: ({ onExportObj }: any) => (
    <div data-testid="ssm-3d-viewer-mock">
      <span>SSM 3D Surface Mesh Viewer</span>
      <button onClick={onExportObj} data-testid="mock-export-btn">Export OBJ</button>
    </div>
  ),
}));

describe('DualViewXRayPage Feature', () => {
  it('renders Dual-View CXR Lab title, separate dropzones, and SSM header badge', async () => {
    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <DualViewXRayPage />
      </BrowserRouter>
    );

    // Title & Header Badges
    expect(screen.getByText(/Dual-View X-Ray Research Lab/i)).toBeInTheDocument();
    expect(screen.getByText('Dual-View CXR Lab')).toBeInTheDocument();
    expect(screen.getByText('Bi-Planar Ingestion')).toBeInTheDocument();
    expect(screen.getByText('Phase 2B: 3D SSM Engine')).toBeInTheDocument();

    // Dual Upload Zones
    expect(screen.getByText('Slot A: Frontal View')).toBeInTheDocument();
    expect(screen.getByText('Slot B: Lateral View')).toBeInTheDocument();

    // Viewport Controls
    expect(screen.getByText('✓ Synchronized Zoom')).toBeInTheDocument();
    expect(screen.getByText('Standard X-Ray')).toBeInTheDocument();

    // Workflow Tabs
    expect(screen.getByText(/3D Lung Reconstruction/i)).toBeInTheDocument();
    expect(screen.getByText(/2D Pathology Screening/i)).toBeInTheDocument();
    expect(screen.getByText(/Scientific Architecture & Roadmap/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(reconstructionApi.getModelInfo).toHaveBeenCalled();
    });
  });

  it('toggles synchronized zoom and image inversion settings', async () => {
    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <DualViewXRayPage />
      </BrowserRouter>
    );

    const syncBtn = screen.getByText('✓ Synchronized Zoom');
    fireEvent.click(syncBtn);
    expect(screen.getByText('Independent Zoom')).toBeInTheDocument();

    const invertBtn = screen.getByText('Standard X-Ray');
    fireEvent.click(invertBtn);
    expect(screen.getByText('Inverted (Bone White)')).toBeInTheDocument();

    await waitFor(() => {
      expect(reconstructionApi.getModelInfo).toHaveBeenCalled();
    });
  });

  it('navigates between workflow tabs seamlessly', async () => {
    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <DualViewXRayPage />
      </BrowserRouter>
    );

    // Switch to Roadmap Tab
    const roadmapTab = screen.getByText(/Scientific Architecture & Roadmap/i);
    fireEvent.click(roadmapTab);
    expect(screen.getByText(/3D Reconstruction Scientific Roadmap & Methodology/i)).toBeInTheDocument();
    expect(screen.getByText(/Phase 1: Dual-View Review/i)).toBeInTheDocument();
    expect(screen.getByText(/Phase 2: Morphable Shape Models/i)).toBeInTheDocument();

    // Switch to Screening Tab
    const screeningTab = screen.getByText(/2D Pathology Screening/i);
    fireEvent.click(screeningTab);
    expect(screen.getByText(/Frontal Radiograph Neural Inference/i)).toBeInTheDocument();
    expect(screen.getByText(/No Dual-View Analysis Executed/i)).toBeInTheDocument();

    // Switch back to Reconstruction Tab
    const reconTab = screen.getByText(/3D Lung Reconstruction/i);
    fireEvent.click(reconTab);
    expect(screen.getByText(/Patient-Specific 3D Lung Surface Mesh/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(reconstructionApi.getModelInfo).toHaveBeenCalled();
    });
  });

  it('executes 3D SSM surface reconstruction and displays organ volumetry & quality metrics', async () => {
    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <DualViewXRayPage />
      </BrowserRouter>
    );

    // Find and click Reconstruct button
    const reconBtn = screen.getByRole('button', { name: /Reconstruct 3D Lung Surface/i });
    fireEvent.click(reconBtn);

    // Wait for reconstruction results to appear
    await waitFor(() => {
      expect(screen.getByTestId('ssm-3d-viewer-mock')).toBeInTheDocument();
      expect(screen.getByText(/1,042.5 mL/i)).toBeInTheDocument();
      expect(screen.getByText(/563.0 mL/i)).toBeInTheDocument();
      expect(screen.getByText(/479.5 mL/i)).toBeInTheDocument();
      expect(screen.getByText(/91.5%/i)).toBeInTheDocument();
      expect(screen.getByText(/1.75 mm/i)).toBeInTheDocument();
    });
  });
});
