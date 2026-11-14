import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import { CTExplorerPage } from '@/features/ct/CTExplorerPage';
import { ctApi } from '@/services/api/ctApi';

vi.mock('@/services/api/ctApi', () => ({
  ctApi: {
    getMetadata: vi.fn(),
    getAnnotations: vi.fn(),
    getMeasurements: vi.fn(),
    getMeshQuality: vi.fn(),
    getSlice: vi.fn(),
    getMeshObj: vi.fn(),
    getDRR: vi.fn(),
  },
}));

// Mock ThreeMeshCanvas since WebGL is not available in JSDOM
vi.mock('@/features/ct/ThreeMeshCanvas', () => ({
  ThreeMeshCanvas: () => <div data-testid="three-canvas-mock">Three.js Mock Canvas</div>,
}));

describe('CTExplorerPage Feature', () => {
  const mockMetadata = {
    scan_id: 'demo_thoracic_scan',
    filename: 'demo_scan.mhd',
    dimensions: { x: 128, y: 128, z: 64 },
    spacing_mm: [0.742, 0.742, 1.25] as [number, number, number],
    origin_mm: [0, 0, 0] as [number, number, number],
    direction_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    supported_views: ['axial', 'coronal', 'sagittal'],
    window_presets: {
      lung: { center: -600, width: 1500 },
    },
    anatomical_structures: ['lungs', 'left_lung', 'right_lung'],
    labels_present: [0, 3, 4, 5],
    annotation_count: 12,
    geometry_validation: { status: 'PASSED' as const, mismatches: [] },
    disclaimer: 'For research only',
  };

  const mockMeasurements = {
    scan_id: 'demo_thoracic_scan',
    has_segmentation: true,
    geometry_validation_status: 'PASSED',
    physical_voxel_volume_mm3: 0.6885,
    lungs: {
      left_lung_ml: 61.54,
      right_lung_ml: 67.44,
      total_lung_volume_ml: 128.98,
      volume_balance: {
        absolute_difference_ml: 5.9,
        relative_difference_pct: 9.59,
      },
    },
    airway: {
      trachea_volume_ml: 2.31,
      trachea_voxels: 3360,
    },
    label_audit: [],
    disclaimer: 'Research only',
  };

  const mockMeshQuality = {
    scan_id: 'demo_thoracic_scan',
    structure: 'lungs',
    watertight: true,
    euler_number: 2,
    is_winding_consistent: true,
    vertex_count: 14200,
    face_count: 28400,
    degenerate_face_count: 0,
    surface_area_mm2: 56200.5,
    disclaimer: 'Research only',
  };

  const mockAnnotations = {
    success: true,
    scan_id: 'demo_thoracic_scan',
    total_annotations: 1,
    inside_volume_count: 1,
    annotations: [
      {
        nodule_id: 'N1',
        seriesuid: 'demo',
        coordX_mm: -12.4,
        coordY_mm: 45.2,
        coordZ_mm: -180.5,
        voxel_x: 64,
        voxel_y: 64,
        voxel_z: 32,
        diameter_mm: 8.5,
        inside_volume: true,
      },
    ],
    disclaimer: 'Research only',
  };

  it('renders CT Explorer page and displays quantitative volumetry without throwing', async () => {
    vi.mocked(ctApi.getMetadata).mockResolvedValueOnce(mockMetadata);
    vi.mocked(ctApi.getMeasurements).mockResolvedValueOnce(mockMeasurements);
    vi.mocked(ctApi.getMeshQuality).mockResolvedValueOnce(mockMeshQuality);
    vi.mocked(ctApi.getAnnotations).mockResolvedValueOnce(mockAnnotations);

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <CTExplorerPage />
      </BrowserRouter>
    );

    expect(screen.getByText('3D Computed Tomography Explorer')).toBeInTheDocument();
    expect(screen.getByText('Quantitative Lung Volumetry')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('129.0 mL')).toBeInTheDocument(); // total lung volume
      expect(screen.getByText('61.5 mL')).toBeInTheDocument();  // left lung
      expect(screen.getByText('67.4 mL')).toBeInTheDocument();  // right lung
      expect(screen.getByText('2.3 mL')).toBeInTheDocument();   // trachea
    });
  });

  it('safely handles missing or empty measurements without runtime crash', async () => {
    vi.mocked(ctApi.getMetadata).mockResolvedValueOnce(mockMetadata);
    // Return empty lungs and airway structures
    vi.mocked(ctApi.getMeasurements).mockResolvedValueOnce({
      scan_id: 'demo_thoracic_scan',
      has_segmentation: true,
      geometry_validation_status: 'PASSED',
      lungs: {} as any,
      label_audit: [],
      disclaimer: 'Research only',
    });
    vi.mocked(ctApi.getMeshQuality).mockResolvedValueOnce(mockMeshQuality);
    vi.mocked(ctApi.getAnnotations).mockResolvedValueOnce(mockAnnotations);

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <CTExplorerPage />
      </BrowserRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('Quantitative Lung Volumetry')).toBeInTheDocument();
    });

    // Confirms fallback 0.0 mL is rendered instead of throwing TypeError on toFixed
    expect(screen.getAllByText('0.0 mL').length).toBeGreaterThanOrEqual(1);
  });

  it('safely renders MPR slicer with undefined physical_coordinate_mm without crashing', async () => {
    vi.mocked(ctApi.getMetadata).mockResolvedValueOnce(mockMetadata);
    vi.mocked(ctApi.getMeasurements).mockResolvedValueOnce(mockMeasurements);
    vi.mocked(ctApi.getMeshQuality).mockResolvedValueOnce(mockMeshQuality);
    vi.mocked(ctApi.getAnnotations).mockResolvedValueOnce(mockAnnotations);
    vi.mocked(ctApi.getSlice).mockResolvedValue({
      scan_id: 'demo_thoracic_scan',
      view: 'axial',
      slice_index: 10,
      max_slice_index: 63,
      physical_coordinate_mm: undefined as any,
      window_applied: 'lung',
      has_segmentation_overlay: true,
      image_base64: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
      nodules_on_slice: [],
    });

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <CTExplorerPage />
      </BrowserRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('3D Computed Tomography Explorer')).toBeInTheDocument();
    });

    // Click on Tab 2: Orthogonal Slicer (MPR)
    const slicerTab = screen.getByText('Multi-Planar Reconstruction (MPR)');
    fireEvent.click(slicerTab);

    // Verify slicer renders and calculates fallback physical coordinate without throwing
    await waitFor(() => {
      expect(screen.getByText('AXIAL Orthogonal Viewport')).toBeInTheDocument();
      expect(screen.getByText('Coord:')).toBeInTheDocument();
    });
  });

  it('renders DRR Synthesis & 3D Bridge tab and simulates radiographic projection', async () => {
    const mockDRR = {
      success: true,
      scan_id: 'demo_thoracic_scan',
      view: 'frontal',
      view_name: 'Frontal (PA)',
      image_base64: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
      dimensions: { width: 128, height: 64 },
      physical_fov_mm: { width_mm: 300.0, height_mm: 350.0 },
      geometry: {
        sdd_mm: 1800.0,
        sod_mm: 1450.0,
        magnification_factor: 1.241,
        axis_integrated: 'Y-axis (Posterior -> Anterior)',
        detector_pixel_pitch_mm: 0.742,
      },
      attenuation_stats: {
        min_line_integral: 0.05,
        max_line_integral: 12.4,
        mean_line_integral: 2.31,
      },
      computation_latency_ms: 45.2,
      disclaimer: 'DRRs are synthetic radiographic projections simulated via Beer-Lambert linear attenuation for research and decision-support, not certified diagnostic radiograms.',
    };

    vi.mocked(ctApi.getMetadata).mockResolvedValueOnce(mockMetadata);
    vi.mocked(ctApi.getMeasurements).mockResolvedValueOnce(mockMeasurements);
    vi.mocked(ctApi.getMeshQuality).mockResolvedValueOnce(mockMeshQuality);
    vi.mocked(ctApi.getAnnotations).mockResolvedValueOnce(mockAnnotations);
    vi.mocked(ctApi.getDRR).mockResolvedValue(mockDRR);

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <CTExplorerPage />
      </BrowserRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('3D Computed Tomography Explorer')).toBeInTheDocument();
    });

    // Click on Tab 4: DRR Synthesis & 3D Bridge
    const drrTab = screen.getByText('DRR Synthesis & 3D Bridge');
    fireEvent.click(drrTab);

    // Verify DRR Viewport and controls render
    await waitFor(() => {
      expect(screen.getByText('CT-to-DRR Radiographic Synthesis & Dual-View 3D Bridge')).toBeInTheDocument();
      expect(screen.getByText('Synthesized Frontal (PA) Radiograph')).toBeInTheDocument();
      expect(screen.getByText('Export to Dual-View Lab')).toBeInTheDocument();
      expect(screen.getByText('Save Radiograph')).toBeInTheDocument();
      expect(screen.getByText('1.241×')).toBeInTheDocument();
      expect(screen.getByText('Quantum Photon Poisson Noise')).toBeInTheDocument();
    });

    expect(ctApi.getDRR).toHaveBeenCalledWith(
      'demo_thoracic_scan',
      'frontal',
      1800,
      1450,
      false,
      false
    );
  });
});

