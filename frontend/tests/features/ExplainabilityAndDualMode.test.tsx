import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import { PatientFriendlyXRayView } from '@/features/xray/components/PatientFriendlyXRayView';
import { DoctorResearcherXRayView } from '@/features/xray/components/DoctorResearcherXRayView';
import { DiseaseExplainabilityModal } from '@/features/xray/components/DiseaseExplainabilityModal';
import { MedicalGlossaryModal } from '@/features/xray/components/MedicalGlossaryModal';
import { XRayPredictionResponse } from '@/types/xray';

const mockMultiLabelResult: XRayPredictionResponse = {
  success: true,
  model: {
    id: 'densenet121_nih_6disease',
    architecture: 'DenseNet121',
    task: 'Multi-label chest X-ray disease classification',
    training_dataset: 'NIH ChestX-ray14',
    checkpoint_epoch: 4,
  },
  predictions: {
    Cardiomegaly: { probability: 0.1245, threshold: 0.950195, experimental_flag: false },
    Edema: { probability: 0.0832, threshold: 0.876953, experimental_flag: false },
    Consolidation: { probability: 0.1542, threshold: 0.732910, experimental_flag: false },
    Pneumonia: { probability: 0.0431, threshold: 0.752930, experimental_flag: false },
    Atelectasis: { probability: 0.2225, threshold: 0.749512, experimental_flag: false },
    Pneumothorax: { probability: 0.0123, threshold: 0.582520, experimental_flag: false },
  },
  summary: {
    flagged_conditions_count: 0,
    highest_probability_disease: 'Atelectasis',
    highest_probability_value: 0.2225,
  },
  inference_time_ms: 135.4,
  disclaimer: 'Research decision-support output.',
  timestamp: new Date().toISOString(),
};

describe('Clinical Explainability & Dual-Mode UI', () => {
  it('renders Patient-Friendly view answering all 6 core questions and explaining highest model score', () => {
    const onOpenExplain = vi.fn();
    const onOpenGlossary = vi.fn();

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <PatientFriendlyXRayView
          fileName="patient_chest_radiograph.png"
          fileSize={2 * 1024 * 1024}
          multiLabelResult={mockMultiLabelResult}
          binaryResult={null}
          activeModelType="multilabel"
          onOpenExplainability={onOpenExplain}
          onOpenGlossary={onOpenGlossary}
        />
      </BrowserRouter>
    );

    // Question 1
    expect(screen.getByText(/What image was analyzed\?/i)).toBeInTheDocument();
    expect(screen.getByText(/patient_chest_radiograph\.png/i)).toBeInTheDocument();

    // Question 2
    expect(screen.getByText(/What did the AI evaluate\?/i)).toBeInTheDocument();

    // Question 3
    expect(screen.getByText(/What did the model return\?/i)).toBeInTheDocument();
    expect(screen.getByText(/Highest Model Score:/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Atelectasis/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/22\.25%/i).length).toBeGreaterThan(0);

    // Question 4: Must forbid "all clear" or "normal X-ray" claims
    expect(screen.getByText(/What do these results mean\?/i)).toBeInTheDocument();
    expect(screen.getByText(/Important Medical Notice:/i)).toBeInTheDocument();
    expect(screen.getByText(/prove that the lungs are completely healthy, normal, or "all clear\."/i)).toBeInTheDocument();

    // Question 5
    expect(screen.getByText(/What are the limitations of this analysis\?/i)).toBeInTheDocument();

    // Question 6
    expect(screen.getByText(/What should you do next\?/i)).toBeInTheDocument();
    expect(screen.getByText(/Always consult a qualified doctor or radiologist/i)).toBeInTheDocument();
  });

  it('renders Doctor / Researcher view with exact raw telemetry, checkpoint ID, and JSON export', () => {
    const onOpenExplain = vi.fn();
    const onOpenGlossary = vi.fn();

    render(
      <BrowserRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <DoctorResearcherXRayView
          fileName="patient_chest_radiograph.png"
          fileSize={2 * 1024 * 1024}
          multiLabelResult={mockMultiLabelResult}
          binaryResult={null}
          activeModelType="multilabel"
          onOpenExplainability={onOpenExplain}
          onOpenGlossary={onOpenGlossary}
        />
      </BrowserRouter>
    );

    expect(screen.getByText('Doctor / Researcher Telemetry')).toBeInTheDocument();
    expect(screen.getByText('Export JSON')).toBeInTheDocument();
    expect(screen.getByText('Print Report')).toBeInTheDocument();
    expect(screen.getByText(/densenet121_nih_best\.pth/i)).toBeInTheDocument();
    expect(screen.getByText(/135\.4 ms/i)).toBeInTheDocument();
    expect(screen.getByText(/CheXpert Drop/i)).toBeInTheDocument();

    // Verify raw float precision display
    expect(screen.getByText(/0\.222500/i)).toBeInTheDocument();
  });

  it('renders Disease Explainability Modal with medical definitions and diagnostic caveats', () => {
    const onClose = vi.fn();

    render(
      <DiseaseExplainabilityModal
        isOpen={true}
        onClose={onClose}
        initialDisease="Atelectasis"
        currentProbabilities={mockMultiLabelResult.predictions}
      />
    );

    expect(screen.getByText('Clinical Pathology Guide & AI Explainability')).toBeInTheDocument();
    expect(screen.getAllByText('Atelectasis').length).toBeGreaterThan(0);
    expect(screen.getByText(/Partial Lung Collapse \/ Volume Loss/i)).toBeInTheDocument();
    expect(screen.getByText(/Crucial Clinical & Diagnostic Caveats/i)).toBeInTheDocument();
    expect(screen.getByText(/Understanding the Difference: 4 Distinct Levels/i)).toBeInTheDocument();

    // Close button
    const closeBtn = screen.getByText('Close Guide');
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalled();
  });

  it('renders Medical Glossary Modal with search filter', () => {
    const onClose = vi.fn();

    render(
      <MedicalGlossaryModal
        isOpen={true}
        onClose={onClose}
      />
    );

    expect(screen.getByText('Medical & AI Radiology Glossary')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Search medical or technical terms/i)).toBeInTheDocument();
    expect(screen.getByText('AUROC (Area Under ROC Curve)')).toBeInTheDocument();
    expect(screen.getByText('Youden Index (J)')).toBeInTheDocument();

    const searchInput = screen.getByPlaceholderText(/Search medical or technical terms/i);
    fireEvent.change(searchInput, { target: { value: 'Youden' } });

    expect(screen.getByText('Youden Index (J)')).toBeInTheDocument();
    expect(screen.queryByText('AUROC (Area Under ROC Curve)')).not.toBeInTheDocument();
  });
});
