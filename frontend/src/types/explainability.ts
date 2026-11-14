import { DiseaseName } from './xray';

export interface DiseaseClinicalProfile {
  name: DiseaseName;
  patientFriendlyName: string;
  whatItIs: string;
  clinicalMeaning: string;
  modelScoreMeaning: string;
  thresholdMeaning: string;
  diagnosticCaveat: string;
  radiographicSigns: string[];
  calibratedThreshold: number;
}

export interface GlossaryTerm {
  term: string;
  category: 'clinical' | 'technical' | 'regulatory';
  shortDefinition: string;
  detailedExplanation: string;
}

export const DISEASE_CLINICAL_PROFILES: Record<DiseaseName, DiseaseClinicalProfile> = {
  Cardiomegaly: {
    name: 'Cardiomegaly',
    patientFriendlyName: 'Enlarged Heart Silhouette',
    whatItIs: 'An apparent enlargement of the heart shadow on the chest radiograph, defined classically as a cardiothoracic ratio (CTR) greater than 0.50 (50%) on a standard upright PA projection.',
    clinicalMeaning: 'Cardiomegaly often reflects left or right ventricular hypertrophy, chamber dilatation, pericardial effusion (fluid around the heart), or multi-chamber disease secondary to chronic hypertension, ischemic heart disease, or valvular disorder.',
    modelScoreMeaning: 'The model probability represents the neural network’s mathematical confidence that visual image features (e.g. transverse cardiac diameter vs thoracic width) match patterns labeled as Cardiomegaly in training radiographs.',
    thresholdMeaning: 'The experimental threshold of 0.9502 (95.02%) is the operating cutoff determined on the NIH ChestX-ray14 held-out test split via Youden Index J optimization to minimize false positives while detecting marked cardiac enlargement.',
    diagnosticCaveat: 'An AP projection, supine positioning, shallow inspiration, or obesity can artificially magnify the cardiac silhouette without true anatomical cardiomegaly. A high score is not an echocardiogram or clinical diagnosis.',
    radiographicSigns: [
      'Cardiothoracic ratio (CTR) > 50% on PA radiograph',
      'Left heart border displacement toward the lateral chest wall',
      'Elevation or blunting of the cardiac apex'
    ],
    calibratedThreshold: 0.950195,
  },
  Edema: {
    name: 'Edema',
    patientFriendlyName: 'Fluid in Lungs / Pulmonary Congestion',
    whatItIs: 'Abnormal accumulation of extravascular fluid in the interstitial spaces or alveolar air sacs of the lungs.',
    clinicalMeaning: 'Most commonly caused by acute or decompensated congestive heart failure with elevated pulmonary capillary wedge pressure (cardiogenic edema). Non-cardiogenic causes include ARDS, severe volume overload, sepsis, and renal failure.',
    modelScoreMeaning: 'The continuous score reflects visual density patterns corresponding to perihilar bat-wing haziness, vascular pedicle enlargement, and loss of distinct vessel margins.',
    thresholdMeaning: 'The experimental threshold of 0.8770 (87.70%) represents the calibrated decision boundary balancing early interstitial congestion detection against normal vascular fullness.',
    diagnosticCaveat: 'Subtle interstitial edema is difficult to distinguish from chronic interstitial lung disease, fluid overload from IV hydration, or bilateral viral pneumonia on a single radiograph without clinical history and BNP/NT-proBNP correlation.',
    radiographicSigns: [
      'Bilateral perihilar indistinctness (bat-wing / butterfly pattern)',
      'Kerley B lines (short horizontal peripheral linear opacities)',
      'Peribronchial cuffing and pleural fluid accumulation'
    ],
    calibratedThreshold: 0.876953,
  },
  Consolidation: {
    name: 'Consolidation',
    patientFriendlyName: 'Dense Lung Opacity / Airspace Filling',
    whatItIs: 'A region of normally compressible lung tissue that has become filled with liquid (inflammatory exudate, transudate, pus, blood, or neoplastic cells) instead of air.',
    clinicalMeaning: 'Classic hallmark of bacterial lobar pneumonia. Also seen in pulmonary contusion, pulmonary hemorrhage, severe aspiration, or organizing pneumonia.',
    modelScoreMeaning: 'Reflects dense, non-homogenous or homogenous radiographic attenuation that obscures underlying pulmonary vascular structures while sometimes sparing air bronchograms.',
    thresholdMeaning: 'The experimental threshold of 0.7329 (73.29%) was tuned on the NIH test set to separate dense alveolar filling from minor atelectatic bands or overlying soft tissue densities.',
    diagnosticCaveat: 'Consolidation and obstructive atelectasis frequently co-exist or look identical on plain films. A positive flag does not identify the causal organism (e.g., bacterial vs fungal vs viral).',
    radiographicSigns: [
      'Air bronchograms (branching radiolucent bronchial columns)',
      'Silhouette sign (effacement of adjacent diaphragm or cardiac border)',
      'Dense homogeneous lobar or segmental opacification'
    ],
    calibratedThreshold: 0.732910,
  },
  Pneumonia: {
    name: 'Pneumonia',
    patientFriendlyName: 'Pulmonary Infection / Infiltrate',
    whatItIs: 'Acute microbial infection and inflammation of the pulmonary parenchyma involving terminal bronchioles and alveoli.',
    clinicalMeaning: 'Presents clinically with fever, productive cough, pleuritic chest pain, and dyspnea. Etiologies span bacterial (Streptococcus pneumoniae), atypical, and viral pathogens.',
    modelScoreMeaning: 'The model score is derived from pattern recognition of focal alveolar opacities, patchy bronchopneumonic infiltrates, and air-space consolidation features.',
    thresholdMeaning: 'The experimental threshold of 0.7529 (75.29%) represents the empirical operating point selected on the NIH validation set.',
    diagnosticCaveat: 'Pneumonia is a CLINICAL-RADIOGRAPHIC diagnosis. It cannot be diagnosed from an X-ray alone without clinical signs of infection (fever, leukocytosis, crackles). Conversely, early dehydration or immunocompromised status can delay radiographic findings.',
    radiographicSigns: [
      'Patchy or confluent airspace opacities',
      'Segmental or lobar distribution without marked volume loss',
      'Associated parapneumonic pleural fluid or reactive effusion'
    ],
    calibratedThreshold: 0.752930,
  },
  Atelectasis: {
    name: 'Atelectasis',
    patientFriendlyName: 'Partial Lung Collapse / Volume Loss',
    whatItIs: 'Loss of lung volume resulting from the collapse of previously expanded pulmonary tissue, either involving subsegmental plates, a whole lobe, or an entire lung.',
    clinicalMeaning: 'Extremely common post-operatively, after general anesthesia, from shallow hypoventilation, mucus plugging, extrinsic airway compression (lymphadenopathy, tumor), or pleural effusion.',
    modelScoreMeaning: 'Neural network activation triggered by linear plate-like opacities (discoid atelectasis), triangular lobar collapse shadows, and secondary shifts of fissures and mediastinum.',
    thresholdMeaning: 'The experimental threshold of 0.7495 (74.95%) filters out routine baseline dependent discoid basilar atelectasis so that only prominent volume loss triggers an experimental flag.',
    diagnosticCaveat: 'Very high prevalence in supine or ICU radiographs. A high score indicates volume loss but does NOT distinguish benign hypoventilation from an obstructing central airway lesion.',
    radiographicSigns: [
      'Linear horizontal opacities (discoid/Platelike atelectasis) at lung bases',
      'Displacement of interlobar fissures toward the collapsed region',
      'Elevation of the hemidiaphragm and ipsilateral mediastinal shift'
    ],
    calibratedThreshold: 0.749512,
  },
  Pneumothorax: {
    name: 'Pneumothorax',
    patientFriendlyName: 'Trapped Air in Pleural Space / Collapsed Lung',
    whatItIs: 'Presence of gas or air in the pleural cavity between the visceral and parietal pleura, disrupting the negative intrapleural pressure and causing partial or complete lung collapse.',
    clinicalMeaning: 'Can be spontaneous (primary in tall thin individuals, secondary in COPD), traumatic, or iatrogenic (after central venous line placement or lung biopsy). Tension pneumothorax is a critical surgical emergency requiring immediate needle decompression.',
    modelScoreMeaning: 'Probability responding to a thin, razor-sharp visceral pleural line separated from the chest wall by a completely radiolucent (black) peripheral zone devoid of pulmonary vascular markings.',
    thresholdMeaning: 'The experimental threshold of 0.5825 (58.25%) is lower than other pathologies to maintain high sensitivity for this acute and time-critical condition.',
    diagnosticCaveat: 'Skin folds, companion shadows of ribs, patient garments, or external plastic tubing can mimic a pneumothorax line (pseudopneumothorax). Expiratory or lateral decubitus views are often required for subtle apical pneumothoraces.',
    radiographicSigns: [
      'Sharp, thin white visceral pleural line',
      'Total absence of bronchovascular markings peripheral to the pleural line',
      'Contralateral mediastinal shift and diaphragm depression in tension cases'
    ],
    calibratedThreshold: 0.582520,
  },
};

export const MEDICAL_GLOSSARY: GlossaryTerm[] = [
  {
    term: 'Model Probability',
    category: 'technical',
    shortDefinition: 'Continuous mathematical output between 0.00 and 1.00 produced by the neural network.',
    detailedExplanation: 'Computed using the Sigmoid activation function applied to raw model logits. A probability of 0.75 does NOT mean a 75% clinical chance the patient has the disease; it indicates where the model’s internal pattern activation lies on a 0 to 1 scale.',
  },
  {
    term: 'Calibrated Decision Threshold',
    category: 'technical',
    shortDefinition: 'A pre-set cutoff point used to convert a continuous probability into an experimental flag.',
    detailedExplanation: 'Determined using Youden’s J statistic (Sensitivity + Specificity - 1) on the held-out NIH ChestX-ray14 dataset. If a model output is above this cutoff, it produces an experimental flag; below it, no flag is produced.',
  },
  {
    term: 'Experimental Threshold Flag',
    category: 'technical',
    shortDefinition: 'A binary true/false indicator showing whether the model probability met or exceeded the calibration threshold.',
    detailedExplanation: 'This is an algorithmic flag created for benchmark research and triage testing. It is NOT a clinical diagnosis and does not confirm pathology.',
  },
  {
    term: 'Clinical Diagnosis',
    category: 'clinical',
    shortDefinition: 'A comprehensive medical determination made by a licensed healthcare provider.',
    detailedExplanation: 'Synthesizes physical examination, medical history, vital signs, laboratory tests, and multi-view imaging interpreted by a qualified radiologist. An AI model output alone cannot make a clinical diagnosis.',
  },
  {
    term: 'AUROC (Area Under ROC Curve)',
    category: 'technical',
    shortDefinition: 'A metric measuring the model’s ability to discriminate between patients with and without a condition.',
    detailedExplanation: 'Ranges from 0.5 (random guessing) to 1.0 (perfect classification). DenseNet121 achieved an overall macro AUROC of 0.8206 on the NIH held-out test split, but dropped to 0.6277 when tested on external Stanford CheXpert data due to domain shift.',
  },
  {
    term: 'Youden Index (J)',
    category: 'technical',
    shortDefinition: 'An objective mathematical criterion for selecting optimal decision thresholds.',
    detailedExplanation: 'Defined as J = Sensitivity + Specificity - 1. It identifies the operating threshold on the Receiver Operating Characteristic (ROC) curve where the balance between true positives and false positives is maximized.',
  },
  {
    term: 'Domain Shift',
    category: 'technical',
    shortDefinition: 'Performance degradation when an AI model encounters data from a different hospital or scanner.',
    detailedExplanation: 'Occurs because different hospitals utilize different X-ray machines, beam energies, patient demographics, and radiologist labeling guidelines. A model performing with 0.82 AUROC on NIH may perform with 0.62 AUROC on CheXpert.',
  },
  {
    term: 'PA Projection (Posteroanterior)',
    category: 'clinical',
    shortDefinition: 'Standard chest X-ray taken with X-rays traveling from the patient’s back to the chest detector.',
    detailedExplanation: 'The gold standard for thoracic imaging. Minimizes heart magnification and allows clear assessment of lung fields and cardiac borders.',
  },
  {
    term: 'AP Projection (Anteroposterior)',
    category: 'clinical',
    shortDefinition: 'Chest X-ray taken with X-rays passing from the front of the chest to a detector placed behind the back.',
    detailedExplanation: 'Commonly used in portable or bedbound ICU examinations. Naturally causes geometric magnification of the heart silhouette, often mimicking cardiomegaly.',
  },
  {
    term: 'Lateral Chest Radiograph',
    category: 'clinical',
    shortDefinition: 'Side-view chest X-ray that provides three-dimensional depth perspective.',
    detailedExplanation: 'Crucial for inspecting anatomical regions obscured on frontal views: the retrosternal clear space, retrocardiac space, and posterior costophrenic angles.',
  },
];

