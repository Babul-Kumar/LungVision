# LungVision AI — Frontend Visual Design System & UX Specification (DESIGN.md)

**Document Version:** 1.0.0  
**Status:** Approved / Active Governance  
**Author:** Staff Frontend Architect & Medical UI/UX Lead  
**Governing Skill:** Gravitas Engineering Reliability Runtime  
**Created:** 2026-11-12  

---

## 1. Visual Identity & Design Philosophy

LungVision AI is styled as a state-of-the-art clinical intelligence workstation. The aesthetic draws inspiration from modern medical PACS workstations, diagnostic imaging consoles, and scientific computing interfaces:
- **Surface Palette:** Deep slate and midnight navy surfaces reduce ocular fatigue during prolonged radiologic image review.
- **Accents:** Crisp cyan (`#06b6d4`) and surgical teal (`#0d9488`) signal precision, active selection, and system readiness.
- **Pathology Signaling:** Dual-level status coding:
  - Normal / Below Threshold: Subtle Slate / Muted Emerald (`#10b981`).
  - Flagged / Above Threshold: High-visibility Amber (`#f59e0b`) or Clinical Coral (`#ef4444`).
- **Typography:** Modern clean sans-serif (`Inter`, `system-ui`) for clinical prose; high-precision tabular monospace (`JetBrains Mono`, `ui-monospace`) for Hounsfield Units, spatial millimetre coordinates, and sigmoid probabilities.

---

## 2. Color Palette & Design Tokens

```css
:root {
  /* Surfaces */
  --bg-primary: #070b14;       /* Main canvas background */
  --bg-secondary: #0d1424;     /* Card & panel surface */
  --bg-tertiary: #141f36;      /* Hover states & nested cards */
  --border-subtle: #1e293b;    /* Card & section borders */
  --border-active: #0284c7;    /* Selected viewport or focus outline */

  /* Text & Foreground */
  --text-primary: #f8fafc;     /* High contrast clinical text */
  --text-secondary: #94a3b8;   /* Labels, captions, descriptions */
  --text-muted: #64748b;       /* Metadata, timestamps, minor units */

  /* Medical Accents */
  --accent-cyan: #06b6d4;      /* Primary interactive elements, left lung */
  --accent-teal: #14b8a6;      /* Active tabs, right lung, healthy state */
  --accent-blue: #3b82f6;      /* Secondary buttons, links */
  --accent-amber: #f59e0b;     /* Positive pathology threshold flag */
  --accent-rose: #f43f5e;      /* Critical abnormality, error banner */
  --accent-emerald: #10b981;   /* Normal diagnosis, online status */
}
```

---

## 3. Typography & Hierarchy

| Role | Font Family | Size / Line Height | Weight | Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Display Title** | Sans-serif | 28px / 36px | Bold (700) | Main module headers |
| **Section Header**| Sans-serif | 18px / 26px | Semi-Bold (600)| Card titles, viewport labels |
| **Body Primary**  | Sans-serif | 14px / 20px | Regular (400) | Descriptions, clinical notes |
| **Body Small**    | Sans-serif | 12px / 16px | Medium (500) | Secondary labels, badges |
| **Numerical Data**| Monospace | 13px / 18px | Medium (500) | Probabilities, mm coords, HU |

---

## 4. Reusable Component System

### 4.1 Buttons (`Button`)
- `primary`: Solid cyan background with dark text, subtle glow on hover.
- `secondary`: Slate background with subtle border, light text.
- `outline`: Transparent with subtle border, cyan text on hover.
- `danger`: Red outline/fill for resetting data.

### 4.2 Diagnostic Metric Card (`ProbabilityBar`)
- Displays pathology name, continuous probability $[0.0, 1.0]$, and a dual-colored progress track.
- A vertical tick marker indicates the exact **calibrated NIH threshold**.
- If `probability >= threshold`, the bar shifts to Amber/Coral with an `ALERT` badge.

### 4.3 Viewport Container (`MPRViewport`)
- High-contrast canvas container with overlaid slice index indicators, physical coordinate readout $(Z\text{ mm})$, and synchronized crosshairs.
- Top toolbar for window presets (`Lung`, `Mediastinum`, `Bone`) and overlay toggles (`Mask`, `Nodules`).

### 4.4 3D Mesh Viewer (`ThreeCanvas`)
- Embedded WebGL viewport featuring interactive OrbitControls, FPS counter, structure selector pill buttons, and reset view trigger.

---

## 5. Page Layouts & Interaction Flows

### 5.1 Main Layout (`AppLayout`)
- **Top Navigation Bar:** Persistent brand title `LUNGVISION AI`, global system health pill (`Online / Degraded`), quick action links, settings gear icon.
- **Left Sidebar:** Icon-accompanied navigation tabs with active indicator glow (`Dashboard`, `X-Ray Screening`, `3D CT Explorer`, `Research Benchmarks`, `Model Registry`, `History`).
- **Main Viewport:** Generously padded responsive workspace with subtle entrance fade transitions.
- **Persistent Footer:** Regulatory disclaimer: *"Experimental Medical AI Decision-Support Platform. Not for Diagnostic Use."*

### 5.2 X-Ray Analysis Flow
1. User drops an X-ray image into the upload dropzone.
2. Client validates magic bytes and displays instant thumbnail preview.
3. User selects model (Multi-label DenseNet121 or Binary ResNet18).
4. User clicks "Run AI Screening".
5. Real-time inference executed against FastAPI backend.
6. Results rendered instantly with probability meters, threshold markers, and experimental flags.

### 5.3 3D CT Exploration Flow
1. User selects demo scan (`demo_thoracic_scan`) or uploads a custom CT archive.
2. System loads volume dimensions, voxel spacing, and available labels.
3. Multi-Planar Reconstruction (MPR) panel renders synchronized Axial, Coronal, Sagittal views with slider controls.
4. Three.js canvas streams the 3D Wavefront OBJ mesh and renders anatomical lungs with LUNA16 nodule markers.
5. Volumetry card displays calculated Left, Right, and Total lung volumes in mL.

