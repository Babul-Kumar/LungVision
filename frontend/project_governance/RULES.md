# LungVision AI — Frontend Engineering & Architectural Rules (RULES.md)

**Document Version:** 1.0.0  
**Status:** Mandatory Compliance  
**Author:** Staff Frontend Architect & Medical UI/UX Lead  
**Governing Skill:** Gravitas Engineering Reliability Runtime  
**Created:** 2026-11-12  

---

## 1. Core Engineering Commandments

1. **Never Fabricate Model Outputs or Statistics:** All probabilities, thresholds, flags, and measurements presented in the UI must originate directly from verified backend endpoints. Never hardcode mock patient findings or fabricate clinical inferences.
2. **Never Claim Clinical Diagnostic Efficacy:** All UI headers, cards, and export views must carry the clear regulatory disclaimer:  
   *"LungVision AI is an experimental decision-support research platform. Not approved for clinical diagnosis or treatment."*
3. **Lock 6-Disease Label Ordering:** The multi-label DenseNet121 output labels must strictly retain their verified NIH order across all tables, lists, and charts:
   1. `Cardiomegaly`
   2. `Edema`
   3. `Consolidation`
   4. `Pneumonia`
   5. `Atelectasis`
   6. `Pneumothorax`
4. **Preserve Backend Code Integrity:** Never modify existing backend models, weights, thresholds, or API routes to simplify frontend code. If an API contract discrepancy occurs, document the finding and propose the minimal backward-compatible patch.
5. **Strict TypeScript Standards:** Never use `any` as an escape hatch for typing issues. All API payloads, component props, and event handlers must have explicit, strictly checked TypeScript types.
6. **No Mock Prediction Fallbacks in Production:** In the event of a backend connection error, display a clear, informative error state. Never fall back to fake prediction numbers.
7. **Clean Three.js WebGL Resource Disposal:** Every Three.js scene, geometry, material, and texture must be explicitly disposed of on component unmounting to prevent memory leaks and WebGL context crashes.
8. **Responsive Across All Form Factors:** Every view must seamlessly adapt between Desktop ($\ge 1280\text{px}$), Tablet ($768\text{px} - 1279\text{px}$), and Mobile ($< 768\text{px}$).

---

## 2. Component Design & Architecture Rules

- **Clean Layer Separation:** Visual UI components must not execute direct fetch/axios calls. All network requests must flow through `src/services/api/` modules.
- **Pure & Accessible Markup:** Use semantic HTML (`<main>`, `<nav>`, `<aside>`, `<section>`, `<article>`, `<button>`). Ensure form inputs have matching `<label>` elements and buttons have clear `aria-label` tags.
- **Restrained Motion:** Keep animations subtle, functional, and purposeful (e.g. smooth tab transitions, loading skeletons). Avoid distracting decorative loops.
- **Fail-Safe State Management:** Every asynchronous view must handle three standard states: `loading`, `error`, and `data`. Never leave blank uninformative screens during network requests.

---

## 3. Medical Imaging & CT Representation Rules

- **Respect Coordinate Conventions:** Medical volumes use $(Z, Y, X)$ array memory order and physical $(X, Y, Z)$ coordinates in millimeters. Never accidentally mirror left and right lungs.
- **Distinguish Annotations from Predictions:** LUNA16 ground-truth nodule coordinates must be clearly labelled as reference spatial annotations, never as automated model detections.
- **Hounsfield Unit Color Consistency:** Use standard clinical windowing presets (`lung`: -600/1500, `mediastinum`: 50/350, `bone`: 400/1800) and standard contour colors for segmentation overlays (Left lung: cyan/blue, Right lung: teal/green, Trachea: amber/orange).

