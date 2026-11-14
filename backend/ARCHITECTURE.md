# LungVision AI Backend — Architecture & System Design

**Document Version:** 1.1.0  
**Status:** Implemented, Integrated & Verified  
**Author:** Staff Backend Architect & MLOps Lead  

---

## 1. System Topology

The LungVision AI Backend is an asynchronous microservice engineered with Python, FastAPI, and PyTorch, designed to decouple machine learning and 3D volumetric pipelines from client presentation layers.

```mermaid
graph TD
    Client["React + TypeScript Frontend (with Three.js Viewer)"]
    
    subgraph Gateway ["FastAPI Gateway (Uvicorn / Port 8000)"]
        CORS["CORS Middleware"]
        Router["/api/v1 Router"]
        ErrorH["RFC 7807 Exception Handlers"]
    end
    
    subgraph API_Endpoints ["API Endpoints"]
        HealthEP["/health & /ready"]
        XRayEP["/xray/predict & /xray/predict-binary"]
        CTEP["/ct/* (upload, metadata, slice, annotations, mesh, quality, measurements)"]
        ModelEP["/models & /models/{id}"]
        ResearchEP["/research/metrics"]
    end
    
    subgraph Services ["Application Service Layer"]
        XRaySvc["XRayService"]
        CTSvc["CTService Facade"]
        RegistrySvc["RegistryService"]
        Preproc["XRayPreprocessor (224x224 RGB ImageNet)"]
    end

    subgraph CT_Submodules ["Modular CT Pipeline (from Notebook Audit)"]
        CTLoader["CT Loader (SimpleITK & Reference Volume)"]
        CTGeom["CT Geometry (Physical-to-Voxel & ZYX Transforms)"]
        CTSlices["CT Slices (HU Windowing & Contour Overlays)"]
        CTAnn["CT Annotation (LUNA16 Nodule Mapping)"]
        CTMesh["CT Mesh (Padded Marching Cubes & Trimesh Audit)"]
        CTMeas["CT Measurement (Quantitative Volumes in mL)"]
    end
    
    subgraph MLEngine ["ML Runtime Engine"]
        Loader["ModelLoader (Singleton)"]
        DenseNet["DenseNet121 (6 Pathologies)"]
        ResNetOrig["ResNet18 (Original Binary)"]
        ResNetAug["ResNet18 (Blur-Augmented Binary)"]
    end

    Client -->|HTTP/REST Multipart & JSON| CORS
    CORS --> Router
    Router --> ErrorH
    
    Router --> HealthEP
    Router --> XRayEP
    Router --> CTEP
    Router --> ModelEP
    Router --> ResearchEP
    
    XRayEP --> XRaySvc
    CTEP --> CTSvc
    ModelEP --> RegistrySvc
    ResearchEP --> RegistrySvc
    
    CTSvc --> CTLoader
    CTSvc --> CTGeom
    CTSvc --> CTSlices
    CTSvc --> CTAnn
    CTSvc --> CTMesh
    CTSvc --> CTMeas

    XRaySvc --> Preproc
    Preproc --> Loader
    Loader --> DenseNet
    Loader --> ResNetOrig
    Loader --> ResNetAug
```

---

## 2. Integrated 3D CT Architecture (Ported from Interactive Notebook)

The 3D CT exploration engine preserves the mathematical rigor of the interactive notebook (`lungvision-interactive-viewer.ipynb`):

1. **Geometry Engine (`geometry.py`):**
   - Implements tolerant floating-point comparison (`atol=1e-3`) between CT and segmentation masks.
   - Converts physical coordinates $\mathbf{p} = (X, Y, Z)$ to continuous voxel coordinates via $\mathbf{c} = \mathbf{D}^{-1}(\mathbf{p} - \mathbf{O}) \oslash \mathbf{S}$.
   - Transforms marching cubes ZYX vertices into physical millimeters via $\mathbf{P} = \mathbf{O} + \mathbf{D}(\mathbf{v}_{xyz} \odot \mathbf{S})$.
2. **Multi-Planar Slice Service (`slice_service.py`):**
   - Generates Axial ($Z$), Coronal ($Y$), and Sagittal ($X$) cross-sections.
   - Applies standard clinical Hounsfield Unit windowing (Lung: -600/1500, Mediastinum: 50/350, Bone: 400/1800).
   - Alpha-blends color contours for Left Lung (label 3), Right Lung (label 4), and Trachea (label 5).
   - Projects intersecting LUNA16 nodule markers onto the 2D slice plane.
3. **Nodule Annotation Service (`annotation_service.py`):**
   - Parses LUNA16 CSV manifests, checks volume bounds, and samples the center label.
4. **Watertight Mesh Service (`mesh_service.py`):**
   - Utilizes 1-voxel constant zero padding before marching cubes to guarantee watertight surface extraction at CT boundary borders.
   - Evaluates mesh topology using Trimesh (watertightness, winding consistency, Euler number, degenerate faces).
   - Streams standard Wavefront OBJ files for direct consumption by Three.js.
5. **Quantitative Measurement Service (`measurement_service.py`):**
   - Computes exact anatomical lung volume in milliliters ($V = N \times s_x s_y s_z / 1000$).
   - Calculates lung symmetry balance ratios and label distribution breakdowns.
