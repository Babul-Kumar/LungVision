# LungVision AI — Dependency & Execution Environment Forensic Audit

**Audit Date:** November 13, 2026  
**Auditor:** Principal Software Architect & MLOps Environment Specialist  
**Runtime:** Gravitas Engineering Reliability Runtime (`udaysharmadev/Gravitas`)  
**Operating System:** Windows 10/11 Enterprise x86_64  

---

## 1. Python Execution Environments & AppLocker Audit

The audit detected multiple Python environments on the host system. **A critical Application Control (AppLocker / WDAC) policy was identified that affects Python 3.14 DLL loading.**

| Python Binary | Version | Status | AppLocker / Policy Behavior | Recommendation |
| :--- | :--- | :---: | :--- | :--- |
| `C:\Python314\python.exe` | Python 3.14.0a2 | **RESTRICTED** | Blocks DLL imports (`SimpleITK`, `scipy`) via AppLocker | **DO NOT USE FOR ML PIPELINES** |
| `C:\Users\babul\AppData\Local\Programs\Python\Python311\python.exe` | **Python 3.11.9** | **FULLY FUNCTIONAL** | All C-extensions & DLLs load cleanly without restriction | **APPROVED DEFAULT RUNTIME** |

### Python Package Compatibility Matrix (Python 3.11.9 Environment)
| Package | Installed Version | Required in `backend/requirements.txt` | Import Status | Purpose |
| :--- | :-: | :-: | :---: | :--- |
| **`torch`** | `2.6.0+cpu` | `>=2.0.0` | VERIFIED | Deep learning inference & loss backpropagation |
| **`torchvision`** | `0.21.0+cpu` | `>=0.15.0` | VERIFIED | Model architectures (DenseNet121, ResNet18) |
| **`SimpleITK`** | `2.5.7` | `>=2.2.0` | VERIFIED | Medical volume I/O (.mhd/.raw, .nii, spacing) |
| **`trimesh`** | `5.2.0` | `>=3.20.0` | VERIFIED | 3D mesh processing, manifold verification |
| **`scipy`** | `1.15.2` | `>=1.10.0` | VERIFIED | Spatial transformations & statistical testing |
| **`numpy`** | `1.26.4` | `>=1.24.0` | VERIFIED | N-dimensional array processing |
| **`fastapi`** | `0.115.6` | `>=0.100.0` | VERIFIED | High-performance asynchronous API framework |
| **`uvicorn`** | `0.34.0` | `>=0.22.0` | VERIFIED | ASGI web server |
| **`pydantic`** | `2.10.4` | `>=2.0.0` | VERIFIED | Data validation & API schema generation |
| **`scikit-image`**| `0.25.1` | `>=0.20.0` | VERIFIED | Marching Cubes 3D surface extraction |
| **`pillow`** | `11.1.0` | `>=9.5.0` | VERIFIED | Image manipulation & format conversion |

---

## 2. Frontend Execution Environment (Node.js & npm)

| Tool / Library | Version | Status | Location / Role |
| :--- | :-: | :---: | :--- |
| **Node.js** | `v20.x` / `v22.x` | VERIFIED | JavaScript runtime environment |
| **npm** | `10.x` | VERIFIED | Package manager |
| **React** | `18.3.1` | VERIFIED | Frontend component rendering engine |
| **TypeScript** | `5.3.3` | VERIFIED | Static type safety and contract enforcement |
| **Vite** | `5.1.4` | VERIFIED | Next-generation frontend build tool |
| **Three.js** | `0.160.0` | VERIFIED | 3D WebGL rendering engine |
| **Tailwind CSS**| `3.4.1` | VERIFIED | Utility-first medical design styling |
| **Vitest** | `1.2.2` | VERIFIED | Unit and component testing framework |
| **Lucide React**| `0.344.0` | VERIFIED | Clinical iconography system |

---

## 3. Hardware Compute Profile

- **Processor:** Intel(R) Core(TM) x86_64 architecture.
- **Physical RAM:** 16+ GB host memory.
- **GPU Acceleration:** CPU fallback active on local development workstation; GPU acceleration (NVIDIA CUDA) designed for cloud execution (Kaggle/Colab).
- **Disk I/O:** Fast NVMe SSD storage.
