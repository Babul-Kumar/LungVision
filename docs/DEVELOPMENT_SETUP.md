# LungVision AI — Development & Deployment Setup

## 1. System Requirements

- **Operating System**: Windows 10/11, macOS 12+, or Ubuntu 20.04+ Linux
- **Python**: 3.11, 3.12, 3.13, or 3.14 (64-bit)
- **Node.js**: 18.x, 20.x, or 22.x LTS
- **Package Managers**: `pip` and `npm`

---

## 2. Environment Setup

### 2.1 Backend Setup (Python)

1. Clone or navigate to the repository directory:
   ```bash
   cd "computer vision"
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   - **Linux / macOS**:
     ```bash
     source .venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

5. Verify the backend test suite:
   ```bash
   python -m pytest backend/tests -v
   ```
   *Expected outcome: 29 passed.*

6. Start the FastAPI development gateway:
   ```bash
   uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   Or simply run the provided batch script on Windows:
   ```cmd
   start_backend.bat
   ```

---

### 2.2 Frontend Setup (Node.js & React)

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Verify static TypeScript types:
   ```bash
   npx tsc --noEmit
   ```
   *Expected outcome: Exits cleanly with zero errors.*

4. Run the frontend test suite:
   ```bash
   npm test -- --run
   ```
   *Expected outcome: 8 test files, 15 tests passed.*

5. Build the production bundle:
   ```bash
   npm run build
   ```
   *Expected outcome: Assets compiled into `frontend/dist/` with code-split chunks.*

6. Start the local Vite development server:
   ```bash
   npm run dev
   ```
   Or simply run the provided batch script on Windows:
   ```cmd
   start_frontend.bat
   ```

---

## 3. Configuration & Environment Variables

The backend supports configuration via environment variables or a `.env` file at the root.

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HOST` | `127.0.0.1` | Gateway server binding address |
| `PORT` | `8000` | Gateway server port |
| `DEBUG` | `False` | Verbose debug and trace logs |
| `TORCH_DEVICE` | `auto` | Execution device (`auto`, `cpu`, `cuda`) |
| `MAX_IMAGE_SIZE_BYTES` | `20971520` (20MB) | Max allowed 2D radiograph upload size |
| `MAX_CT_SIZE_BYTES` | `157286400` (150MB) | Max allowed 3D CT volume archive size |

Frontend configuration is set in `frontend/src/config/environment.ts` or via Vite environment variables:
- `VITE_API_BASE_URL`: Defaults to `http://localhost:8000/api/v1`.

---

## 4. Health & Verification Probes

Once both processes are active, inspect the following diagnostic endpoints in your browser:
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`
- **Backend Liveness Probe**: `http://localhost:8000/api/v1/health`
- **Model Readiness Probe**: `http://localhost:8000/api/v1/ready`
- **Frontend Medical Workstation**: `http://localhost:5173`

