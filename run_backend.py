"""
LungVision AI - Direct Backend Launcher
Launches the verified FastAPI backend on http://127.0.0.1:8000
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    import uvicorn
    print("==================================================")
    print("   Starting LungVision AI Research Backend API    ")
    print("   Endpoint: http://127.0.0.1:8000                ")
    print("   Docs:     http://127.0.0.1:8000/docs           ")
    print("==================================================")
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

