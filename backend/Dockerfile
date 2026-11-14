# Python 3.11-slim container for LungVision AI Backend
FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install system runtime dependencies for OpenCV, SimpleITK, and healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend application code
COPY backend /app/backend

# Copy required research artifacts and checkpoints
COPY lungVision /app/lungVision
COPY lungvision_disease_classification /app/lungvision_disease_classification
COPY LungVision_AI_Final_Research_Bundle /app/LungVision_AI_Final_Research_Bundle
COPY LungVision_AI_Model_Registry /app/LungVision_AI_Model_Registry
COPY data /app/data

# Expose FastAPI backend port
EXPOSE 8000

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch uvicorn
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

