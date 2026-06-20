# Multi-service Python container for GDOFS
FROM python:3.11-slim

# Install system dependencies (build-essential for standard libraries if required)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency specifications
COPY requirements.txt .

# Install dependencies (use --no-cache-dir to keep image light)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application directories and scripts
COPY api/ ./api/
COPY app/ ./app/
COPY config/ ./config/
COPY src/ ./src/
COPY tests/ ./tests/
COPY run_pipeline.py .

# Create log and model directories
RUN mkdir -p logs models data/raw data/processed data/db

# Copy entrypoint runner script
COPY start.sh .
RUN chmod +x start.sh

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000
EXPOSE 8501

# Run the entrypoint script
CMD ["./start.sh"]
