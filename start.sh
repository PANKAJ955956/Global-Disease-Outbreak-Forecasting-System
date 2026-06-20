#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

echo "========================================="
echo " GDOFS CONTAINER BOOTSTRAP STARTING      "
echo "========================================="

# 1. Run the data ingestion, preprocessing, and model training pipeline
echo "Step 1: Running Model compilation pipeline..."
python run_pipeline.py

# 2. Spin up FastAPI REST endpoints in the background
echo "Step 2: Starting FastAPI Service on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# 3. Spin up Streamlit Dashboard in the foreground
echo "Step 3: Starting Streamlit Dashboard on port 8501..."
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
