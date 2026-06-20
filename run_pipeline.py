import os
import sys
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")

def run_step(step_name, module_path, func_name="main"):
    """
    Dynamically loads and runs a pipeline step function.
    """
    logger.info(f"====================================================================")
    logger.info(f"STARTING PIPELINE STEP: {step_name}")
    logger.info(f"====================================================================")
    start_time = time.time()
    
    try:
        # Import the module dynamically
        # Add src folder to sys path if not already there
        src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
        if src_path not in sys.path:
            sys.path.append(src_path)
            
        import importlib
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        
        # Execute the function
        func()
        
        duration = time.time() - start_time
        logger.info(f"✅ SUCCESS: {step_name} completed in {duration:.2f} seconds.")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ FAILED: {step_name} crashed after {duration:.2f} seconds.")
        logger.error(f"Error details: {e}")
        # Print stack trace for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    pipeline_start = time.time()
    logger.info("Initializing GDOFS End-to-End Orchestrator Pipeline")
    
    # 1. Run Data Ingestion
    run_step("1. Ingestion Layer (WHO/WB/Weather APIs & Simulators)", "src.ingest")
    
    # 2. Run Data Cleaning & Standardization
    run_step("2. Preprocessing & Clean Layer (standardization, rates, merges)", "src.clean")
    
    # 3. Run Feature Engineering & Labeling
    run_step("3. Feature Extraction Layer (lags, rolling mean/std, season/growth rates)", "src.features")
    
    # 4. Train Random Forest & XGBoost Classifiers
    run_step("4. ML Classification Training Layer (RF & XGBoost Risk)", "src.train_rf")
    
    # 5. Train Stacked LSTM Sequence Forecaster
    run_step("5. Deep Learning LSTM Forecaster Layer (PyTorch Stacked LSTM)", "src.train_lstm")
    
    # 6. Fit ARIMA & SARIMA Baselines for Comparison
    run_step("6. Advanced Statistical Forecaster Comparison (ARIMA/SARIMA vs. LSTM)", "src.train_advanced")
    
    total_duration = time.time() - pipeline_start
    logger.info(f"====================================================================")
    logger.info(f"🎉 GDOFS PIPELINE EXECUTION SUCCESSFUL. Total Time: {total_duration/60.0:.2f} minutes.")
    logger.info(f"====================================================================")

if __name__ == "__main__":
    main()
