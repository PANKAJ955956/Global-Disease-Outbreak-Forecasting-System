import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODELS_DIR, SEQUENCE_LENGTH

# PyTorch Stacked LSTM class definition
class StackedLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=3, output_dim=1):
        super(StackedLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def get_risk_label(class_idx):
    labels = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
    return labels.get(class_idx, "Unknown")

def predict_risk(input_data: dict, model_type="rf"):
    """
    Predicts the current outbreak risk level and score.
    Input dictionary should contain:
    - country (str)
    - disease (str)
    - season (str)
    - temperature (float)
    - humidity (float)
    - rainfall (float)
    - pressure (float)
    - wind_speed (float)
    - cases_lag1, case_rate_lag1, cases_lag2, case_rate_lag2, etc.
    - cases_rolling_mean_4, cases_rolling_std_4, case_rate_rolling_mean_4, case_rate_rolling_std_4, etc.
    - growth_rate (float)
    - month (int)
    - quarter (int)
    - humidity_index (float)
    - rainfall_index (float)
    - temp_humidity_index (float)
    - population_density (float)
    """
    # 1. Load models and encoders
    try:
        if model_type == "xgb":
            model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
        else:
            model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
            
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        le_country = joblib.load(os.path.join(MODELS_DIR, "label_encoder_country.pkl"))
        le_disease = joblib.load(os.path.join(MODELS_DIR, "label_encoder_disease.pkl"))
        le_season = joblib.load(os.path.join(MODELS_DIR, "label_encoder_season.pkl"))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    except Exception as e:
        return {"error": f"Failed to load modeling artifacts: {e}"}

    # 2. Encode categorical inputs
    data = input_data.copy()
    
    try:
        # Handle unseen values by falling back safely to index 0 or simple lookup
        try:
            data["country_enc"] = le_country.transform([data["country"]])[0]
        except Exception:
            data["country_enc"] = 0
            
        try:
            data["disease_enc"] = le_disease.transform([data["disease"]])[0]
        except Exception:
            data["disease_enc"] = 0
            
        try:
            data["season_enc"] = le_season.transform([data["season"]])[0]
        except Exception:
            data["season_enc"] = 0
    except Exception as e:
        return {"error": f"Failed to encode categories: {e}"}

    # 3. Create a pandas row in the exact correct order of training columns
    try:
        input_df = pd.DataFrame([data])
        # Ensure all columns exist, if not fill with 0
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0.0
                
        # Select features in the precise training order
        features_ordered = input_df[feature_cols]
        
        # 4. Scale features
        features_scaled = scaler.transform(features_ordered)
    except Exception as e:
        return {"error": f"Failed during feature parsing or scaling: {e}"}

    # 5. Model Inference
    try:
        risk_class = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Associated probability of the chosen class
        score = float(probabilities[risk_class]) * 100.0
        
        # Outbreak risk is the probability of High + Critical risk classes combined
        # which is extremely useful for alerting (gives the actual cumulative threat probability)
        # Class indices: 0: Low, 1: Medium, 2: High, 3: Critical
        outbreak_probability = (probabilities[2] + probabilities[3]) * 100.0
        
        return {
            "risk_level": get_risk_label(risk_class),
            "risk_score": round(score, 2),
            "outbreak_probability": round(outbreak_probability, 2),
            "all_probabilities": {get_risk_label(i): round(float(p) * 100.0, 2) for i, p in enumerate(probabilities)}
        }
    except Exception as e:
        return {"error": f"Model prediction failed: {e}"}

def forecast_lstm(recent_case_rates: list, weeks_ahead=4):
    """
    Runs multi-step autoregressive forecasting on case rates.
    """
    # 1. Load LSTM model and scaler
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
        model_path = os.path.join(MODELS_DIR, "lstm_model.pt")
        
        # Define model structure
        model = StackedLSTM(input_dim=1, hidden_dim=64, num_layers=3, output_dim=1)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
    except Exception as e:
        return {"error": f"Failed to load LSTM model or scaler: {e}"}

    # 2. Preprocess input sequence (need exactly last 12 values)
    if len(recent_case_rates) < SEQUENCE_LENGTH:
        # Pad with historical mean or zeroes if insufficient
        padding = [np.mean(recent_case_rates) if recent_case_rates else 0.0] * (SEQUENCE_LENGTH - len(recent_case_rates))
        recent_case_rates = padding + recent_case_rates
        
    tail = recent_case_rates[-SEQUENCE_LENGTH:]
    
    try:
        # Scale inputs
        scaled_seq = scaler.transform(np.array(tail).reshape(-1, 1)).flatten().tolist()
    except Exception as e:
        return {"error": f"Failed to scale input sequence: {e}"}

    # 3. Autoregressive prediction loop
    predictions = []
    current_seq = list(scaled_seq)
    
    try:
        with torch.no_grad():
            for _ in range(weeks_ahead):
                # Format sequence for PyTorch: batch_size=1, seq_length=12, features=1
                input_tensor = torch.tensor(current_seq[-SEQUENCE_LENGTH:], dtype=torch.float32).reshape(1, SEQUENCE_LENGTH, 1)
                pred_scaled = model(input_tensor).item()
                
                predictions.append(pred_scaled)
                # Append predicted value back to the sequence for the next step prediction
                current_seq.append(pred_scaled)
                
        # Inverse scale predictions back to original values
        forecasts = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        forecasts = np.clip(forecasts, 0.0, None) # Floor at 0.0 case rate
        
        return [round(float(f), 4) for f in forecasts]
    except Exception as e:
        return {"error": f"LSTM prediction loop failed: {e}"}
