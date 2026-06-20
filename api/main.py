import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import pipeline functions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DB_PATH, TARGET_COUNTRIES, TARGET_DISEASES
from src.predict import predict_risk, forecast_lstm
from src.alerts import check_and_trigger_alerts

# Initialize FastAPI App
app = FastAPI(
    title="Global Disease Outbreak Forecasting System (GDOFS) API",
    description="AI-powered public health platform for outbreak risk classification and case-rate forecasting.",
    version="1.0.0"
)

# Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class PredictRequest(BaseModel):
    country: str = Field(..., example="India")
    disease: str = Field(..., example="Dengue")
    season: str = Field(..., example="Summer")
    temperature: float = Field(..., example=30.5)
    humidity: float = Field(..., example=82.0)
    rainfall: float = Field(..., example=120.0)
    pressure: float = Field(1008.5, example=1008.5)
    wind_speed: float = Field(6.2, example=6.2)
    model_type: Optional[str] = Field("rf", example="rf", description="Model to use: 'rf' (Random Forest) or 'xgb' (XGBoost)")

class PredictResponse(BaseModel):
    country: str
    disease: str
    risk_level: str
    risk_score: float
    outbreak_probability: float
    all_probabilities: Dict[str, float]
    alert_triggered: bool
    alert_message: Optional[str] = None

class ForecastRequest(BaseModel):
    country: str = Field(..., example="India")
    disease: str = Field(..., example="Dengue")
    weeks_ahead: int = Field(4, ge=1, le=8, example=4)

class ForecastPoint(BaseModel):
    week_offset: int
    date: str
    predicted_case_rate: float

class ForecastResponse(BaseModel):
    country: str
    disease: str
    forecasts: List[ForecastPoint]

class AlertRecord(BaseModel):
    id: int
    timestamp: str
    country: str
    disease: str
    risk_level: str
    risk_score: float
    outbreak_probability: float
    message: str
    status: str

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Global Disease Outbreak Forecasting System (GDOFS)",
        "docs": "/docs"
    }

@app.get("/countries")
def get_metadata():
    """
    Returns tracked countries and pathogens.
    """
    return {
        "countries": TARGET_COUNTRIES,
        "diseases": TARGET_DISEASES
    }

@app.post("/predict", response_model=PredictResponse)
def api_predict_risk(req: PredictRequest):
    """
    Classifies the current risk of an outbreak based on weather conditions.
    Lags and rolling averages are dynamically fetched from the database
    to construct the full feature vector.
    """
    # Verify inputs are tracked
    if req.country not in TARGET_COUNTRIES or req.disease not in TARGET_DISEASES:
        raise HTTPException(status_code=400, detail="Country or disease is not tracked in the current GDOFS config.")
        
    # Fetch latest features from SQLite to populate lags and rolling mean/std
    conn = sqlite3.connect(DB_PATH)
    try:
        query = """
            SELECT * FROM features 
            WHERE country = ? AND disease = ? 
            ORDER BY date DESC LIMIT 1
        """
        latest_row = pd.read_sql_query(query, conn, params=(req.country, req.disease))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database lookup failed: {e}")
    finally:
        conn.close()
        
    if latest_row.empty:
        raise HTTPException(status_code=404, detail="No historical feature tracking found for the specified country and disease. Run the pipeline first.")
        
    # Build complete feature dictionary for predict_risk
    # We override the weather parameters with user-provided POST values
    input_data = latest_row.iloc[0].to_dict()
    input_data["temperature"] = req.temperature
    input_data["humidity"] = req.humidity
    input_data["rainfall"] = req.rainfall
    input_data["pressure"] = req.pressure
    input_data["wind_speed"] = req.wind_speed
    input_data["season"] = req.season
    
    # Recalculate environmental indices
    input_data["humidity_index"] = req.humidity / 100.0
    input_data["rainfall_index"] = req.rainfall / 100.0
    input_data["temp_humidity_index"] = req.temperature * input_data["humidity_index"]
    
    # Execute risk classification
    pred_res = predict_risk(input_data, model_type=req.model_type)
    if "error" in pred_res:
        raise HTTPException(status_code=500, detail=pred_res["error"])
        
    # Check alert engine
    alert_res = check_and_trigger_alerts(
        country=req.country,
        disease=req.disease,
        risk_level=pred_res["risk_level"],
        risk_score=pred_res["risk_score"],
        outbreak_prob=pred_res["outbreak_probability"]
    )
    
    return PredictResponse(
        country=req.country,
        disease=req.disease,
        risk_level=pred_res["risk_level"],
        risk_score=pred_res["risk_score"],
        outbreak_probability=pred_res["outbreak_probability"],
        all_probabilities=pred_res["all_probabilities"],
        alert_triggered=alert_res.get("alert_triggered", False),
        alert_message=alert_res.get("message")
    )

@app.post("/forecast", response_model=ForecastResponse)
def api_forecast_lstm(req: ForecastRequest):
    """
    Generates multi-week autoregressive case-rate predictions using the Stacked LSTM network.
    """
    if req.country not in TARGET_COUNTRIES or req.disease not in TARGET_DISEASES:
        raise HTTPException(status_code=400, detail="Country or disease is not tracked in the current GDOFS config.")
        
    # Fetch last 12 weeks of case rates
    conn = sqlite3.connect(DB_PATH)
    try:
        query = """
            SELECT case_rate, date FROM features 
            WHERE country = ? AND disease = ? 
            ORDER BY date DESC LIMIT 12
        """
        history_df = pd.read_sql_query(query, conn, params=(req.country, req.disease))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database lookup failed: {e}")
    finally:
        conn.close()
        
    if history_df.empty:
        raise HTTPException(status_code=404, detail="No historical case rate sequences found for this target. Run ingestion first.")
        
    # Reverse to chronological order (database DESC limit 12 gives descending, we need ascending)
    history_df = history_df.iloc[::-1].reset_index(drop=True)
    recent_rates = history_df["case_rate"].tolist()
    
    # Execute LSTM autoregressive forecast
    forecasts = forecast_lstm(recent_rates, weeks_ahead=req.weeks_ahead)
    if isinstance(forecasts, dict) and "error" in forecasts:
        raise HTTPException(status_code=500, detail=forecasts["error"])
        
    # Build date offsets (increment by 7 days for weekly increments)
    last_date_str = history_df["date"].values[-1]
    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
    
    points = []
    for i, rate in enumerate(forecasts):
        forecast_date = last_date + timedelta(weeks=i+1)
        points.append(ForecastPoint(
            week_offset=i+1,
            date=forecast_date.strftime("%Y-%m-%d"),
            predicted_case_rate=rate
        ))
        
    return ForecastResponse(
        country=req.country,
        disease=req.disease,
        forecasts=points
    )

@app.get("/risk", response_model=PredictResponse)
def api_get_current_risk(
    country: str = Query(..., example="India"),
    disease: str = Query(..., example="Dengue"),
    model_type: str = Query("rf", example="rf")
):
    """
    Retrieves risk level for country and disease based on the latest database snapshot.
    """
    # Fetch latest features from SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        query = """
            SELECT * FROM features 
            WHERE country = ? AND disease = ? 
            ORDER BY date DESC LIMIT 1
        """
        latest_row = pd.read_sql_query(query, conn, params=(country, disease))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database lookup failed: {e}")
    finally:
        conn.close()
        
    if latest_row.empty:
        raise HTTPException(status_code=404, detail="No feature tracking found for the specified target.")
        
    input_data = latest_row.iloc[0].to_dict()
    
    # Execute risk classification
    pred_res = predict_risk(input_data, model_type=model_type)
    if "error" in pred_res:
        raise HTTPException(status_code=500, detail=pred_res["error"])
        
    # Check alerts
    alert_res = check_and_trigger_alerts(
        country=country,
        disease=disease,
        risk_level=pred_res["risk_level"],
        risk_score=pred_res["risk_score"],
        outbreak_prob=pred_res["outbreak_probability"]
    )
    
    return PredictResponse(
        country=country,
        disease=disease,
        risk_level=pred_res["risk_level"],
        risk_score=pred_res["risk_score"],
        outbreak_probability=pred_res["outbreak_probability"],
        all_probabilities=pred_res["all_probabilities"],
        alert_triggered=alert_res.get("alert_triggered", False),
        alert_message=alert_res.get("message")
    )

@app.get("/alerts", response_model=List[AlertRecord])
def api_get_alerts(limit: int = Query(50, ge=1, le=200)):
    """
    Retrieves the record list of triggered health warnings.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?"
        alerts_df = pd.read_sql_query(query, conn, params=(limit,))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query alerts database: {e}")
    finally:
        conn.close()
        
    records = []
    for _, row in alerts_df.iterrows():
        records.append(AlertRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            country=row["country"],
            disease=row["disease"],
            risk_level=row["risk_level"],
            risk_score=row["risk_score"],
            outbreak_probability=row["outbreak_probability"],
            message=row["message"],
            status=row["status"]
        ))
        
    return records
