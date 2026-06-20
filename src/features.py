import os
import sqlite3
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    DB_PATH, PROCESSED_DATA_DIR, MODELS_DIR,
    RISK_LOW, RISK_MEDIUM, RISK_HIGH
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("features")

# Country land areas in square kilometers (to calculate population density)
LAND_AREAS = {
    "India": 3287263,
    "Nigeria": 923768,
    "Brazil": 8515767,
    "Indonesia": 1904569,
    "Pakistan": 796095,
    "Bangladesh": 148460,
    "Ethiopia": 1104300,
    "Philippines": 300000
}

def get_season(month):
    """
    Returns seasonal category based on month.
    """
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def assign_risk(rate):
    """
    Categorizes the risk level based on case rate per 100k population.
    """
    if rate < RISK_LOW:
        return 0  # Low
    elif rate < RISK_MEDIUM:
        return 1  # Medium
    elif rate < RISK_HIGH:
        return 2  # High
    else:
        return 3  # Critical

def engineer_features(df):
    """
    Main feature engineering pipeline.
    """
    df = df.copy()
    
    # 1. Ensure date is datetime and sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["country", "disease", "date"]).reset_index(drop=True)
    
    # 2. Lag Features
    logger.info("Generating lag features...")
    grouped = df.groupby(["country", "disease"])
    
    for lag in [1, 2, 4, 8]:
        df[f"cases_lag{lag}"] = grouped["cases"].shift(lag)
        df[f"case_rate_lag{lag}"] = grouped["case_rate"].shift(lag)
        
    # 3. Rolling Window Features (4 weeks)
    logger.info("Generating rolling window features...")
    df["cases_rolling_mean_4"] = grouped["cases"].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df["cases_rolling_std_4"] = grouped["cases"].transform(lambda x: x.rolling(window=4, min_periods=1).std())
    df["case_rate_rolling_mean_4"] = grouped["case_rate"].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df["case_rate_rolling_std_4"] = grouped["case_rate"].transform(lambda x: x.rolling(window=4, min_periods=1).std())
    
    # Fill standard deviation NaNs with 0 (e.g. for the first observation in a window)
    df["cases_rolling_std_4"] = df["cases_rolling_std_4"].fillna(0)
    df["case_rate_rolling_std_4"] = df["case_rate_rolling_std_4"].fillna(0)
    
    # 4. Growth Rate
    logger.info("Computing growth rates...")
    df["growth_rate"] = grouped["cases"].pct_change()
    df["growth_rate"] = df["growth_rate"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # 5. Seasonality
    logger.info("Deriving seasonal and calendar features...")
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["season"] = df["month"].apply(get_season)
    
    # 6. Environmental Indices
    logger.info("Building environmental indices...")
    df["humidity_index"] = df["humidity"] / 100.0
    df["rainfall_index"] = df["rainfall"] / 100.0  # normalize
    # Temp-Humidity Index (THI): simple temperature-humidity product
    df["temp_humidity_index"] = df["temperature"] * df["humidity_index"]
    
    # 7. Demographic Variables (Population Density)
    logger.info("Calculating population density...")
    df["population_density"] = df.apply(
        lambda row: row["population"] / LAND_AREAS.get(row["country"], 1e6),
        axis=1
    )
    
    # 8. Outbreak Risk Label Target
    logger.info("Assigning four-tier risk labels...")
    df["risk_label"] = df["case_rate"].apply(assign_risk)
    
    # 9. Handle any remaining missing values from lags
    # Backfill/forwardfill within the group or pad with 0
    lag_cols = [c for c in df.columns if "lag" in c]
    for col in lag_cols:
        df[col] = df.groupby(["country", "disease"])[col].ffill().bfill().fillna(0)
        
    return df

def encode_and_scale(df):
    """
    Performs label encoding for categoricals and scaling for numeric inputs.
    Saves the transformer objects for downstream inference.
    """
    df = df.copy()
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Categorical encoders
    logger.info("Encoding country and disease labels...")
    le_country = LabelEncoder()
    df["country_enc"] = le_country.fit_transform(df["country"])
    joblib.dump(le_country, os.path.join(MODELS_DIR, "label_encoder_country.pkl"))
    
    le_disease = LabelEncoder()
    df["disease_enc"] = le_disease.fit_transform(df["disease"])
    joblib.dump(le_disease, os.path.join(MODELS_DIR, "label_encoder_disease.pkl"))
    
    # Season encoder (one-hot or label encoded, let's use label encoder for simplicity)
    le_season = LabelEncoder()
    df["season_enc"] = le_season.fit_transform(df["season"])
    joblib.dump(le_season, os.path.join(MODELS_DIR, "label_encoder_season.pkl"))
    
    # Define features to scale
    feature_cols = [
        "temperature", "humidity", "rainfall", "pressure", "wind_speed",
        "cases_lag1", "case_rate_lag1", "cases_lag2", "case_rate_lag2",
        "cases_lag4", "case_rate_lag4", "cases_lag8", "case_rate_lag8",
        "cases_rolling_mean_4", "cases_rolling_std_4",
        "case_rate_rolling_mean_4", "case_rate_rolling_std_4",
        "growth_rate", "month", "quarter", "humidity_index", "rainfall_index",
        "temp_humidity_index", "population_density", "country_enc", "disease_enc", "season_enc"
    ]
    
    # Save the list of feature column names
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    
    logger.info("Scaling features using MinMaxScaler...")
    scaler = MinMaxScaler()
    df_scaled_features = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index
    )
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    
    # Combine original metadata/targets with scaled features
    meta_cols = ["country", "disease", "year", "week", "date", "cases", "deaths", "population", "case_rate", "death_rate", "risk_label"]
    df_final = pd.concat([df[meta_cols], df_scaled_features], axis=1)
    
    return df_final

def main():
    logger.info("Executing Feature Engineering and Labeling Pipeline")
    
    # 1. Load cleaned dataset from SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        df_clean = pd.read_sql_query("SELECT * FROM cleaned_data", conn)
        logger.info(f"Loaded cleaned dataset from DB. Shape: {df_clean.shape}")
        
        # 2. Process features
        df_featured = engineer_features(df_clean)
        logger.info(f"Features engineered. Shape: {df_featured.shape}")
        
        # 3. Encode categoricals and scale numerical features
        df_final = encode_and_scale(df_featured)
        logger.info(f"Feature scaling and encoding complete. Shape: {df_final.shape}")
        
        # 4. Save featured data to CSV
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        csv_path = os.path.join(PROCESSED_DATA_DIR, "featured_data.csv")
        df_final.to_csv(csv_path, index=False)
        logger.info(f"Featured dataset CSV saved to: {csv_path}")
        
        # 5. Persist back to SQLite
        # Convert date to string format for SQLite compatibility
        df_final["date"] = df_final["date"].dt.strftime("%Y-%m-%d")
        df_final.to_sql("features", conn, if_exists="replace", index=False)
        logger.info("Saved 'features' table to SQLite database.")
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    main()
