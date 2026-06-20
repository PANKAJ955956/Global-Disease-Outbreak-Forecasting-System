import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# File paths (relative to workspace root)
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
DB_PATH = "data/db/disease.db"
MODELS_DIR = "models/"

# External API Configuration
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "your_key_here")

# Time-series settings
SEQUENCE_LENGTH = 12      # weeks of history for LSTM input sequence
FORECAST_WEEKS = 4       # default forecast horizon

# Model Hyperparameters
RF_ESTIMATORS = 200
XGB_ESTIMATORS = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Risk Thresholds (case rate per 100k population)
RISK_LOW = 1.0
RISK_MEDIUM = 10.0
RISK_HIGH = 50.0

# Target scope
TARGET_COUNTRIES = [
    "India", "Nigeria", "Brazil", "Indonesia",
    "Pakistan", "Bangladesh", "Ethiopia", "Philippines"
]
TARGET_DISEASES = ["Dengue", "Cholera", "Malaria", "Measles", "Yellow Fever"]

# Geographic coordinates for Folium Map
COUNTRY_COORDS = {
    "India": [20.5937, 78.9629],
    "Nigeria": [9.0820, 8.6753],
    "Brazil": [-14.2350, -51.9253],
    "Indonesia": [-0.7893, 113.9213],
    "Pakistan": [30.3753, 69.3451],
    "Bangladesh": [23.6850, 90.3563],
    "Ethiopia": [9.1450, 40.4897],
    "Philippines": [12.8797, 121.7740]
}
