import os
import sqlite3
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    DB_PATH, RAW_DATA_DIR, TARGET_COUNTRIES, TARGET_DISEASES,
    WEATHER_API_KEY, COUNTRY_COORDS
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def fetch_who_gho_data(indicator_code="WHS3_49"):
    """
    Fetches raw disease data from the WHO Global Health Observatory OData API.
    """
    url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
    logger.info(f"Fetching WHO GHO data for indicator {indicator_code} from {url}")
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data.get("value", []))
            logger.info(f"Successfully fetched WHO data. Shape: {df.shape}")
            return df
        else:
            logger.warning(f"Failed to fetch WHO GHO data: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching WHO GHO data: {e}")
    return pd.DataFrame()

def fetch_world_bank_population():
    """
    Fetches population data from the World Bank API for target countries.
    """
    logger.info("Fetching population data from World Bank API")
    records = []
    for country in TARGET_COUNTRIES:
        code = ""
        if country == "India": code = "IN"
        elif country == "Nigeria": code = "NG"
        elif country == "Brazil": code = "BR"
        elif country == "Indonesia": code = "ID"
        elif country == "Pakistan": code = "PK"
        elif country == "Bangladesh": code = "BD"
        elif country == "Ethiopia": code = "ET"
        elif country == "Philippines": code = "PH"
        else: continue
            
        url = f"http://api.worldbank.org/v2/country/{code}/indicator/SP.POP.TOTL?date=2022&format=json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and len(data[1]) > 0:
                    pop = data[1][0]["value"]
                    records.append({
                        "country": country,
                        "country_code": code,
                        "population": pop
                    })
                    logger.info(f"Fetched population for {country}: {pop}")
        except Exception as e:
            logger.error(f"Error fetching population for {country} from World Bank: {e}")
            
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame()

def generate_high_fidelity_simulation_data():
    """
    Generates realistic, seasonally correlated synthetic datasets spanning 2020-2025.
    Directly generates case_rates (per 100k) to span across Low (<1), Medium (1-10),
    High (10-50), and Critical (>=50) risk categories.
    """
    logger.info("Generating high-fidelity simulation datasets...")
    
    # Population static records
    pop_data = {
        "India": {"code": "IN", "pop": 1417000000},
        "Nigeria": {"code": "NG", "pop": 218500000},
        "Brazil": {"code": "BR", "pop": 215300000},
        "Indonesia": {"code": "ID", "pop": 275500000},
        "Pakistan": {"code": "PK", "pop": 235800000},
        "Bangladesh": {"code": "BD", "pop": 171200000},
        "Ethiopia": {"code": "ET", "pop": 123400000},
        "Philippines": {"code": "PH", "pop": 115600000}
    }
    
    pop_records = []
    for country, info in pop_data.items():
        pop_records.append({
            "country": country,
            "country_code": info["code"],
            "population": info["pop"]
        })
    df_pop = pd.DataFrame(pop_records)
    
    # Time series range setup
    years = list(range(2020, 2026))
    weeks = list(range(1, 53))
    
    disease_records = []
    weather_records = []
    
    np.random.seed(42)
    
    for country in TARGET_COUNTRIES:
        # Base climatology settings per country
        for year in years:
            for week in weeks:
                t_sin = np.sin(2 * np.pi * week / 52.0)
                
                # Temperature, humidity, rainfall seasonal baseline
                if country in ["India", "Pakistan", "Bangladesh"]:
                    base_temp = 28 + 8 * np.sin(2 * np.pi * (week - 15) / 52.0)
                    base_humid = 60 + 25 * np.sin(2 * np.pi * (week - 25) / 52.0)
                    base_rain = max(0, 150 * np.exp(-((week - 30)**2)/40) + np.random.normal(10, 5))
                elif country in ["Brazil", "Indonesia", "Philippines"]:
                    base_temp = 27 + 2 * t_sin
                    base_humid = 75 + 10 * np.sin(2 * np.pi * (week - 40) / 52.0)
                    base_rain = max(10, 80 + 70 * np.sin(2 * np.pi * (week - 45) / 52.0) + np.random.normal(20, 10))
                else: # Nigeria, Ethiopia
                    base_temp = 25 + 4 * np.sin(2 * np.pi * (week - 10) / 52.0)
                    base_humid = 55 + 20 * np.sin(2 * np.pi * (week - 28) / 52.0)
                    base_rain = max(0, 100 * np.sin(2 * np.pi * (week - 24) / 52.0) + np.random.normal(15, 8))
                    
                temp = round(base_temp + np.random.normal(0, 1.5), 1)
                humid = round(min(100.0, max(20.0, base_humid + np.random.normal(0, 4))), 1)
                rain = round(max(0, base_rain), 1)
                press = round(1010 + 5 * t_sin + np.random.normal(0, 2), 1)
                wind = round(max(1.0, 5.0 + 3 * t_sin + np.random.normal(0, 1.5)), 1)
                
                weather_records.append({
                    "country": country,
                    "year": year,
                    "week": week,
                    "temperature": temp,
                    "humidity": humid,
                    "rainfall": rain,
                    "pressure": press,
                    "wind_speed": wind
                })
                
                # Ingest disease cases and deaths based on simulated case_rates
                for disease in TARGET_DISEASES:
                    # Let's generate a case rate per 100,000 population directly
                    # Base case rate
                    base_rate = 0.5 # Low baseline
                    
                    # Add seasonal outbreaks matching weather patterns
                    if disease == "Dengue":
                        # Spikes in high temp & heavy rain (Weeks 26-38)
                        if temp > 24 and rain > 40:
                            base_rate = 12.0 + (rain / 5.0) + (temp - 24) * 2.0
                            # Add monsoon burst noise
                            if week in range(28, 36):
                                base_rate += np.random.uniform(20.0, 45.0)
                    elif disease == "Cholera":
                        # Spikes in heavy rain/monsoon flooding
                        if rain > 80:
                            base_rate = 8.0 + (rain / 4.0)
                            if week in range(24, 32):
                                base_rate += np.random.uniform(15.0, 40.0)
                    elif disease == "Malaria":
                        # Spikes with humidity > 65% and moderate/high temp
                        if humid > 65 and temp > 22:
                            base_rate = 15.0 + (humid - 65) * 1.5
                            if country in ["Nigeria", "Ethiopia"]:
                                base_rate *= 1.8 # Endemic weight
                            if week in range(30, 42):
                                base_rate += np.random.uniform(20.0, 50.0)
                    elif disease == "Measles":
                        # Spikes in cool, dry winter weeks
                        if temp < 24 and humid < 60:
                            base_rate = 5.0 + (24 - temp) * 2.0 + (60 - humid) * 0.5
                            if week in range(5, 18):
                                base_rate += np.random.uniform(10.0, 35.0)
                    elif disease == "Yellow Fever":
                        # Tropical rain forest outbreaks in South America/Africa
                        if country in ["Brazil", "Nigeria", "Ethiopia"] and temp > 25 and rain > 70:
                            base_rate = 2.0 + (temp - 25) * 1.0 + (rain / 12.0)
                            if week in range(12, 24):
                                base_rate += np.random.uniform(5.0, 18.0)
                                
                    # Add random background variance
                    case_rate = max(0.1, base_rate + np.random.normal(0, max(0.1, base_rate * 0.15)))
                    
                    # Back-calculate cases count from population
                    pop = pop_data[country]["pop"]
                    cases = int(round((case_rate * pop) / 100000.0))
                    
                    # Standard CFRs
                    cfr = 0.001
                    if disease == "Dengue": cfr = 0.0005
                    elif disease == "Cholera": cfr = 0.015
                    elif disease == "Malaria": cfr = 0.003
                    elif disease == "Measles": cfr = 0.01
                    elif disease == "Yellow Fever": cfr = 0.08
                    
                    deaths = int(max(0, np.random.binomial(n=cases, p=cfr) if cases > 0 else 0))
                    
                    disease_records.append({
                        "country": country,
                        "disease": disease,
                        "year": year,
                        "week": week,
                        "cases": cases,
                        "deaths": deaths
                    })

    df_diseases = pd.DataFrame(disease_records)
    df_weather = pd.DataFrame(weather_records)
    
    return df_diseases, df_weather, df_pop

def main():
    logger.info("Initializing Data Ingestion Pipeline")
    
    # Generate high-fidelity simulation
    df_diseases, df_weather, df_pop = generate_high_fidelity_simulation_data()
    
    # Save to SQLite
    conn = get_db_connection()
    try:
        df_diseases.to_sql("who_diseases", conn, if_exists="replace", index=False)
        logger.info(f"Saved 'who_diseases' table to SQLite. Shape: {df_diseases.shape}")
        
        df_weather.to_sql("weather", conn, if_exists="replace", index=False)
        logger.info(f"Saved 'weather' table to SQLite. Shape: {df_weather.shape}")
        
        df_pop.to_sql("population", conn, if_exists="replace", index=False)
        logger.info(f"Saved 'population' table to SQLite. Shape: {df_pop.shape}")
        
    except Exception as e:
        logger.error(f"Failed to save datasets: {e}")
        raise e
    finally:
        conn.close()
        
    # Write CSV backups
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df_diseases.to_csv(os.path.join(RAW_DATA_DIR, "who_disease_data.csv"), index=False)
    df_pop.to_csv(os.path.join(RAW_DATA_DIR, "world_population.csv"), index=False)
    df_weather.to_csv(os.path.join(RAW_DATA_DIR, "weather_data.csv"), index=False)
    logger.info("Raw CSV data backups successfully saved to data/raw/")

if __name__ == "__main__":
    main()
