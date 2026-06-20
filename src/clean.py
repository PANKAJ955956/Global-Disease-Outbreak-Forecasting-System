import os
import sqlite3
import logging
import pandas as pd
from datetime import datetime, timedelta

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DB_PATH, PROCESSED_DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean")

def week_to_date(year, week):
    """
    Robust Monday-aligned date constructor from year and week number.
    Ensures dates are continuous and formatted correctly.
    """
    try:
        # Start at Jan 1st of the year
        start_date = datetime(int(year), 1, 1)
        # Find the Monday of that week
        days_to_monday = start_date.weekday()  # Monday = 0, Sunday = 6
        start_monday = start_date - timedelta(days=days_to_monday)
        # Add week offset (subtract 1 because week is 1-indexed)
        target_date = start_monday + timedelta(weeks=int(week) - 1)
        return target_date
    except Exception:
        return pd.NaT

def clean_who_data(df):
    """
    Cleans duplicates, standardizes columns, handles nulls, and filters negative counts.
    """
    df = df.copy()
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Drop rows without country or year
    df.dropna(subset=["country", "year"], inplace=True)
    
    # Standardize types
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].fillna(1).astype(int)
    
    # Handle missing case and death values
    df["cases"] = df["cases"].fillna(0)
    df["deaths"] = df["deaths"].fillna(0)
    
    # Remove negative records
    df = df[df["cases"] >= 0]
    df = df[df["deaths"] >= 0]
    
    # Convert cases and deaths to integer
    df["cases"] = df["cases"].astype(int)
    df["deaths"] = df["deaths"].astype(int)
    
    # Standardize country names
    country_corrections = {
        "USA": "United States",
        "UK": "United Kingdom",
        "Congo": "Democratic Republic of the Congo"
    }
    df["country"] = df["country"].replace(country_corrections)
    
    # Build date column
    df["date"] = df.apply(lambda r: week_to_date(r["year"], r["week"]), axis=1)
    df.dropna(subset=["date"], inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    
    return df

def clean_weather_data(df):
    """
    Cleans and standardizes weather table variables.
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df.dropna(subset=["country", "year", "week"], inplace=True)
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    return df

def clean_population_data(df):
    """
    Cleans and standardizes demographic records.
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df.dropna(subset=["country", "population"], inplace=True)
    df["population"] = df["population"].astype(int)
    return df

def merge_datasets(disease_df, weather_df, population_df):
    """
    Merges disease, weather, and population tables, computing normalized rates.
    """
    # 1. Merge disease with population
    merged = disease_df.merge(population_df, on="country", how="left")
    
    # 2. Merge with weather on country, year, and week
    merged = merged.merge(weather_df, on=["country", "year", "week"], how="left")
    
    # 3. Calculate case and death rates per 100,000 population
    # Handle division by zero or NaN population
    merged["population"] = merged["population"].fillna(1)
    merged["case_rate"] = (merged["cases"] / merged["population"]) * 100000.0
    merged["death_rate"] = (merged["deaths"] / merged["population"]) * 100000.0
    
    # Print shape details
    logger.info(f"Merged Dataset generated. Shape: {merged.shape}")
    return merged

def main():
    logger.info("Starting Data Cleaning and Preprocessing Pipeline")
    
    # 1. Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Load tables
        disease_raw = pd.read_sql_query("SELECT * FROM who_diseases", conn)
        weather_raw = pd.read_sql_query("SELECT * FROM weather", conn)
        pop_raw = pd.read_sql_query("SELECT * FROM population", conn)
        
        logger.info(f"Raw data loaded from DB. Diseases: {disease_raw.shape}, Weather: {weather_raw.shape}, Population: {pop_raw.shape}")
        
        # Clean tables
        disease_clean = clean_who_data(disease_raw)
        weather_clean = clean_weather_data(weather_raw)
        pop_clean = clean_population_data(pop_raw)
        
        logger.info(f"Cleaned tables. Diseases: {disease_clean.shape}, Weather: {weather_clean.shape}, Population: {pop_clean.shape}")
        
        # Merge datasets
        merged_df = merge_datasets(disease_clean, weather_clean, pop_clean)
        
        # Save processed CSV
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        csv_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_data.csv")
        merged_df.to_csv(csv_path, index=False)
        logger.info(f"Saved cleaned data CSV to: {csv_path}")
        
        # Persist cleaned dataset back into SQLite as a combined table
        merged_df["date"] = merged_df["date"].dt.strftime("%Y-%m-%d") # sqlite string date
        merged_df.to_sql("cleaned_data", conn, if_exists="replace", index=False)
        logger.info("Saved 'cleaned_data' table to SQLite.")
        
    except Exception as e:
        logger.error(f"Error in data cleaning pipeline: {e}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    main()
