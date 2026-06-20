import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add project path to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clean import clean_who_data
from src.features import assign_risk, engineer_features

def test_no_negative_cases():
    """
    Test that negative case counts (common reporting errors) are successfully filtered out.
    """
    df = pd.DataFrame({
        "country": ["India", "Brazil", "Nigeria"],
        "year": [2022, 2022, 2022],
        "week": [1, 2, 3],
        "cases": [-5, 100, 0],
        "deaths": [0, 2, -1],
        "disease": ["Dengue", "Cholera", "Malaria"]
    })
    
    cleaned = clean_who_data(df)
    
    # Negative cases and negative deaths should be filtered
    assert (cleaned["cases"] >= 0).all()
    assert (cleaned["deaths"] >= 0).all()
    assert len(cleaned) == 1  # Only Brazil (cases=100, deaths=2) is valid since deaths=-1 and cases=-5 are dropped

def test_no_missing_country():
    """
    Test that records lacking country identifier names are rejected.
    """
    df = pd.DataFrame({
        "country": [None, "India"],
        "year": [2022, 2022],
        "week": [10, 10],
        "cases": [50, 100],
        "deaths": [1, 2],
        "disease": ["Dengue", "Cholera"]
    })
    
    cleaned = clean_who_data(df)
    
    assert cleaned["country"].isnull().sum() == 0
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["country"] == "India"

def test_assign_risk_labels():
    """
    Test the four-tier risk label assignment boundaries.
    Tiers (case rate per 100k): Low (<1), Medium (<10), High (<50), Critical (>=50)
    """
    assert assign_risk(0.5) == 0      # Low
    assert assign_risk(1.0) == 1      # Medium
    assert assign_risk(9.9) == 1      # Medium
    assert assign_risk(10.0) == 2     # High
    assert assign_risk(49.9) == 2     # High
    assert assign_risk(50.0) == 3     # Critical
    assert assign_risk(125.5) == 3    # Critical

def test_growth_rate_handling():
    """
    Test that growth rate computation handles divide-by-zero (inf) safely.
    """
    # Create sample structured dataframe
    df = pd.DataFrame({
        "country": ["India", "India", "India"],
        "disease": ["Dengue", "Dengue", "Dengue"],
        "year": [2022, 2022, 2022],
        "week": [1, 2, 3],
        "date": ["2022-01-03", "2022-01-10", "2022-01-17"],
        "cases": [0, 10, 20],  # 0 to 10 is an infinite increase percentage
        "deaths": [0, 0, 1],
        "population": [10000000, 10000000, 10000000],
        "case_rate": [0.0, 0.1, 0.2],
        "death_rate": [0.0, 0.0, 0.01],
        "temperature": [25.0, 26.0, 27.0],
        "humidity": [60.0, 62.0, 64.0],
        "rainfall": [10.0, 15.0, 20.0],
        "pressure": [1010.0, 1009.0, 1008.0],
        "wind_speed": [5.0, 4.5, 4.0]
    })
    
    # Run engineering
    featured = engineer_features(df)
    
    # Growth rate from index 0 to 1 (0 -> 10 cases) would normally be inf, must be handled to 0.0
    assert not np.isinf(featured["growth_rate"]).any()
    assert not featured["growth_rate"].isnull().any()
