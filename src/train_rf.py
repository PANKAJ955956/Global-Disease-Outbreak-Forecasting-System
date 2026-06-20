import os
import sqlite3
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DB_PATH, MODELS_DIR, RANDOM_STATE, TEST_SIZE, RF_ESTIMATORS, XGB_ESTIMATORS

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_rf")

def get_risk_label(class_idx):
    labels = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
    return labels.get(class_idx, f"Label {class_idx}")

def load_data():
    """
    Loads features dataset from SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM features", conn)
        return df
    except Exception as e:
        logger.error(f"Error loading features table: {e}")
        raise e
    finally:
        conn.close()

def main():
    logger.info("Starting Machine Learning Classification Model Training")
    
    # 1. Load data
    df = load_data()
    logger.info(f"Loaded featured data from database. Shape: {df.shape}")
    
    # 2. Get scaled feature column names and target
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    logger.info(f"Features list loaded: {feature_cols}")
    
    X = df[feature_cols]
    y = df["risk_label"].astype(int)
    
    # Check class distribution
    class_counts = y.value_counts().to_dict()
    logger.info(f"Risk label distribution in dataset: {class_counts}")
    
    # 3. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Split data into Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # 4. Train Random Forest Classifier
    logger.info(f"Training Random Forest Classifier with {RF_ESTIMATORS} trees...")
    rf_model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    logger.info(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    unique_classes = np.unique(np.concatenate([y_test, rf_pred]))
    target_names = [get_risk_label(c) for c in unique_classes]
    logger.info("\n" + classification_report(y_test, rf_pred, labels=unique_classes, target_names=target_names))
    
    # Save RF Model
    rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    joblib.dump(rf_model, rf_path)
    logger.info(f"Random Forest model saved to: {rf_path}")
    
    # Save RF Feature Importance
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    importance_path = os.path.join(MODELS_DIR, "rf_feature_importances.pkl")
    joblib.dump(importances, importance_path)
    logger.info(f"Random Forest feature importance saved to: {importance_path}")
    
    # 5. Train XGBoost Classifier
    logger.info(f"Training XGBoost Classifier with {XGB_ESTIMATORS} estimators...")
    xgb_model = XGBClassifier(
        n_estimators=XGB_ESTIMATORS,
        max_depth=6,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate XGBoost
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    logger.info(f"XGBoost Accuracy: {xgb_acc:.4f}")
    
    xgb_unique_classes = np.unique(np.concatenate([y_test, xgb_pred]))
    xgb_target_names = [get_risk_label(c) for c in xgb_unique_classes]
    logger.info("\n" + classification_report(y_test, xgb_pred, labels=xgb_unique_classes, target_names=xgb_target_names))
    
    # Save XGBoost Model
    xgb_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    joblib.dump(xgb_model, xgb_path)
    logger.info(f"XGBoost model saved to: {xgb_path}")

if __name__ == "__main__":
    main()
