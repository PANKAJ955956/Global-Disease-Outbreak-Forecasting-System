import os
import sqlite3
import logging
from datetime import datetime

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DB_PATH

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("alerts")

# File logger for alerts
alert_file_handler = logging.FileHandler("logs/alerts.log")
alert_file_formatter = logging.Formatter("%(asctime)s - %(message)s")
alert_file_handler.setFormatter(alert_file_formatter)
alert_logger = logging.getLogger("alert_file")
alert_logger.addHandler(alert_file_handler)
alert_logger.setLevel(logging.INFO)

def init_alerts_table():
    """
    Initializes the ALERTS table in SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            country TEXT,
            disease TEXT,
            risk_level TEXT,
            risk_score REAL,
            outbreak_probability REAL,
            message TEXT,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()

def send_simulated_email(country, disease, risk_level, score, message):
    """
    Simulates sending an Email alert.
    Can be easily connected to SMTP/SendGrid.
    """
    email_body = f"""
    ====================================================================
    GDOFS AUTOMATED HEALTH WARNING DISPATCH
    ====================================================================
    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Subject: [ALERT - {risk_level.upper()}] Outbreak Risk Flagged
    
    Warning Message:
    {message}
    
    Details:
    - Country: {country}
    - Tracked Pathogen: {disease}
    - Assigned Severity: {risk_level}
    - Model Confidence Score: {score}%
    
    Action Recommended:
    Mobilize local medical assets and distribute preventative measures.
    ====================================================================
    """
    logger.info(f"Simulating Email Dispatch to WHO and Local Ministry of Health...")
    alert_logger.info(f"[EMAIL DISPATCH] to health-agencies@who.int:\n{email_body}")

def send_simulated_sms(country, disease, risk_level, score, message):
    """
    Simulates sending an SMS alert.
    Can be easily connected to Twilio or other SMS gateways.
    """
    sms_body = f"GDOFS ALERT [{risk_level}]: {disease} in {country}. Risk: {score}%. Action required."
    logger.info(f"Simulating SMS Dispatch to field epidemiologist teams...")
    alert_logger.info(f"[SMS DISPATCH] to +15550199:\n{sms_body}")

def check_and_trigger_alerts(country, disease, risk_level, risk_score, outbreak_prob):
    """
    Checks risk thresholds (>70% or High/Critical levels) and dispatches alerts.
    """
    init_alerts_table()
    
    # Trigger criteria: High or Critical risk levels, or risk score > 70%
    if risk_level in ["High", "Critical"] or risk_score >= 70.0:
        alert_msg = f"ALERT [{risk_level}]: {disease} outbreak risk in {country} — Risk Score: {risk_score}%"
        logger.warning(alert_msg)
        
        # 1. Save alert record to SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            cursor.execute("""
                INSERT INTO alerts (timestamp, country, disease, risk_level, risk_score, outbreak_probability, message, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (now_str, country, disease, risk_level, risk_score, outbreak_prob, alert_msg, "Dispatched"))
            conn.commit()
            logger.info("Outbreak alert stored in database.")
        except Exception as e:
            logger.error(f"Failed to write alert to DB: {e}")
        finally:
            conn.close()
            
        # 2. Trigger dispatches
        send_simulated_email(country, disease, risk_level, risk_score, alert_msg)
        send_simulated_sms(country, disease, risk_level, risk_score, alert_msg)
        
        return {
            "alert_triggered": True,
            "message": alert_msg
        }
        
    return {"alert_triggered": False}

def process_alerts_bulk(predictions_df):
    """
    Processes a bulk DataFrame of predictions and triggers alerts for qualifying rows.
    """
    triggered_alerts = []
    for _, row in predictions_df.iterrows():
        res = check_and_trigger_alerts(
            country=row["country"],
            disease=row["disease"],
            risk_level=row["risk_level"],
            risk_score=row["risk_score"],
            outbreak_prob=row.get("outbreak_probability", row["risk_score"])
        )
        if res["alert_triggered"]:
            triggered_alerts.append(res["message"])
            
    return triggered_alerts
