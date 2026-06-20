import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib

# Add project path to python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DB_PATH, TARGET_COUNTRIES, TARGET_DISEASES, COUNTRY_COORDS, RISK_LOW, RISK_MEDIUM, RISK_HIGH
from src.predict import predict_risk, forecast_lstm

# ----------------- PAGE CONFIG & CUSTOM THEME -----------------
st.set_page_config(
    page_title="GDOFS | Global Outbreak Forecasting",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #0d1117;
        color: #c9d1d9;
    }
    
    /* Title styling */
    .gdofs-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(45deg, #00f2fe, #4facfe, #ff0844);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .gdofs-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    
    /* Metrics card container */
    .metric-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #58a6ff;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }
    
    /* Alerts console style */
    .alert-critical {
        background: rgba(248, 81, 73, 0.15);
        border-left: 5px solid #f85149;
        color: #ff7b72;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
    .alert-high {
        background: rgba(210, 153, 34, 0.15);
        border-left: 5px solid #d29922;
        color: #f0883e;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- UTILITY FUNCTIONS -----------------
def load_db_features():
    """
    Loads features table from SQLite.
    """
    if not os.path.exists(DB_PATH):
        st.error("SQLite Database not found. Please run the data ingestion and cleaning pipelines first.")
        st.stop()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM features", conn)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"Failed to fetch data from DB: {e}")
        st.stop()
    finally:
        conn.close()

def load_db_alerts():
    """
    Loads alerts records from SQLite.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM alerts ORDER BY timestamp DESC", conn)
        return df
    except Exception:
        # Table might not exist yet if no alerts were triggered
        return pd.DataFrame()
    finally:
        conn.close()

# ----------------- STATE INITIALIZATION -----------------
df = load_db_features()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### 🎛️ Control Panel")
selected_country = st.sidebar.selectbox("Country Target", TARGET_COUNTRIES)
selected_disease = st.sidebar.selectbox("Pathogen Target", TARGET_DISEASES)
selected_model = st.sidebar.radio("Risk Classifier Model", ["Random Forest", "XGBoost"])

# Filter features for selected targets
filtered_df = df[(df["country"] == selected_country) & (df["disease"] == selected_disease)].sort_values("date").reset_index(drop=True)

# Get the latest row of data for current prediction input
latest_record = filtered_df.iloc[-1].to_dict() if not filtered_df.empty else {}

# Inject sliders for weather parameter simulation in prediction
st.sidebar.markdown("### 🌤️ Weather Simulation")
sim_temp = st.sidebar.slider("Temperature (°C)", min_value=10.0, max_value=42.0, value=float(latest_record.get("temperature", 25.0)), step=0.5)
sim_humid = st.sidebar.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=float(latest_record.get("humidity", 70.0)), step=1.0)
sim_rain = st.sidebar.slider("Rainfall (mm)", min_value=0.0, max_value=400.0, value=float(latest_record.get("rainfall", 50.0)), step=5.0)

# Forecast Weeks slider
forecast_horizon = st.sidebar.slider("Forecast Horizon (Weeks Ahead)", min_value=1, max_value=8, value=4)

# ----------------- HEADER SECTION -----------------
st.markdown('<div class="gdofs-title">Global Disease Outbreak Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="gdofs-subtitle">Proactive early warning epidemiological intelligence powered by Machine Learning and Deep Learning</div>', unsafe_allow_html=True)

# ----------------- TABS SETUP -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview Dashboard",
    "📈 Pathogen Forecasting",
    "🗺️ GIS Risk Heatmap",
    "🔔 Alerts Console",
    "🔍 Explainability & Analytics"
])

# ==============================================================================
# TAB 1: OVERVIEW DASHBOARD
# ==============================================================================
with tab1:
    if filtered_df.empty:
        st.warning("No data found for this selection.")
    else:
        # Calculate Metric values
        total_cases = int(filtered_df["cases"].sum())
        total_deaths = int(filtered_df["deaths"].sum())
        cfr = (total_deaths / total_cases * 100.0) if total_cases > 0 else 0.0
        avg_case_rate = filtered_df["case_rate"].mean()
        
        # Grid of cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_cases:,}</div><div class="metric-label">Total Cases</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_deaths:,}</div><div class="metric-label">Total Deaths</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{cfr:.2f}%</div><div class="metric-label">Case Fatality Rate (CFR)</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_case_rate:.2f}</div><div class="metric-label">Avg Case Rate (per 100k)</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Perform live risk prediction based on simulation weather inputs + latest lags
        pred_input = latest_record.copy()
        pred_input["temperature"] = sim_temp
        pred_input["humidity"] = sim_humid
        pred_input["rainfall"] = sim_rain
        # Update indices
        pred_input["humidity_index"] = sim_humid / 100.0
        pred_input["rainfall_index"] = sim_rain / 100.0
        pred_input["temp_humidity_index"] = sim_temp * pred_input["humidity_index"]
        
        model_type_code = "rf" if selected_model == "Random Forest" else "xgb"
        prediction = predict_risk(pred_input, model_type=model_type_code)
        
        # Risk assessment Display
        left_col, right_col = st.columns([1, 2])
        
        with left_col:
            st.markdown("### 🛡️ Outbreak Risk Assessment")
            
            risk_level = prediction.get("risk_level", "Unknown")
            risk_score = prediction.get("risk_score", 0.0)
            
            # Select color based on risk severity
            risk_colors = {"Low": "#2ea44f", "Medium": "#d29922", "High": "#f0883e", "Critical": "#f85149"}
            color = risk_colors.get(risk_level, "#58a6ff")
            
            # Radial score dial or text presentation
            st.markdown(f"""
            <div style="background: rgba(22, 27, 34, 0.8); border: 1px solid rgba(48, 54, 61, 0.8); border-radius: 12px; padding: 2rem; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                <div style="font-size: 1.1rem; color: #8b949e; margin-bottom: 0.5rem; text-transform: uppercase;">Current Alert Status</div>
                <div style="font-size: 3rem; font-weight: 900; color: {color}; text-shadow: 0 0 10px {color}33;">{risk_level.upper()}</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #c9d1d9; margin-top: 1rem;">{risk_score}%</div>
                <div style="font-size: 0.9rem; color: #8b949e; margin-top: 0.2rem;">Model Probability Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display other classes probability
            all_prob = prediction.get("all_probabilities", {})
            prob_df = pd.DataFrame(list(all_prob.items()), columns=["Risk Tiers", "Probability (%)"])
            fig_prob = px.bar(prob_df, x="Probability (%)", y="Risk Tiers", orientation="h", color="Risk Tiers",
                              color_discrete_map=risk_colors, title="Model Probability Distribution")
            fig_prob.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9")
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with right_col:
            st.markdown("### 📊 Historical Trends")
            
            # Interactive Plotly chart showing historical cases & deaths
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df["date"], y=filtered_df["cases"], name="Weekly Cases", line=dict(color="#58a6ff", width=2.5)))
            fig.add_trace(go.Scatter(x=filtered_df["date"], y=filtered_df["deaths"], name="Weekly Deaths", line=dict(color="#ff7b72", width=1.5), yaxis="y2"))
            
            fig.update_layout(
                title=f"Epidemiological Trend: {selected_country} - {selected_disease}",
                xaxis=dict(title="Timeline", gridcolor="rgba(48,54,61,0.3)"),
                yaxis=dict(title=dict(text="Cases count", font=dict(color="#58a6ff")), tickfont=dict(color="#58a6ff"), gridcolor="rgba(48,54,61,0.3)"),
                yaxis2=dict(title=dict(text="Deaths count", font=dict(color="#ff7b72")), tickfont=dict(color="#ff7b72"), anchor="x", overlaying="y", side="right"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#c9d1d9",
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Weather variables trend comparison
            st.markdown("#### 🌦️ Local Weather Climatology Correlation")
            fig_weather = px.line(filtered_df, x="date", y=["temperature", "humidity", "rainfall"],
                                  title="Environmental Variable Timeline Tracking")
            fig_weather.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9",
                                      xaxis=dict(gridcolor="rgba(48,54,61,0.3)"), yaxis=dict(gridcolor="rgba(48,54,61,0.3)"))
            st.plotly_chart(fig_weather, use_container_width=True)

# ==============================================================================
# TAB 2: PATHOGEN FORECASTING
# ==============================================================================
with tab2:
    st.markdown("### 📈 Time Series Predictive Modeling")
    st.markdown(f"Running **PyTorch Stacked LSTM network** using the last 12 weeks of historical sequence variables to forecast case rates in **{selected_country}** ({selected_disease}).")
    
    if filtered_df.empty:
        st.warning("No data available.")
    else:
        recent_rates = filtered_df["case_rate"].tolist()
        
        # Run forecast
        forecasts = forecast_lstm(recent_rates, weeks_ahead=forecast_horizon)
        
        if isinstance(forecasts, dict) and "error" in forecasts:
            st.error(forecasts["error"])
        else:
            # Build forecast timeline
            last_date = filtered_df["date"].iloc[-1]
            forecast_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(forecast_horizon)]
            
            # Historical tail for plotting context
            tail_df = filtered_df.tail(12)
            
            # Create interactive plot
            fig_fore = go.Figure()
            
            # Historical context
            fig_fore.add_trace(go.Scatter(
                x=tail_df["date"], y=tail_df["case_rate"],
                mode='lines+markers', name='Historical Case Rate',
                line=dict(color='#58a6ff', width=3)
            ))
            
            # Forecast curve
            # Attach last point of history to start of forecast for visualization continuity
            f_dates = [last_date] + forecast_dates
            f_rates = [tail_df["case_rate"].iloc[-1]] + forecasts
            
            fig_fore.add_trace(go.Scatter(
                x=f_dates, y=f_rates,
                mode='lines+markers', name='LSTM Forecast Projection',
                line=dict(color='#ff7b72', width=3, dash='dash')
            ))
            
            fig_fore.update_layout(
                title=f"Autoregressive Forecast Projection (Next {forecast_horizon} Weeks)",
                xaxis=dict(title="Date Timeline", gridcolor="rgba(48,54,61,0.3)"),
                yaxis=dict(title="Normalized Case Rate per 100k", gridcolor="rgba(48,54,61,0.3)"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#c9d1d9",
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig_fore, use_container_width=True)
            
            # Summary Table
            st.markdown("#### 📋 Projected Numbers Table")
            forecast_df = pd.DataFrame({
                "Forecast Week": [f"+{i+1} Week" for i in range(forecast_horizon)],
                "Projected Date": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                "LSTM Predicted Case Rate (per 100k)": forecasts
            })
            st.dataframe(forecast_df, use_container_width=True)
            
            # Model Comparisons Display (ARIMA / SARIMA comparison metrics)
            st.markdown("#### 🔄 Model Comparison & Validation Metrics")
            comp_path = os.path.join("models", "model_comparisons.pkl")
            if os.path.exists(comp_path):
                comparisons = joblib.load(comp_path)
                comp_df = pd.DataFrame(comparisons).T.rename(columns={"MAE": "Mean Absolute Error (MAE)", "RMSE": "Root Mean Squared Error (RMSE)"})
                
                left_sub, right_sub = st.columns([1, 1])
                with left_sub:
                    st.dataframe(comp_df, use_container_width=True)
                with right_sub:
                    st.markdown("""
                    **Model Insight:**
                    - **Stacked LSTM:** Excels in capturing complex seasonal spikes and meteorological non-linear correlations by using high-dimensional memory.
                    - **SARIMA:** Fits traditional seasonal patterns. Solid baseline but struggles to react dynamically to sudden environmental shifts (e.g. erratic monsoon).
                    - **ARIMA:** Linear baseline. Captures trend well but fails on seasonal spikes.
                    """)
            else:
                st.info("Run the advanced validation scripts to generate comparison tables.")

# ==============================================================================
# TAB 3: GIS RISK HEATMAP
# ==============================================================================
with tab3:
    st.markdown("### 🗺️ GIS Outbreak Risk Heatmap Mapping")
    st.markdown(f"Geospatial visualization of average case rates for the selected pathogen: **{selected_disease}**")
    
    # 1. Prepare map baseline coordinates
    m = folium.Map(location=[15.0, 10.0], zoom_start=2, tiles="cartodbpositron")
    
    # Calculate average case rate per country for chosen disease
    map_data = []
    for c, coords in COUNTRY_COORDS.items():
        sub = df[(df["country"] == c) & (df["disease"] == selected_disease)]
        if not sub.empty:
            rate = float(sub["case_rate"].mean())
            cases = int(sub["cases"].sum())
            deaths = int(sub["deaths"].sum())
            
            # Color coding based on case rate thresholds
            if rate < RISK_LOW:
                color = "#2ea44f" # green
                risk_lvl = "Low"
            elif rate < RISK_MEDIUM:
                color = "#d29922" # orange
                risk_lvl = "Medium"
            elif rate < RISK_HIGH:
                color = "#f0883e" # dark orange
                risk_lvl = "High"
            else:
                color = "#f85149" # red
                risk_lvl = "Critical"
                
            folium.CircleMarker(
                location=coords,
                radius=max(6, min(30, int(rate * 0.4))),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(f"""
                    <b>{c}</b><br>
                    Pathogen: {selected_disease}<br>
                    Avg Case Rate: {rate:.2f} per 100k<br>
                    Cumulative Cases: {cases:,}<br>
                    Cumulative Deaths: {deaths:,}<br>
                    Classification Tiers: <b>{risk_lvl}</b>
                """, max_width=250)
            ).add_to(m)
            
    # Draw map
    st_folium(m, width=1200, height=600)

# ==============================================================================
# TAB 4: ALERTS CONSOLE
# ==============================================================================
with tab4:
    st.markdown("### 🔔 Active Early Warning Warnings Log")
    st.markdown("The GDOFS Alert Engine flags zones where disease case-rates cross thresholds (>70% risk confidence / Critical severity).")
    
    alerts_df = load_db_alerts()
    
    if alerts_df.empty:
        st.success("🎉 All systems clear. No active outbreak warnings flagged.")
    else:
        # Display alerts in a clean, warnings-style interface
        col_list, col_filter = st.columns([3, 1])
        
        with col_filter:
            st.markdown("#### 🔍 Filter Logs")
            c_filter = st.selectbox("Filter Country", ["All"] + TARGET_COUNTRIES)
            d_filter = st.selectbox("Filter Pathogen", ["All"] + TARGET_DISEASES)
            
            # Filter the alerts df
            filtered_alerts = alerts_df.copy()
            if c_filter != "All":
                filtered_alerts = filtered_alerts[filtered_alerts["country"] == c_filter]
            if d_filter != "All":
                filtered_alerts = filtered_alerts[filtered_alerts["disease"] == d_filter]
                
            st.metric("Total Warnings Flagged", len(filtered_alerts))
            
        with col_list:
            for _, row in filtered_alerts.head(20).iterrows():
                risk_lvl = row["risk_level"]
                style_class = "alert-critical" if risk_lvl == "Critical" else "alert-high"
                
                st.markdown(f"""
                <div class="{style_class}">
                    <strong>[{row['timestamp']}] {row['message']}</strong><br>
                    <span style="font-size: 0.85rem; color: #c9d1d9;">
                        Transmission Status: {row['status']} | Outbreak Risk Score: {row['risk_score']}% | Pathogen: {row['disease']}
                    </span>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# TAB 5: EXPLAINABILITY & ANALYTICS
# ==============================================================================
with tab5:
    st.markdown("### 🔍 Model Explainability (XAI)")
    st.markdown("Understanding features driving the prediction outputs of the Random Forest and XGBoost risk classifiers.")
    
    left_xai, right_xai = st.columns([1, 1])
    
    with left_xai:
        st.markdown("#### 📊 Random Forest Feature Importances")
        # Load and plot feature importances
        imp_path = os.path.join("models", "rf_feature_importances.pkl")
        if os.path.exists(imp_path):
            importances = joblib.load(imp_path)
            imp_df = pd.DataFrame(importances).reset_index().rename(columns={"index": "Features", 0: "Importance"})
            
            # Select top 12 features for display
            top_imp = imp_df.head(12)
            fig_imp = px.bar(top_imp, x="Importance", y="Features", orientation="h", color="Importance",
                             color_continuous_scale="Viridis", title="Top 12 Features Influencing Outbreak Risk Tiers")
            fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Train the classifiers to generate feature importance profiles.")
            
    with right_xai:
        st.markdown("#### 📐 Mathematical & Feature Glossary")
        st.markdown("""
        The GDOFS AI model relies on advanced temporal and environmental indicators:
        
        1. **Lag Features (`cases_lag1`, `case_rate_lag4`, etc.):**
           Captures historical momentum. An upward trend in case rates over the past 1-4 weeks is the strongest predictor of a full-scale outbreak.
           
        2. **Growth Rate (`growth_rate`):**
           Calculates week-over-week growth:
           $$\Delta \% = \\frac{C_t - C_{t-1}}{C_{t-1} + \epsilon}$$
           Sudden exponential growths trigger critical escalations.
           
        3. **Temp-Humidity Index (THI):**
           Combined vector tracking vector-borne pathogen suitabilities. Highly correlated with mosquito breeding speed (Dengue, Malaria):
           $$THI = T_{Celsius} \\times \\left(\\frac{Humidity\%}{100}\\right)$$
           
        4. **Rolling Standard Deviation (4-week window):**
           Measures historical epidemiological volatility. Unstable case rates indicate high risk of transmission cycles.
        """)
        
        # Pearson correlation matrix heatmap
        st.markdown("#### 🧬 Pearson Feature Correlations")
        numeric_df = filtered_df.select_dtypes(include="number")
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
                                 title="Indicator Correlation Grid")
            fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9")
            st.plotly_chart(fig_corr, use_container_width=True)
