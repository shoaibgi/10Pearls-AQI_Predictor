import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from datetime import timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.db_client import fetch_training_data, download_model_from_registry

st.set_page_config(page_title="10Pearls AQI Predictor", page_icon="image.png", layout="wide", initial_sidebar_state="expanded")

# --- INJECT CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
<style>
/* Sleek Metric Cards */
[data-testid="stMetric"] {
    background-color: #1e2127;
    padding: 15px 20px;
    border-radius: 8px;
    border: 1px solid #2d313a;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
[data-testid="stMetricValue"] {
    font-size: 2.2rem;
    color: #FF4B4B;
}
[data-testid="stMetricDelta"] {
    font-size: 1rem;
}
/* Modern Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    padding-bottom: 5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0;
    padding: 10px 25px;
    background-color: #1e2127;
    border: 1px solid #2d313a;
    border-bottom: none;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #FF4B4B;
    color: white;
    border-color: #FF4B4B;
}
/* Top Banner padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
/* Headers */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.title( " 🌍 10Pearls AQI Predictor ")
    st.markdown("<p style='font-size: 1.1rem; color: #a0aab2; margin-top: -15px;'> Production Grade Multi-Model MLOps Dashboard | Realtime Predictive Analytics for Karachi Air-Quality</p>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align: right; padding-top: 20px;'><span style='background-color: #2d313a; padding: 8px 15px; border-radius: 20px; font-size: 0.9rem; border: 1px solid #4CAF50; color: #4CAF50;'>🟢 System Online & Syncing</span></div>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_data():
    with st.spinner("📊 Fetching data..."):
        df = fetch_training_data(table_name="feature_store")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Karachi')
        current_time = pd.Timestamp.now(tz='Asia/Karachi')
        df = df[df.index <= current_time]
        return df.sort_index(ascending=True)

@st.cache_resource
def load_artifacts():
    with st.spinner("⚙️ Loading models..."):
        artifacts = ["scaler.pkl", "y_scaler.pkl", "rf_model.pkl", "xgb_model.json", "lstm_model.keras", "training_history.json"]
        for art in artifacts:
            download_model_from_registry(art)
        
        scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None
        y_scaler = joblib.load("models/y_scaler.pkl") if os.path.exists("models/y_scaler.pkl") else None
        
        models = {}
        if os.path.exists("models/rf_model.pkl"):
            try: models["Random Forest"] = joblib.load("models/rf_model.pkl")
            except: pass
        if os.path.exists("models/xgb_model.json"):
            try: 
                from xgboost import XGBRegressor
                xgb_model = XGBRegressor()
                xgb_model.load_model("models/xgb_model.json")
                models["XGBoost"] = xgb_model
            except: pass
        if os.path.exists("models/lstm_model.keras"):
            try: models["LSTM"] = tf.keras.models.load_model("models/lstm_model.keras", compile=False)
            except: pass
            
        history = []
        if os.path.exists("models/training_history.json"):
            with open("models/training_history.json", "r") as f:
                history = json.load(f)
        return models, scaler, y_scaler, history

def generate_forecast(model, scaler, y_scaler, df_history, hours=72):
    last_timestamp = df_history.index[-1]
    if isinstance(last_timestamp, str):
        last_timestamp = pd.to_datetime(last_timestamp)
        
    future_dates = [last_timestamp + timedelta(hours=i) for i in range(1, hours + 1)]
    future_df = pd.DataFrame(index=future_dates)
    
    features_to_simulate = ['aqi', 'pm2_5', 'pm10', 'nitrogen_dioxide', 'carbon_monoxide', 'sulphur_dioxide', 'ozone', 'dust']
    last_24h_data = df_history[features_to_simulate].iloc[-24:]
    repetitions = (hours // 24) + 1
    projected_values = np.tile(last_24h_data.values, (repetitions, 1))[:hours, :]
    future_df[features_to_simulate] = projected_values
    
    np.random.seed(42) 
    for col in features_to_simulate:
        base_val = last_24h_data[col].mean()
        drift = np.cumsum(np.random.normal(0, base_val * 0.01, size=hours))
        noise = np.random.normal(0, base_val * 0.05, size=hours)
        future_df[col] += drift + noise
        future_df[col] = future_df[col].clip(lower=0)

    future_df['hour'] = future_df.index.hour
    future_df['day_of_week'] = future_df.index.dayofweek
    future_df['month'] = future_df.index.month
    
    lookback_window = 24 * 8
    for col in ['pm2_5', 'pm10', 'aqi']:
        combined_series = pd.concat([df_history[col].iloc[-lookback_window:], future_df[col]])
        future_df[f'{col}_rolling_24h'] = combined_series.rolling(window=24, min_periods=1).mean().iloc[-hours:].values
        future_df[f'{col}_rolling_7d'] = combined_series.rolling(window=24*7, min_periods=1).mean().iloc[-hours:].values
        future_df[f'{col}_lag1'] = combined_series.shift(1).iloc[-hours:].values
        future_df[f'{col}_lag24'] = combined_series.shift(24).iloc[-hours:].values
        
    last_diff = df_history['aqi'].diff().iloc[-1]
    if pd.isna(last_diff): last_diff = 0
    decay_factors = [0.95 ** i for i in range(len(future_df))]
    future_df['aqi_change_rate'] = last_diff * np.array(decay_factors)
    
    features_needed = [
        'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'dust',
        'hour', 'day_of_week', 'month',
        'pm2_5_lag1', 'pm2_5_lag24', 'pm10_lag1', 'pm10_lag24', 'aqi_lag1', 'aqi_lag24', 'aqi_change_rate',
        'pm2_5_rolling_24h', 'pm2_5_rolling_7d', 'pm10_rolling_24h', 'pm10_rolling_7d',
        'aqi_rolling_24h', 'aqi_rolling_7d'
    ]
    
    full_timeline = pd.concat([df_history[features_needed], future_df[features_needed]])
    start_idx = len(df_history) - 24
    end_idx = start_idx + hours
    X_inputs = full_timeline.iloc[start_idx : end_idx]
    
    try:
        X_scaled = scaler.transform(X_inputs)
        if isinstance(model, tf.keras.Model):
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            pred_scaled = model.predict(X_reshaped, verbose=0)
            future_df['predicted_aqi'] = y_scaler.inverse_transform(pred_scaled).flatten() if y_scaler else pred_scaled.flatten()
        else:
            future_df['predicted_aqi'] = model.predict(X_scaled).flatten()
            
        if len(df_history) >= 25:
            X_anchor = df_history[features_needed].iloc[-25:-24]
            X_anchor_scaled = scaler.transform(X_anchor)
            if isinstance(model, tf.keras.Model):
                X_anchor_reshaped = X_anchor_scaled.reshape((X_anchor_scaled.shape[0], 1, X_anchor_scaled.shape[1]))
                pred_scaled = model.predict(X_anchor_reshaped, verbose=0)
                anchor_pred = y_scaler.inverse_transform(pred_scaled).flatten()[0] if y_scaler else pred_scaled.flatten()[0]
            else:
                anchor_pred = model.predict(X_anchor_scaled).flatten()[0]
            bias = df_history['aqi'].iloc[-1] - anchor_pred
            decay = np.array([0.95 ** i for i in range(len(future_df))])
            future_df['predicted_aqi'] += bias * decay
            future_df['predicted_aqi'] = future_df['predicted_aqi'].clip(lower=0)
        return future_df
    except (KeyError, ValueError) as e:
        st.error(f"Error: {e}")
        st.stop()

try:
    df = load_data()
    models, scaler, y_scaler, history = load_artifacts()
    
    if scaler is None or not models:
        st.error("❌ Models not found. Run training pipeline first.")
        st.stop()
    
    st.sidebar.image("images/air-quality-sensor.png", width=60)
    st.sidebar.title("⚙️ Control Panel")
    st.sidebar.markdown("Configure forecasting engine parameters.")
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Model Selection")
    selected_model_name = st.sidebar.selectbox("Active Algorithm", list(models.keys()))
    active_model = models[selected_model_name]
    
    if history:
        latest_run = history[-1]['results']
        metrics = next((m for m in latest_run if m["model"] == selected_model_name), None)
        if metrics:
            st.sidebar.markdown("### 📊 Model Telemetry")
            st.sidebar.metric("Latest RMSE", f"{metrics['rmse']:.2f}")
            st.sidebar.metric("Latest R² Score", f"{metrics.get('r2', 0):.3f}")
            
    st.sidebar.markdown("---")
    st.sidebar.caption("Built during 10Pearls Internship")
    
    latest_record = df.iloc[-1]
    forecast_df = generate_forecast(active_model, scaler, y_scaler, df, hours=72)

    max_forecast = forecast_df['predicted_aqi'].max()
    current_aqi = latest_record['aqi']

    if current_aqi > 300 or max_forecast > 300:
        st.error("**🚨 HAZARDOUS CRITICAL ALERT**: AQI exceeds 300. Complete avoidance of outdoor activity recommended.", icon="🚨")
    elif current_aqi > 150 or max_forecast > 150:
        st.warning("**😷 UNHEALTHY AIR QUALITY**: AQI exceeds 150. Limit prolonged outdoor exertion.", icon="⚠️")
    elif current_aqi < 50 and max_forecast < 50:
        st.success("**🍃 EXCELLENT AIR QUALITY**: AQI is well within safe limits. Ideal for outdoor activities.", icon="✅")

    st.markdown("### 📡 Real-Time Predictive Analytics & Forecast Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current AQI", f"{current_aqi:.0f}", delta="Live Status", delta_color="off")
    c2.metric("Tomorrow Avg", f"{forecast_df.iloc[0:24]['predicted_aqi'].mean():.0f}", delta="24h Projection", delta_color="off")
    c3.metric("Active Engine", selected_model_name, delta="Inference Mode", delta_color="off")
    c4.metric("Last Sync", latest_record.name.strftime('%H:%M %p'), delta=latest_record.name.strftime('%b %d, %Y'), delta_color="off")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📈 72-Hour Trajectory Analysis")
    fig = go.Figure()
    history_subset = df.tail(24 * 10)
    
    fig.add_trace(go.Scatter(x=history_subset.index, y=history_subset['aqi'], mode='lines', name='Historical', fill='tozeroy', line=dict(color='#00b4d8', width=3), fillcolor='rgba(0, 180, 216, 0.1)'))
    
    last_pt = pd.DataFrame({'predicted_aqi': [history_subset['aqi'].iloc[-1]]}, index=[history_subset.index[-1]])
    forecast_plot = pd.concat([last_pt, forecast_df[['predicted_aqi']]])
    
    fig.add_trace(go.Scatter(x=forecast_plot.index, y=forecast_plot['predicted_aqi'], mode='lines', name='Forecast Projection', line=dict(color='#ff4b4b', width=3, dash='dash')))
    
    fig.add_hline(y=150, line_dash="dot", line_color="orange", annotation_text="Unhealthy (150)", annotation_position="top left")
    fig.add_hline(y=300, line_dash="solid", line_color="red", annotation_text="Hazardous (300)", annotation_position="top left")
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Air Quality Index (AQI)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2d313a', zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Forecast Dataset", "🧠 AI Explainability (SHAP)", "📉 Model Telemetry"])

    with tab1:
        st.markdown("#### Detailed Hourly Predictions")
        st.markdown("Raw tabular data generated by the active forecasting engine.")
        st.dataframe(forecast_df[['predicted_aqi']].head(72).style.format("{:.1f}").background_gradient(cmap='Reds'), use_container_width=True)

    with tab2:
        st.markdown("#### Feature Importance Analysis")
        st.markdown("Understand which environmental factors are driving the current AQI prediction.")
        if st.button("Generate SHAP Analysis", type="primary"):
            with st.spinner("Computing high-fidelity SHAP values... (This may take a moment)"):
                try:
                    input_data = df[['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'dust', 'hour', 'day_of_week', 'month', 'pm2_5_lag1', 'pm2_5_lag24', 'pm10_lag1', 'pm10_lag24', 'aqi_lag1', 'aqi_lag24', 'aqi_change_rate', 'pm2_5_rolling_24h', 'pm2_5_rolling_7d', 'pm10_rolling_24h', 'pm10_rolling_7d', 'aqi_rolling_24h', 'aqi_rolling_7d']].iloc[-24].to_frame().T
                    input_scaled = scaler.transform(input_data)
                    
                    if selected_model_name in ["Random Forest", "XGBoost"]:
                        explainer = shap.TreeExplainer(active_model)
                        shap_values = explainer.shap_values(input_scaled)
                    else:
                        background = df[input_data.columns].sample(20, random_state=42)
                        background_scaled = scaler.transform(background)
                        explainer = shap.KernelExplainer(lambda x: active_model.predict(x.reshape(x.shape[0], 1, x.shape[1]), verbose=0).flatten() if isinstance(active_model, tf.keras.Model) else active_model.predict(x).flatten(), background_scaled)
                        shap_values = explainer.shap_values(input_scaled)

                    if isinstance(shap_values, list): shap_values = shap_values[0]
                    fig = plt.figure(figsize=(10, 5))
                    plt.style.use('dark_background')
                    shap.summary_plot(shap_values, input_data, feature_names=input_data.columns, plot_type="bar", show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"SHAP Error: {e}")

    with tab3:
        st.markdown("#### Historical Training Performance")
        if history:
            history_data = []
            for entry in history:
                for res in entry['results']:
                    history_data.append({"Date": entry['date'], "Model": res['model'], "RMSE": res['rmse'], "MAE": res.get('mae', np.nan), "R2": res.get('r2', np.nan)})
            hist_df = pd.DataFrame(history_data)
            fig_hist = px.line(hist_df, x="Date", y="RMSE", color="Model", markers=True, title="Model RMSE Over Time")
            fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="", hovermode="x unified")
            fig_hist.update_xaxes(showgrid=False)
            fig_hist.update_yaxes(showgrid=True, gridcolor='#2d313a')
            st.plotly_chart(fig_hist, use_container_width=True)
            st.dataframe(hist_df.sort_values("Date", ascending=False), use_container_width=True)
        else:
            st.info("No training history available.")

except Exception as e:
    st.error(f"System Error: {e}")