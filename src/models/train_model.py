import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from pytz import timezone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.db_client import fetch_training_data, upload_model_to_registry, download_model_from_registry

TARGET = 'target_aqi_next_hour'
FEATURES = [
    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'dust',
    'hour', 'day_of_week', 'month',
    'pm2_5_lag1', 'pm2_5_lag24', 'pm10_lag1', 'pm10_lag24', 'aqi_lag1', 'aqi_lag24',
    'aqi_change_rate', 'pm2_5_rolling_24h', 'pm2_5_rolling_7d',
    'pm10_rolling_24h', 'pm10_rolling_7d', 'aqi_rolling_24h', 'aqi_rolling_7d'
]

def calculate_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"📊 {name}: RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}

if __name__ == "__main__":
    print("🚀 Starting daily training pipeline...")
    
    df = fetch_training_data(table_name="feature_store")
    
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Asia/Karachi')
    
    current_time = pd.Timestamp.now(tz='Asia/Karachi')
    df = df[df.index <= current_time]
    df.sort_index(ascending=True, inplace=True)
    
    df['target_aqi_next_hour'] = df['aqi'].shift(-1)
    df.dropna(subset=['aqi_lag1', 'aqi_lag24', TARGET], inplace=True)
    
    if len(df) < 200:
        sys.exit("❌ Insufficient data. Run: python src/features/feature_pipeline.py backfill")
    
    X = df[FEATURES]
    y = df[TARGET]
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📊 Training on {len(df)} rows | Train: {len(X_train)} | Test: {len(X_test)}")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []

    print("\n🌲 Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=10, 
                               min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    results.append(calculate_metrics("Random Forest", y_test, rf.predict(X_test_scaled)))

    print("🚀 Training XGBoost...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, 
                       subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(X_train_scaled, y_train)
    results.append(calculate_metrics("XGBoost", y_test, xgb.predict(X_test_scaled)))

    print("🧠 Training LSTM...")
    
    val_split_idx_lstm = int(len(X_train_scaled) * 0.8)
    X_train_base, X_val = X_train_scaled[:val_split_idx_lstm], X_train_scaled[val_split_idx_lstm:]
    y_train_base, y_val = y_train.iloc[:val_split_idx_lstm], y_train.iloc[val_split_idx_lstm:]

    y_scaler = MinMaxScaler()
    y_train_base_scaled = y_scaler.fit_transform(y_train_base.values.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))
    
    X_train_lstm = X_train_base.reshape((X_train_base.shape[0], 1, X_train_base.shape[1]))
    X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    lstm = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(1, X_train_base.shape[1])),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    lstm.fit(X_train_lstm, y_train_base_scaled, epochs=50, batch_size=32, 
             validation_data=(X_val_lstm, y_val_scaled),
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
             verbose=0)

    lstm_preds = y_scaler.inverse_transform(lstm.predict(X_test_lstm, verbose=0)).flatten()
    results.append(calculate_metrics("LSTM", y_test, lstm_preds))

    best_model_stats = min(results, key=lambda x: x['rmse'])
    print(f"\n🏆 WINNER: {best_model_stats['model']} (RMSE: {best_model_stats['rmse']:.4f})")

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(y_scaler, "models/y_scaler.pkl")
    joblib.dump(rf, "models/rf_model.pkl", compress=('zlib', 9))
    xgb.save_model("models/xgb_model.json")
    lstm.save("models/lstm_model.keras")
    
    upload_model_to_registry("models/scaler.pkl", "scaler.pkl")
    upload_model_to_registry("models/y_scaler.pkl", "y_scaler.pkl")
    upload_model_to_registry("models/rf_model.pkl", "rf_model.pkl")
    upload_model_to_registry("models/xgb_model.json", "xgb_model.json")
    upload_model_to_registry("models/lstm_model.keras", "lstm_model.keras")

    history_file = "models/training_history.json"
    if download_model_from_registry("training_history.json") is None:
        history = []
    else:
        with open(history_file, "r") as f:
            history = json.load(f)
            
    # Append today's results
    new_entry = {
        "date": datetime.now(timezone('Asia/Karachi')).strftime("%Y-%m-%d %H:%M:%S"),
        "winner": best_model_stats['model'],
        "results": results
    }
    history.append(new_entry)
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
        
    upload_model_to_registry(history_file, "training_history.json")
    
    print("--- TRAINING PIPELINE COMPLETE ---")
    
    