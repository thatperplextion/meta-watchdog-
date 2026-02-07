"""
Meta-Watchdog Real-Time Monitoring Dashboard
A Flask-based web interface for presenting ML model health monitoring.
SELF-CONTAINED VERSION - No external meta_watchdog imports needed.
"""

from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from dataclasses import dataclass
from typing import Tuple, Optional

app = Flask(__name__)

# ============================================================================
# EMBEDDED DATA LOADER (from meta_watchdog.data)
# ============================================================================
@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    name: str
    n_samples: int
    n_features: int
    feature_names: list
    target_name: str

def load_electricity_data(data_path: str, sample_size: int = 5000) -> Tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Load electricity consumption dataset."""
    csv_file = os.path.join(data_path, 'household_power_consumption.txt')
    
    feature_names = ['Global_reactive_power', 'Voltage', 'Global_intensity', 
                    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    if not os.path.exists(csv_file):
        # Generate synthetic data if file not found
        print("[Data] CSV not found, generating synthetic electricity data...")
        np.random.seed(42)
        n = sample_size
        
        X = np.column_stack([
            np.random.uniform(0.1, 0.5, n),    # Global_reactive_power
            np.random.uniform(230, 250, n),    # Voltage
            np.random.uniform(0.5, 20, n),     # Global_intensity
            np.random.uniform(0, 50, n),       # Sub_metering_1
            np.random.uniform(0, 50, n),       # Sub_metering_2
            np.random.uniform(0, 20, n),       # Sub_metering_3
        ])
        
        # Target: combination of features with noise
        y = 0.5 * X[:, 2] + 0.1 * X[:, 0] + 0.01 * X[:, 3] + np.random.normal(0, 0.2, n)
        
    else:
        print(f"[Data] Loading from {csv_file}")
        df = pd.read_csv(csv_file, sep=';', low_memory=False)
        
        # Clean data
        df = df.replace('?', np.nan)
        numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=numeric_cols)
        
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        X = df[feature_names].values.astype(float)
        y = df['Global_active_power'].values.astype(float)
    
    info = DatasetInfo(
        name='Electricity Consumption',
        n_samples=len(y),
        n_features=X.shape[1],
        feature_names=feature_names,
        target_name='Global_active_power'
    )
    
    print(f"[Data] Loaded {info.n_samples} samples with {info.n_features} features")
    return X, y, info

# ============================================================================
# EMBEDDED ORCHESTRATOR (from meta_watchdog.orchestrator)
# ============================================================================
class SimpleOrchestrator:
    """Simplified Meta-Watchdog orchestrator for monitoring."""
    
    def __init__(self):
        self.observations = []
        self.drift_score = 0.0
        self.health_score = 1.0
    
    def observe(self, X, y_true, y_pred, confidence):
        """Record an observation."""
        mae = np.mean(np.abs(y_pred - y_true))
        self.drift_score = min(mae / 0.5, 1.0)
        self.health_score = np.mean(confidence)
        
        self.observations.append({
            'timestamp': datetime.now(),
            'samples': len(y_true),
            'mae': mae,
            'confidence': self.health_score,
            'drift_score': self.drift_score
        })
    
    def get_health(self):
        return self.health_score
    
    def get_drift_score(self):
        return self.drift_score

# ============================================================================
# DASHBOARD STATE
# ============================================================================
class DashboardState:
    def __init__(self):
        self.model = None
        self.orchestrator = None
        self.X_test = None
        self.y_test = None
        self.metrics_history = []
        self.current_batch = 0
        self.is_running = False
        self.drift_mode = False
        self.r2_score = 0
        self.total_samples = 0
        self.feature_names = []
        
state = DashboardState()

def initialize_system():
    """Initialize the ML model and monitoring system."""
    print("[Dashboard] Initializing system...")
    
    # Find data path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(os.path.dirname(current_dir), 'data'),
        '/opt/render/project/src/data',
        os.path.join(current_dir, 'data'),
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        data_path = possible_paths[0]
        os.makedirs(data_path, exist_ok=True)
    
    print(f"[Dashboard] Using data path: {data_path}")
    
    # Load data
    X, y, info = load_electricity_data(data_path, sample_size=5000)
    state.total_samples = info.n_samples
    state.feature_names = info.feature_names
    
    # Split data
    X_train, state.X_test, y_train, state.y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    state.model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    state.model.fit(X_train, y_train)
    
    # Calculate R2
    y_pred = state.model.predict(state.X_test)
    state.r2_score = r2_score(state.y_test, y_pred)
    
    # Initialize orchestrator
    state.orchestrator = SimpleOrchestrator()
    
    print(f"[Dashboard] System ready! R2 = {state.r2_score:.4f}")

# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current system status."""
    return jsonify({
        'initialized': state.model is not None,
        'is_running': state.is_running,
        'drift_mode': state.drift_mode,
        'current_batch': state.current_batch,
        'r2_score': round(state.r2_score, 4) if state.r2_score else 0,
        'total_samples': state.total_samples,
        'metrics_count': len(state.metrics_history)
    })

@app.route('/api/metrics')
def get_metrics():
    """Get metrics history."""
    return jsonify({
        'history': state.metrics_history[-50:],
        'current_batch': state.current_batch
    })

@app.route('/api/process_batch/<mode>')
def process_batch(mode):
    """Process a single batch of data."""
    if state.model is None:
        return jsonify({'error': 'System not initialized'}), 400
    
    batch_size = 100
    idx = state.current_batch * batch_size
    end_idx = min(idx + batch_size, len(state.X_test))
    
    if idx >= len(state.X_test):
        state.current_batch = 0
        idx = 0
        end_idx = batch_size
    
    X_batch = state.X_test[idx:end_idx].copy()
    y_batch = state.y_test[idx:end_idx].copy()
    
    # Apply drift if requested
    drift_applied = False
    if mode == 'drift':
        drift_amount = 1.0 + (state.current_batch % 5) * 0.3
        X_batch = X_batch + drift_amount * np.std(X_batch, axis=0)
        drift_applied = True
    
    # Make predictions
    y_pred = state.model.predict(X_batch)
    confidence = np.clip(1 - np.abs(y_pred - y_batch) / (np.abs(y_batch) + 1), 0.5, 1.0)
    
    # Observe with orchestrator
    state.orchestrator.observe(X_batch, y_batch, y_pred, confidence)
    
    # Calculate metrics
    mae = float(np.mean(np.abs(y_pred - y_batch)))
    rmse = float(np.sqrt(np.mean((y_pred - y_batch) ** 2)))
    avg_confidence = float(np.mean(confidence))
    
    # Determine status
    if mae > 0.5:
        status = 'critical'
        status_text = 'DRIFT DETECTED!'
    elif mae > 0.1:
        status = 'warning'
        status_text = 'Warning'
    else:
        status = 'normal'
        status_text = 'Normal'
    
    metrics = {
        'batch': state.current_batch + 1,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'samples': len(X_batch),
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'confidence': round(avg_confidence, 3),
        'status': status,
        'status_text': status_text,
        'drift_applied': drift_applied
    }
    
    state.metrics_history.append(metrics)
    state.current_batch += 1
    
    return jsonify(metrics)

@app.route('/api/reset')
def reset():
    """Reset the monitoring state."""
    state.metrics_history = []
    state.current_batch = 0
    state.is_running = False
    return jsonify({'success': True})

@app.route('/api/initialize')
def api_initialize():
    """Initialize or reinitialize the system."""
    try:
        initialize_system()
        return jsonify({'success': True, 'r2_score': state.r2_score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# STARTUP
# ============================================================================
# Initialize on import (for gunicorn)
try:
    initialize_system()
except Exception as e:
    print(f"[Dashboard] Init warning: {e}")
    print("[Dashboard] Will generate synthetic data")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("   META-WATCHDOG DASHBOARD")
    print("="*60)
    print("   Open your browser and go to:")
    print("   http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
