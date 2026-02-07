"""
Meta-Watchdog Real-Time Monitoring Dashboard
A Flask-based web interface for presenting ML model health monitoring.
"""

from flask import Flask, render_template, jsonify
import numpy as np
import sys
import os
from datetime import datetime
import threading
import time

# Add parent directory to path (works on both local and Render)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Also try Render's path
render_path = '/opt/render/project/src'

for path in [parent_dir, render_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

from meta_watchdog.data import DatasetLoader
from meta_watchdog.orchestrator import MetaWatchdogOrchestrator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

# Global state
class DashboardState:
    def __init__(self):
        self.model = None
        self.orchestrator = None
        self.X_test = None
        self.y_test = None
        self.loader = None
        self.metrics_history = []
        self.current_batch = 0
        self.is_running = False
        self.drift_mode = False
        self.r2_score = 0
        self.total_samples = 0
        
state = DashboardState()

def initialize_system():
    """Initialize the ML model and monitoring system."""
    print("[Dashboard] Initializing system...")
    
    # Load data - use parent directory path
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    state.loader = DatasetLoader(data_path)
    X, y, info = state.loader.load_electricity(sample_size=5000)
    state.total_samples = info.n_samples
    
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
    state.orchestrator = MetaWatchdogOrchestrator()
    
    print(f"[Dashboard] System ready! R² = {state.r2_score:.4f}")

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
        'history': state.metrics_history[-50:],  # Last 50 entries
        'current_batch': state.current_batch
    })

@app.route('/api/process_batch/<mode>')
def process_batch(mode):
    """Process a single batch of data."""
    if state.model is None:
        return jsonify({'error': 'System not initialized'}), 400
    
    # Get batch index
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
    
    # Create metrics entry
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

if __name__ == '__main__':
    # Initialize on startup
    initialize_system()
    
    print("\n" + "="*60)
    print("   META-WATCHDOG DASHBOARD")
    print("="*60)
    print("   Open your browser and go to:")
    print("   http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
