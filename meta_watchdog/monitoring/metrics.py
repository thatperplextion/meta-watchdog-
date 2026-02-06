"""
Metric Definitions and Aggregation Utilities

This module provides metric computation and aggregation for monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import deque
import numpy as np
from numpy.typing import NDArray


class MetricType(Enum):
    """Types of metrics that can be tracked."""
    # Performance metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MAE = "mean_absolute_error"
    MSE = "mean_squared_error"
    RMSE = "root_mean_squared_error"
    R2 = "r2_score"
    
    # Confidence metrics
    MEAN_CONFIDENCE = "mean_confidence"
    CONFIDENCE_STD = "confidence_std"
    CONFIDENCE_ACCURACY_GAP = "confidence_accuracy_gap"
    BRIER_SCORE = "brier_score"
    
    # Calibration metrics
    CALIBRATION_ERROR = "calibration_error"
    OVERCONFIDENCE_RATIO = "overconfidence_ratio"
    UNDERCONFIDENCE_RATIO = "underconfidence_ratio"
    
    # Stability metrics
    PREDICTION_VARIANCE = "prediction_variance"
    CONFIDENCE_VARIANCE = "confidence_variance"
    
    # Custom
    CUSTOM = "custom"


@dataclass
class MetricValue:
    """A single metric value with metadata."""
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    window_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"MetricValue({self.metric_type.value}={self.value:.4f})"


class RollingWindow:
    """
    A rolling window for time-series data.
    
    Maintains a fixed-size window of values for computing
    rolling statistics and detecting trends.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize rolling window.
        
        Args:
            window_size: Maximum number of elements to keep
        """
        self.window_size = window_size
        self._values: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)
    
    def add(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Add a value to the window.
        
        Args:
            value: Value to add
            timestamp: Optional timestamp (defaults to now)
        """
        self._values.append(value)
        self._timestamps.append(timestamp or datetime.now())
    
    def add_batch(
        self, 
        values: List[float], 
        timestamps: Optional[List[datetime]] = None
    ) -> None:
        """Add multiple values at once."""
        if timestamps is None:
            timestamps = [datetime.now()] * len(values)
        
        for v, t in zip(values, timestamps):
            self.add(v, t)
    
    @property
    def values(self) -> NDArray[np.floating]:
        """Get values as numpy array."""
        return np.array(list(self._values))
    
    @property
    def timestamps(self) -> List[datetime]:
        """Get timestamps."""
        return list(self._timestamps)
    
    def __len__(self) -> int:
        return len(self._values)
    
    def is_full(self) -> bool:
        """Check if window is at capacity."""
        return len(self._values) >= self.window_size
    
    # ========== Statistics ==========
    
    def mean(self) -> float:
        """Compute mean of window."""
        if len(self._values) == 0:
            return 0.0
        return float(np.mean(self.values))
    
    def std(self) -> float:
        """Compute standard deviation."""
        if len(self._values) < 2:
            return 0.0
        return float(np.std(self.values))
    
    def variance(self) -> float:
        """Compute variance."""
        if len(self._values) < 2:
            return 0.0
        return float(np.var(self.values))
    
    def min(self) -> float:
        """Get minimum value."""
        if len(self._values) == 0:
            return 0.0
        return float(np.min(self.values))
    
    def max(self) -> float:
        """Get maximum value."""
        if len(self._values) == 0:
            return 0.0
        return float(np.max(self.values))
    
    def percentile(self, p: float) -> float:
        """Compute percentile."""
        if len(self._values) == 0:
            return 0.0
        return float(np.percentile(self.values, p))
    
    def median(self) -> float:
        """Compute median."""
        return self.percentile(50)
    
    # ========== Trend Analysis ==========
    
    def trend(self) -> float:
        """
        Compute trend (slope) of values over time.
        
        Positive = increasing, Negative = decreasing.
        
        Returns:
            Trend coefficient (slope of linear fit)
        """
        if len(self._values) < 2:
            return 0.0
        
        x = np.arange(len(self._values))
        y = self.values
        
        # Simple linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)
    
    def trend_strength(self) -> float:
        """
        Compute strength of trend (R² of linear fit).
        
        Returns:
            R² value in [0, 1], higher = stronger trend
        """
        if len(self._values) < 2:
            return 0.0
        
        x = np.arange(len(self._values))
        y = self.values
        
        # Linear fit
        slope = self.trend()
        intercept = np.mean(y) - slope * np.mean(x)
        y_pred = slope * x + intercept
        
        # R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return float(1 - (ss_res / ss_tot))
    
    def is_degrading(self, threshold: float = -0.01) -> bool:
        """
        Check if metric is showing degradation trend.
        
        Args:
            threshold: Trend threshold for degradation (default: -0.01)
            
        Returns:
            True if degrading
        """
        return self.trend() < threshold and self.trend_strength() > 0.3
    
    def is_improving(self, threshold: float = 0.01) -> bool:
        """
        Check if metric is showing improvement trend.
        
        Args:
            threshold: Trend threshold for improvement
            
        Returns:
            True if improving
        """
        return self.trend() > threshold and self.trend_strength() > 0.3
    
    # ========== Change Detection ==========
    
    def detect_change_point(self, sensitivity: float = 2.0) -> Optional[int]:
        """
        Detect if there's a significant change point in the window.
        
        Uses CUSUM-like detection for simplicity.
        
        Args:
            sensitivity: Number of std devs for significance
            
        Returns:
            Index of change point, or None if no change detected
        """
        if len(self._values) < 10:
            return None
        
        values = self.values
        cumsum = np.cumsum(values - np.mean(values))
        
        # Find maximum deviation
        max_idx = np.argmax(np.abs(cumsum))
        max_dev = np.abs(cumsum[max_idx])
        
        threshold = sensitivity * np.std(values) * np.sqrt(len(values))
        
        if max_dev > threshold:
            return int(max_idx)
        
        return None
    
    def recent_vs_historical(
        self, recent_fraction: float = 0.2
    ) -> Tuple[float, float, bool]:
        """
        Compare recent values to historical.
        
        Args:
            recent_fraction: Fraction of window to consider "recent"
            
        Returns:
            Tuple of (recent_mean, historical_mean, is_significantly_different)
        """
        if len(self._values) < 10:
            return 0.0, 0.0, False
        
        values = self.values
        split_idx = int(len(values) * (1 - recent_fraction))
        
        historical = values[:split_idx]
        recent = values[split_idx:]
        
        hist_mean = float(np.mean(historical))
        recent_mean = float(np.mean(recent))
        
        # Simple significance test
        pooled_std = np.std(values)
        if pooled_std == 0:
            return recent_mean, hist_mean, False
        
        z_score = abs(recent_mean - hist_mean) / (pooled_std / np.sqrt(len(recent)))
        is_significant = z_score > 2.0
        
        return recent_mean, hist_mean, is_significant
    
    def clear(self) -> None:
        """Clear all values."""
        self._values.clear()
        self._timestamps.clear()


class MetricAggregator:
    """
    Aggregates multiple metrics over time.
    
    Provides a unified interface for tracking all system metrics.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metric aggregator.
        
        Args:
            window_size: Default window size for all metrics
        """
        self.window_size = window_size
        self._windows: Dict[str, RollingWindow] = {}
        self._latest_values: Dict[str, MetricValue] = {}
        self._custom_computers: Dict[str, Callable] = {}
    
    def register_metric(
        self, 
        name: str, 
        metric_type: MetricType = MetricType.CUSTOM,
        window_size: Optional[int] = None
    ) -> None:
        """
        Register a metric to track.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            window_size: Optional custom window size
        """
        ws = window_size or self.window_size
        self._windows[name] = RollingWindow(ws)
    
    def record(
        self, 
        name: str, 
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Value to record
            timestamp: Optional timestamp
            metadata: Optional metadata
        """
        if name not in self._windows:
            self.register_metric(name)
        
        self._windows[name].add(value, timestamp)
        
        metric_type = MetricType.CUSTOM
        for mt in MetricType:
            if mt.value == name:
                metric_type = mt
                break
        
        self._latest_values[name] = MetricValue(
            metric_type=metric_type,
            value=value,
            timestamp=timestamp or datetime.now(),
            window_size=1,
            metadata=metadata or {}
        )
    
    def record_batch(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric_name -> value
            timestamp: Optional shared timestamp
        """
        ts = timestamp or datetime.now()
        for name, value in metrics.items():
            self.record(name, value, ts)
    
    def get_window(self, name: str) -> Optional[RollingWindow]:
        """Get the rolling window for a metric."""
        return self._windows.get(name)
    
    def get_latest(self, name: str) -> Optional[MetricValue]:
        """Get the latest value for a metric."""
        return self._latest_values.get(name)
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Returns:
            Dictionary with mean, std, min, max, trend, etc.
        """
        window = self._windows.get(name)
        if window is None or len(window) == 0:
            return {}
        
        return {
            "mean": window.mean(),
            "std": window.std(),
            "min": window.min(),
            "max": window.max(),
            "median": window.median(),
            "trend": window.trend(),
            "trend_strength": window.trend_strength(),
            "count": len(window),
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_statistics(name) for name in self._windows}
    
    def get_trends(self) -> Dict[str, float]:
        """Get trends for all metrics."""
        return {
            name: window.trend() 
            for name, window in self._windows.items()
            if len(window) > 1
        }
    
    def get_degrading_metrics(self) -> List[str]:
        """Get list of metrics showing degradation."""
        return [
            name for name, window in self._windows.items()
            if len(window) > 1 and window.is_degrading()
        ]
    
    def compute_health_indicators(self) -> Dict[str, Any]:
        """
        Compute overall health indicators from all metrics.
        
        Returns:
            Dictionary with health indicators
        """
        degrading = self.get_degrading_metrics()
        trends = self.get_trends()
        
        # Count concerning trends
        concerning_count = sum(1 for t in trends.values() if t < -0.01)
        
        return {
            "degrading_metrics": degrading,
            "degrading_count": len(degrading),
            "concerning_trend_count": concerning_count,
            "metrics_tracked": len(self._windows),
            "average_trend": np.mean(list(trends.values())) if trends else 0.0,
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._windows.clear()
        self._latest_values.clear()
    
    def reset_metric(self, name: str) -> None:
        """Reset a specific metric."""
        if name in self._windows:
            self._windows[name].clear()
        if name in self._latest_values:
            del self._latest_values[name]


def compute_accuracy(
    predictions: NDArray, 
    actuals: NDArray
) -> float:
    """Compute classification accuracy."""
    return float(np.mean(predictions == actuals))


def compute_mae(
    predictions: NDArray[np.floating], 
    actuals: NDArray[np.floating]
) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(predictions - actuals)))


def compute_mse(
    predictions: NDArray[np.floating], 
    actuals: NDArray[np.floating]
) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((predictions - actuals) ** 2))


def compute_rmse(
    predictions: NDArray[np.floating], 
    actuals: NDArray[np.floating]
) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(compute_mse(predictions, actuals)))


def compute_confidence_accuracy_gap(
    confidence: NDArray[np.floating],
    was_correct: NDArray[np.bool_]
) -> float:
    """
    Compute gap between confidence and actual accuracy.
    
    Positive = overconfident, Negative = underconfident.
    """
    mean_confidence = float(np.mean(confidence))
    actual_accuracy = float(np.mean(was_correct))
    return mean_confidence - actual_accuracy


def compute_calibration_error(
    confidence: NDArray[np.floating],
    was_correct: NDArray[np.bool_],
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Lower is better calibrated.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidence)
    
    for i in range(n_bins):
        mask = (confidence >= bin_boundaries[i]) & (confidence < bin_boundaries[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(was_correct[mask])
            bin_conf = np.mean(confidence[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / total) * abs(bin_acc - bin_conf)
    
    return float(ece)


def compute_brier_score(
    confidence: NDArray[np.floating],
    was_correct: NDArray[np.bool_]
) -> float:
    """
    Compute Brier score for probability calibration.
    
    Lower is better.
    """
    return float(np.mean((confidence - was_correct.astype(float)) ** 2))
