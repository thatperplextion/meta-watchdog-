"""
Trend Analyzer

Analyzes trends in performance metrics to support failure prediction.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import numpy as np
from numpy.typing import NDArray


@dataclass
class TrendResult:
    """Result of trend analysis."""
    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1
    trend_slope: float
    confidence: float
    forecast_next: Optional[float] = None
    forecast_horizon: int = 0
    change_points: List[int] = field(default_factory=list)


class TrendAnalyzer:
    """
    Analyzes trends in time series data to detect patterns
    that may indicate future model failure.
    
    Features:
    - Linear and non-linear trend detection
    - Change point detection
    - Seasonality detection
    - Trend forecasting
    """
    
    def __init__(
        self,
        min_samples: int = 10,
        trend_threshold: float = 0.01,
    ):
        """
        Initialize trend analyzer.
        
        Args:
            min_samples: Minimum samples required for trend analysis
            trend_threshold: Minimum slope to consider a trend
        """
        self.min_samples = min_samples
        self.trend_threshold = trend_threshold
    
    def analyze_trend(
        self,
        values: NDArray[np.floating],
        timestamps: Optional[List[datetime]] = None
    ) -> TrendResult:
        """
        Analyze trend in a time series.
        
        Args:
            values: Array of values
            timestamps: Optional timestamps
            
        Returns:
            TrendResult with analysis
        """
        if len(values) < self.min_samples:
            return TrendResult(
                metric_name="unknown",
                trend_direction="stable",
                trend_strength=0.0,
                trend_slope=0.0,
                confidence=0.0,
            )
        
        # Compute linear trend
        x = np.arange(len(values))
        slope, intercept, r_squared = self._linear_fit(x, values)
        
        # Determine direction
        if abs(slope) < self.trend_threshold:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Confidence based on R² and sample size
        sample_factor = min(1.0, len(values) / 50)
        confidence = r_squared * sample_factor
        
        # Forecast
        forecast_horizon = min(10, len(values) // 4)
        forecast_x = len(values) + forecast_horizon
        forecast_value = slope * forecast_x + intercept
        
        # Detect change points
        change_points = self._detect_change_points(values)
        
        return TrendResult(
            metric_name="unknown",
            trend_direction=direction,
            trend_strength=abs(r_squared),
            trend_slope=float(slope),
            confidence=confidence,
            forecast_next=float(forecast_value),
            forecast_horizon=forecast_horizon,
            change_points=change_points,
        )
    
    def analyze_multiple_trends(
        self,
        metrics: Dict[str, NDArray[np.floating]]
    ) -> Dict[str, TrendResult]:
        """
        Analyze trends for multiple metrics.
        
        Args:
            metrics: Dictionary of metric_name -> values
            
        Returns:
            Dictionary of metric_name -> TrendResult
        """
        results = {}
        for name, values in metrics.items():
            result = self.analyze_trend(values)
            result.metric_name = name
            results[name] = result
        return results
    
    def _linear_fit(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating]
    ) -> Tuple[float, float, float]:
        """
        Fit a linear model and return slope, intercept, R².
        """
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Slope
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0, float(y_mean), 0.0
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return float(slope), float(intercept), float(r_squared)
    
    def _detect_change_points(
        self,
        values: NDArray[np.floating],
        sensitivity: float = 2.0
    ) -> List[int]:
        """
        Detect change points in the time series.
        
        Uses cumulative sum (CUSUM) method.
        """
        if len(values) < 10:
            return []
        
        change_points = []
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return []
        
        # Cumulative sum of deviations
        cumsum = np.cumsum(values - mean_val)
        cumsum_range = cumsum.max() - cumsum.min()
        
        # Threshold for change detection
        threshold = sensitivity * std_val * np.sqrt(len(values))
        
        # Find points where cumsum exceeds threshold
        for i in range(1, len(cumsum) - 1):
            local_max = (
                cumsum[i] > cumsum[i-1] and 
                cumsum[i] > cumsum[i+1] and
                cumsum[i] > threshold
            )
            local_min = (
                cumsum[i] < cumsum[i-1] and 
                cumsum[i] < cumsum[i+1] and
                cumsum[i] < -threshold
            )
            
            if local_max or local_min:
                change_points.append(i)
        
        return change_points
    
    def detect_acceleration(
        self,
        values: NDArray[np.floating]
    ) -> Tuple[float, str]:
        """
        Detect if trend is accelerating or decelerating.
        
        Args:
            values: Time series values
            
        Returns:
            Tuple of (acceleration_rate, status)
        """
        if len(values) < 20:
            return 0.0, "insufficient_data"
        
        # Split into halves
        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]
        
        # Compute trend for each half
        slope1, _, _ = self._linear_fit(np.arange(len(first_half)), first_half)
        slope2, _, _ = self._linear_fit(np.arange(len(second_half)), second_half)
        
        # Acceleration is change in slope
        acceleration = slope2 - slope1
        
        if abs(acceleration) < 0.001:
            status = "stable"
        elif acceleration > 0:
            if slope2 > 0:
                status = "accelerating_increase"
            else:
                status = "decelerating_decrease"
        else:
            if slope2 < 0:
                status = "accelerating_decrease"
            else:
                status = "decelerating_increase"
        
        return float(acceleration), status
    
    def compute_volatility(
        self,
        values: NDArray[np.floating],
        window: int = 5
    ) -> float:
        """
        Compute volatility (rolling standard deviation).
        
        Args:
            values: Time series values
            window: Rolling window size
            
        Returns:
            Average volatility
        """
        if len(values) < window:
            return float(np.std(values))
        
        rolling_std = []
        for i in range(window, len(values)):
            rolling_std.append(np.std(values[i-window:i]))
        
        return float(np.mean(rolling_std))
    
    def forecast(
        self,
        values: NDArray[np.floating],
        horizon: int = 10,
        method: str = "linear"
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Forecast future values.
        
        Args:
            values: Historical values
            horizon: Number of steps to forecast
            method: "linear" or "weighted"
            
        Returns:
            Tuple of (forecasts, confidence_intervals)
        """
        x = np.arange(len(values))
        
        if method == "linear":
            slope, intercept, r_squared = self._linear_fit(x, values)
            
            future_x = np.arange(len(values), len(values) + horizon)
            forecasts = slope * future_x + intercept
            
            # Confidence interval widens with horizon
            base_std = np.std(values)
            confidence = np.array([
                base_std * (1 + 0.1 * i) for i in range(horizon)
            ])
        
        else:  # weighted - more weight to recent values
            weights = np.exp(np.linspace(-1, 0, len(values)))
            weights /= weights.sum()
            
            weighted_mean = np.average(values, weights=weights)
            weighted_trend = np.average(np.diff(values), weights=weights[1:])
            
            forecasts = np.array([
                weighted_mean + weighted_trend * (i + 1)
                for i in range(horizon)
            ])
            
            base_std = np.sqrt(np.average((values - np.mean(values))**2, weights=weights))
            confidence = np.array([
                base_std * (1 + 0.15 * i) for i in range(horizon)
            ])
        
        return forecasts, confidence
    
    def get_trend_summary(
        self,
        metrics: Dict[str, NDArray[np.floating]]
    ) -> Dict[str, Any]:
        """
        Get comprehensive trend summary for multiple metrics.
        
        Args:
            metrics: Dictionary of metric_name -> values
            
        Returns:
            Summary dictionary
        """
        trends = self.analyze_multiple_trends(metrics)
        
        concerning_trends = [
            name for name, result in trends.items()
            if result.trend_direction == "decreasing" 
            and result.trend_strength > 0.3
        ]
        
        accelerating = []
        for name, values in metrics.items():
            if len(values) >= 20:
                accel, status = self.detect_acceleration(values)
                if "accelerating_decrease" in status:
                    accelerating.append(name)
        
        return {
            "trends": {
                name: {
                    "direction": r.trend_direction,
                    "strength": r.trend_strength,
                    "slope": r.trend_slope,
                    "confidence": r.confidence,
                }
                for name, r in trends.items()
            },
            "concerning_trends": concerning_trends,
            "accelerating_decline": accelerating,
            "change_points_detected": any(len(r.change_points) > 0 for r in trends.values()),
            "overall_health": "concerning" if concerning_trends else "stable",
        }
