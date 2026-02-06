"""
Meta-Failure Prediction Model

A meta-level ML model that predicts when and why the base model will fail.
This is the core "self-aware" component of the system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque
import numpy as np
from numpy.typing import NDArray

from meta_watchdog.core.data_structures import (
    FailurePrediction,
    PerformanceMetrics,
    ReliabilityScore,
    Severity,
)
from meta_watchdog.meta_prediction.trend_analyzer import TrendAnalyzer


@dataclass
class FailureIndicator:
    """A single indicator of potential failure."""
    name: str
    value: float
    threshold: float
    severity: Severity
    contribution: float  # Contribution to failure probability
    description: str


class MetaFailurePredictor:
    """
    Meta-ML model that predicts future model failure.
    
    This is the "brain" that watches the watcher. It observes:
    - Performance trends
    - Confidence patterns
    - Feature distribution changes
    - Historical failure patterns
    
    And predicts:
    - Probability of failure
    - Estimated time to failure
    - Contributing factors
    - Failure type
    
    The model is itself interpretable, outputting human-readable
    explanations for its predictions.
    """
    
    def __init__(
        self,
        failure_threshold: float = 0.5,
        warning_threshold: float = 0.3,
        lookback_window: int = 100,
        prediction_horizon: int = 50,
    ):
        """
        Initialize the meta-failure predictor.
        
        Args:
            failure_threshold: Probability threshold for failure alert
            warning_threshold: Probability threshold for warning
            lookback_window: Number of past observations to consider
            prediction_horizon: How far ahead to predict (in predictions)
        """
        self.failure_threshold = failure_threshold
        self.warning_threshold = warning_threshold
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        
        # Trend analyzer
        self._trend_analyzer = TrendAnalyzer()
        
        # Historical data
        self._metrics_history: deque = deque(maxlen=lookback_window * 2)
        self._reliability_history: deque = deque(maxlen=lookback_window * 2)
        self._failure_history: deque = deque(maxlen=100)  # Historical failures
        
        # Prediction history
        self._prediction_history: deque = deque(maxlen=lookback_window)
        
        # Indicator weights (learned/tunable)
        self._indicator_weights = {
            "performance_decline": 0.25,
            "confidence_degradation": 0.15,
            "calibration_error": 0.15,
            "stability_decline": 0.15,
            "feature_drift": 0.15,
            "model_staleness": 0.10,
            "trend_acceleration": 0.05,
        }
        
        # Failure patterns (learned from history)
        self._failure_patterns: List[Dict[str, Any]] = []
        
        # State
        self._last_prediction: Optional[FailurePrediction] = None
        self._consecutive_warnings = 0
    
    # ========== Main Prediction Interface ==========
    
    def predict_failure(
        self,
        current_metrics: PerformanceMetrics,
        reliability_score: ReliabilityScore,
        feature_stats: Optional[Dict[str, Any]] = None,
        model_age: int = 0,
    ) -> FailurePrediction:
        """
        Predict the probability and timing of model failure.
        
        Args:
            current_metrics: Current performance metrics
            reliability_score: Current reliability score
            feature_stats: Feature health statistics
            model_age: Number of predictions since last training
            
        Returns:
            FailurePrediction with failure probability and time estimate
        """
        # Store in history
        self._metrics_history.append({
            "timestamp": datetime.now(),
            "metrics": current_metrics,
        })
        self._reliability_history.append({
            "timestamp": datetime.now(),
            "score": reliability_score.score,
            "status": reliability_score.status,
        })
        
        # Compute failure indicators
        indicators = self._compute_failure_indicators(
            current_metrics, reliability_score, feature_stats, model_age
        )
        
        # Compute base failure probability
        base_probability = self._compute_base_probability(indicators)
        
        # Adjust for trends
        trend_adjustment = self._compute_trend_adjustment()
        
        # Adjust for historical patterns
        pattern_adjustment = self._compute_pattern_adjustment(indicators)
        
        # Final probability
        failure_probability = min(1.0, max(0.0,
            base_probability + trend_adjustment + pattern_adjustment
        ))
        
        # Estimate time to failure
        time_to_failure = self._estimate_time_to_failure(
            failure_probability, indicators
        )
        
        # Determine contributing factors
        contributing_factors = self._get_contributing_factors(indicators)
        
        # Predict failure type
        failure_type = self._predict_failure_type(indicators)
        
        # Compute probability trend
        prob_trend = self._compute_probability_trend(failure_probability)
        
        # Compute confidence in this prediction
        prediction_confidence = self._compute_prediction_confidence(indicators)
        
        # Create prediction
        prediction = FailurePrediction(
            failure_probability=failure_probability,
            time_to_failure=time_to_failure,
            prediction_confidence=prediction_confidence,
            contributing_factors=contributing_factors,
            factor_weights={f: indicators[i].contribution 
                          for i, f in enumerate(contributing_factors) 
                          if i < len(indicators)},
            predicted_failure_type=failure_type,
            probability_trend=prob_trend,
        )
        
        # Update state
        self._prediction_history.append({
            "timestamp": datetime.now(),
            "probability": failure_probability,
            "time_to_failure": time_to_failure,
        })
        
        self._last_prediction = prediction
        self._update_warning_count(failure_probability)
        
        return prediction
    
    # ========== Indicator Computation ==========
    
    def _compute_failure_indicators(
        self,
        metrics: PerformanceMetrics,
        reliability: ReliabilityScore,
        feature_stats: Optional[Dict[str, Any]],
        model_age: int,
    ) -> List[FailureIndicator]:
        """
        Compute individual failure indicators.
        
        Each indicator represents a signal that may predict failure.
        """
        indicators = []
        
        # 1. Performance Decline Indicator
        perf_value = 1.0 - (reliability.performance_score / 100.0)
        indicators.append(FailureIndicator(
            name="performance_decline",
            value=perf_value,
            threshold=0.3,  # 30% decline is concerning
            severity=self._value_to_severity(perf_value, 0.2, 0.4, 0.6),
            contribution=perf_value * self._indicator_weights["performance_decline"],
            description=f"Performance has declined by {perf_value*100:.1f}%"
        ))
        
        # 2. Confidence Degradation Indicator
        conf_value = 1.0 - (reliability.calibration_score / 100.0)
        indicators.append(FailureIndicator(
            name="confidence_degradation",
            value=conf_value,
            threshold=0.25,
            severity=self._value_to_severity(conf_value, 0.15, 0.3, 0.5),
            contribution=conf_value * self._indicator_weights["confidence_degradation"],
            description=f"Confidence calibration degraded by {conf_value*100:.1f}%"
        ))
        
        # 3. Calibration Error Indicator
        cal_error = metrics.confidence_accuracy_gap if metrics else 0.0
        cal_value = min(1.0, abs(cal_error) * 5)  # Scale to 0-1
        indicators.append(FailureIndicator(
            name="calibration_error",
            value=cal_value,
            threshold=0.3,
            severity=self._value_to_severity(cal_value, 0.2, 0.4, 0.6),
            contribution=cal_value * self._indicator_weights["calibration_error"],
            description=f"Calibration error: {abs(cal_error)*100:.1f}% gap"
        ))
        
        # 4. Stability Decline Indicator
        stability_value = 1.0 - (reliability.stability_score / 100.0)
        indicators.append(FailureIndicator(
            name="stability_decline",
            value=stability_value,
            threshold=0.25,
            severity=self._value_to_severity(stability_value, 0.15, 0.3, 0.5),
            contribution=stability_value * self._indicator_weights["stability_decline"],
            description=f"Prediction stability declined by {stability_value*100:.1f}%"
        ))
        
        # 5. Feature Drift Indicator
        if feature_stats:
            drift_value = feature_stats.get("drift_score", 0.0)
        else:
            drift_value = 0.0
        indicators.append(FailureIndicator(
            name="feature_drift",
            value=drift_value,
            threshold=0.2,
            severity=self._value_to_severity(drift_value, 0.1, 0.25, 0.5),
            contribution=drift_value * self._indicator_weights["feature_drift"],
            description=f"Feature drift detected: {drift_value*100:.1f}%"
        ))
        
        # 6. Model Staleness Indicator
        staleness = min(1.0, model_age / 10000)  # Normalize to prediction count
        indicators.append(FailureIndicator(
            name="model_staleness",
            value=staleness,
            threshold=0.5,
            severity=self._value_to_severity(staleness, 0.3, 0.6, 0.8),
            contribution=staleness * self._indicator_weights["model_staleness"],
            description=f"Model has made {model_age} predictions since training"
        ))
        
        # 7. Trend Acceleration Indicator (if we have history)
        accel_value = self._compute_acceleration_indicator()
        indicators.append(FailureIndicator(
            name="trend_acceleration",
            value=accel_value,
            threshold=0.3,
            severity=self._value_to_severity(accel_value, 0.2, 0.4, 0.6),
            contribution=accel_value * self._indicator_weights["trend_acceleration"],
            description=f"Negative trend accelerating" if accel_value > 0.3 else "Trend stable"
        ))
        
        return indicators
    
    def _value_to_severity(
        self, 
        value: float, 
        medium: float, 
        high: float, 
        critical: float
    ) -> Severity:
        """Convert value to severity level."""
        if value >= critical:
            return Severity.CRITICAL
        elif value >= high:
            return Severity.HIGH
        elif value >= medium:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _compute_acceleration_indicator(self) -> float:
        """Compute indicator for accelerating negative trends."""
        if len(self._reliability_history) < 20:
            return 0.0
        
        scores = np.array([h["score"] for h in self._reliability_history])
        
        # Analyze acceleration
        _, status = self._trend_analyzer.detect_acceleration(scores)
        
        if "accelerating_decrease" in status:
            return 0.7
        elif "decelerating_increase" in status:
            return 0.3
        else:
            return 0.1
    
    # ========== Probability Computation ==========
    
    def _compute_base_probability(
        self, indicators: List[FailureIndicator]
    ) -> float:
        """
        Compute base failure probability from indicators.
        
        Uses weighted sum of indicator contributions.
        """
        total_contribution = sum(ind.contribution for ind in indicators)
        
        # Apply non-linear transformation
        # Higher values grow faster (convex function)
        probability = 1 - np.exp(-2 * total_contribution)
        
        return float(probability)
    
    def _compute_trend_adjustment(self) -> float:
        """
        Adjust probability based on observed trends.
        
        If things are getting worse faster, increase probability.
        """
        if len(self._reliability_history) < 10:
            return 0.0
        
        scores = np.array([h["score"] for h in self._reliability_history])[-50:]
        trend_result = self._trend_analyzer.analyze_trend(scores)
        
        adjustment = 0.0
        
        # Negative trend increases failure probability
        if trend_result.trend_direction == "decreasing":
            adjustment += abs(trend_result.trend_slope) * 10
            
            # Stronger trends have more impact
            adjustment *= (1 + trend_result.trend_strength)
        
        return min(0.3, adjustment)  # Cap at 30% adjustment
    
    def _compute_pattern_adjustment(
        self, indicators: List[FailureIndicator]
    ) -> float:
        """
        Adjust probability based on historical failure patterns.
        
        If current state matches past failure patterns, increase probability.
        """
        if not self._failure_patterns:
            return 0.0
        
        # Compute similarity to historical failure patterns
        current_vector = np.array([ind.value for ind in indicators])
        
        max_similarity = 0.0
        for pattern in self._failure_patterns:
            pattern_vector = np.array(pattern.get("indicator_values", []))
            if len(pattern_vector) == len(current_vector):
                # Cosine similarity
                similarity = np.dot(current_vector, pattern_vector) / (
                    np.linalg.norm(current_vector) * np.linalg.norm(pattern_vector) + 1e-8
                )
                max_similarity = max(max_similarity, similarity)
        
        # High similarity to failure pattern increases probability
        return max_similarity * 0.2  # Max 20% adjustment
    
    # ========== Time to Failure Estimation ==========
    
    def _estimate_time_to_failure(
        self,
        failure_probability: float,
        indicators: List[FailureIndicator]
    ) -> Optional[int]:
        """
        Estimate how many predictions until failure.
        
        Uses trend extrapolation and indicator analysis.
        """
        if failure_probability < 0.1:
            return None  # Too early to estimate
        
        if len(self._reliability_history) < 10:
            # Not enough history, use simple estimate
            if failure_probability > 0.8:
                return 10
            elif failure_probability > 0.5:
                return 50
            else:
                return 100
        
        # Get reliability score trend
        scores = np.array([h["score"] for h in self._reliability_history])
        
        if len(scores) < 5:
            return None
        
        # Forecast when score will hit critical threshold (40)
        forecasts, _ = self._trend_analyzer.forecast(scores, horizon=100)
        
        # Find first point below threshold
        critical_threshold = 40
        for i, forecast in enumerate(forecasts):
            if forecast < critical_threshold:
                return i + 1
        
        # If trend doesn't reach critical in horizon, estimate based on probability
        if failure_probability > 0.5:
            return int(100 * (1 - failure_probability))
        
        return None
    
    # ========== Contributing Factors ==========
    
    def _get_contributing_factors(
        self, indicators: List[FailureIndicator]
    ) -> List[str]:
        """Get list of factors contributing to failure risk."""
        # Sort by contribution
        sorted_indicators = sorted(
            indicators, 
            key=lambda x: x.contribution, 
            reverse=True
        )
        
        # Return names of significant contributors
        factors = []
        for ind in sorted_indicators:
            if ind.contribution > 0.05:  # Threshold for significance
                factors.append(ind.name)
        
        return factors[:5]  # Top 5 factors
    
    def _predict_failure_type(
        self, indicators: List[FailureIndicator]
    ) -> str:
        """Predict the type of failure that's most likely."""
        # Find dominant indicator
        max_indicator = max(indicators, key=lambda x: x.contribution)
        
        failure_type_map = {
            "performance_decline": "performance_degradation",
            "confidence_degradation": "calibration_failure",
            "calibration_error": "confidence_miscalibration",
            "stability_decline": "prediction_instability",
            "feature_drift": "data_drift",
            "model_staleness": "model_aging",
            "trend_acceleration": "rapid_degradation",
        }
        
        return failure_type_map.get(max_indicator.name, "unknown")
    
    # ========== Confidence and Trends ==========
    
    def _compute_prediction_confidence(
        self, indicators: List[FailureIndicator]
    ) -> float:
        """
        Compute confidence in the failure prediction itself.
        
        Meta-meta level: how confident are we in our failure prediction?
        """
        confidence = 0.5  # Base confidence
        
        # More history = more confidence
        history_factor = min(1.0, len(self._reliability_history) / 50)
        confidence += 0.2 * history_factor
        
        # Consistent indicators = more confidence
        contributions = [ind.contribution for ind in indicators]
        if len(contributions) > 0:
            cv = np.std(contributions) / (np.mean(contributions) + 1e-8)  # Coefficient of variation
            consistency_factor = 1 / (1 + cv)  # Lower CV = higher confidence
            confidence += 0.2 * consistency_factor
        
        # Strong trend = more confidence
        if len(self._reliability_history) >= 10:
            scores = np.array([h["score"] for h in self._reliability_history])
            trend_result = self._trend_analyzer.analyze_trend(scores)
            confidence += 0.1 * trend_result.trend_strength
        
        return min(1.0, confidence)
    
    def _compute_probability_trend(
        self, current_probability: float
    ) -> float:
        """Compute trend in failure probability."""
        if len(self._prediction_history) < 2:
            return 0.0
        
        probs = [p["probability"] for p in self._prediction_history]
        probs.append(current_probability)
        
        if len(probs) < 3:
            return 0.0
        
        # Simple trend: difference of recent averages
        recent = np.mean(probs[-3:])
        older = np.mean(probs[-6:-3]) if len(probs) >= 6 else np.mean(probs[:-3])
        
        return float(recent - older)
    
    def _update_warning_count(self, probability: float) -> None:
        """Track consecutive warnings."""
        if probability >= self.warning_threshold:
            self._consecutive_warnings += 1
        else:
            self._consecutive_warnings = 0
    
    # ========== Historical Pattern Learning ==========
    
    def record_failure(
        self,
        indicators: List[FailureIndicator],
        failure_type: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a failure event for pattern learning.
        
        Args:
            indicators: Indicators at time of failure
            failure_type: Type of failure that occurred
            timestamp: When failure occurred
        """
        pattern = {
            "timestamp": timestamp or datetime.now(),
            "failure_type": failure_type,
            "indicator_values": [ind.value for ind in indicators],
            "indicator_names": [ind.name for ind in indicators],
        }
        
        self._failure_history.append(pattern)
        self._failure_patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self._failure_patterns) > 50:
            self._failure_patterns = self._failure_patterns[-50:]
    
    # ========== Analysis Methods ==========
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive failure analysis.
        
        Returns:
            Dictionary with detailed analysis
        """
        if not self._last_prediction:
            return {"status": "no_prediction"}
        
        pred = self._last_prediction
        
        return {
            "failure_probability": pred.failure_probability,
            "time_to_failure": pred.time_to_failure,
            "urgency": pred.urgency.name,
            "prediction_confidence": pred.prediction_confidence,
            "contributing_factors": pred.contributing_factors,
            "predicted_failure_type": pred.predicted_failure_type,
            "probability_trend": pred.probability_trend,
            "trend_direction": "increasing" if pred.probability_trend > 0 else "stable_or_decreasing",
            "consecutive_warnings": self._consecutive_warnings,
            "recommendation": self._get_failure_recommendation(pred),
        }
    
    def _get_failure_recommendation(
        self, prediction: FailurePrediction
    ) -> str:
        """Generate recommendation based on failure prediction."""
        if prediction.failure_probability < 0.2:
            return "No immediate action required. Continue monitoring."
        
        elif prediction.failure_probability < 0.5:
            factors = ", ".join(prediction.contributing_factors[:2])
            return f"Warning: Monitor closely. Key concerns: {factors}"
        
        elif prediction.failure_probability < 0.8:
            if prediction.time_to_failure and prediction.time_to_failure < 50:
                return f"High Risk: Prepare for intervention. Estimated {prediction.time_to_failure} predictions until critical."
            return "High Risk: Schedule model review and potential retraining."
        
        else:
            return "CRITICAL: Immediate action required. Consider fallback or emergency retraining."
    
    def get_prediction_history(
        self, last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get historical failure predictions."""
        history = list(self._prediction_history)
        if last_n:
            history = history[-last_n:]
        return history
    
    def reset(self) -> None:
        """Reset all state."""
        self._metrics_history.clear()
        self._reliability_history.clear()
        self._prediction_history.clear()
        self._last_prediction = None
        self._consecutive_warnings = 0
