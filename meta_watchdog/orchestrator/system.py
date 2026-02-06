"""
Meta-Watchdog System Orchestrator

Central orchestration layer that coordinates all components:
- Model monitoring
- Reliability scoring
- Failure prediction
- Root cause analysis
- Action recommendation
- Explanation generation

This is the "brain" that runs the entire self-aware ML system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
import logging
from enum import Enum

from meta_watchdog.core.interfaces import PredictionModel
from meta_watchdog.core.data_structures import (
    Prediction,
    ReliabilityScore,
    FailurePrediction,
    SystemState,
)
from meta_watchdog.monitoring.performance_monitor import PerformanceMonitor
from meta_watchdog.monitoring.reliability_scorer import ReliabilityScoringEngine
from meta_watchdog.meta_prediction.failure_predictor import MetaFailurePredictor
from meta_watchdog.simulation.counterfactual import CounterfactualSimulator
from meta_watchdog.simulation.sensitivity import SensitivityMapper
from meta_watchdog.analysis.root_cause import RootCauseAnalyzer, RootCauseReport
from meta_watchdog.recommendations.action_engine import (
    ActionRecommendationEngine,
    ActionPlan,
)
from meta_watchdog.explainability.explanation_engine import (
    ExplainabilityEngine,
    ExplanationAudience,
)


class SystemMode(Enum):
    """Operating modes for the system."""
    MONITORING = "monitoring"          # Normal operation
    INVESTIGATION = "investigation"    # Detailed analysis
    EMERGENCY = "emergency"            # Critical issues
    LEARNING = "learning"              # Updating baselines


class AlertLevel(Enum):
    """Alert levels."""
    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """System alert."""
    level: AlertLevel
    message: str
    component: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = None


@dataclass
class HealthSnapshot:
    """Complete health snapshot at a point in time."""
    timestamp: datetime
    reliability_score: ReliabilityScore
    failure_prediction: FailurePrediction
    root_cause_report: Optional[RootCauseReport]
    action_plan: Optional[ActionPlan]
    alerts: List[Alert]
    mode: SystemMode
    explanation: str


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    # Thresholds
    reliability_warning_threshold: float = 60.0
    reliability_critical_threshold: float = 40.0
    failure_probability_alert: float = 0.5
    
    # Timing
    health_check_interval: int = 60  # seconds
    deep_analysis_interval: int = 300  # seconds
    
    # Features
    enable_auto_recommendations: bool = True
    enable_auto_explanations: bool = True
    enable_proactive_simulation: bool = False
    
    # Limits
    max_alerts_per_hour: int = 100
    max_recommendations: int = 10


class MetaWatchdogOrchestrator:
    """
    Central orchestrator for the Meta-Watchdog system.
    
    Coordinates all components to provide:
    1. Continuous model health monitoring
    2. Proactive failure prediction
    3. Root cause analysis
    4. Actionable recommendations
    5. Human-readable explanations
    
    Usage:
        orchestrator = MetaWatchdogOrchestrator(model)
        
        # For each prediction batch
        result = orchestrator.observe(X, y_true, predictions, confidences)
        
        # Get current health status
        health = orchestrator.get_health_snapshot()
    """
    
    def __init__(
        self,
        model: Optional[PredictionModel] = None,
        config: Optional[OrchestratorConfig] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            model: Model to monitor (optional)
            config: Orchestrator configuration
            feature_names: Names of input features
        """
        self.model = model
        self.config = config or OrchestratorConfig()
        self.feature_names = feature_names
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor()
        self.reliability_scorer = ReliabilityScoringEngine()
        self.failure_predictor = MetaFailurePredictor()
        self.simulator = CounterfactualSimulator()
        self.sensitivity_mapper = SensitivityMapper()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.action_engine = ActionRecommendationEngine()
        self.explainability_engine = ExplainabilityEngine()
        
        # State
        self._mode = SystemMode.MONITORING
        self._alerts: List[Alert] = []
        self._health_history: List[HealthSnapshot] = []
        self._observation_count = 0
        self._last_deep_analysis: Optional[datetime] = None
        
        # Callbacks
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Logging
        self._logger = logging.getLogger("meta_watchdog.orchestrator")
    
    # ========== Core Operations ==========
    
    def observe(
        self,
        X: NDArray[np.floating],
        y_true: Optional[NDArray] = None,
        predictions: Optional[NDArray] = None,
        confidences: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, Any]:
        """
        Observe a batch of predictions and update system state.
        
        Args:
            X: Input features
            y_true: Ground truth (if available)
            predictions: Model predictions
            confidences: Prediction confidences
            
        Returns:
            Dictionary with observation results
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self._observation_count += 1
        
        # Make predictions if model is available
        if predictions is None and self.model is not None:
            predictions, confidences = self.model.predict_with_confidence(X)
        
        # Update performance monitor
        if y_true is not None and predictions is not None:
            self.performance_monitor.observe(
                y_true=y_true,
                y_pred=predictions,
                confidence=confidences,
            )
        elif confidences is not None:
            # Observe confidence only
            for conf in np.atleast_1d(confidences):
                self.performance_monitor._confidence_history.append(float(conf))
        
        # Compute reliability
        reliability = self._compute_reliability()
        
        # Predict failures
        failure_prediction = self._predict_failures()
        
        # Check for alerts
        alerts = self._check_for_alerts(reliability, failure_prediction)
        self._alerts.extend(alerts)
        
        # Trigger callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self._logger.error(f"Alert callback failed: {e}")
        
        # Build result
        result = {
            "observation_count": self._observation_count,
            "reliability": reliability,
            "failure_prediction": failure_prediction,
            "alerts": alerts,
            "mode": self._mode.value,
        }
        
        # Auto-generate explanation
        if self.config.enable_auto_explanations and (alerts or reliability.score < 70):
            result["explanation"] = self.explainability_engine.explain_reliability(
                reliability, ExplanationAudience.OPERATIONS
            ).summary
        
        return result
    
    def _compute_reliability(self) -> ReliabilityScore:
        """Compute current reliability score."""
        metrics = self.performance_monitor.get_metrics()
        return self.reliability_scorer.compute_score(metrics)
    
    def _predict_failures(self) -> FailurePrediction:
        """Predict potential failures."""
        metrics = self.performance_monitor.get_metrics()
        return self.failure_predictor.predict(
            metrics,
            self._health_history[-10:] if self._health_history else []
        )
    
    def _check_for_alerts(
        self,
        reliability: ReliabilityScore,
        failure_prediction: FailurePrediction,
    ) -> List[Alert]:
        """Check for conditions requiring alerts."""
        alerts = []
        
        # Reliability alerts
        if reliability.score < self.config.reliability_critical_threshold:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Critical reliability drop: {reliability.score:.1f}/100",
                component="reliability",
                data={"score": reliability.score},
            ))
            self._mode = SystemMode.EMERGENCY
        elif reliability.score < self.config.reliability_warning_threshold:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Reliability warning: {reliability.score:.1f}/100",
                component="reliability",
                data={"score": reliability.score},
            ))
        
        # Failure prediction alerts
        if failure_prediction.probability >= self.config.failure_probability_alert:
            level = (AlertLevel.CRITICAL 
                     if failure_prediction.probability >= 0.8 
                     else AlertLevel.WARNING)
            alerts.append(Alert(
                level=level,
                message=f"Failure predicted: {failure_prediction.probability:.0%} probability",
                component="failure_predictor",
                data={
                    "probability": failure_prediction.probability,
                    "type": failure_prediction.failure_type,
                },
            ))
        
        return alerts
    
    # ========== Analysis Operations ==========
    
    def run_deep_analysis(
        self,
        X: Optional[NDArray[np.floating]] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive deep analysis.
        
        Args:
            X: Optional data sample for simulation
            
        Returns:
            Comprehensive analysis results
        """
        self._mode = SystemMode.INVESTIGATION
        self._last_deep_analysis = datetime.now()
        
        # Get current state
        reliability = self._compute_reliability()
        failure_prediction = self._predict_failures()
        
        # Root cause analysis
        performance_history = {
            "accuracy": list(self.performance_monitor._rolling_accuracy.values),
            "confidence": list(self.performance_monitor._confidence_history[-100:]),
        }
        
        # Build feature stats from monitor
        feature_stats = self._build_feature_stats()
        
        confidence_data = {
            "avg_confidence": np.mean(self.performance_monitor._confidence_history[-100:]) 
                            if self.performance_monitor._confidence_history else 0.5,
            "accuracy": reliability.performance_score,
        }
        
        root_cause_report = self.root_cause_analyzer.analyze(
            performance_history=performance_history,
            feature_stats=feature_stats,
            confidence_data=confidence_data,
            feature_names=self.feature_names,
        )
        
        # Generate recommendations
        action_plan = None
        if self.config.enable_auto_recommendations:
            action_plan = self.action_engine.generate_recommendations(
                root_cause_report=root_cause_report,
                reliability_score=reliability,
                failure_prediction=failure_prediction,
            )
        
        # Run simulation if data provided
        simulation_results = None
        sensitivity_map = None
        if X is not None and self.model is not None:
            if self.config.enable_proactive_simulation:
                simulation_results = self.simulator.run_stress_test_suite(
                    self.model, X
                )
            
            sensitivity_map = self.sensitivity_mapper.generate_sensitivity_map(
                self.model, X, feature_names=self.feature_names
            )
        
        # Generate comprehensive explanation
        explanation = self.explainability_engine.generate_full_report(
            reliability=reliability,
            failure_prediction=failure_prediction,
            root_cause_report=root_cause_report,
            action_plan=action_plan,
        )
        
        # Create health snapshot
        snapshot = HealthSnapshot(
            timestamp=datetime.now(),
            reliability_score=reliability,
            failure_prediction=failure_prediction,
            root_cause_report=root_cause_report,
            action_plan=action_plan,
            alerts=self._alerts[-10:],
            mode=self._mode,
            explanation=explanation,
        )
        
        self._health_history.append(snapshot)
        self._mode = SystemMode.MONITORING
        
        return {
            "reliability": reliability,
            "failure_prediction": failure_prediction,
            "root_cause_report": root_cause_report,
            "action_plan": action_plan,
            "simulation_results": simulation_results,
            "sensitivity_map": sensitivity_map,
            "explanation": explanation,
            "snapshot": snapshot,
        }
    
    def _build_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Build feature statistics for analysis."""
        # Placeholder - would be populated with real feature monitoring
        return {}
    
    # ========== Health Status ==========
    
    def get_health_snapshot(self) -> HealthSnapshot:
        """Get current health snapshot."""
        reliability = self._compute_reliability()
        failure_prediction = self._predict_failures()
        
        explanation = self.explainability_engine.explain_reliability(
            reliability
        ).summary
        
        return HealthSnapshot(
            timestamp=datetime.now(),
            reliability_score=reliability,
            failure_prediction=failure_prediction,
            root_cause_report=None,  # Only from deep analysis
            action_plan=None,
            alerts=self._alerts[-5:],
            mode=self._mode,
            explanation=explanation,
        )
    
    def get_system_state(self) -> SystemState:
        """Get complete system state."""
        reliability = self._compute_reliability()
        
        # Determine health status
        if reliability.score >= 80:
            health_status = "healthy"
        elif reliability.score >= 50:
            health_status = "degraded"
        else:
            health_status = "critical"
        
        return SystemState(
            timestamp=datetime.now(),
            health_status=health_status,
            reliability_score=reliability.score,
            active_alerts=[a.message for a in self._alerts[-10:]],
            pending_actions=[],
            mode=self._mode.value,
        )
    
    def get_quick_status(self) -> Dict[str, Any]:
        """Get quick status summary."""
        reliability = self._compute_reliability()
        failure_prediction = self._predict_failures()
        
        return {
            "status": "healthy" if reliability.score >= 70 else "needs_attention",
            "reliability_score": reliability.score,
            "failure_probability": failure_prediction.probability,
            "active_alerts": len([a for a in self._alerts if a.level != AlertLevel.NONE]),
            "mode": self._mode.value,
            "observations": self._observation_count,
        }
    
    # ========== Configuration ==========
    
    def register_alert_callback(
        self, callback: Callable[[Alert], None]
    ) -> None:
        """Register callback for alerts."""
        self._alert_callbacks.append(callback)
    
    def set_model(self, model: PredictionModel) -> None:
        """Set or update the monitored model."""
        self.model = model
    
    def set_feature_names(self, names: List[str]) -> None:
        """Set feature names."""
        self.feature_names = names
    
    def reset(self) -> None:
        """Reset the orchestrator state."""
        self.performance_monitor = PerformanceMonitor()
        self.reliability_scorer = ReliabilityScoringEngine()
        self._alerts.clear()
        self._health_history.clear()
        self._observation_count = 0
        self._mode = SystemMode.MONITORING
    
    # ========== History and Reporting ==========
    
    def get_health_history(
        self, n_records: int = 10
    ) -> List[HealthSnapshot]:
        """Get recent health history."""
        return self._health_history[-n_records:]
    
    def get_alert_history(
        self, n_records: int = 50
    ) -> List[Alert]:
        """Get recent alerts."""
        return self._alerts[-n_records:]
    
    def export_state(self) -> Dict[str, Any]:
        """Export current state for persistence."""
        return {
            "observation_count": self._observation_count,
            "mode": self._mode.value,
            "alerts_count": len(self._alerts),
            "health_history_count": len(self._health_history),
            "config": {
                "reliability_warning": self.config.reliability_warning_threshold,
                "reliability_critical": self.config.reliability_critical_threshold,
                "failure_probability_alert": self.config.failure_probability_alert,
            },
        }
