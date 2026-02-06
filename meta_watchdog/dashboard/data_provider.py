"""
Dashboard Data Provider

Provides structured data for dashboard visualization:
- Health metrics and trends
- Alert summaries
- Recommendation priorities
- Component status
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from meta_watchdog.core.data_structures import ReliabilityScore, FailurePrediction
from meta_watchdog.orchestrator.system import (
    MetaWatchdogOrchestrator,
    HealthSnapshot,
    Alert,
    AlertLevel,
)


@dataclass
class MetricTrend:
    """Trend data for a metric."""
    name: str
    current_value: float
    previous_value: float
    change_percent: float
    trend_direction: str  # "up", "down", "stable"
    history: List[Tuple[datetime, float]]


@dataclass
class ComponentStatus:
    """Status of a system component."""
    name: str
    status: str  # "healthy", "warning", "error", "unknown"
    last_active: Optional[datetime]
    metrics: Dict[str, float]


@dataclass
class DashboardData:
    """Complete data package for dashboard rendering."""
    timestamp: datetime
    
    # Overall status
    system_status: str
    reliability_score: float
    failure_risk: float
    
    # Metrics
    metric_trends: List[MetricTrend]
    
    # Alerts
    active_alerts: List[Dict[str, Any]]
    alert_summary: Dict[str, int]
    
    # Components
    component_statuses: List[ComponentStatus]
    
    # Recommendations
    top_recommendations: List[Dict[str, Any]]
    
    # Charts data
    reliability_history: List[Tuple[datetime, float]]
    failure_risk_history: List[Tuple[datetime, float]]


class DashboardDataProvider:
    """
    Provides formatted data for dashboard visualization.
    
    Extracts and formats data from the orchestrator for display
    in various dashboard interfaces (web, terminal, etc.).
    """
    
    def __init__(
        self,
        orchestrator: MetaWatchdogOrchestrator,
    ):
        """
        Initialize the dashboard data provider.
        
        Args:
            orchestrator: The Meta-Watchdog orchestrator
        """
        self.orchestrator = orchestrator
    
    def get_dashboard_data(self) -> DashboardData:
        """Get complete dashboard data package."""
        # Get current state
        status = self.orchestrator.get_quick_status()
        health_history = self.orchestrator.get_health_history(50)
        alerts = self.orchestrator.get_alert_history(20)
        
        # Build metric trends
        metric_trends = self._build_metric_trends(health_history)
        
        # Build alert summary
        active_alerts, alert_summary = self._build_alert_data(alerts)
        
        # Build component statuses
        component_statuses = self._build_component_statuses()
        
        # Build recommendations
        top_recommendations = self._build_recommendations()
        
        # Build history charts
        reliability_history = self._extract_reliability_history(health_history)
        failure_risk_history = self._extract_failure_history(health_history)
        
        return DashboardData(
            timestamp=datetime.now(),
            system_status=status["status"],
            reliability_score=status["reliability_score"],
            failure_risk=status["failure_probability"],
            metric_trends=metric_trends,
            active_alerts=active_alerts,
            alert_summary=alert_summary,
            component_statuses=component_statuses,
            top_recommendations=top_recommendations,
            reliability_history=reliability_history,
            failure_risk_history=failure_risk_history,
        )
    
    def _build_metric_trends(
        self, history: List[HealthSnapshot]
    ) -> List[MetricTrend]:
        """Build metric trend data."""
        trends = []
        
        if not history:
            return trends
        
        current = history[-1] if history else None
        previous = history[-2] if len(history) > 1 else None
        
        # Reliability trend
        if current:
            curr_rel = current.reliability_score.score
            prev_rel = previous.reliability_score.score if previous else curr_rel
            change = ((curr_rel - prev_rel) / (prev_rel + 1e-8)) * 100
            
            trends.append(MetricTrend(
                name="Reliability",
                current_value=curr_rel,
                previous_value=prev_rel,
                change_percent=change,
                trend_direction=self._determine_trend(change),
                history=[
                    (h.timestamp, h.reliability_score.score)
                    for h in history[-20:]
                ],
            ))
        
        # Performance trend
        if current:
            curr_perf = current.reliability_score.performance_score
            prev_perf = previous.reliability_score.performance_score if previous else curr_perf
            change = ((curr_perf - prev_perf) / (prev_perf + 1e-8)) * 100
            
            trends.append(MetricTrend(
                name="Performance",
                current_value=curr_perf,
                previous_value=prev_perf,
                change_percent=change,
                trend_direction=self._determine_trend(change),
                history=[
                    (h.timestamp, h.reliability_score.performance_score)
                    for h in history[-20:]
                ],
            ))
        
        # Calibration trend
        if current:
            curr_cal = current.reliability_score.calibration_score
            prev_cal = previous.reliability_score.calibration_score if previous else curr_cal
            change = ((curr_cal - prev_cal) / (prev_cal + 1e-8)) * 100
            
            trends.append(MetricTrend(
                name="Calibration",
                current_value=curr_cal,
                previous_value=prev_cal,
                change_percent=change,
                trend_direction=self._determine_trend(change),
                history=[
                    (h.timestamp, h.reliability_score.calibration_score)
                    for h in history[-20:]
                ],
            ))
        
        return trends
    
    def _determine_trend(self, change: float) -> str:
        """Determine trend direction."""
        if change > 5:
            return "up"
        elif change < -5:
            return "down"
        else:
            return "stable"
    
    def _build_alert_data(
        self, alerts: List[Alert]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Build alert data for dashboard."""
        # Active alerts (recent)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        active = [
            a for a in alerts
            if a.timestamp > recent_cutoff and a.level != AlertLevel.NONE
        ]
        
        active_alerts = [
            {
                "level": a.level.value,
                "message": a.message,
                "component": a.component,
                "time": a.timestamp.isoformat(),
            }
            for a in active[-10:]
        ]
        
        # Summary
        summary = {
            "critical": sum(1 for a in active if a.level == AlertLevel.CRITICAL),
            "warning": sum(1 for a in active if a.level == AlertLevel.WARNING),
            "info": sum(1 for a in active if a.level == AlertLevel.INFO),
        }
        
        return active_alerts, summary
    
    def _build_component_statuses(self) -> List[ComponentStatus]:
        """Build component status data."""
        # This would be enhanced with actual component health checks
        return [
            ComponentStatus(
                name="Performance Monitor",
                status="healthy",
                last_active=datetime.now(),
                metrics={"observations": self.orchestrator._observation_count},
            ),
            ComponentStatus(
                name="Reliability Scorer",
                status="healthy",
                last_active=datetime.now(),
                metrics={},
            ),
            ComponentStatus(
                name="Failure Predictor",
                status="healthy",
                last_active=datetime.now(),
                metrics={},
            ),
            ComponentStatus(
                name="Root Cause Analyzer",
                status="healthy",
                last_active=self.orchestrator._last_deep_analysis,
                metrics={},
            ),
        ]
    
    def _build_recommendations(self) -> List[Dict[str, Any]]:
        """Build top recommendations."""
        # Get latest snapshot with action plan
        for snapshot in reversed(self.orchestrator._health_history):
            if snapshot.action_plan:
                return [
                    {
                        "title": r.title,
                        "priority": r.priority.value,
                        "category": r.category.value,
                        "impact": r.estimated_impact,
                    }
                    for r in snapshot.action_plan.recommendations[:5]
                ]
        return []
    
    def _extract_reliability_history(
        self, history: List[HealthSnapshot]
    ) -> List[Tuple[datetime, float]]:
        """Extract reliability score history."""
        return [
            (h.timestamp, h.reliability_score.score)
            for h in history
        ]
    
    def _extract_failure_history(
        self, history: List[HealthSnapshot]
    ) -> List[Tuple[datetime, float]]:
        """Extract failure risk history."""
        return [
            (h.timestamp, h.failure_prediction.probability)
            for h in history
        ]
    
    # ========== Specific Views ==========
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for quick display."""
        status = self.orchestrator.get_quick_status()
        
        return {
            "status": status["status"],
            "score": status["reliability_score"],
            "score_label": self._score_to_label(status["reliability_score"]),
            "risk": status["failure_probability"],
            "risk_label": self._risk_to_label(status["failure_probability"]),
            "alerts": status["active_alerts"],
        }
    
    def _score_to_label(self, score: float) -> str:
        """Convert score to label."""
        if score >= 90:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Fair"
        elif score >= 30:
            return "Poor"
        else:
            return "Critical"
    
    def _risk_to_label(self, risk: float) -> str:
        """Convert risk to label."""
        if risk < 0.2:
            return "Low"
        elif risk < 0.5:
            return "Moderate"
        elif risk < 0.8:
            return "High"
        else:
            return "Critical"
    
    def get_alerts_panel(self) -> Dict[str, Any]:
        """Get alerts panel data."""
        alerts = self.orchestrator.get_alert_history(50)
        active_alerts, summary = self._build_alert_data(alerts)
        
        return {
            "total_active": sum(summary.values()),
            "summary": summary,
            "alerts": active_alerts,
        }
    
    def get_metrics_panel(self) -> List[Dict[str, Any]]:
        """Get metrics panel data."""
        history = self.orchestrator.get_health_history(20)
        trends = self._build_metric_trends(history)
        
        return [
            {
                "name": t.name,
                "value": t.current_value,
                "change": t.change_percent,
                "trend": t.trend_direction,
            }
            for t in trends
        ]
    
    def export_to_json(self) -> Dict[str, Any]:
        """Export dashboard data to JSON-serializable format."""
        data = self.get_dashboard_data()
        
        return {
            "timestamp": data.timestamp.isoformat(),
            "system_status": data.system_status,
            "reliability_score": data.reliability_score,
            "failure_risk": data.failure_risk,
            "metrics": [
                {
                    "name": m.name,
                    "current": m.current_value,
                    "previous": m.previous_value,
                    "change": m.change_percent,
                    "trend": m.trend_direction,
                }
                for m in data.metric_trends
            ],
            "alerts": data.active_alerts,
            "alert_summary": data.alert_summary,
            "recommendations": data.top_recommendations,
        }
