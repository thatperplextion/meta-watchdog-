"""
Action Recommendation Engine

Generates actionable recommendations based on:
- Detected failures and root causes
- Model reliability state
- Historical effectiveness of actions
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from datetime import datetime

from meta_watchdog.analysis.root_cause import (
    CauseCategory,
    IdentifiedCause,
    RootCauseReport,
)
from meta_watchdog.core.data_structures import (
    FailurePrediction,
    ReliabilityScore,
    Recommendation,
    ActionType,
)


class ActionPriority(Enum):
    """Priority levels for actions."""
    CRITICAL = "critical"      # Must do immediately
    HIGH = "high"              # Do soon
    MEDIUM = "medium"          # Should do when possible
    LOW = "low"                # Nice to have
    INFORMATIONAL = "info"     # Just for awareness


class ActionCategory(Enum):
    """Categories of recommended actions."""
    RETRAINING = "retraining"
    MONITORING = "monitoring"
    DATA_PIPELINE = "data_pipeline"
    FEATURE_ENGINEERING = "feature_engineering"
    CALIBRATION = "calibration"
    ALERTING = "alerting"
    INVESTIGATION = "investigation"
    ROLLBACK = "rollback"
    SCALING = "scaling"


@dataclass
class ActionRecommendation:
    """A single action recommendation."""
    action_id: str
    title: str
    description: str
    category: ActionCategory
    priority: ActionPriority
    estimated_effort: str  # "low", "medium", "high"
    estimated_impact: float  # 0-1
    root_cause: Optional[CauseCategory] = None
    affected_features: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    confidence: float = 0.5  # How confident we are this will help


@dataclass
class ActionPlan:
    """Complete action plan with prioritized recommendations."""
    recommendations: List[ActionRecommendation]
    total_estimated_effort: str
    expected_improvement: float
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)


class ActionRecommendationEngine:
    """
    Generates actionable recommendations for model health issues.
    
    Takes analysis results (root causes, reliability scores, failure predictions)
    and generates a prioritized action plan with specific steps.
    
    This is the "advisor" that tells humans what to do.
    """
    
    def __init__(
        self,
        impact_threshold: float = 0.3,
    ):
        """
        Initialize the recommendation engine.
        
        Args:
            impact_threshold: Minimum impact to include a recommendation
        """
        self.impact_threshold = impact_threshold
        self._action_counter = 0
        
        # Map root causes to action templates
        self._cause_actions = self._build_cause_action_map()
        
        # Track recommendation effectiveness
        self._effectiveness_history: Dict[str, List[bool]] = {}
    
    def _build_cause_action_map(self) -> Dict[CauseCategory, List[Dict[str, Any]]]:
        """Build mapping from root causes to action templates."""
        return {
            CauseCategory.DATA_DRIFT: [
                {
                    "title": "Implement Drift Detection Pipeline",
                    "category": ActionCategory.MONITORING,
                    "effort": "medium",
                    "impact": 0.8,
                    "steps": [
                        "Set up drift detection on input features",
                        "Configure alerts for significant drift",
                        "Create dashboard for drift visualization",
                    ],
                    "outcome": "Early warning system for data drift",
                },
                {
                    "title": "Retrain Model with Recent Data",
                    "category": ActionCategory.RETRAINING,
                    "effort": "high",
                    "impact": 0.9,
                    "steps": [
                        "Collect recent data sample",
                        "Validate data quality",
                        "Retrain model with updated data",
                        "Validate performance on holdout set",
                    ],
                    "outcome": "Model aligned with current data distribution",
                },
            ],
            CauseCategory.FEATURE_INSTABILITY: [
                {
                    "title": "Stabilize Volatile Features",
                    "category": ActionCategory.FEATURE_ENGINEERING,
                    "effort": "medium",
                    "impact": 0.7,
                    "steps": [
                        "Identify most unstable features",
                        "Apply smoothing or aggregation",
                        "Test impact on model performance",
                    ],
                    "outcome": "Reduced feature volatility",
                },
            ],
            CauseCategory.OVERCONFIDENCE: [
                {
                    "title": "Apply Confidence Calibration",
                    "category": ActionCategory.CALIBRATION,
                    "effort": "low",
                    "impact": 0.8,
                    "steps": [
                        "Collect calibration dataset",
                        "Apply Platt scaling or isotonic regression",
                        "Validate calibration on test set",
                    ],
                    "outcome": "Well-calibrated prediction confidence",
                },
                {
                    "title": "Add Uncertainty Quantification",
                    "category": ActionCategory.FEATURE_ENGINEERING,
                    "effort": "high",
                    "impact": 0.7,
                    "steps": [
                        "Implement ensemble or Monte Carlo dropout",
                        "Compute prediction intervals",
                        "Integrate with prediction pipeline",
                    ],
                    "outcome": "Explicit uncertainty estimates",
                },
            ],
            CauseCategory.UNDERCONFIDENCE: [
                {
                    "title": "Adjust Confidence Scoring",
                    "category": ActionCategory.CALIBRATION,
                    "effort": "low",
                    "impact": 0.6,
                    "steps": [
                        "Apply temperature scaling",
                        "Validate on calibration set",
                    ],
                    "outcome": "Better calibrated confidence scores",
                },
            ],
            CauseCategory.DISTRIBUTION_SHIFT: [
                {
                    "title": "Investigate Distribution Change Source",
                    "category": ActionCategory.INVESTIGATION,
                    "effort": "medium",
                    "impact": 0.5,
                    "steps": [
                        "Compare recent vs historical data statistics",
                        "Check data pipeline for changes",
                        "Review external factors",
                    ],
                    "outcome": "Understanding of distribution shift cause",
                },
            ],
            CauseCategory.CORRELATION_BREAKDOWN: [
                {
                    "title": "Review Feature Dependencies",
                    "category": ActionCategory.INVESTIGATION,
                    "effort": "medium",
                    "impact": 0.6,
                    "steps": [
                        "Analyze feature correlation matrix",
                        "Check upstream data sources",
                        "Consider feature reconstruction",
                    ],
                    "outcome": "Restored feature relationships",
                },
            ],
            CauseCategory.OUTLIER_INFLUENCE: [
                {
                    "title": "Implement Outlier Detection",
                    "category": ActionCategory.DATA_PIPELINE,
                    "effort": "medium",
                    "impact": 0.7,
                    "steps": [
                        "Add outlier detection in preprocessing",
                        "Configure outlier handling strategy",
                        "Monitor outlier rates",
                    ],
                    "outcome": "Robust handling of outliers",
                },
            ],
            CauseCategory.MISSING_DATA_PATTERN: [
                {
                    "title": "Improve Imputation Strategy",
                    "category": ActionCategory.DATA_PIPELINE,
                    "effort": "medium",
                    "impact": 0.6,
                    "steps": [
                        "Analyze missing patterns",
                        "Implement appropriate imputation",
                        "Validate impact on predictions",
                    ],
                    "outcome": "Better handling of missing data",
                },
            ],
            CauseCategory.DATA_QUALITY: [
                {
                    "title": "Add Data Validation",
                    "category": ActionCategory.DATA_PIPELINE,
                    "effort": "medium",
                    "impact": 0.7,
                    "steps": [
                        "Define data quality rules",
                        "Implement validation checks",
                        "Set up quality monitoring",
                    ],
                    "outcome": "Consistent data quality",
                },
            ],
        }
    
    def generate_recommendations(
        self,
        root_cause_report: Optional[RootCauseReport] = None,
        reliability_score: Optional[ReliabilityScore] = None,
        failure_prediction: Optional[FailurePrediction] = None,
    ) -> ActionPlan:
        """
        Generate action recommendations based on analysis results.
        
        Args:
            root_cause_report: Results from root cause analysis
            reliability_score: Current reliability assessment
            failure_prediction: Predicted failure information
            
        Returns:
            ActionPlan with prioritized recommendations
        """
        recommendations: List[ActionRecommendation] = []
        
        # Process root causes
        if root_cause_report:
            recommendations.extend(
                self._recommendations_from_root_causes(root_cause_report)
            )
        
        # Process reliability issues
        if reliability_score:
            recommendations.extend(
                self._recommendations_from_reliability(reliability_score)
            )
        
        # Process failure predictions
        if failure_prediction:
            recommendations.extend(
                self._recommendations_from_failure_prediction(failure_prediction)
            )
        
        # Deduplicate and prioritize
        recommendations = self._deduplicate_recommendations(recommendations)
        recommendations = self._prioritize_recommendations(recommendations)
        
        # Filter by impact threshold
        recommendations = [
            r for r in recommendations 
            if r.estimated_impact >= self.impact_threshold
        ]
        
        # Create action plan
        total_effort = self._calculate_total_effort(recommendations)
        expected_improvement = self._calculate_expected_improvement(recommendations)
        summary = self._generate_plan_summary(recommendations)
        
        return ActionPlan(
            recommendations=recommendations,
            total_estimated_effort=total_effort,
            expected_improvement=expected_improvement,
            summary=summary,
        )
    
    def _recommendations_from_root_causes(
        self, report: RootCauseReport
    ) -> List[ActionRecommendation]:
        """Generate recommendations from root cause analysis."""
        recommendations = []
        
        # Process primary cause
        if report.primary_cause:
            recs = self._create_recommendations_for_cause(
                report.primary_cause, 
                is_primary=True
            )
            recommendations.extend(recs)
        
        # Process contributing causes
        for cause in report.contributing_causes:
            recs = self._create_recommendations_for_cause(
                cause,
                is_primary=False
            )
            recommendations.extend(recs)
        
        return recommendations
    
    def _create_recommendations_for_cause(
        self,
        cause: IdentifiedCause,
        is_primary: bool,
    ) -> List[ActionRecommendation]:
        """Create recommendations for a specific cause."""
        recommendations = []
        
        action_templates = self._cause_actions.get(cause.category, [])
        
        for template in action_templates:
            self._action_counter += 1
            action_id = f"ACT-{self._action_counter:04d}"
            
            # Adjust priority based on cause severity
            priority = self._determine_priority(
                cause.severity, 
                cause.confidence,
                is_primary
            )
            
            rec = ActionRecommendation(
                action_id=action_id,
                title=template["title"],
                description=f"Address {cause.category.value}: {cause.description}",
                category=template["category"],
                priority=priority,
                estimated_effort=template["effort"],
                estimated_impact=template["impact"] * cause.severity,
                root_cause=cause.category,
                affected_features=cause.affected_features.copy(),
                steps=template["steps"].copy(),
                expected_outcome=template["outcome"],
                confidence=cause.confidence,
            )
            
            recommendations.append(rec)
        
        return recommendations
    
    def _recommendations_from_reliability(
        self, score: ReliabilityScore
    ) -> List[ActionRecommendation]:
        """Generate recommendations from reliability assessment."""
        recommendations = []
        
        # Check each component
        if score.calibration_score < 0.5:
            self._action_counter += 1
            recommendations.append(ActionRecommendation(
                action_id=f"ACT-{self._action_counter:04d}",
                title="Address Calibration Issues",
                description="Model confidence not well calibrated with actual performance",
                category=ActionCategory.CALIBRATION,
                priority=ActionPriority.HIGH,
                estimated_effort="medium",
                estimated_impact=0.7,
                steps=[
                    "Collect calibration validation set",
                    "Apply calibration method",
                    "Validate and deploy",
                ],
                expected_outcome="Improved confidence calibration",
                confidence=0.8,
            ))
        
        if score.stability_score < 0.5:
            self._action_counter += 1
            recommendations.append(ActionRecommendation(
                action_id=f"ACT-{self._action_counter:04d}",
                title="Improve Model Stability",
                description="Model showing unstable performance",
                category=ActionCategory.INVESTIGATION,
                priority=ActionPriority.MEDIUM,
                estimated_effort="medium",
                estimated_impact=0.6,
                steps=[
                    "Analyze stability patterns",
                    "Identify stability factors",
                    "Implement stabilization measures",
                ],
                expected_outcome="More stable model predictions",
                confidence=0.7,
            ))
        
        if score.freshness_score < 0.5:
            self._action_counter += 1
            recommendations.append(ActionRecommendation(
                action_id=f"ACT-{self._action_counter:04d}",
                title="Update Model Training",
                description="Model trained on stale data",
                category=ActionCategory.RETRAINING,
                priority=ActionPriority.HIGH,
                estimated_effort="high",
                estimated_impact=0.8,
                steps=[
                    "Collect recent training data",
                    "Validate data quality",
                    "Retrain and evaluate",
                    "Deploy updated model",
                ],
                expected_outcome="Model trained on current data",
                confidence=0.85,
            ))
        
        return recommendations
    
    def _recommendations_from_failure_prediction(
        self, prediction: FailurePrediction
    ) -> List[ActionRecommendation]:
        """Generate recommendations from failure predictions."""
        recommendations = []
        
        if prediction.probability < 0.3:
            return recommendations  # Low failure probability
        
        # Urgency based on time to failure
        if prediction.estimated_time_to_failure:
            ttf = prediction.estimated_time_to_failure
            if ttf < 3600:  # Less than 1 hour
                priority = ActionPriority.CRITICAL
            elif ttf < 86400:  # Less than 1 day
                priority = ActionPriority.HIGH
            else:
                priority = ActionPriority.MEDIUM
        else:
            priority = ActionPriority.HIGH if prediction.probability > 0.7 else ActionPriority.MEDIUM
        
        self._action_counter += 1
        recommendations.append(ActionRecommendation(
            action_id=f"ACT-{self._action_counter:04d}",
            title="Prepare for Predicted Failure",
            description=f"Failure predicted with {prediction.probability:.0%} probability",
            category=ActionCategory.ALERTING,
            priority=priority,
            estimated_effort="low",
            estimated_impact=0.9,
            steps=[
                "Alert relevant stakeholders",
                "Prepare rollback plan",
                "Monitor closely",
            ],
            expected_outcome="Proactive failure management",
            confidence=prediction.confidence,
        ))
        
        # Add specific recommendations based on failure type
        if prediction.failure_type:
            if "drift" in prediction.failure_type.lower():
                self._action_counter += 1
                recommendations.append(ActionRecommendation(
                    action_id=f"ACT-{self._action_counter:04d}",
                    title="Address Impending Drift",
                    description=f"Drift-related failure expected",
                    category=ActionCategory.MONITORING,
                    priority=priority,
                    estimated_effort="medium",
                    estimated_impact=0.7,
                    steps=[
                        "Intensify drift monitoring",
                        "Prepare retraining pipeline",
                        "Consider feature updates",
                    ],
                    expected_outcome="Drift mitigation ready",
                    confidence=prediction.confidence,
                ))
        
        return recommendations
    
    def _determine_priority(
        self,
        severity: float,
        confidence: float,
        is_primary: bool,
    ) -> ActionPriority:
        """Determine action priority."""
        combined = severity * confidence
        
        if combined > 0.7 and is_primary:
            return ActionPriority.CRITICAL
        elif combined > 0.5 or is_primary:
            return ActionPriority.HIGH
        elif combined > 0.3:
            return ActionPriority.MEDIUM
        else:
            return ActionPriority.LOW
    
    def _deduplicate_recommendations(
        self, recommendations: List[ActionRecommendation]
    ) -> List[ActionRecommendation]:
        """Remove duplicate recommendations."""
        seen_titles = set()
        unique = []
        
        for rec in recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique.append(rec)
            else:
                # Update priority if duplicate has higher priority
                for existing in unique:
                    if existing.title == rec.title:
                        if self._priority_value(rec.priority) > self._priority_value(existing.priority):
                            existing.priority = rec.priority
                        break
        
        return unique
    
    def _priority_value(self, priority: ActionPriority) -> int:
        """Convert priority to numeric value for comparison."""
        mapping = {
            ActionPriority.CRITICAL: 5,
            ActionPriority.HIGH: 4,
            ActionPriority.MEDIUM: 3,
            ActionPriority.LOW: 2,
            ActionPriority.INFORMATIONAL: 1,
        }
        return mapping.get(priority, 0)
    
    def _prioritize_recommendations(
        self, recommendations: List[ActionRecommendation]
    ) -> List[ActionRecommendation]:
        """Sort recommendations by priority and impact."""
        return sorted(
            recommendations,
            key=lambda r: (
                self._priority_value(r.priority),
                r.estimated_impact,
                r.confidence,
            ),
            reverse=True,
        )
    
    def _calculate_total_effort(
        self, recommendations: List[ActionRecommendation]
    ) -> str:
        """Calculate total estimated effort."""
        effort_scores = {"low": 1, "medium": 2, "high": 3}
        
        total = sum(
            effort_scores.get(r.estimated_effort, 2)
            for r in recommendations
        )
        
        if total <= 3:
            return "low"
        elif total <= 7:
            return "medium"
        else:
            return "high"
    
    def _calculate_expected_improvement(
        self, recommendations: List[ActionRecommendation]
    ) -> float:
        """Calculate expected improvement if all actions taken."""
        if not recommendations:
            return 0.0
        
        # Diminishing returns for multiple actions
        impacts = sorted(
            [r.estimated_impact * r.confidence for r in recommendations],
            reverse=True,
        )
        
        improvement = 0.0
        remaining = 1.0
        
        for impact in impacts:
            contribution = impact * remaining
            improvement += contribution
            remaining *= (1 - impact * 0.5)  # Diminishing returns
        
        return min(improvement, 1.0)
    
    def _generate_plan_summary(
        self, recommendations: List[ActionRecommendation]
    ) -> str:
        """Generate summary of the action plan."""
        if not recommendations:
            return "No actions required at this time."
        
        critical = sum(1 for r in recommendations if r.priority == ActionPriority.CRITICAL)
        high = sum(1 for r in recommendations if r.priority == ActionPriority.HIGH)
        
        summary_parts = [f"Generated {len(recommendations)} recommendations."]
        
        if critical:
            summary_parts.append(f"{critical} critical priority action(s) require immediate attention.")
        if high:
            summary_parts.append(f"{high} high priority action(s) should be addressed soon.")
        
        # Most impactful
        top_rec = recommendations[0]
        summary_parts.append(
            f"Top recommendation: {top_rec.title} (expected impact: {top_rec.estimated_impact:.0%})."
        )
        
        return " ".join(summary_parts)
    
    def record_action_outcome(
        self, action_id: str, was_effective: bool
    ) -> None:
        """Record whether an action was effective for learning."""
        if action_id not in self._effectiveness_history:
            self._effectiveness_history[action_id] = []
        
        self._effectiveness_history[action_id].append(was_effective)
    
    def get_action_effectiveness(self, action_title: str) -> Optional[float]:
        """Get historical effectiveness of an action type."""
        # Search by title pattern in history
        matching_outcomes = []
        
        for action_id, outcomes in self._effectiveness_history.items():
            matching_outcomes.extend(outcomes)
        
        if not matching_outcomes:
            return None
        
        return sum(matching_outcomes) / len(matching_outcomes)
