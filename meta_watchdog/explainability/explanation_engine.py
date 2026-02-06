"""
Explainability Layer

Generates human-readable explanations for:
- Model predictions
- Failure predictions
- Reliability assessments
- Root cause analysis
- Action recommendations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from meta_watchdog.core.data_structures import (
    Prediction,
    ReliabilityScore,
    FailurePrediction,
    Explanation,
)
from meta_watchdog.analysis.root_cause import (
    RootCauseReport,
    IdentifiedCause,
    CauseCategory,
)
from meta_watchdog.recommendations.action_engine import (
    ActionPlan,
    ActionRecommendation,
    ActionPriority,
)


class ExplanationAudience(Enum):
    """Target audience for explanations."""
    TECHNICAL = "technical"      # ML engineers, data scientists
    BUSINESS = "business"        # Business stakeholders
    EXECUTIVE = "executive"      # High-level summary
    OPERATIONS = "operations"    # Ops team


class ExplanationVerbosity(Enum):
    """Level of detail in explanations."""
    BRIEF = "brief"          # One sentence
    STANDARD = "standard"    # Paragraph
    DETAILED = "detailed"    # Full explanation
    COMPREHENSIVE = "comprehensive"  # Everything


@dataclass
class ExplanationSection:
    """A section of an explanation."""
    title: str
    content: str
    importance: float  # 0-1
    technical_level: int  # 1-5
    supporting_data: Optional[Dict[str, Any]] = None


@dataclass
class StructuredExplanation:
    """A structured multi-section explanation."""
    summary: str
    sections: List[ExplanationSection]
    audience: ExplanationAudience
    verbosity: ExplanationVerbosity
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_plain_text(self) -> str:
        """Convert to plain text."""
        parts = [self.summary, ""]
        
        for section in self.sections:
            parts.append(f"## {section.title}")
            parts.append(section.content)
            parts.append("")
        
        return "\n".join(parts)
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        parts = [f"# Explanation\n\n{self.summary}\n"]
        
        for section in self.sections:
            parts.append(f"## {section.title}\n\n{section.content}\n")
        
        return "\n".join(parts)


class ExplainabilityEngine:
    """
    Generates human-readable explanations for Meta-Watchdog outputs.
    
    Takes technical analysis results and converts them to clear,
    actionable explanations tailored to different audiences.
    
    This is the "translator" between the system and humans.
    """
    
    def __init__(
        self,
        default_audience: ExplanationAudience = ExplanationAudience.TECHNICAL,
        default_verbosity: ExplanationVerbosity = ExplanationVerbosity.STANDARD,
    ):
        """
        Initialize the explainability engine.
        
        Args:
            default_audience: Default target audience
            default_verbosity: Default verbosity level
        """
        self.default_audience = default_audience
        self.default_verbosity = default_verbosity
        
        # Explanation templates
        self._templates = self._build_templates()
    
    def _build_templates(self) -> Dict[str, Dict[str, str]]:
        """Build explanation templates for different scenarios."""
        return {
            "reliability_good": {
                ExplanationAudience.TECHNICAL: 
                    "Model reliability is strong with score {score:.1f}/100. "
                    "Calibration ({calibration:.0%}), stability ({stability:.0%}), "
                    "and performance ({performance:.0%}) are all within acceptable ranges.",
                ExplanationAudience.BUSINESS:
                    "The model is performing reliably. You can trust its predictions.",
                ExplanationAudience.EXECUTIVE:
                    "Model health: Good. No action required.",
                ExplanationAudience.OPERATIONS:
                    "All systems nominal. Reliability score: {score:.1f}/100.",
            },
            "reliability_warning": {
                ExplanationAudience.TECHNICAL:
                    "Model reliability warning. Score: {score:.1f}/100. "
                    "Issues detected: {issues}. "
                    "Consider monitoring closely and preparing interventions.",
                ExplanationAudience.BUSINESS:
                    "The model is showing some warning signs. "
                    "Predictions may be less reliable than usual.",
                ExplanationAudience.EXECUTIVE:
                    "Model health: Warning. Monitoring recommended.",
                ExplanationAudience.OPERATIONS:
                    "Warning state. Score: {score:.1f}/100. Issues: {issues}.",
            },
            "reliability_critical": {
                ExplanationAudience.TECHNICAL:
                    "CRITICAL: Model reliability severely degraded. Score: {score:.1f}/100. "
                    "Primary issues: {issues}. Immediate intervention recommended.",
                ExplanationAudience.BUSINESS:
                    "ALERT: The model is not performing reliably. "
                    "Consider using backup processes until resolved.",
                ExplanationAudience.EXECUTIVE:
                    "Model health: Critical. Action required immediately.",
                ExplanationAudience.OPERATIONS:
                    "CRITICAL ALERT. Score: {score:.1f}/100. Escalate immediately.",
            },
            "failure_predicted": {
                ExplanationAudience.TECHNICAL:
                    "Failure predicted with {probability:.0%} confidence. "
                    "Type: {failure_type}. Estimated time: {time_estimate}. "
                    "Contributing factors: {factors}.",
                ExplanationAudience.BUSINESS:
                    "The system predicts potential issues ahead. "
                    "Probability: {probability:.0%}. Action should be taken soon.",
                ExplanationAudience.EXECUTIVE:
                    "Risk alert: {probability:.0%} chance of model issues. "
                    "Time estimate: {time_estimate}.",
                ExplanationAudience.OPERATIONS:
                    "Predicted failure. Probability: {probability:.0%}. "
                    "ETA: {time_estimate}. Type: {failure_type}.",
            },
        }
    
    # ========== Main Explanation Methods ==========
    
    def explain_reliability(
        self,
        reliability: ReliabilityScore,
        audience: Optional[ExplanationAudience] = None,
        verbosity: Optional[ExplanationVerbosity] = None,
    ) -> StructuredExplanation:
        """
        Explain a reliability score.
        
        Args:
            reliability: Reliability score to explain
            audience: Target audience
            verbosity: Level of detail
            
        Returns:
            StructuredExplanation
        """
        audience = audience or self.default_audience
        verbosity = verbosity or self.default_verbosity
        
        # Determine status
        if reliability.score >= 80:
            template_key = "reliability_good"
            status = "good"
        elif reliability.score >= 50:
            template_key = "reliability_warning"
            status = "warning"
        else:
            template_key = "reliability_critical"
            status = "critical"
        
        # Build issues list
        issues = []
        if reliability.calibration_score < 0.6:
            issues.append("calibration")
        if reliability.stability_score < 0.6:
            issues.append("stability")
        if reliability.performance_score < 0.6:
            issues.append("performance")
        if reliability.freshness_score < 0.6:
            issues.append("data freshness")
        
        issues_str = ", ".join(issues) if issues else "none"
        
        # Generate summary
        template = self._templates[template_key].get(audience, "")
        summary = template.format(
            score=reliability.score,
            calibration=reliability.calibration_score,
            stability=reliability.stability_score,
            performance=reliability.performance_score,
            issues=issues_str,
        )
        
        # Build sections based on verbosity
        sections = self._build_reliability_sections(
            reliability, status, verbosity, audience
        )
        
        return StructuredExplanation(
            summary=summary,
            sections=sections,
            audience=audience,
            verbosity=verbosity,
            confidence=reliability.confidence if hasattr(reliability, 'confidence') else 0.8,
        )
    
    def _build_reliability_sections(
        self,
        reliability: ReliabilityScore,
        status: str,
        verbosity: ExplanationVerbosity,
        audience: ExplanationAudience,
    ) -> List[ExplanationSection]:
        """Build explanation sections for reliability."""
        sections = []
        
        if verbosity == ExplanationVerbosity.BRIEF:
            return sections  # Only summary
        
        # Performance section
        sections.append(ExplanationSection(
            title="Performance",
            content=self._explain_metric(
                "Performance", reliability.performance_score, audience
            ),
            importance=0.9,
            technical_level=2,
            supporting_data={"score": reliability.performance_score},
        ))
        
        # Calibration section
        sections.append(ExplanationSection(
            title="Calibration",
            content=self._explain_metric(
                "Calibration", reliability.calibration_score, audience
            ),
            importance=0.8,
            technical_level=3,
            supporting_data={"score": reliability.calibration_score},
        ))
        
        if verbosity in [ExplanationVerbosity.DETAILED, ExplanationVerbosity.COMPREHENSIVE]:
            # Stability section
            sections.append(ExplanationSection(
                title="Stability",
                content=self._explain_metric(
                    "Stability", reliability.stability_score, audience
                ),
                importance=0.7,
                technical_level=3,
                supporting_data={"score": reliability.stability_score},
            ))
            
            # Freshness section
            sections.append(ExplanationSection(
                title="Data Freshness",
                content=self._explain_metric(
                    "Data freshness", reliability.freshness_score, audience
                ),
                importance=0.6,
                technical_level=2,
                supporting_data={"score": reliability.freshness_score},
            ))
        
        return sections
    
    def explain_failure_prediction(
        self,
        prediction: FailurePrediction,
        audience: Optional[ExplanationAudience] = None,
        verbosity: Optional[ExplanationVerbosity] = None,
    ) -> StructuredExplanation:
        """
        Explain a failure prediction.
        
        Args:
            prediction: Failure prediction to explain
            audience: Target audience
            verbosity: Level of detail
            
        Returns:
            StructuredExplanation
        """
        audience = audience or self.default_audience
        verbosity = verbosity or self.default_verbosity
        
        # Format time estimate
        if prediction.estimated_time_to_failure:
            ttf = prediction.estimated_time_to_failure
            if ttf < 3600:
                time_estimate = f"{ttf/60:.0f} minutes"
            elif ttf < 86400:
                time_estimate = f"{ttf/3600:.1f} hours"
            else:
                time_estimate = f"{ttf/86400:.1f} days"
        else:
            time_estimate = "unknown"
        
        # Build summary
        factors = prediction.contributing_factors[:3] if prediction.contributing_factors else ["unknown"]
        
        template = self._templates["failure_predicted"].get(audience, "")
        summary = template.format(
            probability=prediction.probability,
            failure_type=prediction.failure_type or "general degradation",
            time_estimate=time_estimate,
            factors=", ".join(factors),
        )
        
        # Build sections
        sections = []
        
        if verbosity != ExplanationVerbosity.BRIEF:
            sections.append(ExplanationSection(
                title="Risk Assessment",
                content=self._explain_failure_risk(prediction, audience),
                importance=1.0,
                technical_level=2,
            ))
            
            if prediction.contributing_factors and verbosity in [
                ExplanationVerbosity.DETAILED, 
                ExplanationVerbosity.COMPREHENSIVE
            ]:
                sections.append(ExplanationSection(
                    title="Contributing Factors",
                    content=self._explain_factors(prediction.contributing_factors, audience),
                    importance=0.8,
                    technical_level=3,
                ))
        
        return StructuredExplanation(
            summary=summary,
            sections=sections,
            audience=audience,
            verbosity=verbosity,
            confidence=prediction.confidence,
        )
    
    def explain_root_cause(
        self,
        report: RootCauseReport,
        audience: Optional[ExplanationAudience] = None,
        verbosity: Optional[ExplanationVerbosity] = None,
    ) -> StructuredExplanation:
        """
        Explain root cause analysis results.
        
        Args:
            report: Root cause report to explain
            audience: Target audience
            verbosity: Level of detail
            
        Returns:
            StructuredExplanation
        """
        audience = audience or self.default_audience
        verbosity = verbosity or self.default_verbosity
        
        # Build summary
        if report.primary_cause:
            summary = self._explain_cause_summary(report.primary_cause, audience)
        else:
            summary = "No significant root causes identified. Model appears healthy."
        
        # Build sections
        sections = []
        
        if report.primary_cause and verbosity != ExplanationVerbosity.BRIEF:
            sections.append(ExplanationSection(
                title="Primary Cause",
                content=self._explain_cause_detail(report.primary_cause, audience),
                importance=1.0,
                technical_level=3,
            ))
        
        if report.contributing_causes and verbosity in [
            ExplanationVerbosity.DETAILED,
            ExplanationVerbosity.COMPREHENSIVE
        ]:
            for i, cause in enumerate(report.contributing_causes[:3]):
                sections.append(ExplanationSection(
                    title=f"Contributing Factor {i+1}",
                    content=self._explain_cause_detail(cause, audience),
                    importance=0.6 - i*0.1,
                    technical_level=3,
                ))
        
        if verbosity == ExplanationVerbosity.COMPREHENSIVE and report.causal_chain:
            sections.append(ExplanationSection(
                title="Causal Chain",
                content=self._explain_causal_chain(report.causal_chain, audience),
                importance=0.5,
                technical_level=4,
            ))
        
        return StructuredExplanation(
            summary=summary,
            sections=sections,
            audience=audience,
            verbosity=verbosity,
            confidence=report.analysis_confidence,
        )
    
    def explain_action_plan(
        self,
        plan: ActionPlan,
        audience: Optional[ExplanationAudience] = None,
        verbosity: Optional[ExplanationVerbosity] = None,
    ) -> StructuredExplanation:
        """
        Explain an action plan.
        
        Args:
            plan: Action plan to explain
            audience: Target audience
            verbosity: Level of detail
            
        Returns:
            StructuredExplanation
        """
        audience = audience or self.default_audience
        verbosity = verbosity or self.default_verbosity
        
        # Build summary
        summary = plan.summary
        
        # Build sections
        sections = []
        
        if verbosity != ExplanationVerbosity.BRIEF and plan.recommendations:
            # Top recommendations
            top_recs = plan.recommendations[:3]
            
            for i, rec in enumerate(top_recs):
                sections.append(ExplanationSection(
                    title=f"Action {i+1}: {rec.title}",
                    content=self._explain_recommendation(rec, audience, verbosity),
                    importance=rec.estimated_impact,
                    technical_level=2 if audience == ExplanationAudience.TECHNICAL else 1,
                ))
        
        if verbosity == ExplanationVerbosity.COMPREHENSIVE:
            sections.append(ExplanationSection(
                title="Expected Outcomes",
                content=f"If all recommended actions are taken, we expect approximately "
                        f"{plan.expected_improvement:.0%} improvement in model health. "
                        f"Total effort required: {plan.total_estimated_effort}.",
                importance=0.7,
                technical_level=2,
            ))
        
        return StructuredExplanation(
            summary=summary,
            sections=sections,
            audience=audience,
            verbosity=verbosity,
            confidence=0.8,
        )
    
    # ========== Helper Methods ==========
    
    def _explain_metric(
        self,
        metric_name: str,
        score: float,
        audience: ExplanationAudience,
    ) -> str:
        """Generate explanation for a single metric."""
        if score >= 0.8:
            status = "excellent"
            assessment = "well above acceptable thresholds"
        elif score >= 0.6:
            status = "good"
            assessment = "within acceptable range"
        elif score >= 0.4:
            status = "moderate"
            assessment = "below optimal but functional"
        else:
            status = "poor"
            assessment = "below acceptable thresholds"
        
        if audience == ExplanationAudience.TECHNICAL:
            return f"{metric_name} score: {score:.2f} ({status}). This is {assessment}."
        elif audience == ExplanationAudience.BUSINESS:
            return f"{metric_name} is {status}."
        else:
            return f"{metric_name}: {status.upper()}"
    
    def _explain_failure_risk(
        self,
        prediction: FailurePrediction,
        audience: ExplanationAudience,
    ) -> str:
        """Explain failure risk level."""
        prob = prediction.probability
        
        if prob >= 0.8:
            risk = "very high"
            action = "immediate action required"
        elif prob >= 0.6:
            risk = "high"
            action = "action recommended soon"
        elif prob >= 0.4:
            risk = "moderate"
            action = "monitor closely"
        else:
            risk = "low"
            action = "continue normal monitoring"
        
        if audience == ExplanationAudience.TECHNICAL:
            return (f"Failure probability: {prob:.1%}. Risk level: {risk}. "
                    f"Recommendation: {action}. "
                    f"Analysis confidence: {prediction.confidence:.0%}.")
        else:
            return f"Risk level is {risk}. {action.capitalize()}."
    
    def _explain_factors(
        self,
        factors: List[str],
        audience: ExplanationAudience,
    ) -> str:
        """Explain contributing factors."""
        if audience == ExplanationAudience.TECHNICAL:
            factor_list = "\n".join(f"• {f}" for f in factors)
            return f"The following factors contribute to this prediction:\n{factor_list}"
        else:
            return f"Main factors: {', '.join(factors[:3])}."
    
    def _explain_cause_summary(
        self,
        cause: IdentifiedCause,
        audience: ExplanationAudience,
    ) -> str:
        """Generate summary for a root cause."""
        cause_names = {
            CauseCategory.DATA_DRIFT: "data drift",
            CauseCategory.FEATURE_INSTABILITY: "unstable features",
            CauseCategory.OVERCONFIDENCE: "model overconfidence",
            CauseCategory.UNDERCONFIDENCE: "model underconfidence",
            CauseCategory.DISTRIBUTION_SHIFT: "distribution shift",
            CauseCategory.CORRELATION_BREAKDOWN: "correlation breakdown",
            CauseCategory.OUTLIER_INFLUENCE: "outlier influence",
            CauseCategory.MISSING_DATA_PATTERN: "missing data patterns",
            CauseCategory.DATA_QUALITY: "data quality issues",
        }
        
        cause_name = cause_names.get(cause.category, cause.category.value)
        
        if audience == ExplanationAudience.TECHNICAL:
            return (f"Root cause identified: {cause_name}. "
                    f"Confidence: {cause.confidence:.0%}, Severity: {cause.severity:.0%}. "
                    f"{cause.description}")
        elif audience == ExplanationAudience.BUSINESS:
            return f"The main issue is {cause_name}. {cause.description}"
        else:
            return f"Primary issue: {cause_name}"
    
    def _explain_cause_detail(
        self,
        cause: IdentifiedCause,
        audience: ExplanationAudience,
    ) -> str:
        """Generate detailed explanation for a cause."""
        parts = [cause.description]
        
        if audience == ExplanationAudience.TECHNICAL:
            if cause.affected_features:
                parts.append(f"Affected features: {', '.join(cause.affected_features[:5])}")
            
            if cause.evidence:
                evidence_text = "; ".join(e.description for e in cause.evidence[:3])
                parts.append(f"Evidence: {evidence_text}")
        
        return " ".join(parts)
    
    def _explain_causal_chain(
        self,
        chain: List[Tuple[str, str]],
        audience: ExplanationAudience,
    ) -> str:
        """Explain the causal chain."""
        if not chain:
            return "No clear causal chain identified."
        
        chain_parts = [f"{cause} → {effect}" for cause, effect in chain]
        
        if audience == ExplanationAudience.TECHNICAL:
            return "Identified causal relationships:\n" + "\n".join(f"• {p}" for p in chain_parts)
        else:
            return f"One issue leads to another: {chain_parts[0]}"
    
    def _explain_recommendation(
        self,
        rec: ActionRecommendation,
        audience: ExplanationAudience,
        verbosity: ExplanationVerbosity,
    ) -> str:
        """Generate explanation for a recommendation."""
        parts = [rec.description]
        
        if audience == ExplanationAudience.TECHNICAL:
            parts.append(f"Priority: {rec.priority.value}. "
                        f"Effort: {rec.estimated_effort}. "
                        f"Expected impact: {rec.estimated_impact:.0%}.")
            
            if verbosity == ExplanationVerbosity.COMPREHENSIVE and rec.steps:
                steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(rec.steps))
                parts.append(f"\nSteps:\n{steps_text}")
        elif audience == ExplanationAudience.BUSINESS:
            priority_text = "urgent" if rec.priority.value in ["critical", "high"] else "recommended"
            parts.append(f"This is {priority_text}.")
        
        if rec.expected_outcome:
            parts.append(f"Expected outcome: {rec.expected_outcome}")
        
        return " ".join(parts)
    
    # ========== Utility Methods ==========
    
    def generate_full_report(
        self,
        reliability: Optional[ReliabilityScore] = None,
        failure_prediction: Optional[FailurePrediction] = None,
        root_cause_report: Optional[RootCauseReport] = None,
        action_plan: Optional[ActionPlan] = None,
        audience: Optional[ExplanationAudience] = None,
    ) -> str:
        """
        Generate a comprehensive report combining all analyses.
        
        Args:
            reliability: Reliability score
            failure_prediction: Failure prediction
            root_cause_report: Root cause analysis
            action_plan: Action recommendations
            audience: Target audience
            
        Returns:
            Combined report as formatted text
        """
        audience = audience or self.default_audience
        parts = ["# Meta-Watchdog Health Report", ""]
        
        if reliability:
            exp = self.explain_reliability(reliability, audience, ExplanationVerbosity.STANDARD)
            parts.append("## Reliability Assessment")
            parts.append(exp.summary)
            parts.append("")
        
        if failure_prediction and failure_prediction.probability > 0.3:
            exp = self.explain_failure_prediction(failure_prediction, audience, ExplanationVerbosity.STANDARD)
            parts.append("## Failure Risk")
            parts.append(exp.summary)
            parts.append("")
        
        if root_cause_report and root_cause_report.primary_cause:
            exp = self.explain_root_cause(root_cause_report, audience, ExplanationVerbosity.STANDARD)
            parts.append("## Root Cause Analysis")
            parts.append(exp.summary)
            parts.append("")
        
        if action_plan and action_plan.recommendations:
            exp = self.explain_action_plan(action_plan, audience, ExplanationVerbosity.STANDARD)
            parts.append("## Recommended Actions")
            parts.append(exp.summary)
            for section in exp.sections[:3]:
                parts.append(f"\n### {section.title}")
                parts.append(section.content)
            parts.append("")
        
        return "\n".join(parts)
