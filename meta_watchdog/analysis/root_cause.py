"""
Root Cause Analysis Engine

Identifies WHY failures occur or are predicted by analyzing:
- Data drift patterns
- Feature instability
- Confidence calibration issues
- Environmental changes
- Historical failure correlations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from numpy.typing import NDArray
from datetime import datetime


class CauseCategory(Enum):
    """Categories of root causes for model failures."""
    DATA_DRIFT = "data_drift"
    FEATURE_INSTABILITY = "feature_instability"
    OVERCONFIDENCE = "overconfidence"
    UNDERCONFIDENCE = "underconfidence"
    DISTRIBUTION_SHIFT = "distribution_shift"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    OUTLIER_INFLUENCE = "outlier_influence"
    MISSING_DATA_PATTERN = "missing_data_pattern"
    CONCEPT_DRIFT = "concept_drift"
    ENVIRONMENTAL_CHANGE = "environmental_change"
    STALE_FEATURES = "stale_features"
    DATA_QUALITY = "data_quality"
    UNKNOWN = "unknown"


@dataclass
class CauseEvidence:
    """Evidence supporting a root cause hypothesis."""
    evidence_type: str
    description: str
    strength: float  # 0-1
    data: Optional[Dict[str, Any]] = None


@dataclass
class IdentifiedCause:
    """A single identified root cause."""
    category: CauseCategory
    confidence: float  # How sure we are this is a cause (0-1)
    severity: float  # How much this impacts the model (0-1)
    description: str
    evidence: List[CauseEvidence]
    affected_features: List[str]
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RootCauseReport:
    """Complete root cause analysis report."""
    primary_cause: Optional[IdentifiedCause]
    contributing_causes: List[IdentifiedCause]
    analysis_confidence: float  # Overall confidence in the analysis
    time_analyzed: datetime
    summary: str
    feature_involvement: Dict[str, float]  # Feature -> involvement score
    causal_chain: List[Tuple[str, str]]  # (cause, effect) relationships


class RootCauseAnalyzer:
    """
    Analyzes model performance issues to identify root causes.
    
    Uses multiple signals to determine WHY a model is failing:
    1. Performance metrics patterns
    2. Feature statistics changes
    3. Prediction confidence patterns
    4. Historical failure correlations
    
    This is the "detective" that explains failures to humans.
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.1,
        correlation_threshold: float = 0.3,
        confidence_calibration_threshold: float = 0.15,
    ):
        """
        Initialize the Root Cause Analyzer.
        
        Args:
            drift_threshold: Threshold for detecting significant drift
            correlation_threshold: Minimum correlation to consider causal
            confidence_calibration_threshold: Threshold for calibration issues
        """
        self.drift_threshold = drift_threshold
        self.correlation_threshold = correlation_threshold
        self.confidence_calibration_threshold = confidence_calibration_threshold
        
        # Cause detection functions
        self._cause_detectors = {
            CauseCategory.DATA_DRIFT: self._detect_data_drift,
            CauseCategory.FEATURE_INSTABILITY: self._detect_feature_instability,
            CauseCategory.OVERCONFIDENCE: self._detect_overconfidence,
            CauseCategory.UNDERCONFIDENCE: self._detect_underconfidence,
            CauseCategory.DISTRIBUTION_SHIFT: self._detect_distribution_shift,
            CauseCategory.CORRELATION_BREAKDOWN: self._detect_correlation_breakdown,
            CauseCategory.OUTLIER_INFLUENCE: self._detect_outlier_influence,
            CauseCategory.MISSING_DATA_PATTERN: self._detect_missing_data_pattern,
            CauseCategory.DATA_QUALITY: self._detect_data_quality_issues,
        }
        
        # Analysis history
        self._analysis_history: List[RootCauseReport] = []
    
    def analyze(
        self,
        performance_history: Dict[str, List[float]],
        feature_stats: Dict[str, Dict[str, float]],
        confidence_data: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> RootCauseReport:
        """
        Perform root cause analysis.
        
        Args:
            performance_history: Dictionary of metric_name -> list of values
            feature_stats: Dictionary of feature statistics
            confidence_data: Optional confidence calibration data
            feature_names: Optional feature names
            
        Returns:
            RootCauseReport
        """
        identified_causes: List[IdentifiedCause] = []
        
        # Run all detectors
        for category, detector in self._cause_detectors.items():
            cause = detector(performance_history, feature_stats, confidence_data)
            if cause is not None:
                identified_causes.append(cause)
        
        # Sort by severity * confidence
        identified_causes.sort(
            key=lambda c: c.severity * c.confidence,
            reverse=True
        )
        
        # Determine primary and contributing causes
        primary_cause = identified_causes[0] if identified_causes else None
        contributing_causes = identified_causes[1:5] if len(identified_causes) > 1 else []
        
        # Compute feature involvement
        feature_involvement = self._compute_feature_involvement(
            identified_causes, feature_names
        )
        
        # Build causal chain
        causal_chain = self._build_causal_chain(identified_causes)
        
        # Generate summary
        summary = self._generate_summary(primary_cause, contributing_causes)
        
        # Overall confidence
        if identified_causes:
            analysis_confidence = float(np.mean([c.confidence for c in identified_causes[:3]]))
        else:
            analysis_confidence = 0.0
        
        report = RootCauseReport(
            primary_cause=primary_cause,
            contributing_causes=contributing_causes,
            analysis_confidence=analysis_confidence,
            time_analyzed=datetime.now(),
            summary=summary,
            feature_involvement=feature_involvement,
            causal_chain=causal_chain,
        )
        
        self._analysis_history.append(report)
        return report
    
    # ========== Cause Detectors ==========
    
    def _detect_data_drift(
        self,
        performance_history: Dict[str, List[float]],
        feature_stats: Dict[str, Dict[str, float]],
        confidence_data: Optional[Dict[str, Any]],
    ) -> Optional[IdentifiedCause]:
        """Detect data drift as a root cause."""
        evidence = []
        affected_features = []
        drift_scores = []
        
        for feature_name, stats in feature_stats.items():
            if "drift_score" in stats:
                drift_score = stats["drift_score"]
                if drift_score > self.drift_threshold:
                    affected_features.append(feature_name)
                    drift_scores.append(drift_score)
                    evidence.append(CauseEvidence(
                        evidence_type="feature_drift",
                        description=f"{feature_name} shows drift score of {drift_score:.3f}",
                        strength=min(drift_score / 0.5, 1.0),
                        data={"feature": feature_name, "drift_score": drift_score}
                    ))
        
        if not affected_features:
            return None
        
        avg_drift = float(np.mean(drift_scores))
        
        return IdentifiedCause(
            category=CauseCategory.DATA_DRIFT,
            confidence=min(avg_drift / 0.3, 1.0),
            severity=avg_drift,
            description=f"Data drift detected in {len(affected_features)} features",
            evidence=evidence,
            affected_features=affected_features,
            recommended_actions=[
                "Implement drift monitoring for affected features",
                "Consider retraining with recent data",
                "Investigate data collection process changes",
            ],
        )
    
    def _detect_feature_instability(
        self,
        performance_history: Dict[str, List[float]],
        feature_stats: Dict[str, Dict[str, float]],
        confidence_data: Optional[Dict[str, Any]],
    ) -> Optional[IdentifiedCause]:
        """Detect unstable features."""
        evidence = []
        affected_features = []
        
        for feature_name, stats in feature_stats.items():
            cv = stats.get("coefficient_of_variation", 0)
            if cv > 0.5:  # High variability
                affected_features.append(feature_name)
                evidence.append(CauseEvidence(
                    evidence_type="high_variability",
                    description=f"{feature_name} has coefficient of variation {cv:.3f}",
                    strength=min(cv, 1.0),
                    data={"feature": feature_name, "cv": cv}
                ))
        
        if not affected_features:
            return None
        
        return IdentifiedCause(
            category=CauseCategory.FEATURE_INSTABILITY,
            confidence=0.7,
            severity=0.5,
            description=f"{len(affected_features)} features show high instability",
            evidence=evidence,
            affected_features=affected_features,
            recommended_actions=[
                "Add smoothing or aggregation to unstable features",
                "Consider feature engineering to stabilize inputs",
                "Implement feature-level monitoring",
            ],
        )
    
    def _detect_overconfidence(
        self,
        performance_history: Dict[str, List[float]],
        feature_stats: Dict[str, Dict[str, float]],
        confidence_data: Optional[Dict[str, Any]],
    ) -> Optional[IdentifiedCause]:
        """Detect model overconfidence."""
        if not confidence_data:
            return None
        
        avg_confidence = confidence_data.get("avg_confidence", 0.5)
        accuracy = confidence_data.get("accuracy", 0.5)
        
        calibration_gap = avg_confidence - accuracy
        
        if calibration_gap < self.confidence_calibration_threshold:
            return None
        
        evidence = [
            CauseEvidence(
                evidence_type="calibration_gap",
                description=f"Model confidence {avg_confidence:.2f} exceeds accuracy {accuracy:.2f}",
                strength=min(calibration_gap * 3, 1.0),
                data={"avg_confidence": avg_confidence, "accuracy": accuracy}
            )
        ]
        
        return IdentifiedCause(
            category=CauseCategory.OVERCONFIDENCE,
            confidence=min(calibration_gap * 2, 0.95),
            severity=calibration_gap,
            description="Model is overconfident in its predictions",
            evidence=evidence,
            affected_features=[],
            recommended_actions=[
                "Apply confidence calibration (Platt scaling, isotonic regression)",
                "Add uncertainty quantification methods",
                "Review training data for bias toward easy examples",
            ],
        )
    
    def _detect_underconfidence(
        self,
        performance_history: Dict[str, List[float]],
        feature_stats: Dict[str, Dict[str, float]],
        confidence_data: Optional[Dict[str, Any]],
    ) -> Optional[IdentifiedCause]:
        """Detect model underconfidence."""
        if not confidence_data:
            return None
        
        avg_confidence = confidence_data.get("avg_confidence", 0.5)
        accuracy = confidence_data.get("accuracy", 0.5)
        
        calibration_gap = accuracy - avg_confidence
        
        if calibration_gap < self.confidence_calibration_threshold:
            return None
        
        evidence = [
            CauseEvidence(
                evidence_type="calibration_gap",
                description=f"Model accuracy {accuracy:.2f} exceeds confidence {avg_confidence:.2f}",
                strength=min(calibration_gap * 3, 1.0),
                data={"avg_confidence": avg_confidence, "accuracy": accuracy}
            )
        ]
        
        return IdentifiedCause(
            category=CauseCategory.UNDERCONFIDENCE,
            confidence=min(calibration_gap * 2, 0.95),
            severity=calibration_gap * 0.7,  # Less severe than overconfidence
            description="Model is underconfident - better than it thinks",
            evidence=evidence,
            affected_features=[],
            recommended_actions=[
                "Apply temperature scaling to increase confidence",
                "Review training process for regularization issues",
            ],
        )
    
    def _detect_distribution_shift(
        self,
        performance_history: Dict[str, List[float]],
        feature_stats: Dict[str, Dict[str, float]],
        confidence_data: Optional[Dict[str, Any]],
    ) -> Optional[IdentifiedCause]:
        """Detect distribution shift in performance."""
        evidence = []
        
        for metric_name, values in performance_history.items():
            if len(values) < 10:
                continue
            
            # Split into halves
            mid = len(values) // 2
            first_half = np.array(values[:mid])
            second_half = np.array(values[mid:])
            
            # Check for significant difference
            first_mean = np.mean(first_half)
            second_mean = np.mean(second_half)
            
            change = abs(second_mean - first_mean) / (first_mean + 1e-8)
            
            if change > 0.2:  # 20% change
                evidence.append(CauseEvidence(
                    evidence_type="metric_shift",
                    description=f"{metric_name} shifted from {first_mean:.3f} to {second_mean:.3f}",
                    strength=min(change, 1.0),
                    data={"metric": metric_name, "change": change}
                ))
        
        if not evidence:
            return None
        
        avg_strength = np.mean([e.strength for e in evidence])
        
        return IdentifiedCause(
            category=CauseCategory.DISTRIBUTION_SHIFT,
            confidence=float(avg_strength),
            severity=float(avg_strength),
            description="Detected shift in performance distribution",
            evidence=evidence,
            affected_features=[],
            recommended_actions=[
                "Investigate recent data changes",
                "Check for external factors affecting predictions",
                "Consider model retraining with recent data",
            ],
        )
    
    def _detect_correlation_breakdown(
        self,
        performance_history: Dict[str, List[float]],
        feature_stats: Dict[str, Dict[str, float]],
        confidence_data: Optional[Dict[str, Any]],
    ) -> Optional[IdentifiedCause]:
        """Detect breakdown in feature correlations."""
        evidence = []
        affected_features = []
        
        for feature_name, stats in feature_stats.items():
            correlation_change = stats.get("correlation_change", 0)
            
            if abs(correlation_change) > self.correlation_threshold:
                affected_features.append(feature_name)
                evidence.append(CauseEvidence(
                    evidence_type="correlation_change",
                    description=f"{feature_name} correlation changed by {correlation_change:.3f}",
                    strength=min(abs(correlation_change), 1.0),
                    data={"feature": feature_name, "change": correlation_change}
                ))
        
        if not evidence:
            return None
        
        return IdentifiedCause(
            category=CauseCategory.CORRELATION_BREAKDOWN,
            confidence=0.75,
            severity=0.6,
            description=f"Correlation structure changed for {len(affected_features)} features",
            evidence=evidence,
            affected_features=affected_features,
            recommended_actions=[
                "Review feature dependencies",
                "Check for upstream data changes",
                "Consider adding correlation monitoring",
            ],
        )
    
    def _detect_outlier_influence(
        self,
        performance_history: Dict[str, List[float]],
        feature_stats: Dict[str, Dict[str, float]],
        confidence_data: Optional[Dict[str, Any]],
    ) -> Optional[IdentifiedCause]:
        """Detect outlier influence on model."""
        evidence = []
        affected_features = []
        
        for feature_name, stats in feature_stats.items():
            outlier_rate = stats.get("outlier_rate", 0)
            
            if outlier_rate > 0.05:  # More than 5% outliers
                affected_features.append(feature_name)
                evidence.append(CauseEvidence(
                    evidence_type="high_outlier_rate",
                    description=f"{feature_name} has {outlier_rate*100:.1f}% outliers",
                    strength=min(outlier_rate * 5, 1.0),
                    data={"feature": feature_name, "outlier_rate": outlier_rate}
                ))
        
        if not evidence:
            return None
        
        return IdentifiedCause(
            category=CauseCategory.OUTLIER_INFLUENCE,
            confidence=0.8,
            severity=0.5,
            description=f"High outlier rates in {len(affected_features)} features",
            evidence=evidence,
            affected_features=affected_features,
            recommended_actions=[
                "Add outlier detection in preprocessing",
                "Consider robust modeling techniques",
                "Investigate source of outliers",
            ],
        )
    
    def _detect_missing_data_pattern(
        self,
        performance_history: Dict[str, List[float]],
        feature_stats: Dict[str, Dict[str, float]],
        confidence_data: Optional[Dict[str, Any]],
    ) -> Optional[IdentifiedCause]:
        """Detect problematic missing data patterns."""
        evidence = []
        affected_features = []
        
        for feature_name, stats in feature_stats.items():
            missing_rate = stats.get("missing_rate", 0)
            
            if missing_rate > 0.1:  # More than 10% missing
                affected_features.append(feature_name)
                evidence.append(CauseEvidence(
                    evidence_type="high_missing_rate",
                    description=f"{feature_name} has {missing_rate*100:.1f}% missing values",
                    strength=min(missing_rate * 2, 1.0),
                    data={"feature": feature_name, "missing_rate": missing_rate}
                ))
        
        if not evidence:
            return None
        
        return IdentifiedCause(
            category=CauseCategory.MISSING_DATA_PATTERN,
            confidence=0.85,
            severity=0.4,
            description=f"High missing rates in {len(affected_features)} features",
            evidence=evidence,
            affected_features=affected_features,
            recommended_actions=[
                "Review data pipeline for missing data sources",
                "Improve imputation strategies",
                "Consider features with lower missing rates",
            ],
        )
    
    def _detect_data_quality_issues(
        self,
        performance_history: Dict[str, List[float]],
        feature_stats: Dict[str, Dict[str, float]],
        confidence_data: Optional[Dict[str, Any]],
    ) -> Optional[IdentifiedCause]:
        """Detect general data quality issues."""
        evidence = []
        affected_features = []
        
        for feature_name, stats in feature_stats.items():
            quality_score = stats.get("quality_score", 1.0)
            
            if quality_score < 0.7:
                affected_features.append(feature_name)
                evidence.append(CauseEvidence(
                    evidence_type="low_quality",
                    description=f"{feature_name} has quality score {quality_score:.2f}",
                    strength=1 - quality_score,
                    data={"feature": feature_name, "quality_score": quality_score}
                ))
        
        if not evidence:
            return None
        
        return IdentifiedCause(
            category=CauseCategory.DATA_QUALITY,
            confidence=0.7,
            severity=0.5,
            description=f"Data quality issues in {len(affected_features)} features",
            evidence=evidence,
            affected_features=affected_features,
            recommended_actions=[
                "Implement data validation at ingestion",
                "Add data quality monitoring",
                "Review data transformation pipelines",
            ],
        )
    
    # ========== Analysis Helpers ==========
    
    def _compute_feature_involvement(
        self,
        causes: List[IdentifiedCause],
        feature_names: Optional[List[str]],
    ) -> Dict[str, float]:
        """Compute how involved each feature is in the failures."""
        involvement: Dict[str, float] = {}
        
        for cause in causes:
            weight = cause.severity * cause.confidence
            for feature in cause.affected_features:
                current = involvement.get(feature, 0.0)
                involvement[feature] = current + weight
        
        # Normalize
        if involvement:
            max_inv = max(involvement.values())
            if max_inv > 0:
                involvement = {k: v/max_inv for k, v in involvement.items()}
        
        return involvement
    
    def _build_causal_chain(
        self,
        causes: List[IdentifiedCause],
    ) -> List[Tuple[str, str]]:
        """Build causal chain showing cause -> effect relationships."""
        chain: List[Tuple[str, str]] = []
        
        # Known causal relationships
        relationships = [
            (CauseCategory.DATA_DRIFT, CauseCategory.DISTRIBUTION_SHIFT),
            (CauseCategory.FEATURE_INSTABILITY, CauseCategory.OVERCONFIDENCE),
            (CauseCategory.OUTLIER_INFLUENCE, CauseCategory.FEATURE_INSTABILITY),
            (CauseCategory.MISSING_DATA_PATTERN, CauseCategory.DATA_QUALITY),
            (CauseCategory.CORRELATION_BREAKDOWN, CauseCategory.DISTRIBUTION_SHIFT),
        ]
        
        cause_categories = {c.category for c in causes}
        
        for cause, effect in relationships:
            if cause in cause_categories and effect in cause_categories:
                chain.append((cause.value, effect.value))
        
        return chain
    
    def _generate_summary(
        self,
        primary_cause: Optional[IdentifiedCause],
        contributing_causes: List[IdentifiedCause],
    ) -> str:
        """Generate human-readable summary."""
        if not primary_cause:
            return "No root causes identified. Model appears healthy."
        
        summary_parts = [
            f"Primary cause: {primary_cause.description} "
            f"(confidence: {primary_cause.confidence:.0%}, severity: {primary_cause.severity:.0%})."
        ]
        
        if contributing_causes:
            contributing_text = ", ".join(
                c.category.value for c in contributing_causes[:3]
            )
            summary_parts.append(f"Contributing factors: {contributing_text}.")
        
        if primary_cause.affected_features:
            features = ", ".join(primary_cause.affected_features[:5])
            summary_parts.append(f"Affected features: {features}.")
        
        return " ".join(summary_parts)
    
    def get_history(self) -> List[RootCauseReport]:
        """Get analysis history."""
        return self._analysis_history.copy()
    
    def clear_history(self) -> None:
        """Clear analysis history."""
        self._analysis_history.clear()
