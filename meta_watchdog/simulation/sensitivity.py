"""
Sensitivity Mapper

Creates failure sensitivity maps showing which features and conditions
are most dangerous for the model.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from meta_watchdog.core.interfaces import PredictionModel, SensitivityMapper as SensitivityMapperInterface
from meta_watchdog.core.data_structures import ScenarioType, SensitivityResult
from meta_watchdog.simulation.scenarios import ScenarioGenerator, ScenarioConfig


@dataclass
class FeatureSensitivity:
    """Sensitivity analysis for a single feature."""
    feature_index: int
    feature_name: str
    overall_sensitivity: float  # 0-1
    scenario_sensitivities: Dict[str, float]
    most_vulnerable_to: str
    risk_level: str  # low, medium, high, critical


@dataclass
class FailureSensitivityMap:
    """
    Complete sensitivity map for a model.
    
    Shows:
    - Which features are most sensitive
    - Which scenarios are most dangerous
    - Feature x Scenario vulnerability matrix
    """
    feature_sensitivities: List[FeatureSensitivity]
    scenario_risks: Dict[str, float]
    vulnerability_matrix: NDArray[np.floating]  # (n_features, n_scenarios)
    feature_names: List[str]
    scenario_names: List[str]
    overall_sensitivity: float
    critical_combinations: List[Tuple[str, str, float]]  # (feature, scenario, sensitivity)


class SensitivityMapper(SensitivityMapperInterface):
    """
    Maps model sensitivity to features and scenarios.
    
    Creates a comprehensive "Failure Sensitivity Map" that shows:
    1. Which features have the most impact on model failure
    2. Which scenarios are most dangerous
    3. Specific combinations that break the model
    
    This is crucial for:
    - Prioritizing monitoring efforts
    - Understanding model vulnerabilities
    - Guiding robustness improvements
    """
    
    def __init__(
        self,
        severity_level: float = 0.5,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the sensitivity mapper.
        
        Args:
            severity_level: Default severity for sensitivity tests
            random_seed: Optional seed for reproducibility
        """
        self.severity_level = severity_level
        self.scenario_generator = ScenarioGenerator(random_seed)
        
        # Cache
        self._last_map: Optional[FailureSensitivityMap] = None
    
    def compute_sensitivity(
        self,
        model: PredictionModel,
        data: NDArray[np.floating],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute overall sensitivity to each feature.
        
        Args:
            model: Model to analyze
            data: Sample data
            feature_names: Optional feature names
            
        Returns:
            Dictionary mapping feature names to sensitivity scores
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_features = data.shape[1]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Get baseline predictions
        base_preds, base_conf = model.predict_with_confidence(data)
        
        sensitivities = {}
        
        for i in range(n_features):
            # Test sensitivity with feature drift
            sensitivity = self._compute_single_feature_sensitivity(
                model, data, i, base_preds, base_conf
            )
            sensitivities[feature_names[i]] = sensitivity
        
        return sensitivities
    
    def _compute_single_feature_sensitivity(
        self,
        model: PredictionModel,
        data: NDArray[np.floating],
        feature_idx: int,
        base_preds: NDArray,
        base_conf: NDArray[np.floating],
    ) -> float:
        """Compute sensitivity for a single feature."""
        sensitivities = []
        
        # Test with multiple scenario types
        for scenario_type in [ScenarioType.FEATURE_DRIFT, ScenarioType.NOISE_INJECTION]:
            config = ScenarioConfig(
                scenario_type=scenario_type,
                severity=self.severity_level,
                affected_features=[feature_idx]
            )
            
            scenario = self.scenario_generator.generate(data, config)
            preds, conf = model.predict_with_confidence(scenario.modified_data)
            
            # Measure change
            if preds.dtype == base_preds.dtype and np.issubdtype(preds.dtype, np.number):
                pred_change = np.mean(np.abs(preds - base_preds))
            else:
                pred_change = np.mean(preds != base_preds)
            
            conf_change = np.mean(np.abs(conf - base_conf))
            
            # Combined sensitivity
            sensitivity = float(pred_change * 0.7 + conf_change * 0.3)
            sensitivities.append(sensitivity)
        
        return np.mean(sensitivities)
    
    def generate_sensitivity_map(
        self,
        model: PredictionModel,
        data: NDArray[np.floating],
        scenarios: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> FailureSensitivityMap:
        """
        Generate a complete failure sensitivity map.
        
        Args:
            model: Model to analyze
            data: Sample data
            scenarios: Scenarios to test (or all)
            feature_names: Optional feature names
            
        Returns:
            FailureSensitivityMap
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_features = data.shape[1]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        if scenarios is None:
            scenario_types = list(ScenarioType)
        else:
            scenario_types = [ScenarioType(s) for s in scenarios]
        
        scenario_names = [s.value for s in scenario_types]
        
        # Get baseline
        base_preds, base_conf = model.predict_with_confidence(data)
        
        # Build vulnerability matrix
        vulnerability_matrix = np.zeros((n_features, len(scenario_types)))
        
        for i, feat_idx in enumerate(range(n_features)):
            for j, scenario_type in enumerate(scenario_types):
                sensitivity = self._compute_feature_scenario_sensitivity(
                    model, data, feat_idx, scenario_type, base_preds, base_conf
                )
                vulnerability_matrix[i, j] = sensitivity
        
        # Compute feature sensitivities
        feature_sensitivities = []
        for i in range(n_features):
            feat_sens = vulnerability_matrix[i, :]
            scenario_sens = {
                scenario_names[j]: float(feat_sens[j]) 
                for j in range(len(scenario_types))
            }
            
            most_vulnerable_idx = np.argmax(feat_sens)
            
            overall = float(np.mean(feat_sens))
            
            feature_sensitivities.append(FeatureSensitivity(
                feature_index=i,
                feature_name=feature_names[i],
                overall_sensitivity=overall,
                scenario_sensitivities=scenario_sens,
                most_vulnerable_to=scenario_names[most_vulnerable_idx],
                risk_level=self._sensitivity_to_risk(overall),
            ))
        
        # Compute scenario risks
        scenario_risks = {
            scenario_names[j]: float(np.mean(vulnerability_matrix[:, j]))
            for j in range(len(scenario_types))
        }
        
        # Find critical combinations
        critical_combinations = self._find_critical_combinations(
            vulnerability_matrix, feature_names, scenario_names
        )
        
        # Overall sensitivity
        overall = float(np.mean(vulnerability_matrix))
        
        sensitivity_map = FailureSensitivityMap(
            feature_sensitivities=feature_sensitivities,
            scenario_risks=scenario_risks,
            vulnerability_matrix=vulnerability_matrix,
            feature_names=feature_names,
            scenario_names=scenario_names,
            overall_sensitivity=overall,
            critical_combinations=critical_combinations,
        )
        
        self._last_map = sensitivity_map
        return sensitivity_map
    
    def _compute_feature_scenario_sensitivity(
        self,
        model: PredictionModel,
        data: NDArray[np.floating],
        feature_idx: int,
        scenario_type: ScenarioType,
        base_preds: NDArray,
        base_conf: NDArray[np.floating],
    ) -> float:
        """Compute sensitivity for a specific feature-scenario combination."""
        config = ScenarioConfig(
            scenario_type=scenario_type,
            severity=self.severity_level,
            affected_features=[feature_idx]
        )
        
        scenario = self.scenario_generator.generate(data, config)
        preds, conf = model.predict_with_confidence(scenario.modified_data)
        
        # Measure prediction change
        if np.issubdtype(preds.dtype, np.number) and np.issubdtype(base_preds.dtype, np.number):
            # Normalize by prediction range
            pred_range = max(np.ptp(base_preds), 1e-8)
            pred_change = np.mean(np.abs(preds - base_preds)) / pred_range
        else:
            pred_change = np.mean(preds != base_preds)
        
        # Confidence change
        conf_change = np.mean(np.abs(conf - base_conf))
        
        # Combined and normalized
        sensitivity = min(1.0, pred_change * 0.6 + conf_change * 0.4)
        
        return float(sensitivity)
    
    def _sensitivity_to_risk(self, sensitivity: float) -> str:
        """Convert sensitivity score to risk level."""
        if sensitivity >= 0.7:
            return "critical"
        elif sensitivity >= 0.5:
            return "high"
        elif sensitivity >= 0.3:
            return "medium"
        else:
            return "low"
    
    def _find_critical_combinations(
        self,
        matrix: NDArray[np.floating],
        feature_names: List[str],
        scenario_names: List[str],
        top_n: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """Find most critical feature-scenario combinations."""
        combinations = []
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                combinations.append((
                    feature_names[i],
                    scenario_names[j],
                    float(matrix[i, j])
                ))
        
        # Sort by sensitivity
        combinations.sort(key=lambda x: x[2], reverse=True)
        
        return combinations[:top_n]
    
    # ========== Analysis Methods ==========
    
    def get_most_sensitive_features(
        self, top_n: int = 5
    ) -> List[FeatureSensitivity]:
        """Get the most sensitive features."""
        if not self._last_map:
            return []
        
        sorted_features = sorted(
            self._last_map.feature_sensitivities,
            key=lambda x: x.overall_sensitivity,
            reverse=True
        )
        
        return sorted_features[:top_n]
    
    def get_most_dangerous_scenarios(
        self, top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """Get the most dangerous scenarios."""
        if not self._last_map:
            return []
        
        sorted_scenarios = sorted(
            self._last_map.scenario_risks.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_scenarios[:top_n]
    
    def get_feature_risk_profile(
        self, feature_name: str
    ) -> Optional[FeatureSensitivity]:
        """Get risk profile for a specific feature."""
        if not self._last_map:
            return None
        
        for fs in self._last_map.feature_sensitivities:
            if fs.feature_name == feature_name:
                return fs
        
        return None
    
    def generate_sensitivity_report(self) -> Dict[str, Any]:
        """Generate a comprehensive sensitivity report."""
        if not self._last_map:
            return {"status": "no_analysis_run"}
        
        map_data = self._last_map
        
        # High risk features
        high_risk = [
            fs for fs in map_data.feature_sensitivities
            if fs.risk_level in ("high", "critical")
        ]
        
        return {
            "overall_sensitivity": map_data.overall_sensitivity,
            "overall_risk_level": self._sensitivity_to_risk(map_data.overall_sensitivity),
            "features_analyzed": len(map_data.feature_names),
            "scenarios_tested": len(map_data.scenario_names),
            "high_risk_features": [
                {
                    "name": fs.feature_name,
                    "sensitivity": fs.overall_sensitivity,
                    "most_vulnerable_to": fs.most_vulnerable_to,
                }
                for fs in high_risk
            ],
            "most_dangerous_scenarios": self.get_most_dangerous_scenarios(),
            "critical_combinations": map_data.critical_combinations[:5],
            "recommendations": self._generate_recommendations(map_data),
        }
    
    def _generate_recommendations(
        self, map_data: FailureSensitivityMap
    ) -> List[str]:
        """Generate recommendations based on sensitivity analysis."""
        recommendations = []
        
        # High overall sensitivity
        if map_data.overall_sensitivity > 0.5:
            recommendations.append(
                "Model shows high overall sensitivity. Consider adding regularization or ensemble methods."
            )
        
        # Critical features
        critical_features = [
            fs for fs in map_data.feature_sensitivities
            if fs.risk_level == "critical"
        ]
        
        if critical_features:
            feature_names = [fs.feature_name for fs in critical_features]
            recommendations.append(
                f"Critical sensitivity detected in features: {', '.join(feature_names)}. "
                "Prioritize monitoring and consider feature engineering."
            )
        
        # Dangerous scenarios
        for scenario, risk in map_data.scenario_risks.items():
            if risk > 0.6:
                scenario_recommendations = {
                    "feature_drift": "Implement drift detection for this model",
                    "noise_injection": "Add input validation and denoising",
                    "outlier_storm": "Implement outlier detection in preprocessing",
                }
                if scenario in scenario_recommendations:
                    recommendations.append(scenario_recommendations[scenario])
        
        return recommendations
    
    def get_monitoring_priorities(self) -> Dict[str, Any]:
        """
        Get prioritized monitoring recommendations.
        
        Returns features and scenarios to monitor most closely.
        """
        if not self._last_map:
            return {"status": "no_analysis_run"}
        
        # Sort features by sensitivity
        sorted_features = sorted(
            self._last_map.feature_sensitivities,
            key=lambda x: x.overall_sensitivity,
            reverse=True
        )
        
        # Sort scenarios
        sorted_scenarios = sorted(
            self._last_map.scenario_risks.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "priority_features": [
                {"name": fs.feature_name, "priority": "high" if i < 3 else "medium"}
                for i, fs in enumerate(sorted_features[:5])
            ],
            "priority_scenarios": [
                {"scenario": s[0], "priority": "high" if i < 2 else "medium"}
                for i, s in enumerate(sorted_scenarios[:4])
            ],
            "critical_pairs": [
                {"feature": c[0], "scenario": c[1]}
                for c in self._last_map.critical_combinations[:3]
            ],
        }
