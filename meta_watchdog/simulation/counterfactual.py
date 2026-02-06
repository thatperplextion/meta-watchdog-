"""
Counterfactual Self-Failure Simulator

The key innovation of Meta-Watchdog: Asking "What if?" to stress-test
models before real-world failures occur.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from meta_watchdog.core.interfaces import PredictionModel, Simulator
from meta_watchdog.core.data_structures import (
    ScenarioType,
    StressTestResult,
)
from meta_watchdog.simulation.scenarios import (
    ScenarioGenerator,
    ScenarioConfig,
    GeneratedScenario,
)


@dataclass
class StressTestSuite:
    """Complete suite of stress test results."""
    timestamp: datetime
    scenarios_tested: int
    results: List[StressTestResult]
    overall_robustness: float  # 0-1, higher = more robust
    most_vulnerable_scenarios: List[str]
    most_robust_scenarios: List[str]
    recommendations: List[str]


class CounterfactualSimulator(Simulator):
    """
    Simulates counterfactual scenarios to stress-test models.
    
    This is the KEY INNOVATION of Meta-Watchdog. Instead of waiting
    for real-world failures, we proactively ask:
    
    "What if the data drifted?"
    "What if noise increased?"
    "What if feature correlations changed?"
    
    By simulating these scenarios, we can:
    1. Identify model vulnerabilities before they're exploited
    2. Quantify robustness across different failure modes
    3. Prioritize what to monitor and protect against
    4. Provide early warning when real data approaches danger zones
    
    The simulator generates synthetic data transformations and
    measures model behavior under each scenario.
    """
    
    def __init__(
        self,
        failure_threshold: float = 0.3,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the simulator.
        
        Args:
            failure_threshold: Performance drop threshold to trigger failure
            random_seed: Optional seed for reproducibility
        """
        self.failure_threshold = failure_threshold
        self.scenario_generator = ScenarioGenerator(random_seed)
        
        # Results cache
        self._last_stress_test: Optional[StressTestSuite] = None
        self._baseline_performance: Optional[float] = None
    
    # ========== Simulator Interface ==========
    
    def generate_scenario(
        self,
        base_data: NDArray[np.floating],
        scenario_type: str,
        severity: float = 0.5
    ) -> NDArray[np.floating]:
        """
        Generate a counterfactual scenario.
        
        Args:
            base_data: Original data to transform
            scenario_type: Type of scenario (as string)
            severity: Severity level 0-1
            
        Returns:
            Transformed data
        """
        # Convert string to enum
        scenario_enum = ScenarioType(scenario_type)
        
        config = ScenarioConfig(
            scenario_type=scenario_enum,
            severity=severity
        )
        
        result = self.scenario_generator.generate(base_data, config)
        return result.modified_data
    
    def get_available_scenarios(self) -> List[str]:
        """Get list of available scenario types."""
        return [st.value for st in ScenarioType]
    
    def run_stress_test(
        self,
        model: PredictionModel,
        base_data: NDArray[np.floating],
        scenarios: Optional[List[str]] = None,
        ground_truth: Optional[NDArray] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive stress tests.
        
        Args:
            model: Model to test
            base_data: Baseline data
            scenarios: Specific scenarios to test (or all)
            ground_truth: Optional ground truth for performance measurement
            
        Returns:
            Comprehensive stress test results
        """
        suite = self.run_stress_test_suite(
            model, base_data, 
            scenarios=[ScenarioType(s) for s in scenarios] if scenarios else None,
            ground_truth=ground_truth
        )
        
        return {
            "overall_robustness": suite.overall_robustness,
            "scenarios_tested": suite.scenarios_tested,
            "most_vulnerable": suite.most_vulnerable_scenarios,
            "most_robust": suite.most_robust_scenarios,
            "results": [
                {
                    "scenario": r.scenario_type.value,
                    "severity": r.severity,
                    "performance_drop": r.performance_drop,
                    "failure_triggered": r.failure_triggered,
                }
                for r in suite.results
            ],
            "recommendations": suite.recommendations,
        }
    
    # ========== Core Stress Testing ==========
    
    def run_stress_test_suite(
        self,
        model: PredictionModel,
        base_data: NDArray[np.floating],
        scenarios: Optional[List[ScenarioType]] = None,
        ground_truth: Optional[NDArray] = None,
        severity_levels: Optional[List[float]] = None,
    ) -> StressTestSuite:
        """
        Run a comprehensive stress test suite.
        
        Args:
            model: Model to stress test
            base_data: Baseline data for testing
            scenarios: Scenarios to test (default: all)
            ground_truth: Optional ground truth for performance
            severity_levels: Severity levels to test
            
        Returns:
            StressTestSuite with all results
        """
        if scenarios is None:
            scenarios = list(ScenarioType)
        
        if severity_levels is None:
            severity_levels = [0.2, 0.5, 0.8]
        
        # Get baseline performance
        baseline_perf, baseline_conf = self._evaluate_baseline(
            model, base_data, ground_truth
        )
        self._baseline_performance = baseline_perf
        
        results: List[StressTestResult] = []
        
        for scenario_type in scenarios:
            for severity in severity_levels:
                result = self._run_single_stress_test(
                    model=model,
                    base_data=base_data,
                    scenario_type=scenario_type,
                    severity=severity,
                    baseline_performance=baseline_perf,
                    baseline_confidence=baseline_conf,
                    ground_truth=ground_truth,
                )
                results.append(result)
        
        # Analyze results
        suite = self._analyze_suite_results(results, scenarios)
        self._last_stress_test = suite
        
        return suite
    
    def _run_single_stress_test(
        self,
        model: PredictionModel,
        base_data: NDArray[np.floating],
        scenario_type: ScenarioType,
        severity: float,
        baseline_performance: float,
        baseline_confidence: float,
        ground_truth: Optional[NDArray] = None,
    ) -> StressTestResult:
        """Run a single stress test scenario."""
        # Generate scenario
        config = ScenarioConfig(scenario_type=scenario_type, severity=severity)
        scenario = self.scenario_generator.generate(base_data, config)
        
        # Get predictions on stressed data
        predictions, confidence = model.predict_with_confidence(scenario.modified_data)
        
        # Evaluate performance under stress
        if ground_truth is not None:
            if model.model_type == "classification":
                stressed_perf = float(np.mean(predictions == ground_truth))
            else:
                stressed_perf = 1.0 - min(1.0, float(np.mean(np.abs(predictions - ground_truth))))
        else:
            # Without ground truth, use proxy metrics
            stressed_perf = self._estimate_performance_proxy(
                predictions, confidence, base_data, scenario.modified_data
            )
        
        # Calculate metrics
        performance_drop = baseline_performance - stressed_perf
        confidence_change = float(np.mean(confidence) - baseline_confidence)
        
        # Compare predictions
        base_predictions, _ = model.predict_with_confidence(base_data)
        if base_predictions.dtype == predictions.dtype:
            predictions_changed = float(np.mean(base_predictions != predictions))
        else:
            predictions_changed = float(np.mean(np.abs(base_predictions - predictions) > 0.01))
        
        # Determine if failure triggered
        failure_triggered = performance_drop > self.failure_threshold
        
        # Find most affected features
        most_affected = scenario.affected_features[:3] if scenario.affected_features else []
        
        return StressTestResult(
            scenario_type=scenario_type,
            severity=severity,
            baseline_performance=baseline_performance,
            stressed_performance=stressed_perf,
            performance_drop=performance_drop,
            predictions_changed_ratio=predictions_changed,
            confidence_change=confidence_change,
            failure_triggered=failure_triggered,
            failure_point=severity if failure_triggered else None,
            most_affected_features=[str(f) for f in most_affected],
        )
    
    def _evaluate_baseline(
        self,
        model: PredictionModel,
        data: NDArray[np.floating],
        ground_truth: Optional[NDArray] = None,
    ) -> Tuple[float, float]:
        """Evaluate baseline performance and confidence."""
        predictions, confidence = model.predict_with_confidence(data)
        
        if ground_truth is not None:
            if model.model_type == "classification":
                performance = float(np.mean(predictions == ground_truth))
            else:
                # For regression, normalize error to 0-1 performance
                mae = float(np.mean(np.abs(predictions - ground_truth)))
                gt_range = np.ptp(ground_truth) + 1e-8
                performance = max(0.0, 1.0 - mae / gt_range)
        else:
            # Use confidence as proxy
            performance = float(np.mean(confidence))
        
        avg_confidence = float(np.mean(confidence))
        
        return performance, avg_confidence
    
    def _estimate_performance_proxy(
        self,
        predictions: NDArray,
        confidence: NDArray[np.floating],
        base_data: NDArray[np.floating],
        stressed_data: NDArray[np.floating],
    ) -> float:
        """
        Estimate performance when ground truth unavailable.
        
        Uses confidence and prediction stability as proxies.
        """
        # Lower confidence under stress suggests worse performance
        conf_factor = float(np.mean(confidence))
        
        # High variance in confidence suggests instability
        conf_stability = 1.0 - min(1.0, float(np.std(confidence)) * 2)
        
        # Combine factors
        proxy_performance = 0.6 * conf_factor + 0.4 * conf_stability
        
        return proxy_performance
    
    def _analyze_suite_results(
        self,
        results: List[StressTestResult],
        scenarios: List[ScenarioType]
    ) -> StressTestSuite:
        """Analyze complete stress test results."""
        # Calculate overall robustness
        avg_perf_drop = np.mean([r.performance_drop for r in results])
        failure_rate = np.mean([r.failure_triggered for r in results])
        
        overall_robustness = max(0.0, 1.0 - avg_perf_drop - failure_rate * 0.5)
        
        # Find most vulnerable scenarios
        scenario_vulnerabilities: Dict[str, float] = {}
        for result in results:
            scenario_name = result.scenario_type.value
            if scenario_name not in scenario_vulnerabilities:
                scenario_vulnerabilities[scenario_name] = []
            scenario_vulnerabilities[scenario_name].append(result.performance_drop)
        
        scenario_avg_drops = {
            s: np.mean(drops) for s, drops in scenario_vulnerabilities.items()
        }
        
        sorted_scenarios = sorted(
            scenario_avg_drops.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        most_vulnerable = [s[0] for s in sorted_scenarios[:3]]
        most_robust = [s[0] for s in sorted_scenarios[-3:]]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            results, most_vulnerable, overall_robustness
        )
        
        return StressTestSuite(
            timestamp=datetime.now(),
            scenarios_tested=len(results),
            results=results,
            overall_robustness=overall_robustness,
            most_vulnerable_scenarios=most_vulnerable,
            most_robust_scenarios=most_robust,
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self,
        results: List[StressTestResult],
        vulnerable_scenarios: List[str],
        robustness: float
    ) -> List[str]:
        """Generate recommendations based on stress test results."""
        recommendations = []
        
        if robustness < 0.5:
            recommendations.append(
                "Model shows low overall robustness. Consider ensemble methods or regularization."
            )
        
        scenario_recommendations = {
            "feature_drift": "Implement drift detection and adaptive retraining",
            "noise_injection": "Consider denoising layers or robust loss functions",
            "trend_shift": "Add trend monitoring and concept drift detection",
            "feature_decay": "Implement feature importance monitoring",
            "outlier_storm": "Add outlier detection and handling in preprocessing",
            "missing_values": "Improve missing value imputation strategy",
            "correlation_breakdown": "Monitor feature correlations over time",
            "scale_shift": "Implement robust normalization that adapts to scale changes",
        }
        
        for scenario in vulnerable_scenarios:
            if scenario in scenario_recommendations:
                recommendations.append(scenario_recommendations[scenario])
        
        # Check for specific failure patterns
        high_severity_failures = [
            r for r in results 
            if r.failure_triggered and r.severity < 0.5
        ]
        
        if high_severity_failures:
            recommendations.append(
                "Model fails at low severity levels. Fundamental robustness improvements needed."
            )
        
        return recommendations
    
    # ========== Targeted Testing ==========
    
    def find_failure_threshold(
        self,
        model: PredictionModel,
        base_data: NDArray[np.floating],
        scenario_type: ScenarioType,
        ground_truth: Optional[NDArray] = None,
        precision: float = 0.05,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Find the severity level at which model fails.
        
        Uses binary search to find the failure threshold.
        
        Args:
            model: Model to test
            base_data: Baseline data
            scenario_type: Scenario to test
            ground_truth: Optional ground truth
            precision: Search precision
            
        Returns:
            Tuple of (threshold_severity, details)
        """
        baseline_perf, baseline_conf = self._evaluate_baseline(
            model, base_data, ground_truth
        )
        
        # Binary search for failure point
        low, high = 0.0, 1.0
        failure_severity = None
        
        while high - low > precision:
            mid = (low + high) / 2
            
            result = self._run_single_stress_test(
                model=model,
                base_data=base_data,
                scenario_type=scenario_type,
                severity=mid,
                baseline_performance=baseline_perf,
                baseline_confidence=baseline_conf,
                ground_truth=ground_truth,
            )
            
            if result.failure_triggered:
                high = mid
                failure_severity = mid
            else:
                low = mid
        
        details = {
            "scenario": scenario_type.value,
            "baseline_performance": baseline_perf,
            "failure_threshold": failure_severity or 1.0,
            "is_robust": failure_severity is None or failure_severity > 0.8,
        }
        
        return failure_severity or 1.0, details
    
    def test_feature_sensitivity(
        self,
        model: PredictionModel,
        base_data: NDArray[np.floating],
        feature_indices: List[int],
        ground_truth: Optional[NDArray] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Test model sensitivity to individual features.
        
        Args:
            model: Model to test
            base_data: Baseline data
            feature_indices: Features to test
            ground_truth: Optional ground truth
            
        Returns:
            Dictionary mapping feature index to sensitivity metrics
        """
        baseline_perf, _ = self._evaluate_baseline(model, base_data, ground_truth)
        
        results = {}
        
        for feat_idx in feature_indices:
            # Test with feature drift
            config = ScenarioConfig(
                scenario_type=ScenarioType.FEATURE_DRIFT,
                severity=0.5,
                affected_features=[feat_idx]
            )
            scenario = self.scenario_generator.generate(base_data, config)
            
            predictions, confidence = model.predict_with_confidence(scenario.modified_data)
            
            if ground_truth is not None:
                if model.model_type == "classification":
                    perf = float(np.mean(predictions == ground_truth))
                else:
                    mae = float(np.mean(np.abs(predictions - ground_truth)))
                    perf = max(0, 1 - mae / (np.ptp(ground_truth) + 1e-8))
            else:
                perf = float(np.mean(confidence))
            
            results[feat_idx] = {
                "performance_drop": baseline_perf - perf,
                "mean_confidence": float(np.mean(confidence)),
                "sensitivity_score": max(0, baseline_perf - perf) / 0.5,  # Normalized
            }
        
        return results
    
    # ========== Utilities ==========
    
    def get_last_results(self) -> Optional[StressTestSuite]:
        """Get results from last stress test."""
        return self._last_stress_test
    
    def get_robustness_report(self) -> Dict[str, Any]:
        """Generate a comprehensive robustness report."""
        if not self._last_stress_test:
            return {"status": "no_tests_run"}
        
        suite = self._last_stress_test
        
        return {
            "timestamp": suite.timestamp.isoformat(),
            "overall_robustness": suite.overall_robustness,
            "robustness_grade": self._robustness_to_grade(suite.overall_robustness),
            "scenarios_tested": suite.scenarios_tested,
            "failures_detected": sum(1 for r in suite.results if r.failure_triggered),
            "most_vulnerable": suite.most_vulnerable_scenarios,
            "most_robust": suite.most_robust_scenarios,
            "average_performance_drop": np.mean([r.performance_drop for r in suite.results]),
            "max_performance_drop": max(r.performance_drop for r in suite.results),
            "recommendations": suite.recommendations,
        }
    
    def _robustness_to_grade(self, robustness: float) -> str:
        """Convert robustness score to letter grade."""
        if robustness >= 0.9:
            return "A"
        elif robustness >= 0.8:
            return "B"
        elif robustness >= 0.7:
            return "C"
        elif robustness >= 0.6:
            return "D"
        else:
            return "F"
