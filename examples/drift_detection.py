"""
Example: Drift Detection with Meta-Watchdog

Demonstrates how Meta-Watchdog detects and responds to data drift.
"""

import numpy as np
from meta_watchdog.core.base_model import BaseModel
from meta_watchdog.orchestrator import MetaWatchdogOrchestrator, OrchestratorConfig
from meta_watchdog.testing import SyntheticDataGenerator, SyntheticDataConfig, DataPattern
from meta_watchdog.dashboard import DashboardDataProvider, TerminalDashboard


class DriftSensitiveModel(BaseModel):
    """Model that performs worse when data drifts from training distribution."""
    
    def __init__(self):
        super().__init__("drift_sensitive_model", "1.0")
        # "Training" distribution: centered at 0
        self._training_mean = 0.0
        self._training_std = 1.0
    
    def _predict_impl(self, X):
        return (np.sum(X[:, :3], axis=1) > 0).astype(int)
    
    def _predict_proba_impl(self, X):
        # Confidence decreases when data is far from training distribution
        distance = np.abs(np.mean(X, axis=1) - self._training_mean)
        confidence_penalty = 1 / (1 + distance)
        
        base_scores = 1 / (1 + np.exp(-np.sum(X[:, :3], axis=1)))
        # Reduce confidence for drifted data
        adjusted = base_scores * confidence_penalty
        adjusted = np.clip(adjusted, 0.1, 0.9)
        
        return np.column_stack([1 - adjusted, adjusted])


def main():
    print("=" * 60)
    print("META-WATCHDOG: Drift Detection Example")
    print("=" * 60)
    
    # Setup
    model = DriftSensitiveModel()
    config = OrchestratorConfig(
        reliability_warning_threshold=65.0,
        reliability_critical_threshold=45.0,
    )
    orchestrator = MetaWatchdogOrchestrator(model=model, config=config)
    
    # Setup alert tracking
    alerts_received = []
    orchestrator.register_alert_callback(lambda a: alerts_received.append(a))
    
    data_config = SyntheticDataConfig(n_features=5, random_seed=42)
    generator = SyntheticDataGenerator(data_config)
    
    print("\n--- Phase 1: Stable Operation (No Drift) ---")
    stable_batches = generator.generate_normal_scenario(n_batches=5, batch_size=100)
    
    for i, batch in enumerate(stable_batches):
        result = orchestrator.observe(
            X=batch["X"],
            y_true=batch["y_true"],
            predictions=batch["y_pred"],
            confidences=batch["confidence"],
        )
        print(f"  Batch {i+1}: Reliability = {result['reliability'].score:.1f}")
    
    print(f"\n  Status: {orchestrator.get_quick_status()['status'].upper()}")
    print(f"  Alerts so far: {len(alerts_received)}")
    
    print("\n--- Phase 2: Gradual Drift ---")
    drift_batches = generator.generate_drift_scenario(
        n_batches=10, 
        batch_size=100,
        drift_type="gradual"
    )
    
    for i, batch in enumerate(drift_batches):
        result = orchestrator.observe(
            X=batch["X"],
            y_true=batch["y_true"],
            predictions=batch["y_pred"],
            confidences=batch["confidence"],
        )
        
        status = "⚠️" if result['reliability'].score < 65 else "✓"
        print(f"  Batch {i+1}: Reliability = {result['reliability'].score:.1f} {status}")
    
    print(f"\n  Status: {orchestrator.get_quick_status()['status'].upper()}")
    print(f"  Alerts generated: {len(alerts_received)}")
    
    # Show alerts
    if alerts_received:
        print("\n  Recent Alerts:")
        for alert in alerts_received[-3:]:
            print(f"    [{alert.level.value.upper()}] {alert.message}")
    
    # Run deep analysis
    print("\n--- Deep Analysis ---")
    analysis = orchestrator.run_deep_analysis(X=drift_batches[-1]["X"])
    
    print(f"  Failure Risk: {analysis['failure_prediction'].probability:.0%}")
    
    if analysis["root_cause_report"].primary_cause:
        cause = analysis["root_cause_report"].primary_cause
        print(f"  Detected Cause: {cause.category.value}")
        print(f"  Description: {cause.description}")
    
    # Show sensitivity analysis
    if analysis["sensitivity_map"]:
        sens_map = analysis["sensitivity_map"]
        print(f"\n  Most Sensitive Features:")
        for fs in sorted(sens_map.feature_sensitivities, 
                        key=lambda x: x.overall_sensitivity, 
                        reverse=True)[:3]:
            print(f"    • {fs.feature_name}: {fs.overall_sensitivity:.2f} "
                  f"(vulnerable to {fs.most_vulnerable_to})")
    
    # Show recommendations
    if analysis["action_plan"]:
        print(f"\n  Recommendations:")
        for rec in analysis["action_plan"].recommendations[:3]:
            print(f"    [{rec.priority.value}] {rec.title}")
            print(f"      Expected Impact: {rec.estimated_impact:.0%}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
