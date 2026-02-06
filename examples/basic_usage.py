"""
Example: Basic Usage of Meta-Watchdog

Demonstrates the fundamental usage of the self-aware ML system.
"""

import numpy as np
from meta_watchdog.core.base_model import BaseModel
from meta_watchdog.orchestrator import MetaWatchdogOrchestrator, OrchestratorConfig
from meta_watchdog.testing import SyntheticDataGenerator, SyntheticDataConfig


# 1. Define your model by extending BaseModel
class MyClassifier(BaseModel):
    """Example classifier that wraps any sklearn-like model."""
    
    def __init__(self, sklearn_model=None):
        super().__init__("my_classifier", "1.0")
        self.model = sklearn_model
        self._threshold = 0.5
    
    def _predict_impl(self, X):
        """Core prediction logic."""
        if self.model is not None:
            return self.model.predict(X)
        # Simple fallback: linear decision
        return (np.sum(X, axis=1) > 0).astype(int)
    
    def _predict_proba_impl(self, X):
        """Probability predictions for confidence estimation."""
        if self.model is not None and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        # Fallback: sigmoid of sum
        scores = 1 / (1 + np.exp(-np.sum(X, axis=1) * 0.5))
        return np.column_stack([1 - scores, scores])


def main():
    """Run basic Meta-Watchdog example."""
    print("=" * 60)
    print("META-WATCHDOG: Basic Usage Example")
    print("=" * 60)
    
    # 2. Create your model
    model = MyClassifier()
    print("\n✓ Created model")
    
    # 3. Configure Meta-Watchdog
    config = OrchestratorConfig(
        reliability_warning_threshold=60.0,
        reliability_critical_threshold=40.0,
        enable_auto_recommendations=True,
        enable_auto_explanations=True,
    )
    
    # 4. Create the orchestrator
    orchestrator = MetaWatchdogOrchestrator(
        model=model,
        config=config,
        feature_names=[f"feature_{i}" for i in range(5)],
    )
    print("✓ Created Meta-Watchdog orchestrator")
    
    # 5. Generate synthetic data for demonstration
    data_config = SyntheticDataConfig(
        n_samples=100,
        n_features=5,
        random_seed=42,
    )
    generator = SyntheticDataGenerator(data_config)
    print("✓ Created synthetic data generator")
    
    # 6. Simulate normal operation
    print("\n--- Simulating Normal Operation ---")
    batches = generator.generate_normal_scenario(n_batches=5, batch_size=50)
    
    for i, batch in enumerate(batches):
        result = orchestrator.observe(
            X=batch["X"],
            y_true=batch["y_true"],
            predictions=batch["y_pred"],
            confidences=batch["confidence"],
        )
        
        print(f"Batch {i+1}: Reliability = {result['reliability'].score:.1f}, "
              f"Failure Risk = {result['failure_prediction'].probability:.1%}")
    
    # 7. Get health status
    print("\n--- Current Health Status ---")
    status = orchestrator.get_quick_status()
    print(f"System Status: {status['status'].upper()}")
    print(f"Reliability Score: {status['reliability_score']:.1f}/100")
    print(f"Failure Probability: {status['failure_probability']:.1%}")
    print(f"Active Alerts: {status['active_alerts']}")
    
    # 8. Run deep analysis
    print("\n--- Running Deep Analysis ---")
    analysis = orchestrator.run_deep_analysis(X=batches[-1]["X"])
    
    if analysis["root_cause_report"].primary_cause:
        cause = analysis["root_cause_report"].primary_cause
        print(f"Root Cause: {cause.category.value}")
        print(f"  Confidence: {cause.confidence:.0%}")
        print(f"  Severity: {cause.severity:.0%}")
    else:
        print("No significant root causes identified - model is healthy!")
    
    if analysis["action_plan"] and analysis["action_plan"].recommendations:
        print("\nTop Recommendations:")
        for rec in analysis["action_plan"].recommendations[:3]:
            print(f"  • [{rec.priority.value.upper()}] {rec.title}")
    
    # 9. Get human-readable explanation
    print("\n--- Explanation ---")
    print(analysis["explanation"][:500] + "..." if len(analysis["explanation"]) > 500 else analysis["explanation"])
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
