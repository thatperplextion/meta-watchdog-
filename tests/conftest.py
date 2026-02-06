"""
Meta-Watchdog test suite configuration.
"""

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield


@pytest.fixture
def sample_features():
    """Generate sample feature data."""
    return np.random.randn(100, 5)


@pytest.fixture
def sample_labels():
    """Generate sample labels."""
    return np.random.randint(0, 2, 100)


@pytest.fixture
def sample_confidences():
    """Generate sample confidence scores."""
    return np.random.uniform(0.5, 1.0, 100)
