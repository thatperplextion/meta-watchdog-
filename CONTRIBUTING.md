# Contributing to Meta-Watchdog

First off, thank you for considering contributing to Meta-Watchdog! It's people like you that make Meta-Watchdog such a great tool for building self-aware ML systems.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

**Bug Report Template:**
```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Initialize with '...'
2. Call method '...'
3. See error

**Expected behavior**
A clear description of what you expected to happen.

**Environment:**
 - OS: [e.g., Ubuntu 22.04]
 - Python version: [e.g., 3.11]
 - Meta-Watchdog version: [e.g., 1.0.0]

**Additional context**
Add any other context about the problem here.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- A clear and descriptive title
- A detailed description of the proposed functionality
- Explain why this enhancement would be useful
- List any additional dependencies required

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- pip or poetry

### Setting Up Your Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/meta-watchdog.git
cd meta-watchdog

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=meta_watchdog --cov-report=html

# Run specific test file
pytest tests/test_core.py -v

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black meta_watchdog tests
isort meta_watchdog tests

# Check linting
flake8 meta_watchdog tests

# Type check
mypy meta_watchdog
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. Install them with:

```bash
pre-commit install
```

The hooks will run automatically on each commit.

## Project Structure

```
meta_watchdog/
├── core/              # Core abstractions and data structures
├── monitoring/        # Performance monitoring components
├── meta_prediction/   # Failure prediction models
├── simulation/        # Counterfactual simulation
├── analysis/          # Root cause analysis
├── recommendations/   # Action recommendations
├── explainability/    # Explanation generation
├── dashboard/         # Dashboard components
├── orchestrator/      # System coordination
├── telemetry/         # Logging and metrics
└── testing/           # Test utilities
```

## Writing Tests

### Test Guidelines

1. **Unit tests** should test individual components in isolation
2. **Integration tests** should test component interactions
3. Use descriptive test names: `test_reliability_score_degrades_on_drift`
4. Use fixtures for common setup
5. Mock external dependencies

### Test Structure

```python
import pytest
from meta_watchdog import SomeComponent

class TestSomeComponent:
    """Tests for SomeComponent."""
    
    @pytest.fixture
    def component(self):
        """Create a component instance for testing."""
        return SomeComponent()
    
    def test_basic_functionality(self, component):
        """Test that basic functionality works."""
        result = component.do_something()
        assert result is not None
    
    def test_edge_case(self, component):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            component.do_something(invalid_input)
```

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def calculate_reliability(
    metrics: PerformanceMetrics,
    weights: Optional[Dict[str, float]] = None,
) -> ReliabilityScore:
    """Calculate the reliability score for a model.
    
    This function computes a composite reliability score based on
    multiple performance indicators and configurable weights.
    
    Args:
        metrics: The performance metrics to evaluate.
        weights: Optional custom weights for score components.
            Defaults to standard weights if not provided.
    
    Returns:
        A ReliabilityScore object containing the composite score
        and individual component scores.
    
    Raises:
        ValueError: If metrics contain invalid values.
        
    Example:
        >>> metrics = PerformanceMetrics(accuracy=0.95, f1=0.92)
        >>> score = calculate_reliability(metrics)
        >>> print(score.overall)
        87.5
    """
```

## Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

## Questions?

Feel free to open an issue with the "question" label if you have any questions about contributing.

Thank you for your contributions! 🙏
