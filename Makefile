# Meta-Watchdog Makefile
# Common development tasks

.PHONY: help install install-dev test lint format type-check clean build docs docker run

# Default target
help:
	@echo "Meta-Watchdog Development Tasks"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install package"
	@echo "  make install-dev    Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-int       Run integration tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linters"
	@echo "  make format         Format code with black and isort"
	@echo "  make type-check     Run mypy type checking"
	@echo "  make check          Run all quality checks"
	@echo ""
	@echo "Build:"
	@echo "  make build          Build distribution packages"
	@echo "  make clean          Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run Docker container"
	@echo "  make docker-test    Run tests in Docker"
	@echo ""
	@echo "Run:"
	@echo "  make demo           Run demo"
	@echo "  make dashboard      Launch terminal dashboard"
	@echo "  make api            Start API server"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,viz]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=meta_watchdog --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/ -v -m "not integration"

test-int:
	pytest tests/ -v -m "integration"

# Code quality
lint:
	flake8 meta_watchdog tests --max-line-length=88 --extend-ignore=E203
	bandit -r meta_watchdog -ll

format:
	black meta_watchdog tests examples
	isort meta_watchdog tests examples

format-check:
	black --check meta_watchdog tests examples
	isort --check-only meta_watchdog tests examples

type-check:
	mypy meta_watchdog --ignore-missing-imports

check: format-check lint type-check test

# Build
build: clean
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Docker
docker-build:
	docker build -t metawatchdog/meta-watchdog:latest .

docker-run:
	docker run --rm -it metawatchdog/meta-watchdog:latest

docker-test:
	docker build --target development -t metawatchdog/meta-watchdog:test .
	docker run --rm metawatchdog/meta-watchdog:test pytest tests/ -v

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Run
demo:
	python -m meta_watchdog.cli demo --batches 30 --batch-size 20

dashboard:
	python -m meta_watchdog.cli dashboard

api:
	python -c "from meta_watchdog.api import create_api_server; s=create_api_server(); s.start(background=False)"

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Release
release-check:
	python -m build
	twine check dist/*

release-test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release:
	twine upload dist/*

# Development utilities
shell:
	python -c "from meta_watchdog import *; import code; code.interact(local=locals())"

profile:
	python -m cProfile -o profile.out -m pytest tests/ -v
	python -c "import pstats; p=pstats.Stats('profile.out'); p.sort_stats('cumtime').print_stats(20)"

benchmark:
	python -c "from examples.advanced_usage import run_advanced_demo; run_advanced_demo()"

# Version management
version:
	@python -c "from meta_watchdog import __version__; print(__version__)"

bump-patch:
	@echo "Update version in meta_watchdog/__init__.py and pyproject.toml"

bump-minor:
	@echo "Update version in meta_watchdog/__init__.py and pyproject.toml"

bump-major:
	@echo "Update version in meta_watchdog/__init__.py and pyproject.toml"
