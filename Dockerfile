# Meta-Watchdog: Self-Aware Machine Learning System
# Multi-stage Dockerfile for optimal image size

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy and install the package
COPY . .
RUN pip install --no-cache-dir .

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim as runtime

# Labels
LABEL org.opencontainers.image.title="Meta-Watchdog"
LABEL org.opencontainers.image.description="Self-Aware Machine Learning System"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="Meta-Watchdog Team"
LABEL org.opencontainers.image.source="https://github.com/meta-watchdog/meta-watchdog"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user
RUN groupadd -r watchdog && useradd -r -g watchdog watchdog

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=watchdog:watchdog meta_watchdog/ ./meta_watchdog/
COPY --chown=watchdog:watchdog config/ ./config/
COPY --chown=watchdog:watchdog examples/ ./examples/

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R watchdog:watchdog /app

# Switch to non-root user
USER watchdog

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MW_LOG_LEVEL=INFO \
    MW_CONFIG_PATH=/app/config/default_config.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from meta_watchdog import __version__; print(__version__)" || exit 1

# Default command
ENTRYPOINT ["python", "-m", "meta_watchdog.cli"]
CMD ["status"]

# ============================================
# Stage 3: Development
# ============================================
FROM runtime as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black isort flake8 mypy

# Copy test files
COPY --chown=watchdog:watchdog tests/ ./tests/

USER watchdog

CMD ["pytest", "tests/", "-v"]
