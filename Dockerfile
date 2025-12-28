# =============================================================================
# BeautyScorer - CPU Docker Image
# =============================================================================
# Multi-stage build for minimal image size
#
# Build:
#   docker build -t beauty-scorer:latest -f Dockerfile .
#
# Run:
#   docker run -it --rm -v $(pwd)/datasets:/app/datasets beauty-scorer:latest
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build

# Copy only dependency files first for better caching
COPY pyproject.toml .
COPY README.md .

# Install CPU-only PyTorch first (smaller image)
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the package
RUN pip install -e ".[dev]"

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Labels
LABEL maintainer="Alessio Savi" \
    version="1.0.0" \
    description="BeautyScorer - Deep learning model for beauty score prediction (CPU)" \
    org.opencontainers.image.source="https://github.com/alessiosavi/BeautyScorer"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set up application directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser beauty_scorer/ ./beauty_scorer/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser pyproject.toml README.md ./

# Create directories for data and outputs
RUN mkdir -p datasets outputs exports logs && \
    chown -R appuser:appuser /app

# Install package in development mode
RUN pip install -e .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from beauty_scorer import create_model; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "from beauty_scorer import create_model; print('BeautyScorer ready!')"]
