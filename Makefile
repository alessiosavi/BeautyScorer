.PHONY: help install install-dev install-all clean clean-all build test test-cov test-fast lint format typecheck check docs docs-serve docker-build docker-build-gpu docker-run docker-run-gpu debug debug-quick hooks hooks-install hooks-uninstall hooks-run hooks-update

# Default Python interpreter
PYTHON := python
PIP := pip

# Project info
PROJECT_NAME := beauty-scorer
VERSION := $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)$(PROJECT_NAME) v$(VERSION)$(NC)"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================

install: ## Install package in production mode
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	$(PIP) install -e ".[dev]"

install-all: ## Install package with all optional dependencies
	$(PIP) install -e ".[dev,tensorboard,wandb,onnx]"

# =============================================================================
# Cleaning
# =============================================================================

clean: ## Remove build artifacts and cache files
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf beauty_scorer.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"

clean-all: clean ## Remove all generated files including outputs and exports
	@echo "$(YELLOW)Cleaning all generated files...$(NC)"
	rm -rf outputs/
	rm -rf exports/
	rm -rf logs/
	rm -rf site/
	@echo "$(GREEN)Full clean complete!$(NC)"

# =============================================================================
# Building
# =============================================================================

build: clean ## Build source and wheel distributions
	@echo "$(YELLOW)Building package...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)Build complete!$(NC)"

build-wheel: ## Build wheel distribution only
	$(PYTHON) -m build --wheel

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=beauty_scorer --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-fast: ## Run tests excluding slow tests
	@echo "$(YELLOW)Running fast tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "not slow"

test-models: ## Run model tests only
	$(PYTHON) -m pytest tests/test_models.py -v

test-training: ## Run training tests only
	$(PYTHON) -m pytest tests/test_training.py -v

test-inference: ## Run inference tests only
	$(PYTHON) -m pytest tests/test_inference.py -v

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linter (ruff)
	@echo "$(YELLOW)Running linter...$(NC)"
	$(PYTHON) -m ruff check beauty_scorer/ tests/ scripts/

lint-fix: ## Run linter and fix issues
	@echo "$(YELLOW)Running linter with auto-fix...$(NC)"
	$(PYTHON) -m ruff check beauty_scorer/ tests/ scripts/ --fix

format: ## Format code with black
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(PYTHON) -m black beauty_scorer/ tests/ scripts/

format-check: ## Check code formatting without modifying
	$(PYTHON) -m black beauty_scorer/ tests/ scripts/ --check

typecheck: ## Run type checker (mypy)
	@echo "$(YELLOW)Running type checker...$(NC)"
	$(PYTHON) -m mypy beauty_scorer/

check: lint format-check typecheck ## Run all code quality checks
	@echo "$(GREEN)All checks passed!$(NC)"

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	@echo "$(YELLOW)Building documentation...$(NC)"
	mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(YELLOW)Serving documentation at http://localhost:8000$(NC)"
	mkdocs serve

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker image (CPU)
	@echo "$(YELLOW)Building CPU Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):$(VERSION) -t $(PROJECT_NAME):latest -f Dockerfile .
	@echo "$(GREEN)Docker image built: $(PROJECT_NAME):$(VERSION)$(NC)"

docker-build-gpu: ## Build Docker image (GPU/CUDA)
	@echo "$(YELLOW)Building GPU Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):$(VERSION)-gpu -t $(PROJECT_NAME):latest-gpu -f Dockerfile.gpu .
	@echo "$(GREEN)Docker image built: $(PROJECT_NAME):$(VERSION)-gpu$(NC)"

docker-run: ## Run Docker container (CPU)
	docker run -it --rm \
		-v $(PWD)/datasets:/app/datasets \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/exports:/app/exports \
		$(PROJECT_NAME):latest

docker-run-gpu: ## Run Docker container (GPU)
	docker run -it --rm --gpus all \
		-v $(PWD)/datasets:/app/datasets \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/exports:/app/exports \
		$(PROJECT_NAME):latest-gpu

docker-test: ## Run tests in Docker container
	docker run --rm $(PROJECT_NAME):latest pytest tests/ -v

docker-shell: ## Open shell in Docker container
	docker run -it --rm \
		-v $(PWD):/app \
		$(PROJECT_NAME):latest /bin/bash

# =============================================================================
# Git Hooks (pre-commit)
# =============================================================================

hooks: hooks-install ## Alias for hooks-install

hooks-install: ## Install pre-commit hooks
	@echo "$(YELLOW)Installing pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

hooks-uninstall: ## Uninstall pre-commit hooks
	@echo "$(YELLOW)Uninstalling pre-commit hooks...$(NC)"
	pre-commit uninstall
	pre-commit uninstall --hook-type commit-msg
	@echo "$(GREEN)Pre-commit hooks uninstalled!$(NC)"

hooks-run: ## Run all pre-commit hooks on all files
	@echo "$(YELLOW)Running pre-commit hooks on all files...$(NC)"
	pre-commit run --all-files

hooks-update: ## Update pre-commit hooks to latest versions
	@echo "$(YELLOW)Updating pre-commit hooks...$(NC)"
	pre-commit autoupdate
	@echo "$(GREEN)Pre-commit hooks updated!$(NC)"

hooks-clean: ## Clean pre-commit cache
	pre-commit clean
	pre-commit gc

hooks-skip: ## Show how to skip hooks (for emergencies)
	@echo "$(YELLOW)To skip pre-commit hooks (use sparingly):$(NC)"
	@echo "  git commit --no-verify -m 'message'"
	@echo "  git push --no-verify"
	@echo ""
	@echo "$(YELLOW)To skip specific hooks:$(NC)"
	@echo "  SKIP=mypy git commit -m 'message'"
	@echo "  SKIP=black,ruff git commit -m 'message'"

# =============================================================================
# Development
# =============================================================================

debug: ## Run debug notebook (all demos)
	$(PYTHON) debug_notebook.py

debug-quick: ## Run quick sanity checks
	$(PYTHON) debug_quick.py

train: ## Run training (requires config file)
	@echo "$(YELLOW)Starting training...$(NC)"
	$(PYTHON) -m scripts.train --config configs/default.yaml

evaluate: ## Run evaluation
	$(PYTHON) -m scripts.evaluate

export-model: ## Export model to ONNX/TorchScript
	$(PYTHON) -m scripts.export_model

# =============================================================================
# Utilities
# =============================================================================

version: ## Show project version
	@echo "$(PROJECT_NAME) v$(VERSION)"

deps: ## Show installed dependencies
	$(PIP) list | grep -E "(torch|numpy|pandas|albumentations|pydantic)"

env-info: ## Show environment information
	@echo "$(BLUE)Environment Information$(NC)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@$(PYTHON) -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')" 2>/dev/null || true

requirements: ## Generate requirements.txt from pyproject.toml
	$(PIP) freeze > requirements.txt
	@echo "$(GREEN)requirements.txt generated$(NC)"
