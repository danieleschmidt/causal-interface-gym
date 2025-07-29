.PHONY: help install install-dev test lint format clean docs

help:
	@echo "Available commands:"
	@echo "  install      Install package"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo "  docs         Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest --cov=causal_interface_gym --cov-report=html --cov-report=term-missing

lint:
	ruff check .
	mypy .

format:
	black .
	ruff check . --fix

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make html