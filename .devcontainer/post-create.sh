#!/bin/bash

set -e

echo "🚀 Setting up Causal Interface Gym development environment..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -e ".[dev,ui,docs]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install --install-hooks

# Install Node.js dependencies if package.json exists
if [ -f "package.json" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p logs tmp data .pytest_cache .mypy_cache .ruff_cache

# Set up git configuration for container
echo "🔧 Configuring git..."
git config --global --add safe.directory /workspace
git config --global init.defaultBranch main

# Run initial tests to verify setup
echo "🧪 Running initial test suite..."
pytest tests/ -x --tb=short || echo "⚠️  Some tests failed, but setup continues..."

# Install Jupyter extensions if Jupyter is available
if command -v jupyter &> /dev/null; then
    echo "📓 Setting up Jupyter extensions..."
    jupyter labextension install @jupyter-widgets/jupyterlab-manager || true
fi

# Generate initial documentation if Sphinx is available
if command -v sphinx-build &> /dev/null; then
    echo "📚 Generating initial documentation..."
    cd docs && make html && cd .. || echo "⚠️  Documentation build failed"
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start:"
echo "  • Run tests: pytest"
echo "  • Start dev server: python -m causal_interface_gym.examples.demo"
echo "  • Format code: black . && ruff --fix ."
echo "  • Type check: mypy src/"
echo "  • Build docs: cd docs && make html"
echo ""
echo "Happy coding! 🎉"