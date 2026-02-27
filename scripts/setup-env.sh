#!/bin/bash
set -e

# Configuration
DEFAULT_PYTHON_VERSION='3.12'

echo "🔧 Setting up FrameCloud environment..."

# Ensure PATH is set for new installations
export PATH="$HOME/.cargo/bin:$PATH"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✓ uv is already installed"
fi

# Set up Python (use argument if provided, default to configured version)
PYTHON_VERSION="${1:-$DEFAULT_PYTHON_VERSION}"
echo "🐍 Setting up Python ${PYTHON_VERSION}..."
uv python install "${PYTHON_VERSION}"

# Install just
if ! command -v just &> /dev/null; then
    echo "⚙️ Installing just..."
    uv tool install rust-just
else
    echo "✓ just is already installed"
fi

# Install dependencies
echo "📚 Installing dependencies..."
uv sync --dev

echo "✅ Environment setup complete!"
