#!/bin/bash
set -e

echo "🔧 Setting up FrameCloud environment..."

# Ensure PATH is set for new installations
export PATH="$HOME/.cargo/bin:$PATH"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    # export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✓ uv is already installed"
fi

# Set up Python 3.12
echo "🐍 Setting up Python 3.12..."
uv python install 3.12

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
