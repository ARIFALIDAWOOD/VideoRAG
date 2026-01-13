#!/bin/bash
# Setup script for VideoRAG backend using uv

set -e

echo "Setting up VideoRAG backend with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed successfully"
    # Source the shell config to get uv in PATH
    source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
fi

# Check if Python 3.11 is available, install if not
echo "Checking for Python 3.11..."
if ! uv python list --only-installed 2>&1 | grep -q "3\.11"; then
    echo "Installing Python 3.11..."
    uv python install 3.11
fi

echo "Creating virtual environment and installing dependencies..."
uv sync --python 3.11

echo "Installing ImageBind from git (no-deps)..."
uv pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git@3fcf5c9039de97f6ff5528ee4a9dce903c5979b3

echo ""
echo "Setup complete!"
echo ""
echo "To start the server, run:"
echo "  uv run python videorag_api.py"
echo ""
echo "Or use the Vimo desktop app which will auto-start the backend."
