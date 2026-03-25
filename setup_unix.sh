#!/bin/bash
# Hand Tracking Setup Script for Linux/macOS
# This script will automatically install Python and dependencies

set -e  # Exit on any error

echo "========================================"
echo "Hand Tracking - Linux/macOS Setup"
echo "========================================"
echo

# Check if uv is already installed
if ! command -v uv &> /dev/null; then
    echo "[1/3] Downloading uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the cargo env to make uv available in this session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo "ERROR: Failed to install uv. Please check your internet connection."
        exit 1
    fi
else
    echo "[1/3] uv package manager already installed"
fi

echo "[2/3] Creating Python environment and installing dependencies..."
uv venv .venv

echo "[3/3] Installing required packages (this may take a minute)..."
uv pip install -r requirements.txt

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "You can now run the hand tracking script with:"
echo "  ./run_unix.sh"
echo
echo "Or activate the environment manually:"
echo "  source .venv/bin/activate"
echo "  python handtracking.py"
echo
