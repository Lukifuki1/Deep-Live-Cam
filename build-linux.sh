#!/bin/bash
# Build script for Deep-Live-Cam Linux executable
# Requires Python 3.11+ and all requirements installed

set -e  # Exit on error

echo "========================================"
echo "Building Deep-Live-Cam for Linux..."
echo "========================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Install PyInstaller
echo "Installing PyInstaller..."
pip3 install pyinstaller

# Create models directory placeholder (models are downloaded at runtime)
mkdir -p models

# Build the executable
echo "Building executable with PyInstaller..."
pyinstaller deep_live_cam.spec --clean --noconfirm

echo "========================================"
echo "Build complete!"
echo "Executable is in dist/deep_live-cam"
echo "========================================"