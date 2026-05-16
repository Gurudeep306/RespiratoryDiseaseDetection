#!/bin/bash

# Setup script for Respiratory Disease Detection Project
# This script sets up the Python environment and installs dependencies

set -e

echo "=================================================="
echo "Respiratory Disease Detection - Setup Script"
echo "=================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Found Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "✓ Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Test the system: python test_pipeline.py"
echo "3. Train models: python train.py --dataset-type sample"
echo "4. Make predictions: python inference.py --audio audio.wav"
echo ""
echo "For more information, see QUICKSTART.md or README.md"
echo ""
