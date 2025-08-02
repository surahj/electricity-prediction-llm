#!/bin/bash
# Build script for Render deployment

echo "Starting build process..."

# Upgrade pip
python3 -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p models

echo "Build completed successfully!" 