#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 0. Install System Dependencies (Cross-platform) ---
echo "Installing system dependencies..."

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first:"
        echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install OpenMP (required for LightGBM on macOS)
    echo "Installing OpenMP via Homebrew..."
    brew install libomp
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux"
    
    # Update package list
    sudo apt-get update
    
    # Install build essentials and OpenMP
    sudo apt-get install -y build-essential libgomp1
else
    echo "Warning: Unsupported OS: $OSTYPE"
    echo "You may need to install OpenMP manually for LightGBM to work."
fi

# --- 1. Create Directories ---
echo "Creating data, logs and models directories..."
mkdir -p data
mkdir -p models
mkdir -p logs

# --- 2. Create Virtual Environment ---
echo "Creating Python virtual environment..."
python3 -m venv venv

# --- 3. Activate Virtual Environment for this script ---
source venv/bin/activate

# --- 4. Install Dependencies ---
echo "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt

echo "--- Setup Complete ---"
