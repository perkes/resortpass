#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

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
pip install -r requirements.txt

echo "--- Setup Complete ---"
