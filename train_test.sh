#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# Set environment variables for macOS OpenMP support
if [[ "$OSTYPE" == "darwin"* ]]; then
  # Set OpenMP library path for macOS
  if [ -d "/opt/homebrew/lib" ]; then
    # Apple Silicon Mac
    export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
  elif [ -d "/usr/local/lib" ]; then
    # Intel Mac
    export DYLD_LIBRARY_PATH="/usr/local/lib:$DYLD_LIBRARY_PATH"
  fi
fi

# Load environment variables (including PYTHONPATH)
if [ -f ".env" ]; then
  echo "Loading environment variables from .env..."
  set -a  # automatically export all variables
  source .env
  set +a
fi

echo "--- Training Model ---"
python3 src/model/train.py

echo "--- Running Prediction ---"
python3 src/model/predict.py

echo "--- Train/Test Cycle Complete ---"
