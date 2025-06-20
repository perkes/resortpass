#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

echo "--- Generating Dummy Data ---"
python3 src/dummy_data/hotels.py
python3 src/dummy_data/searches.py
python3 src/dummy_data/clickthroughs.py