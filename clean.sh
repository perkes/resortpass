#!/bin/bash

# This script removes directories created during the project setup and execution.

echo "Cleaning up the project..."

# Remove directories
rm -rf data
rm -rf logs
rm -rf models
rm -rf venv

# Deactivate virtual environment if active
if command -v deactivate &> /dev/null
then
    echo "Deactivating virtual environment..."
    deactivate
fi

echo "--- Cleanup complete ---"
