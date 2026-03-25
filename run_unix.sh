#!/bin/bash
# Hand Tracking - Run Script for Linux/macOS

# Check if environment exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run ./setup_unix.sh first."
    echo
    exit 1
fi

# Activate environment and run the script
source .venv/bin/activate
python handtracking.py "$@"
