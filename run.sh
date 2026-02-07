#!/bin/bash

# Activate Virtual Environment
# If it doesn't exist, try to run setup first
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup.sh first..."
    chmod +x setup.sh
    ./setup.sh
fi

source venv/bin/activate

# Run the Python script
echo "Starting Garbage Detection..."
python3 main_pi.py

# Keep terminal open if it crashes
read -p "Press Enter to close..."
