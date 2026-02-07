#!/bin/bash

echo "Starting Setup for Garbage Detection..."

# Update system and install dependencies for OpenCV
echo "Installing system dependencies (libgl1, etc)..."
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 python3-venv

# Create Virtual Environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

# Activate Virtual Environment
source venv/bin/activate

# Install Python Requirements
echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup Complete! You can now run the application using ./run.sh"
read -p "Press Enter to exit..."
