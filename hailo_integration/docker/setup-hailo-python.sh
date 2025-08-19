#!/bin/bash

# Script to be run inside the Docker container to install Hailo software

# Exit on error
set -e

echo "Installing Hailo Python packages..."

# Check if running inside Docker
if [ ! -f "/.dockerenv" ]; then
    echo "This script is intended to be run inside the Docker container."
    echo "Please run: docker exec -it hailo-dog-training bash"
    echo "Then run this script again."
    exit 1
fi

# Install Hailo Python packages
pip install hailo-ai

# Check for Hailo device
if [ -e "/dev/hailo0" ]; then
    echo "Hailo device detected at /dev/hailo0"
    
    # Check Hailo device status
    if command -v hailortcli &> /dev/null; then
        echo "Running Hailo device info:"
        hailortcli device-info
    else
        echo "hailortcli not found. Hailo Runtime might not be properly installed."
    fi
else
    echo "No Hailo device detected. Running in simulation mode."
fi

echo "Hailo setup complete!"
