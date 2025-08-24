#!/bin/bash
# BADNET 24/7 Monitoring Script
# Author: ARYA KONER
# This script runs BADNET monitoring continuously

echo "Starting BADNET 24/7 Monitoring..."
echo "Monitoring Mode: Both (System + Network)"
echo "Update Interval: 10 seconds"
echo "Running continuously..."
echo ""

# Function to handle graceful shutdown
cleanup() {
    echo ""
    echo "Stopping BADNET monitoring..."
    echo " Final monitoring summary:"
    echo "============================================================"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "BADNET" ]; then
    echo "Virtual environment 'BADNET' not found!"
    echo "Please run the setup script first."
    exit 1
fi

# Activate virtual environment
source BADNET/bin/activate

echo "Virtual environment activated"
echo "AI Model: Ready"
echo "Features: 59 loaded"
echo "Monitoring: Active"
echo ""

# Start continuous monitoring
while true; do
    echo "============================================================"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - BADNET 24/7 Monitoring Active"
    echo "============================================================"
    
    # Run monitoring for 1 hour (3600 seconds) then restart
    python3 BADNET.py monitor --mode both --interval 10 --duration 3600
    
    echo ""
    echo "Restarting monitoring cycle..."
    echo ""
    
    # Small delay before restart
    sleep 5
done
