#!/bin/bash
# Verbose BADNET Monitoring Script
# Shows detailed prediction results, attack names, timestamps, and feature values
# Author: ARYA KONER

echo "Starting VERBOSE BADNET Monitoring..."
echo "This will show:"
echo "   • Timestamps for each prediction"
echo "   • Attack detection results (YES/NO)"
echo "   • Attack probability percentages"
echo "   • Threat levels (NORMAL/LOW_SUSPICION/MODERATE_SUSPICION/HIGH_PROBABILITY_ATTACK)"
echo "   • System status (CPU, Memory, Process Count, Disk Activity)"
echo "   • Network status (Connections, Packets, Traffic Level)"
echo "   • Prediction history timeline"
echo ""

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
source cyber_env/bin/activate

echo "Virtual environment activated"
echo "AI Model: Ready"
echo "Features: 59 loaded"
echo ""

# Run verbose monitoring
echo "Starting monitoring with detailed output..."
echo "Press Ctrl+C to stop"
echo ""

python3 verbose_monitor.py --interval 5 --duration 0
