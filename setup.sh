#!/bin/bash
# BADNET Setup Script
# Author: ARYA KONER

echo "BADNET - Cyber Attack Detection & Monitoring System"
echo "=================================================="
echo ""

# Check if Python 3.7+ is installed
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv cyber_env
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source cyber_env/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
fi

# Make BADNET executable
echo "Making BADNET executable..."
chmod +x BADNET
chmod +x run_cyber_attack_system.sh

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Train the AI models: ./run_cyber_attack_system.sh"
echo "2. Save the best model: python3 save_best_model.py"
echo "3. Start using BADNET: ./BADNET status"
echo ""
echo "For more information, see README.md"
echo ""
echo "BADNET is ready to protect your system!"
