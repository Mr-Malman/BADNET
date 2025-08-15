#!/bin/bash

echo "CYBER ATTACK PREDICTION SYSTEM"
echo "================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip3 is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv cyber_env

# Activate virtual environment
echo "Activating virtual environment..."
source cyber_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p logs/featureEngineering logs/train/logs logs/train/predicted_probas logs/train/predictions logs/train/viz processed_data/supervised_learning evaluations data/aggregated_data data/train\ test\ sets/Batch\ 0

# Navigate to codes directory
cd ./codes

echo ""
echo "STARTING CYBER ATTACK PREDICTION PIPELINE"
echo "============================================"
echo ""

# Step 1: Data Cleaning
echo "Step 1: Cleaning cyber attack data..."
python3 cyber_clean.py
if [ $? -ne 0 ]; then
    echo "Error in data cleaning step"
    exit 1
fi
echo "Data cleaning completed successfully!"
echo ""

# Step 2: Feature Engineering
echo "Step 2: Engineering cyber security features..."
python3 cyber_featureEngineering.py --ask 0 --verbose 1
if [ $? -ne 0 ]; then
    echo "Error in feature engineering step"
    exit 1
fi
echo "Feature engineering completed successfully!"
echo ""

# Step 3: Model Training
echo "Step 3: Training cyber attack prediction models..."
python3 cyber_train.py --start_clean 1 --ask 0 --verbose 1 --plot 1
if [ $? -ne 0 ]; then
    echo "Error in model training step"
    exit 1
fi
echo "Model training completed successfully!"
echo ""

# Deactivate virtual environment
deactivate

echo ""
echo "CYBER ATTACK PREDICTION SYSTEM COMPLETED!"
echo "============================================"
echo ""
echo "Results available in:"
echo "   • evaluations/ - Model performance metrics"
echo "   • logs/train/viz/ - Visualization charts"
echo "   • processed_data/ - Processed features"
echo ""
echo "Best performing model and results saved to evaluations/"
echo ""
echo "To view results, check:"
echo "   • evaluations/Batch 0 - Evaluations.csv"
echo "   • logs/train/viz/Batch 0/ - Charts and graphs"
echo ""
echo "Cyber attack prediction system is ready for real-world deployment!"
