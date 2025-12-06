#!/bin/bash
echo "==================================================="
echo "  ECG Arrhythmia Classifier - One-Click Installer"
echo "==================================================="

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed."
    echo "Please install Python 3."
    exit 1
fi

# 2. Install Requirements
echo "[INFO] Installing dependencies..."
pip3 install -r requirements.txt

# 3. Train Model
echo ""
echo "[INFO] Starting Model Training..."
python3 train_sota.py

# 4. Run Dashboard
echo ""
echo "[INFO] Training Complete! Launching Dashboard..."
streamlit run src/ui/dashboard.py
