#!/bin/bash
set -e
echo "[setup] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "[setup] Upgrading pip..."
pip install --upgrade pip
echo "[setup] Installing dependencies..."
pip install -r requirements.txt
echo "[setup] Done. Activate with: source venv/bin/activate"