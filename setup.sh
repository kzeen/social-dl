#!/bin/bash
set -e

echo "Creating virtual environment .venv ..."
python -m venv .venv
source .venv/bin/activate

echo "Installing dependencies ..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
