#!/bin/bash
set -e

echo "Do you want to create a virtual environment (.venv)? (y/n)"
read -r create_venv

if [[ "$create_venv" == "y" || "$create_venv" == "Y" ]]; then
    echo "Creating virtual environment .venv ..."
    python3 -m venv .venv

    # Activate depending on OS
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi

    echo "Installing dependencies inside the virtual environment ..."
    pip install --upgrade pip
    pip install -r requirements.txt

    echo ""
    echo "✅ Setup complete!"
    echo "Activate it later with:"
    echo "  source .venv/bin/activate"
else
    echo "Skipping virtual environment creation."
    echo "Installing dependencies globally ..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo ""
    echo "✅ Setup complete (global install)."
fi