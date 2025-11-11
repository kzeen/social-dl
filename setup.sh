#!/bin/bash
set -e

echo "Do you want to create a virtual environment (.venv)? (y/n)"
read -r create_venv

if [[ "$create_venv" == "y" || "$create_venv" == "Y" ]]; then
    echo "Creating virtual environment .venv ..."
    python -m venv .venv

    # Activate venv (Windows vs Unix)
    if [[ "$OS" == "Windows_NT" ]]; then
        # Git Bash / cmd / PowerShell on Windows
        source .venv/Scripts/activate
    else
        # macOS / Linux
        source .venv/bin/activate
    fi

    echo "Installing dependencies inside the virtual environment ..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

    echo ""
    echo "✅ Setup complete!"
    echo "Activate it later with:"
    if [[ "$OS" == "Windows_NT" ]]; then
        echo "  source .venv/Scripts/activate"
    else
        echo "  source .venv/bin/activate"
    fi
else
    echo "Skipping virtual environment creation."
    echo "Installing dependencies globally (or in your current env) ..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

    echo ""
    echo "✅ Setup complete (no venv)."
fi