#!/usr/bin/env bash

set -e

# ============================================================
# Create Python virtual environment for Vilnis replication
# ============================================================

PAPER_DIR="./papers/01_vilnis_2015_gaussian_embeddings"
VENV_DIR="$PAPER_DIR/.venv"

echo
echo "============================================================"
echo "Creating virtual environment"
echo "============================================================"
echo

python3 -m venv "$VENV_DIR"

echo
echo "============================================================"
echo "Activating virtual environment"
echo "============================================================"
echo

source "$VENV_DIR/bin/activate"

echo
echo "============================================================"
echo "Upgrading pip"
echo "============================================================"
echo

python -m pip install --upgrade pip

echo
echo "============================================================"
echo "Installing core packages"
echo "============================================================"
echo

pip install torch
pip install numpy
pip install scipy
pip install matplotlib
pip install pytest
pip install jupyter

echo
echo "============================================================"
echo "Environment setup complete"
echo "============================================================"
echo

echo "Virtual environment:"
echo "$VENV_DIR"

echo
echo "To activate later:"
echo
echo "source $VENV_DIR/bin/activate"
echo

