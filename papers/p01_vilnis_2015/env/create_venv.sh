#!/usr/bin/env bash

set -e

# ============================================================
# replication-core environment setup
#
# Conventions:
# - Run from repository root
# - One environment per paper
# - Use uv for lightweight/faster package management
# ============================================================

PAPER_DIR="./papers/p01_vilnis_2015"
VENV_DIR="$PAPER_DIR/.venv"

echo
echo "============================================================"
echo "Creating virtual environment"
echo "============================================================"
echo

uv venv "$VENV_DIR"

echo
echo "============================================================"
echo "Activating virtual environment"
echo "============================================================"
echo

source "$VENV_DIR/bin/activate"

echo
echo "============================================================"
echo "Installing core packages"
echo "============================================================"
echo

uv pip install \
    torch \
    numpy \
    pytest

echo
echo "============================================================"
echo "Writing requirements lock file"
echo "============================================================"
echo

uv pip freeze > "$PAPER_DIR/requirements.lock.txt"

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
