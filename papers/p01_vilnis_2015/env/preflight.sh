#!/usr/bin/env bash

set -e

echo
echo "============================================================"
echo "replication-core preflight"
echo "============================================================"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. Installing python3..."
    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip
else
    echo "python3 found: $(python3 --version)"
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "python3 venv support missing. Installing python3-venv..."
    sudo apt update
    sudo apt install -y python3-venv
else
    echo "python3 venv support found."
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "curl not found. Installing curl..."
    sudo apt update
    sudo apt install -y curl
else
    echo "curl found."
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv found: $(uv --version)"
fi

echo
echo "============================================================"
echo "Preflight complete"
echo "============================================================"
echo

echo "Versions:"
python3 --version
uv --version

echo
echo "If uv is still not found in a new shell, run:"
echo 'export PATH="$HOME/.local/bin:$PATH"'