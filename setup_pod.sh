#!/bin/bash

# Pod Setup Script for Vision Forge
set -e

echo "🚀 Starting Pod Setup..."

# 1. Update and Install System Dependencies
echo "[1/6] Installing system dependencies (Node.js, Redis, Git)..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get update && apt-get install -y nodejs redis-server git curl libgl1

# 2. Install UV (Fast Python Package Manager)
echo "[2/6] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 3. Clone the Repository
echo "[3/6] Cloning Vision_Forge repository..."
if [ -d "Vision_Forge" ]; then
    echo "Directory Vision_Forge already exists. Pulling latest changes..."
    cd Vision_Forge && git pull
else
    git clone https://github.com/Priyanshu314/Vision_Forge.git
    cd Vision_Forge
fi

# 4. Sync Environment
echo "[4/6] Syncing python environment with uv..."
# We use the locked cu121 index we configured in pyproject.toml
uv sync

# 5. Prepare Data Directories
echo "[5/6] Creating data directories..."
mkdir -p data/runs outputs

# 6. Start Services
echo "[6/6] Setup complete! You can now start the services."

echo "--------------------------------------------------"
echo "To start the project, run these in separate screens or tabs:"
echo "1. Start Redis:      service redis-server start"
echo "2. Start API:        uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
echo "3. Start Worker:     uv run celery -A backend.core.celery_app worker --loglevel=info -P solo"
echo "--------------------------------------------------"
