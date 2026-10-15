#!/bin/bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

# Set other environment variables
export USE_GPU=true
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=src/models
export VOICES_DIR=src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web

export DEVICE_TYPE=mps
# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run FastAPI with GPU extras using uv run
if [ ! -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    uv sync --extra cpu
fi
# Install new core requirements without removing existing extras.
uv pip install --python "$PROJECT_ROOT/.venv/bin/python" -e .
uv run --no-sync python -c 'from api.src.core.espeak import installed_languages; assert installed_languages(), "Bundled eSpeak library/data unavailable"'
uv run --no-sync python -m api.src.core.japanese
uv run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0
exec uv run --no-sync uvicorn api.src.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8880}"
