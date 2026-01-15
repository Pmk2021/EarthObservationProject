#!/usr/bin/env bash
set -e

# Move to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="ipeo_env"

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# IMPORTANT: run as module
python -m Dino.train

conda deactivate
