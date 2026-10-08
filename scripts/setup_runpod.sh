#!/usr/bin/env bash
# One-shot setup for a fresh RunPod PyTorch pod.
#
#   bash scripts/setup_runpod.sh
#
# Safe to re-run. Installs into a venv at .venv so the pod's system Python is untouched.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "==> repo: $ROOT"

# ---------------------------------------------------------------- GPU sanity check
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! nvidia-smi not found. Is this really a GPU pod?" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

# ------------------------------------------------------------------------ venv
if [ ! -d .venv ]; then
  echo "==> creating .venv"
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# --------------------------------------------------------------- dependencies
# Unsloth pins its own transformers/peft/trl/bitsandbytes. If this pod's CUDA/PyTorch
# combination is unusual, let Unsloth choose the build tag instead:
#   wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
echo "==> installing requirements"
pip install -r requirements.txt

# --------------------------------------------------------------------- config
if [ ! -f .env ]; then
  echo "==> creating .env from .env.mock (fill in HF_TOKEN if you need gated repos)"
  cp .env.mock .env
fi

# RunPod convention: keep artifacts on the network volume so they survive a restart.
if [ -d /workspace ] && [ -w /workspace ]; then
  mkdir -p /workspace/outputs
  echo "==> outputs will default to /workspace/outputs"
fi

echo
echo "==> done. Next:"
echo "    source .venv/bin/activate"
echo "    python run.py --print-config"
echo "    python run.py"
echo
echo "    IMPORTANT: download $ROOT/outputs (or /workspace/outputs) before stopping the pod."
