#!/usr/bin/env bash
# OpenVLA LoRA fine-tune on the RLDS dataset under data/rlds/.
# Runs in the training venv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

if [[ ! -d "$REPO_ROOT/.venv" ]]; then
    echo "training venv not found at $REPO_ROOT/.venv — see README.md" >&2
    exit 1
fi

cd "$REPO_ROOT"
source .venv/bin/activate
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa

exec python -m arm_vla.training.finetune_lora "$@"
