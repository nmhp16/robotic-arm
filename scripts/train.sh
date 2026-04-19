#!/usr/bin/env bash
# OpenVLA LoRA fine-tune on our RLDS dataset.
# Runs inside the training venv. Expect 6–12 h wall-clock on GB10 for 50k steps.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

if [[ ! -d "$REPO_ROOT/.venv" ]]; then
    echo "missing training venv at $REPO_ROOT/.venv — see README.md" >&2
    exit 1
fi

cd "$REPO_ROOT"
source .venv/bin/activate

# aarch64 / flash-attn: openvla's default config requests flash-attn; we
# override to sdpa via the training yaml. Expose as env var too in case any
# submodule reads it directly.
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa

exec python -m arm_vla.training.finetune_lora "$@"
