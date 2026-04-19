#!/usr/bin/env bash
# Evaluate a fine-tuned OpenVLA checkpoint in the Isaac Lab sim env.
#
# Flags:
#   --checkpoint PATH   path to LoRA checkpoint dir (required)
#   --num-episodes N    default 50
#   --no-video          skip video recording (faster)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_vla.eval.rollout "$@"
