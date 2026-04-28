#!/usr/bin/env bash
# Evaluate a trained ACT checkpoint in the Isaac Lab sim env.
#
# Flags:
#   --checkpoint PATH   path to checkpoint dir (e.g. checkpoints/act-ur5-pickplace/final)
#   --num-episodes N    default 20
#   --no-video          skip video recording (faster)
#   --action-horizon N  steps between re-plans (default: chunk_size = full open-loop chunk)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_vla.eval.rollout "$@"
