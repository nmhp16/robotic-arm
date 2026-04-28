#!/usr/bin/env bash
# Evaluate a trained ACT checkpoint via Isaac Lab sim rollouts.
#   ./scripts/eval.sh                           # uses --task pick_place
#   ./scripts/eval.sh --task pick_place --num-episodes 5 --no-video
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_vla.eval.rollout "$@"
