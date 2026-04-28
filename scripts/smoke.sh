#!/usr/bin/env bash
# Scene preview / sanity check for a task's env.
#   ./scripts/smoke.sh                           # headless, --task pick_place
#   ./scripts/smoke.sh --task pick_place --visible --random --steps 200
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_vla.cli.smoke "$@"
