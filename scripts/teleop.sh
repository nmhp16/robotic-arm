#!/usr/bin/env bash
# Record keyboard demos for a task. All flags forwarded to arm_act.cli.teleop.
#   ./scripts/teleop.sh                          # uses --task pick_plant_out
#   ./scripts/teleop.sh --task pick_plant_out --num-demos 25
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_act.cli.teleop "$@"
