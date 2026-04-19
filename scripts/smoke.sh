#!/usr/bin/env bash
# Sanity check for the UR5e pick-and-place env.
#
# Flags (forwarded to the Python module):
#   --visible         open the Isaac Sim GUI
#   --random          drive the arm with small random Delta-pose actions
#   --steps N         number of sim steps (default: 20)
#   --dump-cams DIR   write one frame per RGB camera to DIR
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# isaaclab.sh defers to $VIRTUAL_ENV/$CONDA_PREFIX if set; unset both so it
# falls back to the bundled Isaac Sim Python.
exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_vla.tasks.ur5_pick_place.smoke "$@"
