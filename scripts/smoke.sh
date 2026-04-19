#!/usr/bin/env bash
# Headless sanity check for the UR5e pick-and-place env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec "$ISAACLAB/isaaclab.sh" -p -m arm_vla.tasks.ur5_pick_place.smoke "$@"
