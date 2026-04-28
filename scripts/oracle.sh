#!/usr/bin/env bash
# Scripted oracle demo collection (no keyboard).
#   ./scripts/oracle.sh                          # uses --task pick_place
#   ./scripts/oracle.sh --task pick_place --num-demos 25
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_act.cli.oracle "$@"
