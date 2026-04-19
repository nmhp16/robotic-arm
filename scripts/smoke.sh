#!/usr/bin/env bash
# Smoke test the UR5e pick-and-place env under Isaac Lab's bundled python.
# Boots headless, steps the env for ~1 s, prints obs shapes, exits.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

cd "$REPO_ROOT"

# Point PYTHONPATH at our src/ so the bundled python finds arm_vla without
# requiring an editable install into Isaac Lab's env (we may not want to
# pollute it yet).
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec "$ISAACLAB/isaaclab.sh" -p -m arm_vla.tasks.ur5_pick_place.smoke "$@"
