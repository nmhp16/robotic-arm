#!/usr/bin/env bash
# Augment demos via isaaclab_mimic (curobo-based segment replay).
#   ./scripts/mimic.sh                           # uses --task pick_place
#   ./scripts/mimic.sh --task pick_place --num-demos 1000 --num-envs 8
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_vla.cli.mimic "$@"
