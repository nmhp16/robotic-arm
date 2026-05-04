#!/usr/bin/env bash
# DAgger collector — run trained policy in env, relabel with stateless oracle.
#   ./scripts/dagger.sh                            # uses --task pick_plant_out
#   ./scripts/dagger.sh --task <task> --num-episodes 100 --output-suffix iter1
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_act.cli.dagger "$@"
