#!/usr/bin/env bash
# Annotate raw demos with mimic's datagen_info so isaaclab_mimic can use them.
#   ./scripts/annotate.sh                        # uses --task pick_place
#   ./scripts/annotate.sh --task pick_place --input ... --output ...
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_act.cli.annotate "$@"
