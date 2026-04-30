#!/usr/bin/env bash
# Train ACT for a task. Reads defaults.yaml + tasks/<task>.yaml.
# Runs in the local training venv (not Isaac Lab's python).
#   ./scripts/train.sh                          # uses --task pick_plant_out
#   ./scripts/train.sh --task pick_plant_out --max-steps 1000
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

# Default venv path can be overridden via $ARM_ACT_VENV. Putting the venv
# on a fuse.sshfs mount has caused intermittent EPERM on this hardware
# during heavy concurrent imports (torch+vision stack); local NVMe is safer.
VENV_DIR="${ARM_ACT_VENV:-$HOME/arm-act-venv}"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "training venv not found at $VENV_DIR — run ./scripts/setup.sh first" >&2
    exit 1
fi

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"
exec python -u -m arm_act.training.train_act "$@"
