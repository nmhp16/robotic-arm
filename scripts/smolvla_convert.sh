#!/usr/bin/env bash
# Convert oracle hdf5 demos for a task into LeRobotDataset format.
# Output: data/lerobot/local/<task>/ (gitignored under /data/).
#
#   ./scripts/smolvla_convert.sh                                    # uses DEFAULT_TASK
#   ./scripts/smolvla_convert.sh --task pick_vial_from_holder
#   ./scripts/smolvla_convert.sh --task pick_vial_from_holder --overwrite
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

# Same venv convention as scripts/train.sh — must have ``lerobot`` installed.
VENV_DIR="${ARM_ACT_VENV:-$HOME/arm-act-venv}"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "training venv not found at $VENV_DIR — run ./scripts/setup.sh first" >&2
    exit 1
fi

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

# Sanity check: lerobot must be importable for the converter to work.
if ! python -c "import lerobot" 2>/dev/null; then
    cat >&2 <<'EOF'
lerobot is not installed in the training venv. Install with:

    pip install lerobot

(or "pip install lerobot[smolvla]" if you want the SmolVLA extras up front).
EOF
    exit 1
fi

exec python -u -m arm_act.cli.smolvla_convert "$@"
