#!/usr/bin/env bash
# Offline action-prediction eval for a fine-tuned SmolVLA checkpoint.
# Loads the checkpoint, predicts actions on randomly sampled frames from
# the LeRobotDataset, and prints per-dimension L1 / RMSE errors.
#
# Pre-reqs: ./scripts/train.sh --task <task>  has produced a checkpoint.
#
# Usage:
#   ./scripts/smolvla_eval.sh --task pick_vial_from_holder
#   ./scripts/smolvla_eval.sh --task pick_vial_from_holder --num-frames 500
#   ./scripts/smolvla_eval.sh --task pick_vial_from_holder \
#       --checkpoint checkpoints/<task>/smolvla/checkpoints/005000/pretrained_model
#
# This is OFFLINE — it does not run the policy in Isaac Lab. For closed-
# loop sim rollout you need a SmolVLA-aware version of eval/rollout.py
# which is a follow-up (cross-venv split: lerobot is in the training
# venv, Isaac Lab uses its bundled python).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

VENV_DIR="${ARM_ACT_VENV:-$HOME/arm-act-venv}"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "training venv not found at $VENV_DIR" >&2
    exit 1
fi
cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

if ! python -c "import lerobot" 2>/dev/null; then
    echo "lerobot not installed in training venv. Run: pip install -e \".[smolvla]\"" >&2
    exit 1
fi

exec python -u -m arm_act.eval.smolvla_offline "$@"
