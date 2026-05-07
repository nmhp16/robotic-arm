#!/usr/bin/env bash
# Default training path: fine-tune SmolVLA-450M on a LeRobotDataset.
#
# Why SmolVLA is the default (over the vendored ACT in train_act.sh):
#   - language-conditioned (uses task.instruction from the YAML)
#   - pretrained on the LeRobot Hub corpus, so cross-slot/cross-object
#     generalization comes ~for free relative to ACT-from-scratch
#   - actually-maintained training code via `lerobot-train`
#
# Pre-reqs (run once per task):
#   ./scripts/setup.sh                                  # creates the training venv
#   pip install -e ".[smolvla]"                         # adds lerobot to that venv
#   ./scripts/smolvla_convert.sh --task <task>          # hdf5 -> LeRobotDataset
#
# Usage:
#   ./scripts/train.sh --task pick_vial_from_holder
#   ./scripts/train.sh --task pick_vial_from_holder --steps 50000 --batch-size 8
#
# Hyperparameters can be overridden by appending lerobot-train flags after
# the wrapper's own flags. See `lerobot-train --help` for the full list.
#
# To train the older vendored ACT instead, use ./scripts/train_act.sh.
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

# --- Wrapper-level args we extract before forwarding --------------------------
TASK="pick_vial_from_holder"
STEPS=20000
BATCH_SIZE=16
LR=1.0e-5
EVAL_FREQ=5000
SAVE_FREQ=5000
LOG_FREQ=100
NUM_WORKERS=4
PRETRAINED="lerobot/smolvla_base"

REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr|--learning-rate) LR="$2"; shift 2 ;;
        --eval-freq) EVAL_FREQ="$2"; shift 2 ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --log-freq) LOG_FREQ="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --pretrained) PRETRAINED="$2"; shift 2 ;;
        *) REMAINING_ARGS+=("$1"); shift ;;
    esac
done

DATASET_DIR="$REPO_ROOT/data/lerobot/local/$TASK"
OUTPUT_DIR="$REPO_ROOT/checkpoints/$TASK/smolvla"

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "converted dataset not found at $DATASET_DIR" >&2
    echo "run ./scripts/smolvla_convert.sh --task $TASK first" >&2
    exit 1
fi

# lerobot-train refuses to overwrite an existing non-empty output_dir; it
# wants the dir to either not exist or contain a checkpoint to resume.
# Don't pre-create here — let lerobot create it on first save. Allow
# the user to pass --resume=true through REMAINING_ARGS to continue a
# previous run; in that case skip our existence check.
_resume_requested=false
for arg in "${REMAINING_ARGS[@]}"; do
    if [[ "$arg" == "--resume=true" ]] || [[ "$arg" == "--resume" ]]; then
        _resume_requested=true
        break
    fi
done
if [[ "$_resume_requested" == false ]] \
    && [[ -d "$OUTPUT_DIR" ]] \
    && [[ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]]; then
    echo "output dir $OUTPUT_DIR already has contents; rename/remove or pass --resume=true to continue a previous run." >&2
    exit 1
fi

echo "== SmolVLA fine-tune =="
echo "  task       : $TASK"
echo "  dataset    : $DATASET_DIR"
echo "  output     : $OUTPUT_DIR"
echo "  pretrained : $PRETRAINED"
echo "  steps      : $STEPS  batch=$BATCH_SIZE  lr=$LR"
echo

# LeRobot installs a `lerobot-train` console script. Flag schema is
# draccus-style nested config: `--policy.type=smolvla`, top-level
# `--steps`, etc. Verified against lerobot 0.5.1.
exec lerobot-train \
    --policy.type=smolvla \
    --policy.pretrained_path="$PRETRAINED" \
    --policy.repo_id="local/${TASK}_smolvla" \
    --policy.push_to_hub=false \
    --dataset.repo_id="local/$TASK" \
    --dataset.root="$DATASET_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --num_workers="$NUM_WORKERS" \
    --log_freq="$LOG_FREQ" \
    --save_freq="$SAVE_FREQ" \
    --eval_freq="$EVAL_FREQ" \
    --policy.optimizer_lr="$LR" \
    --wandb.enable=false \
    "${REMAINING_ARGS[@]}"
