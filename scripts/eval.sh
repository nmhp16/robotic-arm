#!/usr/bin/env bash
# Evaluate a fine-tuned OpenVLA checkpoint in the Isaac Lab sim env.
#
# Runs inside Isaac Lab's bundled python. Requires transformers + peft + PIL
# + imageio to be installed there:
#
#     ~/IsaacLab/isaaclab.sh -p -m pip install -e ".[sim]"
#
# Flags:
#   --checkpoint PATH       path to LoRA checkpoint dir (required)
#   --num-episodes N        default 50
#   --no-video              skip video recording (faster)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec "$ISAACLAB/isaaclab.sh" -p -m arm_vla.eval.rollout "$@"
