#!/usr/bin/env bash
# Run pretrained OpenVLA-7B on our tasks (no fine-tuning).
#
# Flags:
#   --task pick_place|stack  which task env (required)
#   --episodes N             number of rollouts (default: 3)
#   --max-steps N            per-episode step cap (default: 200)
#   --unnorm-key K           action unnorm key (default: bridge_orig)
#   --out PATH               output mp4 (default: media/openvla_zeroshot_<task>.mp4)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_vla.eval.zeroshot "$@"
