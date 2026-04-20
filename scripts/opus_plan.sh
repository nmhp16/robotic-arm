#!/usr/bin/env bash
# Closed-loop Opus planner: plan → execute primitives → check success →
# replan on failure. Demo that uses Opus for high-level task planning, not
# per-step motor control.
#
# Flags:
#   --max-attempts N   replan up to N times on failure (default: 3)
#   --out PATH         output mp4 (default: media/opus_planner.mp4)
#   --fps N            video fps (default: 20)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_vla.eval.opus_planner "$@"
