#!/usr/bin/env bash
# Drive the UR10 pick-and-place env with Claude Opus 4.7 via the claude CLI.
# Demo only — Opus is a language model, not a VLA, so don't expect successful
# grasps. Useful as a baseline comparison against a fine-tuned OpenVLA.
#
# Requires `claude login` (uses Claude Code auth, no API key needed).
#
# Flags:
#   --steps N       number of sim steps (default: 50)
#   --out PATH      output mp4 (default: media/opus_rollout.mp4)
#   --fps N         video fps (default: 4)
#   --instruction S task instruction (default: pick+place blue cube)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_vla.eval.opus_agent "$@"
