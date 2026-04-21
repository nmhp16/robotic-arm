#!/usr/bin/env bash
# Collect pick-and-place demos via a scripted oracle (no teleop keyboard needed).
#
# Drop-in substitute for teleop.sh — writes the same HDF5 format so mimic /
# convert / train all work unchanged.
#
# Flags:
#   --num-demos N      number of successful episodes to collect (default: 15)
#   --dataset-file F   output HDF5 (default: data/raw/demos.hdf5)
#   --max-steps N      step cap per episode (default: 400)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m arm_vla.datagen.oracle "$@"
