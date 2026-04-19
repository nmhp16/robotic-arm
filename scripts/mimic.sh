#!/usr/bin/env bash
# Augment teleop demos via isaaclab_mimic.
#
# Reads data/raw/demos.hdf5 (from teleop.sh), runs curobo-based segment
# replay across randomized scenes, and writes data/augmented/demos.hdf5.
#
# Flags:
#   --num-demos N   target number of generated demos (default: 500)
#   --input F       source dataset (default: data/raw/demos.hdf5)
#   --output F      output dataset (default: data/augmented/demos.hdf5)
#   --num-envs N    parallel envs during generation (default: 4)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

NUM_DEMOS=500
INPUT="$REPO_ROOT/data/raw/demos.hdf5"
OUTPUT="$REPO_ROOT/data/augmented/demos.hdf5"
NUM_ENVS=4
TASK="Isaac-PickPlace-UR5-IK-Rel-Mimic-v0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-demos) NUM_DEMOS="$2"; shift 2 ;;
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$INPUT" ]]; then
    echo "missing input dataset: $INPUT (run ./scripts/teleop.sh first)" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# Pre-import our datagen package to register the mimic gym id before
# generate_dataset.py looks it up.
exec "$ISAACLAB/isaaclab.sh" -p -c "
import arm_vla.datagen  # registers Isaac-PickPlace-UR5-IK-Rel-Mimic-v0
import runpy
import sys
sys.argv = [
    'generate_dataset.py',
    '--task', '$TASK',
    '--input_file', '$INPUT',
    '--output_file', '$OUTPUT',
    '--generation_num_trials', '$NUM_DEMOS',
    '--num_envs', '$NUM_ENVS',
    '--enable_cameras',
]
runpy.run_path('$ISAACLAB/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py', run_name='__main__')
"
