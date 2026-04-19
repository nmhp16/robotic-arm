#!/usr/bin/env bash
# Record keyboard teleop demonstrations to data/raw/demos.hdf5.
#
# Wraps Isaac Lab's scripts/tools/record_demos.py with task defaults.
#
# Keyboard (Se3Keyboard):
#   W/S   +x / -x     A/D  +y / -y     Q/E  +z / -z
#   Z/X   roll        T/G  pitch       C/V  yaw
#   K     toggle gripper (open/close)
#   R     reset and discard current episode
#
# Flags:
#   --num-demos N   stop after N successful demos (default: 15)
#   --step-hz N     sim step rate (default: 30)
#   --dataset F     output file  (default: data/raw/demos.hdf5)
#
# The env's `cube_on_target` termination auto-advances the episode after
# ~10 consecutive success steps (~0.3 s at 30 Hz).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

NUM_DEMOS=15
STEP_HZ=30
DATASET="$REPO_ROOT/data/raw/demos.hdf5"
TASK="Isaac-PickPlace-UR5-IK-Rel-v0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-demos) NUM_DEMOS="$2"; shift 2 ;;
        --step-hz)   STEP_HZ="$2"; shift 2 ;;
        --dataset)   DATASET="$2"; shift 2 ;;
        --task)      TASK="$2"; shift 2 ;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$DATASET")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# Pre-import the task package so its gym.register side-effect runs before
# record_demos.py looks up the id.
exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -c "
import arm_vla.tasks.ur5_pick_place
import runpy, sys
sys.argv = [
    'record_demos.py',
    '--task', '$TASK',
    '--teleop_device', 'keyboard',
    '--dataset_file', '$DATASET',
    '--step_hz', '$STEP_HZ',
    '--num_demos', '$NUM_DEMOS',
    '--enable_cameras',
]
runpy.run_path('$ISAACLAB/scripts/tools/record_demos.py', run_name='__main__')
"
