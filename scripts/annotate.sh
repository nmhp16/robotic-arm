#!/usr/bin/env bash
# Annotate raw demos with mimic's datagen_info so isaaclab_mimic can use them.
# Mimic's generate_dataset.py refuses raw teleop/oracle demos ("Episode lacks
# datagen_info annotations") — annotate_demos.py plays each demo back in the
# env with --auto to compute the required DatagenInfo fields (eef_pose,
# object_pose, target_eef_pose, subtask_term_signals).
#
# Flags:
#   --input F    raw demos (default: data/raw/demos.hdf5)
#   --output F   annotated demos (default: data/annotated/demos.hdf5)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"

INPUT="$REPO_ROOT/data/raw/demos.hdf5"
OUTPUT="$REPO_ROOT/data/annotated/demos.hdf5"
TASK="Isaac-PickPlace-UR5-IK-Rel-Mimic-v0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)   INPUT="$2"; shift 2 ;;
        --output)  OUTPUT="$2"; shift 2 ;;
        --task)    TASK="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$INPUT" ]]; then
    echo "input not found: $INPUT (run ./scripts/oracle.sh first)" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

exec env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -c "
import arm_vla.datagen  # registers the mimic gym id
import runpy, sys
sys.argv = [
    'annotate_demos.py',
    '--task', '$TASK',
    '--input_file', '$INPUT',
    '--output_file', '$OUTPUT',
    '--auto',
    '--headless',
    '--enable_cameras',
]
runpy.run_path('$ISAACLAB/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py', run_name='__main__')
"
