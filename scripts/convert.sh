#!/usr/bin/env bash
# Convert the mimic-augmented HDF5 dataset to RLDS (TFDS) for OpenVLA.
#
# Runs inside the training venv (needs tensorflow + tensorflow-datasets).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

INPUT="$REPO_ROOT/data/augmented/demos.hdf5"
OUTPUT="$REPO_ROOT/data/rlds"
INSTRUCTION="put the blue cube on the green target"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --instruction) INSTRUCTION="$2"; shift 2 ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$INPUT" ]]; then
    echo "missing: $INPUT (run ./scripts/mimic.sh first)" >&2
    exit 1
fi

if [[ ! -d "$REPO_ROOT/.venv" ]]; then
    echo "missing training venv at $REPO_ROOT/.venv — see README.md" >&2
    exit 1
fi

cd "$REPO_ROOT"
source .venv/bin/activate
exec python -m arm_vla.data.rlds_convert \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --instruction "$INSTRUCTION"
