#!/usr/bin/env bash
# One-shot environment bootstrap.
#
# Creates the training venv, installs torch + the project's runtime deps,
# installs the project into Isaac Lab's python (for sim/eval), and runs
# the URDF->USD conversion. Intended to be re-runnable; existing artifacts
# are reused.
#
# Env overrides:
#   ARM_VLA_VENV   training venv path (default: $HOME/arm-vla-venv)
#   ISAACLAB       Isaac Lab install (default: $HOME/IsaacLab)
#   PYTHON         system python (default: python3.12 if available, else python3)
#   SKIP_USD       set to skip URDF->USD conversion (e.g. if already done)
#   SKIP_ISAAC     set to skip pip install into Isaac Lab's python
#   SKIP_TORCH     set to skip torch install (e.g. on systems without NVIDIA wheels)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

ARM_VLA_VENV="${ARM_VLA_VENV:-$HOME/arm-vla-venv}"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
PYTHON="${PYTHON:-$(command -v python3.12 || command -v python3)}"

log() { printf '[setup] %s\n' "$*"; }
warn() { printf '[setup] WARN: %s\n' "$*" >&2; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -n "${PYTHON:-}" ]] || die "no python3 on PATH; install Python 3.10+ first"
log "using PYTHON=$PYTHON ($($PYTHON --version 2>&1))"

# ---------- 1. Training venv ----------------------------------------------
if [[ -d "$ARM_VLA_VENV" ]]; then
    log "training venv exists at $ARM_VLA_VENV"
else
    log "creating training venv at $ARM_VLA_VENV"
    "$PYTHON" -m venv "$ARM_VLA_VENV"
fi
# shellcheck disable=SC1091
source "$ARM_VLA_VENV/bin/activate"
pip install --upgrade pip wheel >/dev/null

if [[ -z "${SKIP_TORCH:-}" ]]; then
    if python -c "import torch, torchvision" >/dev/null 2>&1; then
        log "torch already installed: $(python -c 'import torch; print(torch.__version__)')"
    else
        log "installing torch + torchvision (NVIDIA cu128 wheels)"
        # cu128 wheels include aarch64 builds; on x86_64 they install too.
        # Override with SKIP_TORCH=1 if you need a different CUDA / CPU build.
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    fi
fi

log "installing arm-vla[ml] into training venv"
pip install -e "$REPO_ROOT[ml]"

# ---------- 2. Isaac Lab pip install --------------------------------------
if [[ -z "${SKIP_ISAAC:-}" ]]; then
    if [[ ! -x "$ISAACLAB/isaaclab.sh" ]]; then
        warn "Isaac Lab not found at $ISAACLAB — set ISAACLAB=... and re-run, or use SKIP_ISAAC=1"
    else
        log "installing arm-vla[sim] into Isaac Lab's python ($ISAACLAB)"
        env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p -m pip install -e "$REPO_ROOT[sim]"
    fi
fi

# ---------- 3. URDF -> USD conversion -------------------------------------
if [[ -z "${SKIP_USD:-}" ]]; then
    if [[ ! -x "$ISAACLAB/isaaclab.sh" ]]; then
        warn "skipping USD conversion (Isaac Lab not found)"
    else
        # Run conversions only when the USD looks like a placeholder (<10 KB)
        # or is missing. The URDF importer is heavy and we want this script
        # to be idempotent.
        for variant in ur5_simple_gripper ur5_2f85; do
            usd="$REPO_ROOT/assets/$variant/$variant.usd"
            script="$REPO_ROOT/scripts/convert_$variant.py"
            if [[ -f "$usd" ]] && [[ "$(stat -c %s "$usd" 2>/dev/null || echo 0)" -gt 10240 ]]; then
                log "USD ok: $usd"
                continue
            fi
            if [[ ! -f "$script" ]]; then
                warn "no converter script for $variant"
                continue
            fi
            log "converting $variant URDF -> USD"
            env -u VIRTUAL_ENV -u CONDA_PREFIX "$ISAACLAB/isaaclab.sh" -p "$script"
        done
    fi
fi

log "done."
log ""
log "next steps:"
log "  ./scripts/teleop.sh --task tasks/pick_blue_on_green.yaml --num-demos 15"
log "  ./scripts/mimic.sh  --task tasks/pick_blue_on_green.yaml --num-demos 500"
log "  ./scripts/train.sh  --task tasks/pick_blue_on_green.yaml"
log "  ./scripts/eval.sh   --task tasks/pick_blue_on_green.yaml --checkpoint <ckpt>"
