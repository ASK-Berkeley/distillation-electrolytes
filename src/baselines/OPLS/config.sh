#!/bin/bash
# =============================================================================
# config.sh  --  configuration for the OPLS-AA/GROMACS baseline pipeline
# =============================================================================
# Every path lives here; nothing is hardcoded in the drivers. Edit the toolchain
# block once per machine. The Python drivers read the same CM_* / GMXBIN vars
# from the environment, so `source config.sh` is all a wrapper needs.
#
# Usage:   source config.sh      (done automatically by the run_*.sh wrappers)
# -----------------------------------------------------------------------------

export OPLS_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Toolchain (set these to YOUR installations) -----------------------------
# Python interpreter with: ase, numpy, networkx, matplotlib.
export OPLS_PYTHON="${OPLS_PYTHON:-python3}"

# GROMACS 2025.x install (the dir that contains bin/gmx and bin/GMXRC).
export GMXDIR="${GMXDIR:-$HOME/gromacs}"
export GMXBIN="${GMXBIN:-$GMXDIR/bin/gmx}"
export GMXRC="${GMXRC:-$GMXDIR/bin/GMXRC}"

# PACKMOL executable (only needed to (re)build boxes; conda-forge packmol).
export PACKMOL="${PACKMOL:-packmol}"

# Interactive GPU; under SLURM the allocation overrides via $CUDA_VISIBLE_DEVICES.
export CUDA_DEVICE="${CUDA_DEVICE:-0}"

# --- Input boxes -------------------------------------------------------------
# OPLS/GROMACS run dirs written by build_opls_boxes.py; each subdir holds
# <name>.gro + <name>.top + the .itp files it #includes.
export CM_BOX_ROOT="${CM_BOX_ROOT:-$OPLS_BASE/opls_ready_boxes}"

# --- Output ------------------------------------------------------------------
# Equilibration (em + npt):  <CM_EQ_ROOT>/<box>/{em,heat,npt}.*
export CM_EQ_ROOT="${CM_EQ_ROOT:-$OPLS_BASE/runs/equilibrated}"
# Production seeds picked out of the NPT trajectories by prepare_production.py:
# <CM_PROD_ROOT>/<box>/{seed.gro,<box>.top,*.itp,seed_meta.json}
export CM_PROD_ROOT="${CM_PROD_ROOT:-$OPLS_BASE/runs/production_ready}"
# Replica tree:  <CM_REP_ROOT>/<box>/r<k>/{prod.*,<box>_r<k>.stress}
export CM_REP_ROOT="${CM_REP_ROOT:-$OPLS_BASE/runs/production_replicas}"

# --- Equilibration protocol knobs (see run_equilibration.py --help) ----------
export CM_HEAT_PS="${CM_HEAT_PS:-200}"       # NVT heating ramp length, ps (only with --heat)
export CM_NPT_NS="${CM_NPT_NS:-1}"           # NPT relaxation length, ns
export CM_START_TEMP="${CM_START_TEMP:-10}"  # ramp start temperature, K

# --- Production protocol knobs (see run_production.py --help) -----------------
export CM_NREPLICAS="${CM_NREPLICAS:-4}"
export CM_PROD_NS="${CM_PROD_NS:-20}"        # NVT production length, ns
export CM_SEG_NS="${CM_SEG_NS:-2}"           # segment length (caps intermediate .edr)
export CM_STRESS_INTERVAL="${CM_STRESS_INTERVAL:-2}"   # steps -> 2 fs at dt=1 fs
export CM_TRAJ_INTERVAL="${CM_TRAJ_INTERVAL:-1000}"    # steps -> 1 ps
export CM_TAU_T="${CM_TAU_T:-10.0}"          # weak v-rescale coupling, ps
