#!/bin/bash
# Meta-launcher: 1 ns UMA-OMol (uma-s-1p1) NPT MD at 298 K for the 16 paper
# systems, one SLURM job per system (1 node / 1 GPU each). Purpose: get the
# 298 K equilibrium densities to sit alongside the existing 293 K teacher runs.
#
# Each job starts from a 298 K seed under $OUT_ROOT/<system>/<system>.traj (the
# first frame of the matching 293 K trajectory, re-drawn with 298 K
# Maxwell-Boltzmann velocities; built by
# src/simulate/fairchem_configs/make_uma_298K_seeds.py). The NPT barostat relaxes
# the cell to the model's equilibrium density.
#
# Reuses src/simulate/fairchem_configs/run_uma_sim.sh as the per-job SLURM
# script; parameters go in via --export. Requeue-safe: a resubmission resumes
# from the last frame of <system>.traj (uma_npt.py appends).
#
# Usage:
#   src/simulate/fairchem_configs/launch_uma_298K_density.sh            # submit all
#   src/simulate/fairchem_configs/launch_uma_298K_density.sh --dry-run  # print only
set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

REPO=/u/iamin/electrolytes
OUT_ROOT=/work/nvme/beye/iamin/electrolytes_data/UMA_298K_1ns
SLURM_SCRIPT=$REPO/src/simulate/fairchem_configs/run_uma_sim.sh
MODEL=/projects/beye/iamin/distillation_project/models/uma-s-1p1.pt

# Shared run settings
TEMPERATURE=298
STEPS=1000000          # 1 ns @ 1 fs
INTERVAL=1000          # 1 ps frame dump
TURBO=0                # fp32 inference; 1 enables fairchem turbo (tf32)
WALLTIME=24:00:00      # requeue-safe; auto-resubmits until the 1 ns is done

# The 16 paper systems (dir names under $OUT_ROOT). DEGDME=diglyme, TEGDME=tgdme.
SYSTEMS=(
  napf6_dme_0_1M   napf6_dme_1M
  napf6_pc_0_1M    napf6_pc_1M
  napf6_diglyme_0_1M  napf6_diglyme_1M
  napf6_tgdme_1M
  naotf_dme_0_1M   naotf_dme_1M
  naotf_pc_0_1M    naotf_pc_1M
  naotf_diglyme_0_1M  naotf_diglyme_1M
  naotf_tgdme_0_1M  naotf_tgdme_1M
  lipf6_dme_0_5M
)

cd "$REPO"
mkdir -p slurm_logs

if [ ! -f "$MODEL" ]; then
  echo "ERROR: UMA model not found -> $MODEL" >&2
  exit 1
fi

submitted=0
skipped=0

for sys in "${SYSTEMS[@]}"; do
  traj_dir="$OUT_ROOT/$sys"
  seed="$traj_dir/$sys.traj"
  job="uma298_${sys}"

  if [ ! -f "$seed" ]; then
    echo "SKIP  $job: seed traj not found -> $seed" >&2
    skipped=$((skipped + 1))
    continue
  fi

  cmd=(sbatch
    --time="$WALLTIME"
    --job-name="$job"
    --export="ALL,TRAJ_DIR=$traj_dir,MODEL=$MODEL,TEMPERATURE=$TEMPERATURE,STEPS=$STEPS,INTERVAL=$INTERVAL,TURBO=$TURBO"
    "$SLURM_SCRIPT")

  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    echo ">> $job  ($traj_dir)"
    "${cmd[@]}"
  fi
  submitted=$((submitted + 1))
done

echo
echo "Done. $submitted job(s) $( [ "$DRY_RUN" -eq 1 ] && echo 'would be submitted' || echo 'submitted'), $skipped skipped."
