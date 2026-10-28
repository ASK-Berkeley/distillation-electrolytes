#!/bin/bash
# Meta-launcher: 1 ns NPT equilibration of the exp04 correct-molarity boxes.
#
# Each system/concentration starts from its UMA-relaxed packed box
# (exp04_correct_molarity_samebox/<sys>/initial_relaxed_<conc>.xyz) and runs 1 ns of NPT
# at 298 K / 1 bar with the system's own PaiNN model. The barostat relaxes the
# cell to the model's equilibrium density -- THIS is the step that sets the true
# density and thereby locks in the target molarity (the boxes were built so that
# at that density the molarity is correct). Output is the equilibrated seed for
# the downstream transport runs.
#
# NPT = LGV thermostat + LGV Langevin-piston barostat (isotropic, 1 bar), set up
# by fennol_npt.py for --ensemble NPT. fp32 (MATMUL_PREC=highest).
#
# Reuses src/simulate/run_fennol_sim.sh as the per-job SLURM script; parameters
# go in via --export (ENSEMBLE=NPT, FROM_XYZ=<relaxed box>, STEPS, ...).
# Requeue-safe: a resubmission resumes from <system>.dyn.restart.
#
# Usage:
#   src/simulate/fennol_configs/launch_npt_equilibration.sh            # submit all
#   src/simulate/fennol_configs/launch_npt_equilibration.sh --dry-run  # print only
set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT=${DATA_ROOT:-$REPO/data}
EXP_DIR=$DATA_ROOT/experiments/exp04_correct_molarity_samebox
SLURM_SCRIPT=$REPO/src/simulate/run_fennol_sim.sh

# Shared run settings
TEMPERATURE=298
ENSEMBLE=NPT
STEPS=1000000          # 1 ns @ 1 fs
INTERVAL=1000          # 1 ps frame dump
MATMUL_PREC=highest    # fp32
WALLTIME=3:00:00      # generous; boxes are 2.7k-3.7k atoms. Requeue-safe anyway.
SEED=1
CKPT=best_model.fnx

# system : comma-separated concentration labels (match initial_relaxed_<conc>.xyz)
SYSTEMS=(
  "lipf6_dme:0_5M"
  "naotf_diglyme:0_1M,1M"
  "naotf_dme:0_1M,1M"
  "naotf_pc:0_1M,1M"
  "naotf_tgdme:0_1M,1M"
  "napf6_diglyme:0_1M,1M"
  "napf6_dme:0_1M,1M"
  "napf6_pc:0_1M,1M"
  "napf6_tgdme:1M"
)

cd "$REPO"
mkdir -p slurm_logs

submitted=0
skipped=0

for entry in "${SYSTEMS[@]}"; do
  sys="${entry%%:*}"
  concs="${entry#*:}"
  model="$EXP_DIR/$sys/model/$CKPT"

  if [ ! -f "$model" ]; then
    echo "SKIP  $sys: model not found -> $model" >&2
    skipped=$((skipped + 1))
    continue
  fi

  IFS=',' read -ra conc_arr <<< "$concs"
  for conc in "${conc_arr[@]}"; do
    xyz="$EXP_DIR/$sys/initial_relaxed_${conc}.xyz"
    run_name="npt_equil_${conc}_${TEMPERATURE}K_1ns"
    run_dir="$EXP_DIR/$sys/$run_name"
    job="nptequil_${sys}_${conc}"

    # Fresh run needs the relaxed seed; a resume (run_dir exists) does not.
    if [ ! -f "$xyz" ] && [ ! -e "$run_dir" ]; then
      echo "SKIP  $job: relaxed box not found -> $xyz" >&2
      skipped=$((skipped + 1))
      continue
    fi

    cmd=(sbatch
      --time="$WALLTIME"
      --job-name="$job"
      --export="ALL,RUN_DIR=$run_dir,MODEL=$model,FROM_XYZ=$xyz,TEMPERATURE=$TEMPERATURE,STEPS=$STEPS,INTERVAL=$INTERVAL,ENSEMBLE=$ENSEMBLE,MATMUL_PREC=$MATMUL_PREC,SEED=$SEED"
      "$SLURM_SCRIPT")

    if [ "$DRY_RUN" -eq 1 ]; then
      printf '%q ' "${cmd[@]}"; echo
    else
      echo ">> $job  ($run_dir)"
      "${cmd[@]}"
    fi
    submitted=$((submitted + 1))
  done
done

echo
echo "Done. $submitted job(s) $( [ "$DRY_RUN" -eq 1 ] && echo 'would be submitted' || echo 'submitted'), $skipped skipped."
