#!/bin/bash
# Launches the exp01 student NPT runs: 20 ns at 298 K and 1 bar for every
# system/concentration, 4 replicas each (64 jobs). All replicas of a box start from
# the same teacher frame and differ only in their FeNNol --seed (initial velocities
# and thermostat/barostat noise).
#
# These runs give the student densities (density_replicas.py, Fig. 2b) and, from
# replica 0, the frames whose density drives the molarity correction
# (select_nvt_seeds.py, README section 5.1).
#
# Per job:
#   - model    : exp01 best_model.fnx for that system
#   - seed box : datasets/version01_mixed_noise/<sys>/simulation_start_<conc>.xyz
#   - output   : exp01/<sys>/npt_<conc>_298K_20ns_1ps_replicas/replica_<k>/
#                npt_<conc>_298K_20ns_1ps.{traj,...}  (the layout the analysis reads)
#   - 20 ns @ 1 fs, 1 ps frames, fp32; requeue-safe (resumes from .dyn.restart)
#
# Each job is run_fennol_sim.sh; everything is passed in via --export.
#
# Usage:
#   src/simulate/launch_exp01_production.sh            # submit all jobs
#   src/simulate/launch_exp01_production.sh --dry-run  # print sbatch commands only
#
set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT=${DATA_ROOT:-$REPO/data}
EXP_DIR=$DATA_ROOT/experiments/exp01_mixed_noise_all_systems 
DATASET_DIR=$DATA_ROOT/datasets/version01_mixed_noise
SLURM_SCRIPT=$REPO/src/simulate/run_fennol_sim.sh

# Shared run settings
TEMPERATURE=298
ENSEMBLE=NPT
STEPS=20000000      # 20 ns @ 1 fs
INTERVAL=1000       # 1 ps  @ 1 fs
MATMUL_PREC=highest # fp32
WALLTIME=30:00:00
NREPLICAS=4
CKPT=best_model.fnx

# system : comma-separated concentration labels (match simulation_start_<label>.xyz)
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
    xyz="$DATASET_DIR/$sys/simulation_start_${conc}.xyz"
    if [ ! -f "$xyz" ]; then
      echo "SKIP  $sys $conc: seed box not found -> $xyz" >&2
      skipped=$((skipped + 1))
      continue
    fi

    for rep in $(seq 0 $((NREPLICAS - 1))); do
      seed=$((rep + 1))                      # distinct per replica
      run_name="npt_${conc}_${TEMPERATURE}K_20ns_1ps"
      run_dir="$EXP_DIR/$sys/${run_name}_replicas/replica_${rep}"
      job="npt_${sys}_${conc}_r${rep}"

      cmd=(sbatch
        --time="$WALLTIME"
        --job-name="$job"
        --export="ALL,RUN_DIR=$run_dir,MODEL=$model,FROM_XYZ=$xyz,TEMPERATURE=$TEMPERATURE,STEPS=$STEPS,INTERVAL=$INTERVAL,ENSEMBLE=$ENSEMBLE,MATMUL_PREC=$MATMUL_PREC,SEED=$seed,SYSTEM_NAME=$run_name"
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
done

echo
echo "Done. $submitted job(s) $( [ "$DRY_RUN" -eq 1 ] && echo 'would be submitted' || echo 'submitted'), $skipped skipped."
