#!/bin/bash
# Launches the paper's production transport runs: for each of the 16 systems,
# NREPLICAS independent 20 ns NVT replicas at 298 K from that system's NPT-equilibrated
# seed frame, differing only in their random seed (initial velocities and thermostat
# noise). Each replica yields
#   * self-diffusivity     (MSD of molecular COMs, from the 1 ps trajectory)
#   * ionic conductivity   (Onsager collective displacement, from the trajectory)
#   * shear viscosity      (Green-Kubo stress ACF, from the 2 fs .stress log)
# The 2 fs charge-current log (.current) is also written, but no analysis in the
# paper reads it.
#
# Why NVT with a weak global Bussi thermostat (gamma = 0.1 THz, tau = 10 ps): NVE
# runs heat up from fp32 force noise, which spoils the V/(kB T) prefactor. tau is
# far shorter than that ns-scale drift but far longer than the sub-ps stress ACF,
# and Bussi rescales velocities by one global factor, so unlike Langevin it adds no
# per-atom friction to bias viscosity or conductivity.
#
# Each job is run_fennol_sim.sh, which requeues itself before the wall limit and
# resumes from .dyn.restart, so one submission runs a replica to completion.
#
# Usage:
#   src/simulate/launch_transport.sh              # submit 16 systems x NREPLICAS
#   src/simulate/launch_transport.sh --dry-run    # print the sbatch commands only
#
# Env overrides:
#   NREPLICAS (default 4)  REP_START (default 0): replicas r<REP_START>..r<NREPLICAS-1>,
#       seed = replica index + 1.
#   DATA_ROOT (default <repo>/data): runs go to $DATA_ROOT/experiments/exp04_correct_molarity_samebox.
#   STEPS (default 20000000 = 20 ns at 1 fs).  GAMMA_THZ (default 0.1).
#   STRESS_INTERVAL / CURRENT_INTERVAL (default 2 = every 2 fs).
#   WALLTIME (default 03:00:00) per requeue window.
set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT=${DATA_ROOT:-$REPO/data}
EXP_DIR=$DATA_ROOT/experiments/exp04_correct_molarity_samebox
SLURM_SCRIPT=$REPO/src/simulate/run_fennol_sim.sh

# Shared run settings
TEMPERATURE=298
ENSEMBLE=NVT               # thermostatted 298 K (no NVE heating)
THERMOSTAT=BUSSI           # gentle global velocity-rescale (CSVR)
GAMMA_THZ=${GAMMA_THZ:-0.1}       # weak coupling, tau=10 ps
STEPS=${STEPS:-20000000}          # 20 ns @ 1 fs
INTERVAL=1000              # trajectory frame every 1 ps (diffusivity + Onsager sigma)
STRESS_INTERVAL=${STRESS_INTERVAL:-2}    # stress dump every 2 fs (viscosity)
CURRENT_INTERVAL=${CURRENT_INTERVAL:-2}  # current dump every 2 fs (not used by the analysis)
MATMUL_PREC=highest        # fp32 (accurate stress)
WALLTIME=${WALLTIME:-03:00:00}
NREPLICAS=${NREPLICAS:-4}
REP_START=${REP_START:-0}
CKPT=best_model.fnx

# system : "system_id:conc_label"  (seed = equilibrated frame for that system/conc)
# All 16 exp04 correct-molarity systems, one conc per entry.
SYSTEMS=(
  "lipf6_dme:0_5M"
  "naotf_diglyme:0_1M"
  "naotf_diglyme:1M"
  "naotf_dme:0_1M"
  "naotf_dme:1M"
  "naotf_pc:0_1M"
  "naotf_pc:1M"
  "naotf_tgdme:0_1M"
  "naotf_tgdme:1M"
  "napf6_diglyme:0_1M"
  "napf6_diglyme:1M"
  "napf6_dme:0_1M"
  "napf6_dme:1M"
  "napf6_pc:0_1M"
  "napf6_pc:1M"
  "napf6_tgdme:1M"
)

cd "$REPO"
mkdir -p slurm_logs

submitted=0
skipped=0

# submit_one <sys> <conc> <seed> <run_name> <job_name>
submit_one() {
  local sys="$1" conc="$2" seed="$3" run_name="$4" job="$5"
  local model="$EXP_DIR/$sys/model/$CKPT"
  # Seed = the NPT-equilibrium frame selected by select_nvt_seeds.py (--layout flat):
  # the frame in the last 300 ps whose density is closest to the last-500 ps mean.
  # NVT holds this box.
  local xyz="$EXP_DIR/$sys/nvt_seed_${conc}_${TEMPERATURE}K.xyz"
  local run_dir="$EXP_DIR/$sys/$run_name"

  if [ ! -f "$model" ]; then
    echo "SKIP  $job: model not found -> $model" >&2; skipped=$((skipped + 1)); return
  fi
  if [ ! -f "$xyz" ] && [ ! -e "$run_dir" ]; then
    echo "SKIP  $job: seed box not found -> $xyz" >&2; skipped=$((skipped + 1)); return
  fi

  local cmd=(sbatch
    --time="$WALLTIME"
    --job-name="$job"
    --export="ALL,RUN_DIR=$run_dir,MODEL=$model,FROM_XYZ=$xyz,TEMPERATURE=$TEMPERATURE,STEPS=$STEPS,INTERVAL=$INTERVAL,ENSEMBLE=$ENSEMBLE,THERMOSTAT=$THERMOSTAT,GAMMA_THZ=$GAMMA_THZ,MATMUL_PREC=$MATMUL_PREC,SEED=$seed,STRESS_LOG=1,STRESS_INTERVAL=$STRESS_INTERVAL,CURRENT_LOG=1,CURRENT_INTERVAL=$CURRENT_INTERVAL,SYSTEM_ID=$sys"
    "$SLURM_SCRIPT")

  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    echo ">> $job  ($run_dir)"
    "${cmd[@]}"
  fi
  submitted=$((submitted + 1))
}

for entry in "${SYSTEMS[@]}"; do
  sys="${entry%%:*}"; conc="${entry#*:}"
  for rep in $(seq "$REP_START" $((NREPLICAS - 1))); do
    seed=$((rep + 1))
    run_name="transport_${conc}_${TEMPERATURE}K_r${rep}"
    submit_one "$sys" "$conc" "$seed" "$run_name" "transport_${sys}_${conc}_r${rep}"
  done
done

echo
echo "Done. $submitted job(s) $( [ "$DRY_RUN" -eq 1 ] && echo 'would be submitted' || echo 'submitted'), $skipped skipped."
