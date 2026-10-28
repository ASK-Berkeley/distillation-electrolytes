#!/bin/bash
# Meta-launcher: 4 extra replicas of yuejian's 1 ns / 293 K NPT PaiNN runs, so the
# RDF-vs-UMA comparison has 5 replicas per system (his r0 + our r1..r4).
#
# 8 systems x 4 replicas = 32 runs = 8 Perlmutter GPU nodes (4 A100s each).
# lipf6_dme is deliberately excluded.
#
# Every replica of a system starts from the SAME frame — the `start.xyz` of the
# existing r0 run — and differs only in FeNNol's --seed, which sets the initial
# Maxwell-Boltzmann velocities and the thermostat/barostat noise stream.
#
# Run settings are copied from the r0 `<run>.fennol.yaml` so the replicas are
# directly comparable to it:
#   NPT, 293 K, LGV thermostat (gamma 1 THz) + LGV piston (tau 200 fs, 20 THz),
#   1.01325 atm, dt 1 fs, 1e6 steps (1 ns), tdump 0.1 ps (--interval 100),
#   matmul_prec=highest (fp32), nblist_skin 2.0, traj_format ase.
#
# The seed frames are READ from yuejian's original tree (SRC_ROOT, read-only to
# us) and the replicas are WRITTEN to a scratch mirror (OUT_ROOT) that keeps the
# same tail layout, so once he grants group-write the runs drop straight in:
#
#   SRC_ROOT/<system>/npt_1M_293K_1ns_100fs/start.xyz   <- read (his r0 seed)
#   OUT_ROOT/<system>/npt_1M_293K_1ns_100fs_r{1..4}/    <- written here
#
# To move them over once he has run  `D=<SRC_ROOT>; chmod g+ws "$D" "$D"/*`:
#   rsync -av --info=progress2 <OUT_ROOT>/ <SRC_ROOT>/
# (/pscratch is purged after ~30 days of no access — copy before then.)
#
# Usage:
#   src/simulate/launch_painn_1ns_replicas.sh --dry-run   # print sbatch commands
#   src/simulate/launch_painn_1ns_replicas.sh             # submit all 8 node jobs
#   QOS=regular src/simulate/launch_painn_1ns_replicas.sh # half the charge, lower priority
#
#   # only some systems (premium caps submitted jobs at 5/user, so the tail of a
#   # full submit can be re-sent under regular):
#   QOS=regular src/simulate/launch_painn_1ns_replicas.sh napf6_dme napf6_pc
#
set -euo pipefail

DRY_RUN=0
WANTED=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -*) echo "Unknown flag: $arg" >&2; exit 1 ;;
    *) WANTED+=("$arg") ;;
  esac
done

REPO=/global/homes/i/ishan_a/electrolytes
CFS=/global/cfs/cdirs/m5250/distillation_proj
EXP_DIR=$CFS/electrolytes_data/experiments/exp01_mixed_noise_all_systems
# Read-only source of the r0 seed frames (yuejian's tree).
SRC_ROOT=$CFS/simulation_results/PAINN/electrolytes_data/simulation/FP32_simulation/In_distribution_exp/NPT_langevin/npt_1ns_293K_for_compare_rdf_with_UMA
# Scratch mirror we write into; same tail layout as SRC_ROOT for a plain rsync later.
OUT_ROOT=/pscratch/sd/i/ishan_a/electrolytes_data/simulation/FP32_simulation/In_distribution_exp/NPT_langevin/npt_1ns_293K_for_compare_rdf_with_UMA
SLURM_SCRIPT=$REPO/src/simulate/run_painn_replicas_nersc.sh

# Run settings — match the existing r0 runs exactly.
RUN_BASE=npt_1M_293K_1ns_100fs   # r0 dir name; replicas append _r<N>
TEMPERATURE=293
ENSEMBLE=NPT
STEPS=1000000        # 1 ns @ 1 fs
INTERVAL=100         # 0.1 ps dump
MATMUL_PREC=highest  # fp32
CKPT=best_model.fnx

# SLURM — must agree with the #SBATCH defaults in run_painn_replicas_nersc.sh
# (these command-line flags override them).
QOS=${QOS:-premium}
WALLTIME=${WALLTIME:-6:00:00}  # a 1 ns PaiNN run is ~3 h 45 m on a solo A100; with 4
                               # replicas sharing a node's CPUs it may exceed 6 h, in
                               # which case the job requeues and resumes from .dyn.restart.
ACCOUNT=${ACCOUNT:-m5168_g}

# 8 systems, 1 M each (lipf6_dme excluded; napf6_tgdme only ever had 1 M).
SYSTEMS=(
  naotf_diglyme
  naotf_dme
  naotf_pc
  naotf_tgdme
  napf6_diglyme
  napf6_dme
  napf6_pc
  napf6_tgdme
)
REPLICAS=(1 2 3 4)   # replica index == FeNNol --seed; r0 is yuejian's existing run

# Restrict to the systems named on the command line, if any.
if [ ${#WANTED[@]} -gt 0 ]; then
  for w in "${WANTED[@]}"; do
    printf '%s\n' "${SYSTEMS[@]}" | grep -qx "$w" || {
      echo "ERROR: '$w' is not one of the known systems:" >&2
      printf '  %s\n' "${SYSTEMS[@]}" >&2
      exit 1
    }
  done
  SYSTEMS=("${WANTED[@]}")
fi

cd "$REPO"
mkdir -p slurm_logs/painn_replicas

# ---------------------------------------------------------------- preflight --
# Check everything up front so a permissions or missing-file problem shows up as
# one clear list, not as 8 jobs that each die 30 s after starting.
errors=0

for sys in "${SYSTEMS[@]}"; do
  model="$EXP_DIR/$sys/model/$CKPT"
  sys_out="$OUT_ROOT/$sys"
  seed_xyz="$SRC_ROOT/$sys/$RUN_BASE/start.xyz"

  [ -f "$model" ]    || { echo "MISSING model    : $model" >&2; errors=$((errors + 1)); }
  [ -f "$seed_xyz" ] || { echo "MISSING seed xyz : $seed_xyz" >&2; errors=$((errors + 1)); }

  # We own the mirror, so just create it (the run dirs themselves are made by
  # fennol_npt.py on a fresh start — creating them here would trip its
  # refuse-to-clobber check).
  [ "$DRY_RUN" -eq 1 ] || mkdir -p "$sys_out"

  for rep in "${REPLICAS[@]}"; do
    d="$sys_out/${RUN_BASE}_r${rep}"
    if [ -e "$d" ]; then
      # Not fatal: the node script drops --from-xyz and resumes from .dyn.restart.
      echo "NOTE  replica dir already exists (will RESUME, not restart): $d" >&2
    fi
  done
done

if [ "$errors" -gt 0 ]; then
  echo >&2
  echo "Preflight failed with $errors problem(s); nothing submitted." >&2
  exit 1
fi

echo "Preflight OK: ${#SYSTEMS[@]} systems x ${#REPLICAS[@]} replicas = $(( ${#SYSTEMS[@]} * ${#REPLICAS[@]} )) runs on ${#SYSTEMS[@]} nodes."
echo "  seeds from : $SRC_ROOT"
echo "  writing to : $OUT_ROOT"
echo

# ------------------------------------------------------------------ submit --
submitted=0
for sys in "${SYSTEMS[@]}"; do
  model="$EXP_DIR/$sys/model/$CKPT"
  sys_out="$OUT_ROOT/$sys"
  seed_xyz="$SRC_ROOT/$sys/$RUN_BASE/start.xyz"
  job="painn_rep_${sys}"

  cmd=(sbatch
    -A "$ACCOUNT"
    -q "$QOS"
    -t "$WALLTIME"
    -J "$job"
    --export="ALL,SYSTEM=$sys,MODEL=$model,SEED_XYZ=$seed_xyz,SYS_OUT_DIR=$sys_out,RUN_BASE=$RUN_BASE,TEMPERATURE=$TEMPERATURE,STEPS=$STEPS,INTERVAL=$INTERVAL,ENSEMBLE=$ENSEMBLE,MATMUL_PREC=$MATMUL_PREC"
    "$SLURM_SCRIPT"
    "${REPLICAS[@]}")

  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    echo ">> $job  -> $sys_out/${RUN_BASE}_r{$(IFS=,; echo "${REPLICAS[*]}")}"
    "${cmd[@]}"
  fi
  submitted=$((submitted + ${#REPLICAS[@]}))
done

echo
echo "Done. $submitted run(s) across ${#SYSTEMS[@]} node job(s) $( [ "$DRY_RUN" -eq 1 ] && echo 'would be submitted' || echo 'submitted')."
