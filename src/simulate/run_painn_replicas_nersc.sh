#!/bin/bash
#SBATCH -A m5168_g
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -q premium
#SBATCH -t 6:00:00
#SBATCH -o slurm_logs/painn_replicas/%x_%j.out
#SBATCH -e slurm_logs/painn_replicas/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@300   # 5 min before the wall limit
#
# One Perlmutter node = one system = 4 PaiNN replicas, one per A100.
#
# Every replica starts from the SAME seed frame (SEED_XYZ) and differs only in
# --seed, which sets FeNNol's initial Maxwell-Boltzmann velocities and the
# thermostat/barostat noise stream. Replica r gets seed r.
#
# Parameters come in via `sbatch --export` (see launch_painn_1ns_replicas.sh);
# the replica indices are positional args:
#
#   sbatch --export=ALL,SYSTEM=...,MODEL=...,... run_painn_replicas_nersc.sh 1 2 3 4
#
# Required env:
#   SYSTEM       system id (e.g. napf6_dme) — labels only
#   MODEL        path to best_model.fnx
#   SEED_XYZ     seed structure, shared by all replicas (the r0 run's start.xyz)
#   SYS_OUT_DIR  output dir for this system; run dirs are $SYS_OUT_DIR/${RUN_BASE}_r<N>
#   RUN_BASE     run-name stem (e.g. npt_1M_293K_1ns_100fs)
# Optional env (defaults match yuejian's r0 .fennol.yaml):
#   TEMPERATURE (293)  STEPS (1000000)  INTERVAL (100)  ENSEMBLE (NPT)
#   MATMUL_PREC (highest)
#
# Requeue-safe: SLURM sends USR1 to THIS batch shell 5 min before the limit. The
# handler can only run while bash is *not* blocked in a foreground command, so
# every python runs in the background and we block in `wait` (interruptible).
# On requeue each replica dir already exists, so --from-xyz is dropped and the
# run resumes from its .dyn.restart.

set -uo pipefail

REPO=/global/homes/i/ishan_a/electrolytes

requeue_on_timeout() {
  echo "Time limit nearly up, requeuing job $SLURM_JOB_ID ..."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0   # let SLURM tear down + resubmit; each replica resumes from .dyn.restart
}
trap requeue_on_timeout USR1

REPS=("$@")
if [ ${#REPS[@]} -eq 0 ]; then
  echo "Usage: sbatch --export=ALL,SYSTEM=...,... run_painn_replicas_nersc.sh <rep> [rep ...]" >&2
  exit 1
fi
if [ ${#REPS[@]} -gt 4 ]; then
  echo "ERROR: got ${#REPS[@]} replicas but a Perlmutter GPU node has only 4 GPUs." >&2
  exit 1
fi

: "${TEMPERATURE:=293}"
: "${STEPS:=1000000}"
: "${INTERVAL:=100}"
: "${ENSEMBLE:=NPT}"
: "${MATMUL_PREC:=highest}"

cd "$REPO"
LOG_DIR=$REPO/slurm_logs/painn_replicas
mkdir -p "$LOG_DIR"

source ~/.bashrc
conda activate fennol
export PYTHONNOUSERSITE=1

echo "=========================================================="
echo "System   : $SYSTEM"
echo "Node     : $(hostname)   job $SLURM_JOB_ID   restarts=${SLURM_RESTART_COUNT:-0}"
echo "Model    : $MODEL"
echo "Seed xyz : $SEED_XYZ"
echo "Out dir  : $SYS_OUT_DIR"
echo "Replicas : ${REPS[*]}   ($ENSEMBLE ${TEMPERATURE}K steps=$STEPS interval=$INTERVAL prec=$MATMUL_PREC)"
echo "GPUs     : $(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
echo "=========================================================="

pids=()
names=()
for i in "${!REPS[@]}"; do
  rep="${REPS[$i]}"
  run_name="${RUN_BASE}_r${rep}"
  run_dir="$SYS_OUT_DIR/$run_name"
  log="$LOG_DIR/${SYSTEM}_${run_name}.log"

  # Fresh start only when the run dir does not exist yet; on a requeue/resume the
  # dir is there and fennol_npt.py continues from <run_name>.dyn.restart.
  FROM=""
  if [ ! -e "$run_dir" ]; then
    FROM="--from-xyz $SEED_XYZ"
    echo "GPU $i: FRESH  $run_dir  (seed $rep)"
  else
    echo "GPU $i: RESUME $run_dir  (seed $rep)"
  fi

  CUDA_VISIBLE_DEVICES=$i python -u -m src.simulate.fennol_npt "$run_dir" \
    --model "$MODEL" $FROM \
    --ensemble "$ENSEMBLE" \
    --temperature "$TEMPERATURE" \
    --pressure 1.0 \
    --timestep 1.0 \
    --steps "$STEPS" \
    --interval "$INTERVAL" \
    --traj-format ase \
    --matmul-prec "$MATMUL_PREC" \
    --seed "$rep" \
    --device cuda:0 \
    >> "$log" 2>&1 &
  pids+=($!)
  names+=("$run_name")
done

echo "All ${#pids[@]} replicas launched; waiting (logs in $LOG_DIR)."

# Wait on each child explicitly so a crashed replica surfaces (a bare `wait`
# would return 0). These waits are what the USR1 trap interrupts on timeout.
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "OK:     ${names[$i]}"
  else
    echo "FAILED: ${names[$i]}  (see $LOG_DIR/${SYSTEM}_${names[$i]}.log)"
    rc=1
  fi
done

echo "$SYSTEM: all replicas finished (rc=$rc) at $(date)"
exit $rc
