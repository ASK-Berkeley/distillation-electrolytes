#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=ghx4
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=beye-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@300   # 5 min before timeout

# Parameterized FeNNol MD launcher. Pass parameters via env (sbatch --export):
#   RUN_DIR     (required) run output directory
#   MODEL       (required) path to .fnx model
#   FROM_XYZ    (optional) seed .xyz — used ONLY for a fresh start (dir absent);
#               on requeue the dir exists so we resume from .dyn.restart instead.
#   TEMPERATURE (required, K)  STEPS (default 1000000)  INTERVAL (default 1000)
#   ENSEMBLE    (default NPT)
#   MATMUL_PREC (optional) default|high|highest (tf32 vs fp32); omit -> driver default
#   SEED        (optional) FeNNol random_seed — set per replica for independent runs
#   SYSTEM_NAME (optional) basename of the output files (<name>.traj, .dyn.restart, ...);
#               omit -> the run directory's basename
#   THERMOSTAT  (optional) LGV|BUSSI|NOSE; omit -> driver default LGV. Use BUSSI
#               for a Green-Kubo transport run.
#   STRESS_LOG  (optional) 1 -> log the off-diagonal stress tensor to
#               <system>.stress for equilibrium (Green-Kubo) viscosity. Use with
#               ENSEMBLE=NVE or a gentle global thermostat (BUSSI). omit / 0 -> off.
#   STRESS_INTERVAL (optional) steps between stress dumps (default driver: 5).
#   GAMMA_THZ   (optional) thermostat coupling rate in THz (tau=1/gamma) for
#               NVT/NPT; omit -> driver default 1.0. Use ~0.1 for a transport run.
#   CURRENT_LOG (optional) 1 -> log the charge current J=sum q_i v_i to
#               <system>.current for Green-Kubo ionic conductivity. omit / 0 -> off.
#   CURRENT_INTERVAL (optional) steps between current dumps (default driver: 2).
#   SYSTEM_ID   (optional) registered system id for the current log's formal charges;
#               omit -> inferred from the run's parent folder.
# Requeue-safe: the same submission resumes after a timeout requeue.

# Requeue-safe timeout handling. SLURM sends USR1 to THIS batch shell 5 min
# before the wall limit (--signal=B:USR1@300). The handler must run *now*, but
# bash defers a trap until the current foreground command returns — so python
# MUST run in the background with `wait` (below), otherwise the foreground
# python blocks this handler until SLURM hard-kills the job and the requeue
# never happens (this was the bug: jobs just TIMED OUT with no requeue).
requeue_on_timeout() {
  echo "Time limit nearly up, requeuing job $SLURM_JOB_ID ..."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0   # let SLURM tear down + resubmit; resumes from .dyn.restart
}
trap requeue_on_timeout USR1

module load cuda/12.6.1 2>/dev/null
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fennol

# The launchers cd to the repo root before sbatch, so the submit dir is the repo.
cd "${REPO:-$SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

: "${TEMPERATURE:?TEMPERATURE (K) must be set}"
: "${STEPS:=1000000}"
: "${INTERVAL:=1000}"
: "${ENSEMBLE:=NPT}"

echo "Job on $(hostname) | RUN_DIR=$RUN_DIR | ENSEMBLE=$ENSEMBLE T=$TEMPERATURE STEPS=$STEPS INTERVAL=$INTERVAL"
nvidia-smi --query-gpu=name --format=csv,noheader

# Fresh start only when the run dir does not exist yet and a seed is given;
# otherwise (resume / requeue) omit --from-xyz and continue from .dyn.restart.
FROM=""
if [ -n "$FROM_XYZ" ] && [ ! -e "$RUN_DIR" ]; then
  FROM="--from-xyz $FROM_XYZ"
fi

MPREC=""
[ -n "$MATMUL_PREC" ] && MPREC="--matmul-prec $MATMUL_PREC"

SEEDARG=""
[ -n "$SEED" ] && SEEDARG="--seed $SEED"

THERMOARG=""
[ -n "$THERMOSTAT" ] && THERMOARG="--thermostat $THERMOSTAT"

STRESSARG=""
if [ -n "$STRESS_LOG" ] && [ "$STRESS_LOG" != "0" ]; then
  STRESSARG="--stress-log"
  [ -n "$STRESS_INTERVAL" ] && STRESSARG="$STRESSARG --stress-interval $STRESS_INTERVAL"
fi

GAMMAARG=""
[ -n "$GAMMA_THZ" ] && GAMMAARG="--gamma-thz $GAMMA_THZ"

CURRENTARG=""
if [ -n "$CURRENT_LOG" ] && [ "$CURRENT_LOG" != "0" ]; then
  CURRENTARG="--current-log"
  [ -n "$CURRENT_INTERVAL" ] && CURRENTARG="$CURRENTARG --current-interval $CURRENT_INTERVAL"
  [ -n "$SYSTEM_ID" ] && CURRENTARG="$CURRENTARG --system-id $SYSTEM_ID"
fi

# Run in the BACKGROUND + `wait` so the USR1 trap above can fire mid-run.
# `wait` is interruptible by the trap; a bare foreground python would not be.
NAMEARG=""
[ -n "${SYSTEM_NAME:-}" ] && NAMEARG="--system-name $SYSTEM_NAME"

python -m src.simulate.fennol_npt "$RUN_DIR" \
  --model "$MODEL" $FROM $MPREC $SEEDARG $THERMOARG $STRESSARG $GAMMAARG $CURRENTARG $NAMEARG \
  --ensemble "$ENSEMBLE" --temperature "$TEMPERATURE" \
  --steps "$STEPS" --interval "$INTERVAL" &
MD_PID=$!

# `wait` blocks until python exits. If USR1 arrives first, the trap requeues and
# exit 0s (so we never reach the line below); on normal completion/crash, wait
# returns python's status and we exit with it (no requeue).
wait "$MD_PID"
exit $?
