#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=beye-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@300   # 5 min before timeout

# Parameterized UMA MD launcher (one trajectory, one GPU, one node).
# Driver: src/simulate/uma_npt.py. Pass parameters via env (sbatch --export):
#   TRAJ_DIR    (required) trajectory directory; must contain <basename>.traj
#               (the seed on a fresh start, or the in-progress run on requeue).
#   MODEL       (required) path to UMA .pt checkpoint.
#   TEMPERATURE (required, K)   STEPS (default 1000000)   INTERVAL (default 1000)
#   TURBO       (optional) 1 -> fairchem 'turbo' inference (tf32=True). omit/0 -> off.
#
# Resume/requeue is automatic: uma_npt.py reads the LAST frame of the .traj and
# appends, so a requeued job continues from wherever it stopped. INTERVAL must
# stay constant across requeues (the driver infers elapsed steps from the frame
# count), which it does because it is fixed by the launcher.

# Requeue-safe timeout handling. SLURM sends USR1 to THIS batch shell 5 min
# before the wall limit (--signal=B:USR1@300). bash defers the trap until the
# current foreground command returns, so python MUST run in the background with
# `wait` (below) or the handler never fires and the job just times out.
requeue_on_timeout() {
  echo "Time limit nearly up, requeuing job $SLURM_JOB_ID ..."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0   # let SLURM tear down + resubmit; resumes from the last .traj frame
}
trap requeue_on_timeout USR1

module load cuda/12.6 2>/dev/null
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairchemV3

# Bypass the Cray CC wrapper: torch.compile C++ codegen needs g++/gcc.
export CXX=g++
export CC=gcc

cd /u/iamin/electrolytes
mkdir -p slurm_logs

: "${TEMPERATURE:?TEMPERATURE (K) must be set}"
: "${STEPS:=1000000}"
: "${INTERVAL:=1000}"

if [ -z "$TRAJ_DIR" ] || [ -z "$MODEL" ]; then
  echo "ERROR: TRAJ_DIR and MODEL must be set (via sbatch --export)." >&2
  exit 1
fi

echo "Job on $(hostname) | TRAJ_DIR=$TRAJ_DIR | T=$TEMPERATURE STEPS=$STEPS INTERVAL=$INTERVAL TURBO=${TURBO:-0}"
nvidia-smi --query-gpu=name --format=csv,noheader

TURBOARG=""
if [ -n "$TURBO" ] && [ "$TURBO" != "0" ]; then
  TURBOARG="--turbo"
fi

# Run in the BACKGROUND + `wait` so the USR1 trap can fire mid-run.
python src/simulate/uma_npt.py "$TRAJ_DIR" \
  --models "$MODEL" \
  --steps "$STEPS" \
  --interval "$INTERVAL" \
  --temperature "$TEMPERATURE" \
  $TURBOARG &
MD_PID=$!

# `wait` blocks until python exits. If USR1 arrives first, the trap requeues and
# exit 0s; on normal completion/crash, wait returns python's status.
wait "$MD_PID"
exit $?
