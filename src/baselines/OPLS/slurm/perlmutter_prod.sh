#!/bin/bash
#SBATCH -A m3278_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 24:00:00
#SBATCH --array=0-3
#SBATCH -J opls_prod
#SBATCH -o /pscratch/sd/i/ishan_a/opls_clp/logs/%x_%A_%a.out
# =============================================================================
# 20 ns NVT production, 16 boxes x 4 replicas = 64 runs over 4 nodes
# (shard k/4: 16 runs per node, 4 per GPU, 4 OpenMP threads each). Premium QOS
# allows 5 submitted jobs per user, hence 4 array tasks beside the prep job.
# Resumable: resubmitting skips finished segments and continues the rest.
# =============================================================================
set -uo pipefail
W=${OPLS_WORK:-/pscratch/sd/i/ishan_a/opls_clp}
CODE=$W/code/src/baselines/OPLS

set +u
source /global/homes/i/ishan_a/mambaforge/etc/profile.d/conda.sh
conda activate fairchemV2
set -u
export PYTHONNOUSERSITE=1
unset OMP_NUM_THREADS
export GMXBIN=$W/gmxenv/bin/gmx
export OPLS_CODE_SHA="$(cat $W/provenance/git_head.txt)+staged_changes.patch"
export CM_TEGDME_PME=gpu
cd "$CODE"

echo "=== $(date) host $(hostname) job ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-0}"
[ -f $W/PREP_OK ] || { echo "FATAL: $W/PREP_OK missing, prep did not finish"; exit 1; }
echo "seeds from: $(cat $W/PRODUCTION_SOURCE)"
nvidia-smi --query-gpu=index,name,compute_mode --format=csv,noheader

if [ -f $W/USE_MPS ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    nvidia-cuda-mps-control -d && sleep 3
fi

CM_PROD_ROOT=$W/production_ready CM_REP_ROOT=$W/production_replicas \
    python run_production.py --gpus 0,1,2,3 --per-gpu 4 --ntomp 4 --shard ${SLURM_ARRAY_TASK_ID}/4
echo "=== shard ${SLURM_ARRAY_TASK_ID} exited   $(date)"
grep -l '"segments_done": 10' $W/production_replicas/*/r*/prod_state.json 2>/dev/null | wc -l | xargs echo "replicas complete (all shards):"
