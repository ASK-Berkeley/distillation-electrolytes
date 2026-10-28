#!/bin/bash
# ============================================================================
# Example: run AllScAIP MD on Delta (ghx4), one trajectory per node.
# Driver: src/simulate/allscaip.py
#
# Difference from the UMA example: per-node temperatures (TEMPS_STR) instead of
# a single shared TEMP, so each trajectory can run at its own temperature.
# Edit TRAJ_DIRS_STR / TEMPS_STR / MODEL / STEPS / INTERVAL, set --nodes to the
# number of trajectories, then: sbatch src/simulate/fairchem_configs/allscaip_example.sh
# ============================================================================
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --job-name=allscaip_example
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=beye-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

trap 'echo "Time limit nearly up, requeuing..."; scontrol requeue $SLURM_JOB_ID; wait' USR1

module load cuda/12.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairchemV3

mkdir -p slurm_logs

echo "Job starting on $(hostname)"
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)"

# One trajectory dir per node, with a matching per-node temperature.
BASE=/work/nvme/beye/iamin/nitesh_side_project_trajs/allscaip_trajs

export TRAJ_DIRS_STR="\
$BASE/sys_0.5m_kpf6_253K \
$BASE/sys_0.5m_kpf6_300K \
$BASE/sys_0.5m_kpf6_353K \
$BASE/sys_0.5m_lipf6_253K"

export TEMPS_STR="253 300 353 253"

export MODEL=/projects/beye/iamin/distillation_project/models/AllScAIP-OMol102M-md-cons.pt

# 1.5 ns @ 1 fs/step = 1,500,000 steps; save every 10 steps
export STEPS=1500000
export INTERVAL=10

srun bash -c '
  read -ra TRAJ_DIRS <<< "$TRAJ_DIRS_STR"
  read -ra TEMPS <<< "$TEMPS_STR"
  IDX=$SLURM_PROCID
  echo "Node $IDX: ${TRAJ_DIRS[$IDX]} @ ${TEMPS[$IDX]} K"
  python /u/iamin/electrolytes/src/simulate/allscaip.py \
    "${TRAJ_DIRS[$IDX]}" \
    --models "$MODEL" \
    --steps "$STEPS" \
    --interval "$INTERVAL" \
    --temperature "${TEMPS[$IDX]}"
' &
wait $!
