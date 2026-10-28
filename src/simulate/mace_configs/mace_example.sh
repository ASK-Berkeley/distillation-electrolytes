#!/bin/bash
# ============================================================================
# Example: run MACE MD on Delta (ghx4), one trajectory per node.
# Driver: src/simulate/mace_md.py
#
# Edit TRAJ_DIRS_STR / MODEL / TEMP / STEPS / INTERVAL below, set --nodes to
# the number of trajectories, then: sbatch src/simulate/mace_configs/mace_example.sh
# ============================================================================
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=ghx4
#SBATCH --time=10:00:00
#SBATCH --job-name=mace_example
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

# One trajectory dir per node (SLURM_PROCID indexes this list).
BASE=/projects/beye/iamin/distillation_project/MACE_POLAR-1-M_trajs

export TRAJ_DIRS_STR="\
$BASE/dme_pure \
$BASE/diglyme_pure \
$BASE/tgdme_pure \
$BASE/propylenecarbonate_pure \
$BASE/diethyleneglycol_pure \
$BASE/dimethylcarbonate_pure"

export MODEL=/projects/beye/iamin/distillation_project/models/MACE-POLAR-1-M.model

# 150 ps @ 1 fs/step = 150,000 steps; save every 100 steps
export STEPS=150000
export INTERVAL=100
export TEMP=298.2

srun bash -c '
  read -ra TRAJ_DIRS <<< "$TRAJ_DIRS_STR"
  IDX=$SLURM_PROCID
  echo "Node $IDX: ${TRAJ_DIRS[$IDX]} @ '"$TEMP"' K"
  python /u/iamin/electrolytes/src/simulate/mace_md.py \
    "${TRAJ_DIRS[$IDX]}" \
    --models "$MODEL" \
    --steps "$STEPS" \
    --interval "$INTERVAL" \
    --temperature '"$TEMP"' \
    --initial_temperature '"$TEMP"'
' &
wait $!
