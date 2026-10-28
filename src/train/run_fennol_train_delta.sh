#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
# A full pass per epoch (nbatch_per_epoch = n_train // batch_size) is ~900 batches
# for the 3600-frame datasets and ~1800 for the pooled 7200-frame ones. At the
# ~160 ms/batch these GH200s sustain that is ~11 h and ~22 h respectively for 275
# epochs, so 8 h was only ever enough for the (wrong) 200-batch default. ghx4
# allows 2 days.
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=beye-dtai-gh
#SBATCH --gpus-per-node=1

if [ -z "$1" ]; then
  echo "Usage: sbatch run_fennol_train_delta.sh <config.yaml>"
  exit 1
fi

CONFIG="$1"
# Derive job name from config filename (strip path and .yaml)
JOB_NAME=$(basename "${CONFIG}" .yaml)
echo "Job name: ${JOB_NAME}"

module load cuda/12.6.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fennol

mkdir -p slurm_logs

echo "Job starting on $(hostname)"
echo "Config: ${CONFIG}"
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)"

cd /u/iamin/electrolytes
fennol_train "${CONFIG}"
