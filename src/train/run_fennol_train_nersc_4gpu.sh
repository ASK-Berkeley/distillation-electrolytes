#!/bin/bash
#SBATCH -A m5250_g
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -q premium
#SBATCH -t 12:00:00
#SBATCH -J fennol_4gpu
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

# Run up to 4 FeNNol training configs concurrently, one per GPU, on a single node.
# Configs are passed as positional arguments (typically by
# launch_fennol_experiment.py --nersc, which batches systems 4 at a time):
#   sbatch run_fennol_train_nersc_4gpu.sh <config1.yaml> [config2 config3 config4]

set -euo pipefail

cd /global/homes/i/ishan_a/electrolytes

source ~/.bashrc
conda activate fennol
export PYTHONNOUSERSITE=1

mkdir -p slurm_logs

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  echo "Usage: sbatch run_fennol_train_nersc_4gpu.sh <config1.yaml> [config2 ... config4]"
  exit 1
fi
if [ ${#CONFIGS[@]} -gt 4 ]; then
  echo "ERROR: got ${#CONFIGS[@]} configs but a node has only 4 GPUs."
  exit 1
fi

echo "Job starting on $(hostname)"
echo "GPUs available: $(nvidia-smi --list-gpus 2>/dev/null | wc -l)"

pids=()
for i in "${!CONFIGS[@]}"; do
  echo "Launching GPU $i: ${CONFIGS[$i]}"
  CUDA_VISIBLE_DEVICES=$i fennol_train "${CONFIGS[$i]}" &
  pids+=($!)
done

echo "All ${#CONFIGS[@]} training runs launched, waiting..."
# Wait on each child explicitly so a crashed run surfaces (a bare `wait` returns 0).
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "OK:     ${CONFIGS[$i]}"
  else
    echo "FAILED: ${CONFIGS[$i]}"
    rc=1
  fi
done
echo "All training runs complete (rc=$rc)."
exit $rc
