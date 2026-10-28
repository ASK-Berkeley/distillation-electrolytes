#!/bin/bash
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH -C gpu&hbm80g
#SBATCH -A m5250_g
#SBATCH -q premium
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -o /global/u2/i/ishan_a/electrolytes/slurm_logs/create_no_noise_%j.out
#SBATCH -e /global/u2/i/ishan_a/electrolytes/slurm_logs/create_no_noise_%j.err
#SBATCH -J create_no_noise

# ABLATION: build the NO-NOISE counterparts of two version01_mixed_noise systems,
# for comparing models trained with vs. without rattled OOD points. Output goes to
# version05_no_noise/, one folder PER system (kept separate so each is its own
# ablation baseline):
#   - napf6_tgdme : 1M   trajectory (pre-labeled)
#   - lipf6_dme   : 0.5M trajectory (position-only)
#
# Each system is sampled over EXACTLY the same two grids as version01: MD frames
# every 50 fs from t=0 (2000) + offset-grid frames every 50 fs from t=20 fs (2000)
# over the first 100 ps = 4000 frames -> 3600 train / 400 val. The only difference
# is --no-noise: the offset grid is kept at its true MD geometry instead of being
# rattled, so every frame is pure MD.
#
# Speed: with --no-noise a pre-labeled trajectory needs NO UMA pass at all
# (napf6_tgdme just reads stored labels), and the position-only 0.5M lipf6_dme
# UMA-labels its 4000 real MD frames — no re-labeling of rattled geometries. Both
# systems run concurrently, one per local GPU, so the job is fast (1 h is ample).
#
# Idempotent: a system whose training_all.pkl already exists is skipped on resubmit.

source ~/.bashrc
conda activate fairchemV3

cd /global/homes/i/ishan_a/electrolytes
mkdir -p slurm_logs

RAW=/global/cfs/cdirs/m5024/distillation_project/data/raw_data_from_UMA_simulation/293K
F05="$RAW/UMA_trajs_293K_0_5M_100ps"
F1="$RAW/UMA_trajs_293K_1M_1ns"

UMA_CKPT=/global/cfs/cdirs/m4558/distillation_project/models/uma-s-1p1.pt
OUT_BASE=/global/u2/i/ishan_a/electrolytes/data/datasets/version05_no_noise

# One record per system: "name ; traj path ; label". Single trajectory each.
JOBS=(
  "napf6_tgdme;$F1/napf6_tgdme.traj;1M"
  "lipf6_dme;$F05/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj;0.5M"
)

pids=()
pnames=()
for (( idx = 0; idx < ${#JOBS[@]}; idx++ )); do
    gpu=$idx                                    # one system per local GPU
    IFS=';' read -r name traj label <<< "${JOBS[$idx]}"
    if [ -f "$OUT_BASE/$name/training_all.pkl" ]; then
        echo "  [$name] -> output exists, skipping"
        continue
    fi
    echo "  [$name] -> GPU $gpu (label: $label, no-noise)"
    CUDA_VISIBLE_DEVICES=$gpu python src/data_prep/dataset_creation/create_rattled_dataset.py \
        --traj "$traj" \
        --labels "$label" \
        --uma-checkpoint "$UMA_CKPT" \
        --output-dir "$OUT_BASE/$name" \
        --no-noise \
        > slurm_logs/create_no_noise_${name}.log 2>&1 &
    pids+=($!)
    pnames+=("$name")
done

# Wait on each child explicitly so a crashed system surfaces (a bare `wait`
# returns 0 regardless and hides failures).
rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  [${pnames[$i]}] OK"
    else
        echo "  [${pnames[$i]}] FAILED (see slurm_logs/create_no_noise_${pnames[$i]}.log)"
        rc=1
    fi
done

echo "Done at $(date) (rc=$rc)"
exit $rc
