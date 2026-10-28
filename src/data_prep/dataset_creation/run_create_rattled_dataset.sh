#!/bin/bash
#SBATCH -N 1
#SBATCH -t 2:30:00
#SBATCH -C gpu&hbm80g
#SBATCH -A m5250_g
#SBATCH -q premium
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --array=0-2
#SBATCH -o /global/u2/i/ishan_a/electrolytes/slurm_logs/create_rattled_%A_%a.out
#SBATCH -e /global/u2/i/ishan_a/electrolytes/slurm_logs/create_rattled_%A_%a.err
#SBATCH -J create_rattled

# Build one rattled+MD dataset PER salt/solvent system for the 293 K UMA
# trajectories, using only the concentrations that have BOTH experimental data
# and a trajectory (see docs/uma_trajectory_coverage_293K.md). Each system gets
# its own output folder under version01_mixed_noise/; a 2-concentration system
# yields 8000 frames (4000/conc), a 1-concentration system yields 4000. UMA
# relabels every noised frame; it also relabels MD frames from position-only
# (0.1/0.5 M) trajectories, but reuses the stored labels in the 1 M trajectories.
#
# Every trajectory is sampled identically over its first 100 ps: MD frames every
# 50 fs (t=0,50,100,...) + noised frames every 50 fs offset by 20 fs
# (t=20,70,120,...) = 2000 MD + 2000 noised per concentration. The 1 ns (1 M)
# trajectories use only their first 100 ps, same as the 100 ps diluted ones.
#
# Parallelism: 10 systems are split across GPUS_PER_NODE=4 GPUs per node, so they
# need 3 nodes (packed 4 / 4 / 2). This is submitted as a 3-task job array
# (--array=0-2) on premium: each array task grabs one full node and runs its slice
# of up to 4 systems concurrently, one per local GPU (CUDA_VISIBLE_DEVICES). Using
# a job array instead of a single -N 3 job keeps everything to the simple,
# robust within-node GPU dispatch (no cross-node srun).
#
# Re-run a single node's slice with: sbatch --array=<t> run_create_rattled_dataset.sh

source ~/.bashrc
conda activate fairchemV3

cd /global/homes/i/ishan_a/electrolytes
mkdir -p slurm_logs

RAW=/global/cfs/cdirs/m5024/distillation_project/data/raw_data_from_UMA_simulation/293K
F01="$RAW/UMA_trajs_293K_0_1M_100ps"
F05="$RAW/UMA_trajs_293K_0_5M_100ps"
F1="$RAW/UMA_trajs_293K_1M_1ns"

UMA_CKPT=/global/cfs/cdirs/m4558/distillation_project/models/uma-s-1p1.pt
OUT_BASE=/global/u2/i/ishan_a/electrolytes/data/datasets/version01_mixed_noise

SIGMA=0.1
GPUS_PER_NODE=4

# One record per system: "name ; space-separated traj paths ; space-separated concs".
# Trajs and concs are positional one-to-one (same order). Systems are dealt out to
# nodes in chunks of GPUS_PER_NODE; keep --array=0-<ceil(N/GPUS_PER_NODE)-1> in sync.
JOBS=(
  "lipf6_dme;$F05/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj;0.5M"
  "naotf_diglyme;$F01/md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj $F1/naotf_diglyme.traj;0.1M 1M"
  "naotf_dme;$F01/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj $F1/naotf_dme.traj;0.1M 1M"
  "naotf_pc;$F01/md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj $F1/naotf_pc.traj;0.1M 1M"
  "naotf_tgdme;$F01/md_omol_naotf_tgdme_1m_s1p1/md_omol_naotf_tgdme_1m_s1p1.traj $F1/naotf_tgdme.traj;0.1M 1M"
  "napf6_diglyme;$F01/md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj $F1/napf6_diglyme.traj;0.1M 1M"
  "napf6_dme;$F01/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj $F1/napf6_dme.traj;0.1M 1M"
  "napf6_pc;$F01/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj $F1/napf6_pc.traj;0.1M 1M"
  "napf6_tgdme;$F1/napf6_tgdme.traj;1M"
)

# This node handles systems [start, start+GPUS_PER_NODE) of JOBS.
start=$((SLURM_ARRAY_TASK_ID * GPUS_PER_NODE))
echo "=== node/array-task $SLURM_ARRAY_TASK_ID: systems $start..$((start + GPUS_PER_NODE - 1)) ==="

pids=()
pnames=()
for (( idx = start; idx < start + GPUS_PER_NODE && idx < ${#JOBS[@]}; idx++ )); do
    gpu=$((idx - start))                        # local GPU = position within node
    IFS=';' read -r name trajs concs <<< "${JOBS[$idx]}"
    # Idempotent: skip systems whose dataset is already built, so a resubmit only
    # rebuilds the missing/failed ones.
    if [ -f "$OUT_BASE/$name/training_all.pkl" ]; then
        echo "  [$name] -> output exists, skipping"
        continue
    fi
    echo "  [$name] -> GPU $gpu (concs: $concs)"
    CUDA_VISIBLE_DEVICES=$gpu python src/data_prep/dataset_creation/create_rattled_dataset.py \
        --traj $trajs \
        --concentrations $concs \
        --uma-checkpoint "$UMA_CKPT" \
        --output-dir "$OUT_BASE/$name" \
        --stdev $SIGMA \
        > slurm_logs/create_${name}.log 2>&1 &
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
        echo "  [${pnames[$i]}] FAILED (see slurm_logs/create_${pnames[$i]}.log)"
        rc=1
    fi
done

echo "Done node/array-task $SLURM_ARRAY_TASK_ID at $(date) (rc=$rc)"
exit $rc
