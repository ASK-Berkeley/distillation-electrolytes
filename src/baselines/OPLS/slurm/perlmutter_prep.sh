#!/bin/bash
#SBATCH -A m3278_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 04:00:00
#SBATCH -J opls_prep
#SBATCH -o /pscratch/sd/i/ishan_a/opls_clp/logs/%x_%j.out
# =============================================================================
# Two-pass molarity correction + production seeds for the 16-box OPLS set.
#
#   pass 1   EM + 6 ns NPT on the existing packed boxes (current force field);
#            a density-determination run, longer than the seed equilibration
#   repack   new solvent counts from the pass-1 equilibrium density
#   pass 2   EM + 1 ns NPT on the repacked boxes
#   seeds    prepare_production.py on pass 2 (falls back to pass 1 if any part
#            of the repack path fails, so production always has seeds)
#
# Markers written to $W: USE_MPS (only if GPU sharing needs MPS),
# PRODUCTION_SOURCE (pass1|pass2), PREP_OK.
# Everything is idempotent; resubmitting continues where it stopped.
# =============================================================================
set -uo pipefail
W=${OPLS_WORK:-/pscratch/sd/i/ishan_a/opls_clp}
CODE=$W/code/src/baselines/OPLS
NBOX=16

set +u
source /global/homes/i/ishan_a/mambaforge/etc/profile.d/conda.sh
conda activate fairchemV2
set -u
export PYTHONNOUSERSITE=1
unset OMP_NUM_THREADS                      # mdrun gets -ntomp; a mismatch is fatal
export GMXBIN=$W/gmxenv/bin/gmx PACKMOL=$W/gmxenv/bin/packmol
export OPLS_CODE_SHA="$(cat $W/provenance/git_head.txt)+staged_changes.patch"
export CM_TEGDME_PME=gpu
cd "$CODE"

echo "=== $(date) host $(hostname) job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv,noheader
$GMXBIN --version | grep -E "GROMACS version|GPU support"

count_npt() {   # finished NPT runs under an equilibration root
    local n=0 d
    for d in "$1"/*/; do
        [ -f "$d/npt.gro" ] && grep -q "Finished mdrun" "$d/npt.log" 2>/dev/null && n=$((n + 1))
    done
    echo $n
}

run_eq() {      # run_eq <box_root> <eq_root> <npt_ns>; second call resumes anything interrupted
    local try
    for try in 1 2; do
        CM_BOX_ROOT=$1 CM_EQ_ROOT=$2 python run_equilibration.py --gpus 0,1,2,3 --per-gpu 4 --ntomp 4 --npt-ns $3
        [ "$(count_npt "$2")" -eq $NBOX ] && return 0
        echo "run_eq: $(count_npt "$2")/$NBOX finished after try $try"
    done
    return 1
}

share_test() {  # two mdruns at once on GPU 0 must both finish
    local T=$W/gputest/eq/napf6_dme_1M_298 k
    [ -f $T/t.tpr ] || { echo "share_test: no test tpr, assuming sharing works"; return 0; }
    rm -f $T/share?.log
    for k in 1 2; do
        (cd $T && CUDA_VISIBLE_DEVICES=0 $GMXBIN mdrun -deffnm share$k -s t.tpr -nsteps 5000 \
            -nb gpu -pme gpu -update cpu -ntmpi 1 -ntomp 4 -pin off -notunepme > share$k.out 2>&1) &
    done
    wait
    grep -q "Finished mdrun" $T/share1.log 2>/dev/null && grep -q "Finished mdrun" $T/share2.log 2>/dev/null
}

start_mps() {
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    nvidia-cuda-mps-control -d && sleep 3
}

echo "=== GPU sharing test"
rm -f $W/USE_MPS
if share_test; then
    echo "two mdruns share one GPU without MPS"
else
    echo "plain sharing failed, trying MPS"
    start_mps
    if share_test; then echo "sharing works with MPS"; touch $W/USE_MPS
    else echo "FATAL: cannot run two mdruns on one GPU"; exit 1; fi
fi

echo "=== pass 1: EM + NPT on the existing packed boxes   $(date)"
if ! run_eq $W/pass1/opls_ready_boxes $W/pass1/equilibrated 6; then
    echo "FATAL: pass 1 did not finish for all $NBOX boxes"; exit 1
fi

PASS=pass1
echo "=== repack at the pass-1 density   $(date)"
if python repack_boxes.py --eq-root $W/pass1/equilibrated --out $W/pass2/boxes_root --workers 16 --window-frac 0.3334 \
   && [ "$(ls $W/pass2/boxes_root/*/initial_*.xyz 2>/dev/null | wc -l)" -eq $NBOX ]; then
    echo "=== pass 2: build + EM + NPT on the repacked boxes   $(date)"
    if python build_opls_boxes.py --src $W/pass2/boxes_root --out $W/pass2/opls_ready_boxes --lipf6-temps 298 \
           > $W/logs/build_pass2.log 2>&1 \
       && run_eq $W/pass2/opls_ready_boxes $W/pass2/equilibrated 1; then
        PASS=pass2
    else
        echo "WARNING: pass 2 incomplete, falling back to pass 1 seeds"
    fi
else
    echo "WARNING: repack failed, falling back to pass 1 seeds"
fi

echo "=== production seeds from $PASS   $(date)"
CM_EQ_ROOT=$W/$PASS/equilibrated CM_PROD_ROOT=$W/production_ready python prepare_production.py
NSEED=$(ls $W/production_ready/*/seed.gro 2>/dev/null | wc -l)
echo "$NSEED/$NBOX seeds written"
if [ "$NSEED" -ne $NBOX ]; then echo "FATAL: missing seeds"; exit 1; fi
echo $PASS > $W/PRODUCTION_SOURCE
touch $W/PREP_OK
echo "=== prep done ($PASS)   $(date)"
