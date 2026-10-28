#!/bin/bash
# Runs INSIDE a Delta allocation: 1 ns NPT at 298 K for the two exp07_no_noise
# models, one per GPU, concurrently. Invoked by launch_exp07_npt_interactive.sh;
# can also be run by hand from a plain `salloc` shell.
#
# The allocation has 4 GH200s but there are only 2 tasks, so tasks 0 and 1 pin
# themselves to GPU 0 and GPU 1 via CUDA_VISIBLE_DEVICES (JAX then sees exactly
# one device and --device cuda:0 refers to it).
#
# Cycle-safe. Each task writes $LOGDIR/task<N>.rc on exit:
#   - a task whose .rc is 0 is SKIPPED on later cycles (finished, or blown up and
#     exited 0 -- either way there is nothing to gain by resuming it)
#   - a task killed mid-run by the 2 h wall leaves no .rc, so the next cycle
#     resumes it from its .dyn.restart (--from-xyz is dropped when the run dir
#     already exists; passing it would make fennol_npt.py refuse to clobber)
#
# Settings mirror src/simulate/fennol_configs/launch_npt_equilibration.sh:
# NPT (LGV thermostat + LGV Langevin piston, isotropic) at 298 K / 1 bar, dt 1 fs,
# 1e6 steps = 1 ns, 1 ps frame dumps, fp32 (matmul_prec=highest), seed 1.
#
# These models are the no-noise ablation and are EXPECTED to be less stable than
# their exp01 counterparts, so every task ends with a density health check --
# an evaporated run still exits 0 and still writes a full-length trajectory.

set -uo pipefail

: "${REPO:=/u/iamin/electrolytes}"
cd "$REPO"   # the driver runs as a module from the repo root
: "${DATA_ROOT:=/work/nvme/beye/iamin/electrolytes_data}"
: "${EXP:=exp07_no_noise}"
: "${SESSION:=exp07npt}"
: "${TASKS:=0 1}"
: "${CONDA_ENV:=fennol}"
: "${CUDA_MODULE:=cuda/12.6.1}"

# Run settings (see launch_npt_equilibration.sh).
: "${TEMPERATURE:=298}"
: "${PRESSURE:=1.0}"
: "${ENSEMBLE:=NPT}"
: "${STEPS:=1000000}"        # 1 ns @ 1 fs
: "${INTERVAL:=1000}"        # 1 ps frame dump
: "${TIMESTEP:=1.0}"
: "${MATMUL_PREC:=highest}"  # fp32
: "${SEED:=1}"
: "${CKPT:=best_model.fnx}"

EXP_DIR="$DATA_ROOT/experiments/$EXP"
DATASETS="$DATA_ROOT/datasets/version05_no_noise"
DRIVER="$REPO/src/simulate/fennol_npt.py"
LOGDIR="$EXP_DIR/pool_logs/$SESSION"

# "<system> <concentration label> <seed xyz basename, in $DATASETS/<system>/>"
RUNS=(
  "napf6_tgdme 1M   simulation_start_1M.xyz"
  "lipf6_dme   0_5M simulation_start_0_5M.xyz"
)

mkdir -p "$LOGDIR"

module load "$CUDA_MODULE" 2>/dev/null
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1

echo "=============================================================="
echo "exp07 NPT pool | node $(hostname) | job ${SLURM_JOB_ID:-none}"
echo "tasks: $TASKS   |   $(date)"
echo "GPUs visible: $(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
echo "settings: $ENSEMBLE ${TEMPERATURE}K ${PRESSURE}bar dt=${TIMESTEP}fs "\
"steps=$STEPS interval=$INTERVAL prec=$MATMUL_PREC seed=$SEED"
echo "logs: $LOGDIR"
echo "=============================================================="

# ------------------------------------------------------------------ one task --
run_task() {
  local idx="$1" gpu="$2"
  read -r sys conc xyz <<< "${RUNS[$idx]}"

  local model="$EXP_DIR/$sys/model/$CKPT"
  local seed_xyz="$DATASETS/$sys/$xyz"
  local run_dir="$EXP_DIR/$sys/npt_${conc}_${TEMPERATURE}K_1ns"
  local log="$LOGDIR/task${idx}.log"

  {
    echo "=============================================================="
    echo "task $idx  $sys ($conc)  GPU $gpu   $(date)"
    echo "model   : $model"
    echo "seed    : $seed_xyz"
    echo "run dir : $run_dir"

    local f
    for f in "$model" "$DRIVER"; do
      [ -f "$f" ] || { echo "ERROR: missing $f"; return 1; }
    done

    # Fresh start needs the seed frame; a resume does not.
    local from=""
    if [ ! -e "$run_dir" ]; then
      [ -f "$seed_xyz" ] || { echo "ERROR: missing seed xyz $seed_xyz"; return 1; }
      from="--from-xyz $seed_xyz"
      echo "mode    : FRESH start"
    else
      echo "mode    : RESUME from $run_dir/*.dyn.restart"
    fi
    echo "=============================================================="

    CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.simulate.fennol_npt "$run_dir" \
      --model "$model" $from \
      --ensemble "$ENSEMBLE" \
      --temperature "$TEMPERATURE" \
      --pressure "$PRESSURE" \
      --timestep "$TIMESTEP" \
      --steps "$STEPS" \
      --interval "$INTERVAL" \
      --traj-format ase \
      --matmul-prec "$MATMUL_PREC" \
      --seed "$SEED" \
      --device cuda:0
    local rc=$?
    echo "task $idx ($sys) exited rc=$rc at $(date)"

    # A run that evaporated still exits 0, so check the physics, not the code.
    if [ $rc -eq 0 ]; then
      python - "$run_dir" <<'PYEOF' || true
import glob, os, sys
from ase.io.trajectory import Trajectory
t = sorted(glob.glob(os.path.join(sys.argv[1], "*.traj")))
if not t:
    print("HEALTH: no trajectory written")
else:
    tr = Trajectory(t[0])
    if not len(tr):
        print("HEALTH: trajectory is empty")
    else:
        a = tr[len(tr) - 1]
        rho = a.get_masses().sum() / a.get_volume() * 1.66053906660
        v = "OK" if 0.7 < rho < 2.0 else "BLOWN UP (evaporated)"
        print(f"HEALTH: {len(tr)} frames, final density {rho:.4f} g/cm^3 -> {v}")
PYEOF
    fi
    return $rc
  } >> "$log" 2>&1
}

# ------------------------------------------------------------------ dispatch --
pids=()
labels=()
gpu=0
for idx in $TASKS; do
  rcfile="$LOGDIR/task${idx}.rc"
  if [ -f "$rcfile" ] && [ "$(cat "$rcfile")" = "0" ]; then
    echo "SKIP  task $idx -- already finished rc=0"
    continue
  fi
  read -r sys conc _ <<< "${RUNS[$idx]}"
  echo ">> task $idx  $sys ($conc)  -> GPU $gpu   log: $LOGDIR/task${idx}.log"
  ( run_task "$idx" "$gpu"; echo $? > "$LOGDIR/task${idx}.rc" ) &
  pids+=($!)
  labels+=("task${idx}:${sys}")
  gpu=$((gpu + 1))
done

if [ ${#pids[@]} -eq 0 ]; then
  echo "Nothing to do -- every requested task already finished rc=0."
  exit 0
fi

echo
echo "${#pids[@]} task(s) running; waiting..."
fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || fail=$((fail + 1))
done

echo
echo "=============================================================="
echo "pool done at $(date)   ($fail task(s) exited non-zero this cycle)"
for idx in $TASKS; do
  rcfile="$LOGDIR/task${idx}.rc"
  read -r sys _ _ <<< "${RUNS[$idx]}"
  if [ -f "$rcfile" ]; then
    printf '  task %s %-14s rc=%s  %s\n' "$idx" "$sys" "$(cat "$rcfile")" \
      "$(grep -h '^HEALTH:' "$LOGDIR/task${idx}.log" 2>/dev/null | tail -1)"
  else
    printf '  task %s %-14s rc=?    (killed by the wall -- resumes next cycle)\n' "$idx" "$sys"
  fi
done
echo "=============================================================="
exit "$fail"
