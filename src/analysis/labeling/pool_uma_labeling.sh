#!/bin/bash
# Runs INSIDE a Delta allocation: label 4 NPT trajectories with UMA at a 2 ps
# interval, one trajectory per GPU, concurrently. Invoked by
# launch_uma_labeling_interactive.sh; can also be run by hand from a `salloc` shell.
#
# The four trajectories pair up for the figure:
#   exp07_no_noise  napf6_tgdme / lipf6_dme  -- no-noise ablation, both evaporated
#   exp04_correct_molarity_samebox  same two systems -- the healthy reference runs
#
# All four dump every 1 ps (tdump[ps]: 1.0), so --save-interval-fs 1000 and
# --every-ps 2 gives stride 2. All four carry FeNNol reference forces/stress/energy
# in the .traj, so each npz gets ref_* arrays next to the uma_* arrays and the
# parity plot needs no second pass.
#
# NOTE the exp07 trajectories are mostly post-evaporation (napf6_tgdme blows up at
# ~7.4 ps of 201, lipf6_dme at ~500 of 1000). Those frames are labeled too and are
# meant to be -- the point is to show where the no-noise model departs from UMA.
# Expect the dilute-gas frames to label FASTER (few neighbours), not slower.
#
# label_traj_with_uma.py defaults its --log to ONE fixed path opened with mode="w",
# so four concurrent tasks would overwrite each other's log. --log is passed
# explicitly below for that reason; do not drop it.
#
# Cycle-safe. Each task writes $LOGDIR/task<N>.rc on exit; a task whose .rc is 0 is
# SKIPPED on a later cycle, and its output npz is left alone.

set -uo pipefail

: "${REPO:=/u/iamin/electrolytes}"
: "${DATA_ROOT:=/work/nvme/beye/iamin/electrolytes_data}"
: "${SESSION:=umalabel}"
: "${TASKS:=0 1 2 3}"
: "${CONDA_ENV:=fairchemV3}"
: "${CUDA_MODULE:=cuda/12.6.1}"

# Always the 1p1 teacher (see label_traj_with_uma.py) -- 1p2 makes non-comparable numbers.
: "${MODEL:=/projects/beye/iamin/distillation_project/models/uma-s-1p1.pt}"
: "${EVERY_PS:=2}"
: "${SAVE_INTERVAL_FS:=1000}"   # all four runs used tdump[ps]: 1.0
: "${LABEL:=uma}"
: "${EXTRA_ARGS:=}"             # extra flags forwarded verbatim to the driver, e.g. --no-compile

DRIVER="$REPO/src/analysis/labeling/label_traj_with_uma.py"
LOGDIR="$DATA_ROOT/labeling_logs/$SESSION"

# One entry per task: "<short name> <path to .traj>"
RUNS=(
  "exp07_napf6_tgdme $DATA_ROOT/experiments/exp07_no_noise/napf6_tgdme/npt_1M_298K_1ns/npt_1M_298K_1ns.traj"
  "exp07_lipf6_dme   $DATA_ROOT/experiments/exp07_no_noise/lipf6_dme/npt_0_5M_298K_1ns/npt_0_5M_298K_1ns.traj"
  "exp04_napf6_tgdme $DATA_ROOT/experiments/exp04_correct_molarity_samebox/napf6_tgdme/npt_equil_1M_298K_1ns/npt_equil_1M_298K_1ns.traj"
  "exp04_lipf6_dme   $DATA_ROOT/experiments/exp04_correct_molarity_samebox/lipf6_dme/npt_equil_0_5M_298K_1ns/npt_equil_0_5M_298K_1ns.traj"
)

mkdir -p "$LOGDIR"

module load "$CUDA_MODULE" 2>/dev/null
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1

echo "=============================================================="
echo "UMA labeling pool | node $(hostname) | job ${SLURM_JOB_ID:-none}"
echo "tasks: $TASKS   |   $(date)"
echo "GPUs visible: $(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
echo "model   : $MODEL"
echo "sampling: every $EVERY_PS ps (frames are $SAVE_INTERVAL_FS fs apart)"
echo "logs    : $LOGDIR"
echo "=============================================================="

# ------------------------------------------------------------------ one task --
run_task() {
  local idx="$1" gpu="$2"
  read -r name traj <<< "${RUNS[$idx]}"

  local log="$LOGDIR/task${idx}_${name}.log"
  local out
  out="$(dirname "$traj")/${LABEL}_labels_every${EVERY_PS}ps.npz"

  {
    echo "=============================================================="
    echo "task $idx  $name  GPU $gpu   $(date)"
    echo "traj  : $traj"
    echo "out   : $out"
    echo "=============================================================="

    local f
    for f in "$DRIVER" "$MODEL" "$traj"; do
      [ -f "$f" ] || { echo "ERROR: missing $f"; return 1; }
    done

    CUDA_VISIBLE_DEVICES="$gpu" python -u "$DRIVER" "$traj" \
      --model "$MODEL" \
      --every-ps "$EVERY_PS" \
      --save-interval-fs "$SAVE_INTERVAL_FS" \
      --label "$LABEL" \
      --output "$out" \
      --log "$LOGDIR/task${idx}_${name}.driver.log" \
      --device cuda $EXTRA_ARGS
    local rc=$?
    echo "task $idx ($name) exited rc=$rc at $(date)"

    # Confirm the npz actually holds what the figure needs, rather than trusting rc.
    if [ $rc -eq 0 ]; then
      python - "$out" <<'PYEOF' || true
import sys
import numpy as np
d = np.load(sys.argv[1])
keys = sorted(d.files)
n = len(d["time_ps"])
has_ref = "ref_forces" in d
line = (f"RESULT: {n} frames, {int(d['n_atoms'])} atoms, "
        f"t={d['time_ps'][0]:.0f}-{d['time_ps'][-1]:.0f} ps, ref={has_ref}")
if has_ref:
    fu, fr = d["uma_forces"], d["ref_forces"]
    line += (f", force MAE {np.abs(fu - fr).mean():.4f} eV/A"
             f", |F|_uma rms {np.sqrt((fu ** 2).mean()):.4f}")
print(line)
print("RESULT: keys =", keys)
PYEOF
    fi
    return $rc
  } >> "$log" 2>&1
}

# ------------------------------------------------------------------ dispatch --
pids=()
gpu=0
for idx in $TASKS; do
  rcfile="$LOGDIR/task${idx}.rc"
  read -r name _ <<< "${RUNS[$idx]}"
  if [ -f "$rcfile" ] && [ "$(cat "$rcfile")" = "0" ]; then
    echo "SKIP  task $idx ($name) -- already finished rc=0"
    continue
  fi
  echo ">> task $idx  $name  -> GPU $gpu   log: $LOGDIR/task${idx}_${name}.log"
  ( run_task "$idx" "$gpu"; echo $? > "$LOGDIR/task${idx}.rc" ) &
  pids+=($!)
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
  read -r name _ <<< "${RUNS[$idx]}"
  if [ -f "$rcfile" ]; then
    printf '  task %s %-18s rc=%s  %s\n' "$idx" "$name" "$(cat "$rcfile")" \
      "$(grep -h '^RESULT: [0-9]' "$LOGDIR/task${idx}_${name}.log" 2>/dev/null | tail -1)"
  else
    printf '  task %s %-18s rc=?    (killed by the wall -- reruns next cycle)\n' "$idx" "$name"
  fi
done
echo "=============================================================="
exit "$fail"
