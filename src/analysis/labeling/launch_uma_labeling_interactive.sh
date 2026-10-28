#!/bin/bash
#
# Label the four NPT trajectories with UMA at 2 ps spacing on an interactive Delta
# node, detached, so no terminal has to stay open.
#
# One node = 4 GH200s = one trajectory per GPU, all four at once. ghx4-interactive
# caps walltime at 2:00:00; ~500 frames of ~3000 atoms should finish well inside
# that, but the loop allocates again if a task gets cut off.
#
# UNLIKE the MD launcher this wraps, labeling has NO resume -- a task killed by the
# wall starts over from frame 0 next cycle. Cycles only help if the task can finish
# inside 2 h at all; if one cannot, raise --every-ps or split the trajectory rather
# than adding cycles.
#
#   bash src/analysis/labeling/launch_uma_labeling_interactive.sh            # all 4
#   bash src/analysis/labeling/launch_uma_labeling_interactive.sh --dry-run
#   TASKS="0 1" bash src/analysis/labeling/launch_uma_labeling_interactive.sh # exp07 only
#   EVERY_PS=5 bash src/analysis/labeling/launch_uma_labeling_interactive.sh
#
# Watch it:   tmux attach -t umalabel                 (detach again with ctrl-b d)
#             tail -f /work/nvme/beye/iamin/electrolytes_data/labeling_logs/umalabel/task0_*.log
# Stop it:    bash src/analysis/labeling/launch_uma_labeling_interactive.sh --stop
#
# Use --stop, NOT `tmux kill-session`. The cycle loop runs in a pipeline into tee,
# so killing the session orphans it (reparented to init) and it happily allocates
# its next cycle with nobody watching. --stop kills the recorded launcher pid
# first, then scancels.
#
# tmux lives on ONE login node -- if you reconnect to a different gh-login you will
# not see the session. This script prints which one to come back to.

set -uo pipefail

: "${ACCOUNT:=beye-dtai-gh}"
: "${PARTITION:=ghx4-interactive}"
: "${WALLTIME:=02:00:00}"        # ghx4-interactive hard cap
: "${CYCLES:=2}"
: "${GPUS:=4}"                   # whole node; one trajectory per GPU
: "${CPUS:=64}"
: "${MEM:=360G}"
: "${EXCLUDE:=gh029,gh093}"      # known interconnect trouble
: "${SESSION:=umalabel}"
: "${TMUX_SESSION:=$SESSION}"
: "${TASKS:=0 1 2 3}"
: "${REPO:=/u/iamin/electrolytes}"
: "${DATA_ROOT:=/work/nvme/beye/iamin/electrolytes_data}"
: "${JOBNAME:=uma_label_pool}"
# Always the 1p1 teacher (see label_traj_with_uma.py) -- 1p2 makes non-comparable numbers.
: "${MODEL:=/projects/beye/iamin/distillation_project/models/uma-s-1p1.pt}"
: "${EVERY_PS:=2}"
: "${SAVE_INTERVAL_FS:=1000}"
: "${EXTRA_ARGS:=}"              # forwarded verbatim to the driver, e.g. --no-compile

SELF="$REPO/src/analysis/labeling/launch_uma_labeling_interactive.sh"
POOL="$REPO/src/analysis/labeling/pool_uma_labeling.sh"
LOGDIR="$DATA_ROOT/labeling_logs/$SESSION"
LAUNCHLOG="$LOGDIR/launcher.log"
PIDFILE="$LOGDIR/launcher.pid"

TRAJS=(
  "$DATA_ROOT/experiments/exp07_no_noise/napf6_tgdme/npt_1M_298K_1ns/npt_1M_298K_1ns.traj"
  "$DATA_ROOT/experiments/exp07_no_noise/lipf6_dme/npt_0_5M_298K_1ns/npt_0_5M_298K_1ns.traj"
  "$DATA_ROOT/experiments/exp04_correct_molarity_samebox/napf6_tgdme/npt_equil_1M_298K_1ns/npt_equil_1M_298K_1ns.traj"
  "$DATA_ROOT/experiments/exp04_correct_molarity_samebox/lipf6_dme/npt_equil_0_5M_298K_1ns/npt_equil_0_5M_298K_1ns.traj"
)

# ---------------------------------------------------------------- stop half --
if [ "${1:-}" = "--stop" ]; then
  if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "killing launcher pid $pid"
      kill -TERM "$pid" 2>/dev/null
      sleep 2
      kill -KILL "$pid" 2>/dev/null
    else
      echo "recorded launcher pid $pid is not running"
    fi
    rm -f "$PIDFILE"
  else
    echo "no pidfile at $PIDFILE"
  fi
  scancel -u "$USER" -n "$JOBNAME" 2>/dev/null && echo "scancelled $JOBNAME jobs"
  tmux kill-session -t "$TMUX_SESSION" 2>/dev/null && echo "killed tmux session $TMUX_SESSION"
  pkill -f "salloc .*--job-name=$JOBNAME" 2>/dev/null && echo "cleaned up stray salloc"
  echo "remaining:"
  squeue -u "$USER" -n "$JOBNAME"
  exit 0
fi

# ---------------------------------------------------------------- inner half --
if [ "${1:-}" = "--inside" ]; then
  export REPO DATA_ROOT SESSION TASKS MODEL EVERY_PS SAVE_INTERVAL_FS EXTRA_ARGS
  mkdir -p "$LOGDIR"
  echo $$ > "$PIDFILE"
  stop_now() {
    echo "# launcher told to stop -- releasing allocation"
    scancel -u "$USER" -n "$JOBNAME" 2>/dev/null
    rm -f "$PIDFILE"
    exit 143
  }
  trap stop_now TERM INT
  trap 'rm -f "$PIDFILE"' EXIT

  for ((c = 1; c <= CYCLES; c++)); do
    echo
    echo "###########################################################"
    echo "# cycle $c/$CYCLES   requesting $GPUS GPUs on $PARTITION for $WALLTIME"
    echo "# $(date)"
    echo "###########################################################"

    salloc --account="$ACCOUNT" --partition="$PARTITION" \
      --nodes=1 --ntasks-per-node=1 --gpus-per-node="$GPUS" \
      --cpus-per-task="$CPUS" --mem="$MEM" --time="$WALLTIME" \
      --exclude="$EXCLUDE" --job-name="$JOBNAME" \
      srun --unbuffered --ntasks=1 --gpus-per-node="$GPUS" bash "$POOL"
    rc=$?
    echo "# cycle $c finished rc=$rc at $(date)"

    left=0
    for idx in $TASKS; do
      f="$LOGDIR/task${idx}.rc"
      { [ -f "$f" ] && [ "$(cat "$f")" = "0" ]; } || left=$((left + 1))
    done
    if [ "$left" -eq 0 ]; then
      echo "# all tasks finished rc=0 -- stopping after cycle $c"
      break
    fi
    echo "# $left task(s) still unfinished -- allocating again (they restart from frame 0)"
  done
  echo
  echo "### launcher done at $(date)"
  exit 0
fi

# ---------------------------------------------------------------- outer half --
for f in "$SELF" "$POOL"; do
  [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done

errors=0
[ -f "$MODEL" ] || { echo "MISSING model : $MODEL" >&2; errors=$((errors + 1)); }
for idx in $TASKS; do
  t="${TRAJS[$idx]}"
  [ -f "$t" ] || { echo "MISSING traj  : $t" >&2; errors=$((errors + 1)); }
  out="$(dirname "$t")/uma_labels_every${EVERY_PS}ps.npz"
  [ -e "$out" ] && echo "NOTE  output exists (will be OVERWRITTEN): $out" >&2
done
[ "$errors" -gt 0 ] && { echo "Preflight failed with $errors problem(s); nothing launched." >&2; exit 1; }

if [ "${1:-}" = "--dry-run" ]; then
  cat <<EOF
DRY RUN -- nothing allocated.

  plan     : up to $CYCLES x $WALLTIME on $PARTITION, $GPUS GPUs, tasks: $TASKS
  pool     : $POOL
  model    : $MODEL
  sampling : every $EVERY_PS ps (frames $SAVE_INTERVAL_FS fs apart)
  logs     : $LOGDIR/task<N>_<name>.log
  outputs  : <traj dir>/uma_labels_every${EVERY_PS}ps.npz

  per cycle:
    salloc --account=$ACCOUNT --partition=$PARTITION --nodes=1 \\
      --ntasks-per-node=1 --gpus-per-node=$GPUS --cpus-per-task=$CPUS \\
      --mem=$MEM --time=$WALLTIME --exclude=$EXCLUDE --job-name=$JOBNAME \\
      srun --unbuffered --ntasks=1 --gpus-per-node=$GPUS bash $POOL

  tasks    : 0 = exp07 napf6_tgdme (GPU 0)   2 = exp04 napf6_tgdme (GPU 2)
             1 = exp07 lipf6_dme   (GPU 1)   3 = exp04 lipf6_dme   (GPU 3)
EOF
  exit 0
fi

mkdir -p "$LOGDIR"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$TMUX_SESSION' already exists on $(hostname -s)." >&2
  echo "       attach with:  tmux attach -t $TMUX_SESSION" >&2
  echo "       or stop it:   bash $SELF --stop" >&2
  exit 1
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: a launcher is already running (pid $(cat "$PIDFILE"))." >&2
  echo "       stop it first:  bash $SELF --stop" >&2
  exit 1
fi

tmux new-session -d -s "$TMUX_SESSION" \
  -e ACCOUNT="$ACCOUNT" -e PARTITION="$PARTITION" -e WALLTIME="$WALLTIME" \
  -e CYCLES="$CYCLES" -e GPUS="$GPUS" -e CPUS="$CPUS" -e MEM="$MEM" \
  -e EXCLUDE="$EXCLUDE" -e SESSION="$SESSION" -e TASKS="$TASKS" \
  -e REPO="$REPO" -e DATA_ROOT="$DATA_ROOT" -e JOBNAME="$JOBNAME" \
  -e MODEL="$MODEL" -e EVERY_PS="$EVERY_PS" -e SAVE_INTERVAL_FS="$SAVE_INTERVAL_FS" \
  -e EXTRA_ARGS="$EXTRA_ARGS" \
  "bash '$SELF' --inside 2>&1 | tee -a '$LAUNCHLOG'; echo; echo '=== launcher exited, window kept open ==='; exec bash"

cat <<EOF
Launched detached.

  login node : $(hostname -s)        <- tmux only exists here
  session    : $TMUX_SESSION
  plan       : up to $CYCLES x $WALLTIME on $PARTITION, $GPUS GPUs, tasks: $TASKS
  launcher   : $LAUNCHLOG
  task logs  : $LOGDIR/task<N>_<name>.log
  outputs    : <traj dir>/uma_labels_every${EVERY_PS}ps.npz

  watch      : tmux attach -t $TMUX_SESSION      (ctrl-b d to detach)
  queue      : squeue -u \$USER
  stop       : bash $SELF --stop            (do NOT just kill the tmux session)

Safe to close this terminal now.
EOF
