#!/bin/bash
#
# Launch the two exp07_no_noise 1 ns NPT runs (298 K) on an interactive Delta
# node, detached, so no terminal has to stay open.
#
# Why interactive: ghx4 has ~1100 jobs pending, ghx4-interactive has a couple
# dozen, so this starts in minutes instead of hours. The catch is that
# ghx4-interactive caps walltime at 2:00:00 and a 1 ns run takes roughly that,
# so this allocates in a loop: each cycle grabs a whole node (4 GH200s), runs
# pool_exp07_npt.sh (2 systems, 1 GPU each) until the wall clock kills it, then
# allocates again. Both runs resume from their .dyn.restart and tasks that
# finished rc=0 are skipped, so cycles are additive and the loop exits early
# once both are done. Two cycles = ~4 h of runtime, as requested.
#
# Everything runs inside a detached tmux session on this login node, so you can
# close the terminal (or the ssh connection) and it keeps going.
#
#   bash src/simulate/fennol_configs/launch_exp07_npt_interactive.sh            # 2 cycles x 2 h
#   bash src/simulate/fennol_configs/launch_exp07_npt_interactive.sh --dry-run  # print the plan only
#   CYCLES=3 bash src/simulate/fennol_configs/launch_exp07_npt_interactive.sh
#   TASKS=0  bash src/simulate/fennol_configs/launch_exp07_npt_interactive.sh   # napf6_tgdme only
#   PARTITION=ghx4 WALLTIME=04:00:00 CYCLES=1 bash ...                          # batch queue instead
#
# Watch it:   tmux attach -t exp07npt                 (detach again with ctrl-b d)
#             tail -f /work/nvme/beye/iamin/electrolytes_data/experiments/exp07_no_noise/pool_logs/exp07npt/task0.log
# Stop it:    bash src/simulate/fennol_configs/launch_exp07_npt_interactive.sh --stop
#
# Use --stop, NOT `tmux kill-session`. The cycle loop runs in a pipeline into
# tee, so killing the session orphans it (reparented to init) and it happily
# allocates its next cycle with nobody watching. --stop kills the recorded
# launcher pid first, then scancels.
#
# tmux lives on ONE login node -- if you reconnect to a different gh-login you
# will not see the session. This script prints which one to come back to.

set -uo pipefail

: "${ACCOUNT:=beye-dtai-gh}"
: "${PARTITION:=ghx4-interactive}"
: "${WALLTIME:=02:00:00}"        # ghx4-interactive hard cap
: "${CYCLES:=2}"                 # 2 x 2 h ~= the 4 h asked for
: "${GPUS:=4}"                   # whole node; the pool only uses 2 of them
: "${CPUS:=64}"
: "${MEM:=360G}"
: "${EXCLUDE:=gh029,gh093}"      # known interconnect trouble
: "${SESSION:=exp07npt}"
: "${TMUX_SESSION:=$SESSION}"
: "${TASKS:=0 1}"
: "${REPO:=/u/iamin/electrolytes}"
: "${DATA_ROOT:=/work/nvme/beye/iamin/electrolytes_data}"
: "${EXP:=exp07_no_noise}"
: "${JOBNAME:=exp07_npt_pool}"

SELF="$REPO/src/simulate/fennol_configs/launch_exp07_npt_interactive.sh"
POOL="$REPO/src/simulate/fennol_configs/pool_exp07_npt.sh"
LOGDIR="$DATA_ROOT/experiments/$EXP/pool_logs/$SESSION"
LAUNCHLOG="$LOGDIR/launcher.log"
PIDFILE="$LOGDIR/launcher.pid"

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
  # Kill the allocation after the launcher, so it cannot start another cycle.
  scancel -u "$USER" -n "$JOBNAME" 2>/dev/null && echo "scancelled $JOBNAME jobs"
  tmux kill-session -t "$TMUX_SESSION" 2>/dev/null && echo "killed tmux session $TMUX_SESSION"
  # A salloc orphaned by an earlier bad stop holds its node until killed.
  pkill -f "salloc .*--job-name=$JOBNAME" 2>/dev/null && echo "cleaned up stray salloc"
  echo "remaining:"
  squeue -u "$USER" -n "$JOBNAME"
  exit 0
fi

# ---------------------------------------------------------------- inner half --
# Runs inside tmux: allocate, run the pool on the node, repeat.
if [ "${1:-}" = "--inside" ]; then
  export REPO DATA_ROOT EXP SESSION TASKS
  mkdir -p "$LOGDIR"
  echo $$ > "$PIDFILE"
  # Without this, a TERM leaves the current salloc holding a node with nothing
  # running on it -- exactly the failure --stop exists to prevent.
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

    # Stop as soon as every requested task has a clean .rc -- no point burning
    # another allocation just to have the pool exit immediately.
    left=0
    for idx in $TASKS; do
      f="$LOGDIR/task${idx}.rc"
      { [ -f "$f" ] && [ "$(cat "$f")" = "0" ]; } || left=$((left + 1))
    done
    if [ "$left" -eq 0 ]; then
      echo "# all tasks finished rc=0 -- stopping after cycle $c"
      break
    fi
    echo "# $left task(s) still unfinished -- allocating again"
  done
  echo
  echo "### launcher done at $(date)"
  exit 0
fi

# ---------------------------------------------------------------- outer half --
for f in "$SELF" "$POOL"; do
  [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done

# Preflight: fail loudly here rather than 30 s into an allocation.
errors=0
declare -A SEEDS=( [napf6_tgdme]=simulation_start_1M.xyz [lipf6_dme]=simulation_start_0_5M.xyz )
for sys in napf6_tgdme lipf6_dme; do
  model="$DATA_ROOT/experiments/$EXP/$sys/model/best_model.fnx"
  seed="$DATA_ROOT/datasets/version05_no_noise/$sys/${SEEDS[$sys]}"
  run_dir="$DATA_ROOT/experiments/$EXP/$sys/npt_"*"_298K_1ns"
  [ -f "$model" ] || { echo "MISSING model : $model" >&2; errors=$((errors + 1)); }
  [ -f "$seed" ]  || { echo "MISSING seed  : $seed"  >&2; errors=$((errors + 1)); }
  for d in $run_dir; do
    [ -e "$d" ] && echo "NOTE  run dir exists (will RESUME, not restart): $d" >&2
  done
done
[ "$errors" -gt 0 ] && { echo "Preflight failed with $errors problem(s); nothing launched." >&2; exit 1; }

if [ "${1:-}" = "--dry-run" ]; then
  cat <<EOF
DRY RUN -- nothing allocated.

  plan     : up to $CYCLES x $WALLTIME on $PARTITION, $GPUS GPUs, tasks: $TASKS
  pool     : $POOL
  logs     : $LOGDIR/task<N>.log
  runs     : $DATA_ROOT/experiments/$EXP/<system>/npt_<conc>_298K_1ns

  per cycle:
    salloc --account=$ACCOUNT --partition=$PARTITION --nodes=1 \\
      --ntasks-per-node=1 --gpus-per-node=$GPUS --cpus-per-task=$CPUS \\
      --mem=$MEM --time=$WALLTIME --exclude=$EXCLUDE --job-name=$JOBNAME \\
      srun --unbuffered --ntasks=1 --gpus-per-node=$GPUS bash $POOL

  tasks    : 0 = napf6_tgdme (1M,   GPU 0)
             1 = lipf6_dme   (0_5M, GPU 1)
  settings : NPT 298 K / 1 bar, dt 1 fs, 1e6 steps (1 ns), 1 ps dumps,
             matmul_prec=highest (fp32), seed 1, best_model.fnx
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

# A live pidfile means a launcher is still cycling even if its tmux session is
# gone (orphaned by a `tmux kill-session`). Two launchers would race on the same
# run dirs, so refuse rather than double-allocate.
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: a launcher is already running (pid $(cat "$PIDFILE"))." >&2
  echo "       stop it first:  bash $SELF --stop" >&2
  exit 1
fi

tmux new-session -d -s "$TMUX_SESSION" \
  -e ACCOUNT="$ACCOUNT" -e PARTITION="$PARTITION" -e WALLTIME="$WALLTIME" \
  -e CYCLES="$CYCLES" -e GPUS="$GPUS" -e CPUS="$CPUS" -e MEM="$MEM" \
  -e EXCLUDE="$EXCLUDE" -e SESSION="$SESSION" -e TASKS="$TASKS" \
  -e REPO="$REPO" -e DATA_ROOT="$DATA_ROOT" -e EXP="$EXP" -e JOBNAME="$JOBNAME" \
  "bash '$SELF' --inside 2>&1 | tee -a '$LAUNCHLOG'; echo; echo '=== launcher exited, window kept open ==='; exec bash"

cat <<EOF
Launched detached.

  login node : $(hostname -s)        <- tmux only exists here
  session    : $TMUX_SESSION
  plan       : up to $CYCLES x $WALLTIME on $PARTITION, $GPUS GPUs, tasks: $TASKS
  launcher   : $LAUNCHLOG
  run logs   : $LOGDIR/task<N>.log
  runs       : $DATA_ROOT/experiments/$EXP/<system>/npt_<conc>_298K_1ns

  watch      : tmux attach -t $TMUX_SESSION      (ctrl-b d to detach)
  queue      : squeue -u \$USER
  stop       : bash $SELF --stop            (do NOT just kill the tmux session)

Safe to close this terminal now.
EOF
