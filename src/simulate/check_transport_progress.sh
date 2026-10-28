#!/bin/bash
# Progress report for the exp04 transport runs launched by launch_transport.sh.
#
# For every <system>/transport_<conc>_<T>K_r<rep> run dir it prints how far the
# simulation has actually got, in ns, taken from the last timestamp in the
# .stress log (written every STRESS_INTERVAL steps, so it is the finest-grained
# and most up-to-date record of simulated time -- the .traj is binary and the
# .dyn.restart only checkpoints every INTERVAL steps).
#
# Usage:
#   src/simulate/check_transport_progress.sh              # snapshot table
#   src/simulate/check_transport_progress.sh --rate       # + live ns/day (20 s sample)
#   src/simulate/check_transport_progress.sh --rate 60    # + live ns/day (60 s sample)
#
# Env overrides:
#   EXP_DIR    experiment dir (default exp04_correct_molarity_samebox)
#   TARGET_NS  target length in ns (default 15, matching STEPS=15000000 @ 1 fs)
#   TEMP       temperature tag in the run-dir name (default 298)
set -uo pipefail

DATA_ROOT=${DATA_ROOT:-/work/nvme/beye/iamin/electrolytes_data}
EXP_DIR=${EXP_DIR:-$DATA_ROOT/experiments/exp04_correct_molarity_samebox}
TARGET_NS=${TARGET_NS:-15}
TEMP=${TEMP:-298}

RATE=0
RATE_SEC=20
if [ "${1:-}" = "--rate" ]; then
  RATE=1
  [ -n "${2:-}" ] && RATE_SEC="$2"
fi

# Last simulated time (ps) in a .stress log. Reads the tail only, and ignores a
# partially-flushed final line by keeping the last row with a numeric time field.
last_ps() {
  tail -5 "$1" 2>/dev/null | awk '$1 ~ /^[0-9]+(\.[0-9]+)?$/ {v=$1} END {if (v=="") print 0; else print v}'
}

# job name -> slurm state, so the table can say whether a run is live or parked.
declare -A STATE
declare -A TLEFT
while read -r name st left; do
  [ -n "$name" ] && STATE["$name"]="$st" && TLEFT["$name"]="$left"
done < <(squeue --me -h -o "%j %T %L" 2>/dev/null)

run_dirs=()
while IFS= read -r d; do run_dirs+=("$d"); done < <(
  find "$EXP_DIR" -mindepth 2 -maxdepth 2 -type d -name "transport_*_${TEMP}K_r*" | sort
)

if [ ${#run_dirs[@]} -eq 0 ]; then
  echo "No transport run dirs found under $EXP_DIR" >&2
  exit 1
fi

# First pass: current position.
declare -A NS_NOW
declare -A JOBNAME
declare -A AGE
for d in "${run_dirs[@]}"; do
  sys=$(basename "$(dirname "$d")")
  base=$(basename "$d")
  rest=${base#transport_}          # e.g. 1M_298K_r0
  conc=${rest%%_${TEMP}K_*}        # e.g. 1M
  rep=${rest##*_}                  # e.g. r0
  JOBNAME["$d"]="transport_${sys}_${conc}_${rep}"

  s=$(ls "$d"/*.stress 2>/dev/null | head -1)
  if [ -n "$s" ]; then
    NS_NOW["$d"]=$(awk -v p="$(last_ps "$s")" 'BEGIN{printf "%.3f", p/1000}')
    AGE["$d"]=$(( ($(date +%s) - $(stat -c %Y "$s")) / 60 ))
  else
    NS_NOW["$d"]=0
    AGE["$d"]=-1
  fi
done

# Optional second pass: sample again to get an instantaneous rate.
declare -A NSDAY
if [ "$RATE" -eq 1 ]; then
  echo "Sampling for ${RATE_SEC}s to measure live ns/day ..." >&2
  sleep "$RATE_SEC"
  for d in "${run_dirs[@]}"; do
    s=$(ls "$d"/*.stress 2>/dev/null | head -1)
    [ -z "$s" ] && continue
    now=$(awk -v p="$(last_ps "$s")" 'BEGIN{printf "%.6f", p/1000}')
    NSDAY["$d"]=$(awk -v a="${NS_NOW[$d]}" -v b="$now" -v t="$RATE_SEC" \
      'BEGIN{d=b-a; if (d<=0) print "-"; else printf "%.1f", d*86400/t}')
  done
fi

printf '%-16s %-6s %-4s %8s %8s  %-22s %-9s %s\n' \
  SYSTEM CONC REP "ns" "%" "PROGRESS" "STATE" "$( [ "$RATE" -eq 1 ] && echo 'ns/day' || echo 'updated')"
printf '%.0s-' {1..104}; echo

total_ns=0
n_done=0
for d in "${run_dirs[@]}"; do
  sys=$(basename "$(dirname "$d")")
  base=$(basename "$d")
  rest=${base#transport_}
  conc=${rest%%_${TEMP}K_*}
  rep=${rest##*_}
  ns=${NS_NOW[$d]}
  job=${JOBNAME[$d]}

  pct=$(awk -v n="$ns" -v t="$TARGET_NS" 'BEGIN{printf "%.0f", 100*n/t}')
  bar=$(awk -v p="$pct" 'BEGIN{f=int(p/5); if(f>20)f=20; for(i=0;i<f;i++)printf "#"; for(i=f;i<20;i++)printf "."}')

  st=${STATE[$job]:-}
  if awk -v n="$ns" -v t="$TARGET_NS" 'BEGIN{exit !(n>=t-0.01)}'; then
    st_txt="DONE"; n_done=$((n_done + 1))
  elif [ "$st" = "RUNNING" ]; then
    st_txt="RUN ${TLEFT[$job]}"
  elif [ "$st" = "PENDING" ]; then
    st_txt="PEND"
  elif [ -z "$st" ]; then
    st_txt="-- none --"
  else
    st_txt="$st"
  fi

  if [ "$RATE" -eq 1 ]; then
    last_col=${NSDAY[$d]:--}
  elif [ "${AGE[$d]}" -lt 0 ]; then
    last_col="no .stress"
  else
    last_col="${AGE[$d]}m ago"
  fi

  printf '%-16s %-6s %-4s %8s %7s%% [%s] %-9s %s\n' \
    "$sys" "$conc" "$rep" "$ns" "$pct" "$bar" "$st_txt" "$last_col"
  total_ns=$(awk -v a="$total_ns" -v b="$ns" 'BEGIN{printf "%.3f", a+b}')
done

n=${#run_dirs[@]}
printf '%.0s-' {1..104}; echo
awk -v tot="$total_ns" -v n="$n" -v t="$TARGET_NS" -v d="$n_done" 'BEGIN{
  printf "%d runs | %.2f / %d ns total (%.0f%%) | mean %.2f ns | %d complete | %.1f ns remaining\n",
    n, tot, n*t, 100*tot/(n*t), tot/n, d, n*t-tot
}'
