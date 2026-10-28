#!/bin/bash
# Rerun the OPLS transport observables directly on a Perlmutter LOGIN node,
# overwriting under data/opls_transport_observables/:
#   viscosity/    (GK shear viscosity, 2 ns eq cut)          -- always
#   diffusivity/  (Einstein self-D, fixed+Fickian, YH corr)  -- always
#   conductivity/ (Onsager + Nernst-Einstein, byteff2_ase)   -- opt-in, RUN_COND=1
#
# Order matters: viscosity runs first and writes viscosity_gk_by_box.csv, which
# the diffusivity driver reads for the sim-eta Yeh-Hummer correction. This script
# runs them back-to-back so that hand-off is automatic. Conductivity is
# independent of the other two (its own MSD path) and rarely needs rerunning, so
# it is documented here for completeness but SKIPPED unless RUN_COND=1 is set.
#
# WHY THESE --jobs VALUES (do not bump blindly):
#   NERSC login nodes enforce a ~30 GiB per-user memory cgroup cap
#   (`cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max`). Each GK
#   viscosity worker holds a ~3.5 GB stress log (10M rows) in RAM, so >~8 workers
#   trips the OOM killer and reaps the whole ProcessPool ("A process in the process
#   pool was terminated abruptly"). VISC_JOBS=6 -> ~21 GB peak, safe headroom.
#   Diffusivity streams the .xtc and is lighter, so DIFF_JOBS=8 is fine.
#   For more parallelism, submit to a CPU compute node instead (512 GB, ~128 cores).
#
# Usage:
#   bash src/analysis/observables/transport/run_opls_transport_login.sh
#   VISC_JOBS=6 DIFF_JOBS=8 bash .../run_opls_transport_login.sh   # override worker counts
#   RUN_COND=1 bash .../run_opls_transport_login.sh                # also rerun conductivity
#
# Env: fennol (needs mdtraj for the .xtc reader). CPU only. ~8 min viscosity + ~4 min diffusivity
# (+ ~4 min conductivity if RUN_COND=1).

set -euo pipefail

VISC_JOBS="${VISC_JOBS:-6}"
DIFF_JOBS="${DIFF_JOBS:-8}"
COND_JOBS="${COND_JOBS:-8}"
RUN_COND="${RUN_COND:-0}"

# Repo root = four levels up from this script (src/analysis/observables/transport).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
PKG="src.analysis.observables.transport"

source /global/homes/i/ishan_a/mambaforge/etc/profile.d/conda.sh
conda activate fennol
export PYTHONNOUSERSITE=1

cd "$REPO"
mkdir -p slurm_logs
LOG="slurm_logs/opls_transport_login.log"
: > "$LOG"

echo "=== start $(date) on $(hostname) [login; 30 GiB cap] ===" | tee -a "$LOG"
MEMMAX="$(cat /sys/fs/cgroup/user.slice/user-"$(id -u)".slice/memory.max 2>/dev/null || echo '?')"
echo "    memory.max = $MEMMAX bytes | VISC_JOBS=$VISC_JOBS DIFF_JOBS=$DIFF_JOBS" | tee -a "$LOG"

echo "=== [1/2] GK viscosity (2 ns eq cut) --jobs $VISC_JOBS ===" | tee -a "$LOG"
python -m "$PKG.viscosity_gk_opls_replicas" --jobs "$VISC_JOBS" >> "$LOG" 2>&1

echo "=== [2/3] Einstein diffusivity (fixed+Fickian) --jobs $DIFF_JOBS ===" | tee -a "$LOG"
# default --visc-suffix '' reads the viscosity_gk_by_box.csv just written above
python -m "$PKG.diffusivity_opls_replicas" --jobs "$DIFF_JOBS" >> "$LOG" 2>&1

# Conductivity: Onsager + Nernst-Einstein, default --window byteff2_ase, 298 K boxes
# only (16 boxes / 64 replicas). Independent of the two steps above. Opt-in.
if [ "$RUN_COND" = "1" ]; then
  echo "=== [3/3] Onsager/NE conductivity (byteff2_ase) --jobs $COND_JOBS ===" | tee -a "$LOG"
  python -m "$PKG.conductivity_opls_replicas" --jobs "$COND_JOBS" >> "$LOG" 2>&1
else
  echo "=== [3/3] conductivity SKIPPED (set RUN_COND=1 to rerun). Command would be:" | tee -a "$LOG"
  echo "    python -m $PKG.conductivity_opls_replicas --jobs $COND_JOBS" | tee -a "$LOG"
fi

echo "=== done $(date) -> data/opls_transport_observables/{viscosity,diffusivity$([ "$RUN_COND" = "1" ] && echo ,conductivity)} ===" | tee -a "$LOG"
echo "    full log: $LOG"
