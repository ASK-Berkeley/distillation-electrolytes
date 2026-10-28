#!/bin/bash
# Thin wrapper: source config.sh, then run the production driver.
#   ./run_production.sh --gpus 0 --per-gpu 6
#   ./run_production.sh --dry-run
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
# GMXRC references unset vars on purpose, so it cannot survive `set -u`; it also
# redefines GMXBIN as the bin *directory*, which is not what we mean by GMXBIN.
_GMXBIN="$GMXBIN"
set +u
[ -f "$GMXRC" ] && source "$GMXRC"
set -u
export GMXBIN="$_GMXBIN"
exec "${OPLS_PYTHON:-python}" run_production.py "$@"
