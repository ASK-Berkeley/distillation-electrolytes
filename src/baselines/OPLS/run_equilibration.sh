#!/bin/bash
# Thin wrapper: source config.sh, then run the equilibration driver.
#   ./run_equilibration.sh --gpus 0,1
#   ./run_equilibration.sh --sys lipf6 --dry-run
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
# GMXRC references unset vars on purpose, so it cannot survive `set -u`; it also
# redefines GMXBIN as the bin *directory* (GROMACS' own convention), which is not
# what OPLS_code means by GMXBIN. Save ours and put it back afterwards.
_GMXBIN="$GMXBIN"
set +u
[ -f "$GMXRC" ] && source "$GMXRC"
set -u
export GMXBIN="$_GMXBIN"
exec "$OPLS_PYTHON" run_equilibration.py "$@"
