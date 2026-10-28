#!/usr/bin/env bash
# Render fig1_workflow.html to a PNG using headless Chrome, writing directly
# into the paper_electrolytes submodule (fig1_workflow.png here is a symlink
# to that file).
#
# Usage:
#   ./render.sh                      # -> paper_electrolytes/Figures/main_paper/figure1_workflow.png  (2x, 3840 px = 540 dpi at 180 mm)
#   ./render.sh out.png              # custom output name
#   ./render.sh out.png 1            # custom name + device scale factor (1x for quick previews)
#
# No image tooling (PIL/ImageMagick) is needed — Chrome screenshots the page
# directly at the figure's native 1920x1081 canvas size.
#
# Browser: $CHROME if set, else the macOS Chrome app, else the first of
# google-chrome / chromium / chrome-headless-shell on PATH. On Perlmutter
# there is no system Chrome; install a headless shell with
#   pip install playwright && playwright install chromium-headless-shell
# and point CHROME at .../chromium_headless_shell-*/chrome-*/chrome-headless-shell.
set -euo pipefail

cd "$(dirname "$0")"

HTML="fig1_workflow.html"
OUT="${1:-../../../paper_electrolytes/Figures/main_paper/figure1_workflow.png}"
SCALE="${2:-2}"

W=1920
H=1081

MAC_CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [[ -z "${CHROME:-}" ]]; then
  if [[ -x "$MAC_CHROME" ]]; then
    CHROME="$MAC_CHROME"
  else
    for c in google-chrome chromium chromium-browser chrome-headless-shell; do
      if command -v "$c" >/dev/null; then CHROME="$(command -v "$c")"; break; fi
    done
  fi
fi
if [[ -z "${CHROME:-}" || ! -x "$CHROME" ]]; then
  echo "error: no Chrome/Chromium found; set CHROME=/path/to/chrome" >&2
  exit 1
fi

"$CHROME" \
  --headless --disable-gpu --hide-scrollbars \
  --default-background-color=FFFFFFFF \
  --force-device-scale-factor="$SCALE" \
  --window-size="$W,$H" \
  --screenshot="$OUT" \
  "file://$(pwd)/$HTML" 2>/dev/null

echo "wrote $OUT ($(( W * SCALE ))x$(( H * SCALE )) px)"
