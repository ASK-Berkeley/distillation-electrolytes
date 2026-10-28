"""Appendix figure: the effect of the solute:solvent ratio correction on the
molarity each electrolyte box actually equilibrates to.

Two panels, one per concentration group. For each system an arrow runs from the
initial box's equilibrated molarity to the corrected box's, against a dashed
line at the nominal target. The point is that the initial boxes miss the target
in a systematic direction -- always over-concentrated -- and the corrected ones
land on it.

lipf6_dme is drawn in the 1 M panel to match the other appendix figures, but it
targets 0.5 M, so it gets its own short reference segment rather than being
measured against the panel's 1 M line.

Reads only ``molarity_correction.csv`` (written by
``compute_molarity_correction.py``) and writes the PNG straight into the paper
submodule, like the other figure scripts.

    python plot_molarity_correction.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from src.analysis.observables import paths

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CSV = HERE / "molarity_correction.csv"
OUT_DIR = paths.FIGURES_APPENDIX

SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}
SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}

C_BEFORE = "#C44E52"   # initial boxes -- the ones that miss
C_AFTER = "#4C72B0"    # corrected boxes
C_ARROW = "0.45"

PANELS = [("0.1M", "0.1 M systems"), ("1M", "1 M systems")]

FS_TITLE = 14
FS_LABEL = 14
FS_TICK = 11
FS_LEGEND = 11


def label(row) -> str:
    return f"{SALT_LABEL[row.salt]}/{SOLV_LABEL[row.solvent]}"


def draw_panel(ax, sub: pd.DataFrame, title: str) -> None:
    x = range(len(sub))

    # The panel's own target line, drawn only across the systems that share it,
    # so the 0.5 M outlier is not silently measured against a 1 M line.
    main_target = sub["target_M"].mode().iloc[0]
    shared = [i for i, t in enumerate(sub["target_M"]) if t == main_target]
    ax.hlines(main_target, min(shared) - 0.45, max(shared) + 0.45,
              color="0.15", lw=1.2, ls=(0, (5, 3)), zorder=1)
    for i, t in enumerate(sub["target_M"]):
        if t == main_target:
            continue
        ax.hlines(t, i - 0.45, i + 0.45, color="0.15", lw=1.2, ls=(0, (5, 3)),
                  zorder=1)
        # Left of the segment and above the line: below it would land in the
        # tick labels, and centred would sit on top of this system's arrow.
        ax.annotate(f"{t:g} M target", xy=(i - 0.45, t), xytext=(-4, 4),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=FS_TICK - 1, color="0.25")

    for i, row in enumerate(sub.itertuples()):
        # Arrow first so the markers sit on top of its head.
        ax.annotate("", xy=(i, row.m_production), xytext=(i, row.m_initial),
                    arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4,
                                    shrinkA=3.5, shrinkB=3.5))
        ax.plot(i, row.m_initial, "o", ms=8, mfc="white", mec=C_BEFORE,
                mew=1.8, zorder=3)
        ax.plot(i, row.m_production, "o", ms=8, color=C_AFTER, zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels([label(r) for r in sub.itertuples()], rotation=30,
                       ha="right", fontsize=FS_TICK)
    ax.set_ylabel("equilibrated molarity (M)", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_xlim(-0.6, len(sub) - 0.4)
    # Headroom so no marker sits on the frame; the 0.5 M outlier otherwise
    # lands flush on the panel floor.
    lo = min(sub["m_production"].min(), sub["target_M"].min())
    hi = max(sub["m_initial"].max(), sub["target_M"].max())
    pad = 0.10 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0),
                             gridspec_kw={"width_ratios": [7, 9]})
    for ax, (group, title) in zip(axes, PANELS):
        sub = df[df["group"] == group].reset_index(drop=True)
        if sub.empty:
            print(f"no rows for group {group}")
            continue
        draw_panel(ax, sub, title)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=8, mfc="white", mec=C_BEFORE,
               mew=1.8, label="initial boxes (packed to nominal target)"),
        Line2D([], [], ls="none", marker="o", ms=8, color=C_AFTER,
               label="corrected boxes (fixed solute:solvent ratio)"),
        Line2D([], [], color="0.15", lw=1.2, ls=(0, (5, 3)), label="nominal target"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=FS_LEGEND, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Equilibrated molarity before and after the solute:solvent "
                 "ratio correction", fontsize=FS_TITLE + 2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "figure_molarity_correction.png"
    fig.savefig(p, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
