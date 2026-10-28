"""Appendix figure: what the training-noise ablation does to 1 ns of NPT.

Three panels, four runs, one shared time axis:

    density            force MAE vs UMA        max |dF| vs UMA
    (did it survive)   (typical error)         (worst atom)

Two systems x two models. The models differ ONLY in their training set --
exp01's mixed-noise data versus exp07's true-MD-geometry-only data -- so the
gap between the two pairs of curves is the noise ablation and nothing else.

One colour per curve, all solid, all drawn from the main-paper figure 3 palette --
see CURVE_COLOR and the README for why this departs from colour-by-solvent.

Reads the two CSVs written by compute_no_noise_stability.py --
data/no_noise_stability_exp07.csv (no-noise ablation, produced on Delta) and
data/mixed_noise_stability_exp01.csv (mixed-noise reference, produced on Perlmutter)
-- and concatenates them. Either one missing is a warning, not an error, so the
figure can be previewed with just one half present.

    python plot_no_noise_stability.py
    python plot_no_noise_stability.py --smooth 5    # block-mean overlay on the error panels

Do not reach for --smooth to tidy the error panels: the NaPF6 crash is a single
2 ps frame at 19.3 eV/Ang, and a width-5 rolling mean averages it against its
neighbours down to 6.0, erasing the event the figure exists to show.
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
EXP07_CSV = HERE / "data" / "no_noise_stability_exp07.csv"      # no-noise ablation (Delta)
MIXED_CSV = HERE / "data" / "mixed_noise_stability_exp01.csv"   # mixed-noise reference (Perlmutter)

# Written straight into the paper submodule, as the other appendix scripts do, so
# rebuilding the figure updates the manuscript with no copy step. The copies next
# to this script are symlinks back to these files.
OUT_DIR = paths.FIGURES_APPENDIX
STEM = "figure_no_noise_ablation_stability"

SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}
SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}

MODEL_LABEL = {"exp01": "mixed noise", "exp07": "no noise"}

# Four curves, four distinct colours, all from the main-paper figure 3 palette
# (SOLV_COLOR there: DME #E8850E, DEGDME #1ad73c, PC #E6197D, TEGDME #2E6FBF).
#
# The two reference runs keep their own solvent's colour, so they match figure 3.
# The two ablation runs borrow the DEGDME and PC entries -- those colours carry NO
# solvent meaning here, they are just the remaining palette slots. Colouring by
# model instead (two blues, two reds) made the four curves genuinely hard to tell
# apart, which is why this figure departs from the colour-by-solvent convention.
CURVE_COLOR = {
    ("exp01", "napf6_tgdme"): "#2E6FBF",   # TEGDME, as in figure 3
    ("exp01", "lipf6_dme"): "#E8850E",     # DME, as in figure 3
    ("exp07", "napf6_tgdme"): "#1ad73c",   # borrowed
    ("exp07", "lipf6_dme"): "#E6197D",     # borrowed
}
# All four curves are solid: the colours already separate them, and dashes on the
# noisy log panels only broke the lines up and made them harder to follow.

# Below this the box is no longer a liquid. Same threshold the run health checks
# use; it sits far from both the liquid densities (~1.0-1.2) and the evaporated
# ones (~3e-4), so nothing hinges on the exact value.
EVAP_RHO = 0.7

# (column, panel title, y label, log y)
PANELS = [
    ("density_g_cm3", "Density", r"$\rho$ (g/cm$^3$)", False),
    ("force_mae_eV_A", "Force MAE vs UMA", r"MAE$(F)$ (eV/$\mathrm{\AA}$)", True),
    ("max_force_err_eV_A", "Max force error vs UMA",
     r"max$_i\,\|\Delta F_i\|$ (eV/$\mathrm{\AA}$)", True),
]

FS_SUPTITLE = 19
FS_TITLE = 18
FS_LABEL = 16.5
FS_TICK = 12.5
FS_LEGEND = 12.5

# exp01 first so the reference is drawn under the ablation, and napf6 before
# lipf6 within each model -- matches the order the legend and the text use.
ORDER = [("exp01", "napf6_tgdme"), ("exp01", "lipf6_dme"),
         ("exp07", "napf6_tgdme"), ("exp07", "lipf6_dme")]


def _label(model: str, system: str, conc: str) -> str:
    salt, solvent = system.split("_", 1)
    base = f"{SALT_LABEL[salt]}/{SOLV_LABEL[solvent]}"
    if conc == "0_5M":
        base += " (0.5 M)"
    return f"{base} — {MODEL_LABEL[model]}"


def _evaporation_onset(d: pd.DataFrame) -> float | None:
    """Time the density first drops below EVAP_RHO *having once been above it*.

    The "having once been above it" guard means a run that never reaches liquid
    density is reported as no-onset rather than as an evaporation at t = 0. Both
    current pairs (exp01 replicas and exp07) are seeded from equilibrated frames
    and start well above EVAP_RHO, so the guard is not load-bearing today; it is
    kept because it is the correct definition of an evaporation crossing.
    """
    rho = d["density_g_cm3"].values
    above = rho >= EVAP_RHO
    if not above.any():
        return None
    after = d.iloc[int(above.argmax()):]
    below = after[after["density_g_cm3"] < EVAP_RHO]
    return float(below["time_ps"].iloc[0]) if len(below) else None


def draw(df: pd.DataFrame, smooth: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    t_max = float(df["time_ps"].max())

    for ax, (col, title, ylabel, logy) in zip(axes, PANELS):
        for model, system in ORDER:
            d = df[(df["model"] == model) & (df["system"] == system)]
            if not len(d):
                continue
            color = CURVE_COLOR[(model, system)]
            t = d["time_ps"].values
            y = d[col].values
            if smooth > 1 and logy:
                # Only the error panels are noisy enough to want this; density is
                # already smooth and a rolling mean there would hide the collapse.
                ax.plot(t, y, color=color, lw=0.8, alpha=0.20, zorder=2)
                y = pd.Series(y).rolling(smooth, center=True, min_periods=1).mean().values
            ax.plot(t, y, color=color, lw=2.3, alpha=0.9, zorder=3,
                    solid_capstyle="round")

            # A run that ends early would otherwise just stop in open space and
            # read as missing data; the dot says "this is where it ends".
            if t[-1] < t_max * 0.9:
                ax.plot(t[-1], y[-1], marker="o", ms=6.0, color=color,
                        mec="white", mew=0.9, zorder=4)

        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_xlabel("time (ps)", fontsize=FS_LABEL)
        ax.set_xlim(0, t_max)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.25, lw=0.5)
        if logy:
            ax.set_yscale("log")
        else:
            # Linear: on a log axis the 4-decade fall to ~3e-4 would squeeze the
            # entire liquid range into a sliver and hide the pre-collapse drift,
            # which is the part worth looking at.
            #
            # The evaporated runs settle at ~3e-4 g/cm^3, which on this scale is
            # zero. With the axis starting exactly at 0 their traces are drawn
            # underneath the bottom spine and simply vanish -- it reads as the
            # line stopping rather than as a box with nothing left in it. A little
            # padding below zero lifts them clear of the spine so the flat
            # near-zero tail stays visible; no line width can fix it otherwise.
            top = max(1.35, float(df["density_g_cm3"].max()) * 1.1)
            ax.set_ylim(-0.045 * top, top)
            ax.set_yticks([0.5 * i for i in range(int(top / 0.5) + 1)])

    # The napf6 collapse is over in one 2 ps frame at the very left of a 1000 ps
    # axis, so it is drawn but easy to miss. Point at it rather than leave the
    # reader to find it. The dot at 200 ps marks where the run was stopped by hand
    # once it was unambiguously an ideal gas; see the README.
    d07 = df[(df["model"] == "exp07") & (df["system"] == "napf6_tgdme")]
    onset07 = _evaporation_onset(d07) if len(d07) else None
    if onset07 is not None:
        axes[0].annotate(
            f"evaporates at {onset07:.0f} ps",
            xy=(onset07, 0.55), xytext=(0.10 * t_max, 0.30),
            fontsize=FS_LEGEND - 1, color=CURVE_COLOR[("exp07", "napf6_tgdme")],
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->",
                            color=CURVE_COLOR[("exp07", "napf6_tgdme")],
                            lw=1.0, alpha=0.85,
                            connectionstyle="arc3,rad=-0.2"),
        )

    handles = [
        Line2D([], [], color=CURVE_COLOR[(m, s)], lw=2.4,
               label=_label(m, s, df[(df["model"] == m) & (df["system"] == s)]["conc"].iloc[0]))
        for m, s in ORDER
        if len(df[(df["model"] == m) & (df["system"] == s)])
    ]

    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=FS_LEGEND, handlelength=2.4, columnspacing=2.4,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "1 ns NPT at 298 K — training-noise ablation, scored against the UMA-S teacher",
        fontsize=FS_SUPTITLE,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    return fig


def _load(csv_paths: list[str]) -> pd.DataFrame:
    """Concatenate whichever of the given CSVs exist; missing is a warning."""
    frames = []
    for p in csv_paths:
        if Path(p).exists():
            frames.append(pd.read_csv(p))
        else:
            print(f"warning: {p} missing -- run compute_no_noise_stability.py for it")
    if not frames:
        raise SystemExit(f"no input CSVs found among: {csv_paths}")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp07-csv", default=str(EXP07_CSV),
                    help="no-noise ablation CSV (compute --mode no_noise)")
    ap.add_argument("--mixed-csv", default=str(MIXED_CSV),
                    help="mixed-noise reference CSV (compute --mode mixed)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="block-mean window in frames (1 frame = 2 ps) for the error panels")
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    df = _load([args.mixed_csv, args.exp07_csv])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig = draw(df, args.smooth)
    p = outdir / f"{STEM}.png"
    fig.savefig(p, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
