"""Appendix figure: MD stability and UMA-vs-PaiNN agreement along the 16
production transport trajectories (20 ns, 298 K NVT, Bussi).

Two figures, one per concentration group, 5 panels each plus a legend:

    energy / atom      temperature        pressure          <- what the MD did
    force MAE          max |dF|           (legend)          <- teacher vs student

Top row is the PaiNN simulation's own trace -- PaiNN drove the dynamics, so
this is a stability / equilibration check. (Temperature is purely kinetic and
therefore identical for both models; it can only ever be an MD trace.) Bottom
row is the frame-by-frame disagreement between the UMA-s-1p2 teacher and the
PaiNN student evaluated on the SAME configurations, so it measures accuracy
without any dynamics feedback.

Groups follow the figure's convention rather than the literal molarity:
``0.1M`` is the 7 dilute runs, ``1M`` is the 8 concentrated runs plus
lipf6_dme, which only exists at 0.5 M.

One line per system: colour = solvent, line style = salt. Keyed on the
salt/solvent alone, so a given pair looks identical in both figures.

Reads only the CSV written by ``compute_transport_diagnostics.py``, so the
figure redraws without touching the ~60 GB of trajectories. The CSV lives under
``REPO/data/experiments/exp04_correct_molarity_samebox/`` -- with the runs it
came from, and out of git.

    python plot_transport_stability_uma_agreement.py
    python plot_transport_stability_uma_agreement.py --smooth 10   # block-mean overlay
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
# Written by compute_transport_diagnostics.py. Lives with the runs it
# summarizes (REPO/data is a gitignored symlink to scratch) so a 1.2 MB derived
# artifact does not get committed alongside the figure.
CSV = paths.OBS_ROOT / "experiments" / "exp04_correct_molarity_samebox" / "transport_diagnostics.csv"

# Figures are written straight into the paper submodule, the same as the other
# appendix scripts, so rebuilding one updates the manuscript with no copy step.
# The copies sitting next to this script are symlinks back to these files.
OUT_DIR = paths.FIGURES_APPENDIX

SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}
SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}
# The nine systems are a 4-solvent x 3-salt grid, so we encode that structure
# rather than spend nine crowded hues: colour = solvent, line style = salt. This
# keeps only four well-separated colours (checked CVD-safe on a white surface --
# the one adjacent pair in the 6-8 band is covered by the redundant line style
# and the legend, see the README) and lets a reader see all the traces for one
# solvent group together, which is exactly what the accompanying text discusses.
# Both encodings key on the salt/solvent alone, never on concentration, so a
# given salt/solvent pair is drawn identically in both the 0.1 M and 1 M figures.
SOLV_COLOR = {"dme": "#4C72B0", "diglyme": "#55A868", "pc": "#C44E52", "tgdme": "#8159C6"}
# Solid / dashed / dotted, spaced widely enough to read on a thin translucent
# trace. Kept clearly distinct from the thin black *horizontal* reference line
# (the 298 K / zero setpoints), which is dark grey, not a system colour.
SALT_STYLE = {"napf6": "-", "naotf": (0, (6, 2)), "lipf6": (0, (1, 1.4))}

GROUP_TITLE = {
    "0.1M": "0.1 M systems",
    "1M": "1 M systems (LiPF$_6$/DME at 0.5 M)",
}

# (column, panel title, y label, optional horizontal reference line)
#
# Stress MAE is deliberately NOT shown. These are NVT runs with no barostat, so
# stress never enters the equations of motion and cannot affect the trajectory,
# and Green-Kubo viscosity uses only the off-diagonal stress ACF, to which the
# teacher-student disagreement contributes nothing systematic. The full-tensor
# stress MAE (219-802 bar) is also dominated by a near-constant isotropic
# pressure offset rather than by any error the dynamics could feel -- plotting
# it would imply a defect the simulations do not have. See the README.
PANELS = [
    ("energy_per_atom_eV", "Potential energy", "$E$ / atom (eV)", None),
    ("temperature_K", "Temperature", "$T$ (K)", 298.0),
    ("pressure_bar", "Pressure", "$P$ (bar)", 0.0),
    ("force_mae_eV_A", "Force MAE", r"MAE$(F)$ (eV/$\mathrm{\AA}$)", None),
    ("max_force_err_eV_A", "Max force error", r"max$_i\,\|\Delta F_i\|$ (eV/$\mathrm{\AA}$)", None),
]

# Type sizes, kept together so the whole figure rescales from one place.
FS_SUPTITLE = 17
FS_TITLE = 15
FS_LABEL = 16.5
FS_TICK = 12.5
FS_LEGEND = 12.5

# Instantaneous quantities whose frame-to-frame scatter dwarfs the differences
# between systems -- drawn lighter so the panel stays readable.
NOISY = {"temperature_K", "pressure_bar"}


def _split(system: str) -> tuple[str, str]:
    salt, solvent = system.split("_", 1)
    return salt, solvent


def _label(system: str, conc: str) -> str:
    salt, solvent = _split(system)
    base = f"{SALT_LABEL[salt]}/{SOLV_LABEL[solvent]}"
    return f"{base} (0.5 M)" if conc == "0_5M" else base


def _ordered_systems(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Salt then solvent, so the legend groups the way the text discusses them."""
    seen = df[["system", "conc"]].drop_duplicates()
    rows = [(r.system, r.conc) for r in seen.itertuples()]
    salt_order = {"napf6": 0, "naotf": 1, "lipf6": 2}
    solv_order = {"dme": 0, "diglyme": 1, "pc": 2, "tgdme": 3}
    return sorted(rows, key=lambda sc: (salt_order[_split(sc[0])[0]],
                                        solv_order[_split(sc[0])[1]]))


def draw_group(df: pd.DataFrame, group: str, smooth: int) -> plt.Figure:
    sub = df[df["group"] == group]
    systems = _ordered_systems(sub)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6))
    flat = axes.ravel()
    for ax, (col, title, ylabel, ref) in zip(flat, PANELS):
        for system, conc in systems:
            d = sub[(sub["system"] == system) & (sub["conc"] == conc)]
            salt, solvent = _split(system)
            color = SOLV_COLOR[solvent]
            style = SALT_STYLE[salt]
            t = d["time_ps"].values / 1000.0          # ns reads better than 20000 ps
            y = d[col].values
            if smooth > 1:
                # Raw stays visible underneath; the overlay only guides the eye.
                ax.plot(t, y, ls=style, color=color, lw=0.5, alpha=0.18)
                y = pd.Series(y).rolling(smooth, center=True, min_periods=1).mean().values
                ax.plot(t, y, ls=style, color=color, lw=1.4, alpha=0.95)
            else:
                # T and P swing by hundreds of K / thousands of bar frame to
                # frame, so 9 opaque traces would be a solid block. Thin and
                # translucent keeps the envelope legible.
                lw, alpha = (0.6, 0.65) if col in NOISY else (0.8, 0.85)
                ax.plot(t, y, ls=style, color=color, lw=lw, alpha=alpha)
        if ref is not None:
            # On top: the setpoint is the whole point of these two panels, and
            # underneath the traces it is invisible.
            ax.axhline(ref, color="0.15", lw=1.1, ls=(0, (5, 3)), zorder=5)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_xlabel("time (ns)", fontsize=FS_LABEL)
        ax.set_xlim(0, 20)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.25, lw=0.5)

    handles = [
        Line2D([], [], color=SOLV_COLOR[_split(s)[1]], ls=SALT_STYLE[_split(s)[0]],
               lw=1.8, label=_label(s, c))
        for s, c in systems
    ]
    # Dropping the stress panel frees the sixth cell -- the legend goes there
    # rather than under the figure, so nothing is left looking unfinished and
    # the entries get enough room to stack one per line.
    legend_ax = flat[len(PANELS)]
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", ncol=1, frameon=False,
                     fontsize=FS_LEGEND, handlelength=2.8, labelspacing=0.7)

    fig.suptitle(
        f"{GROUP_TITLE[group]} — 20 ns NVT at 298 K: MD stability (top) and "
        "UMA teacher vs PaiNN student (bottom)",
        fontsize=FS_SUPTITLE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--smooth", type=int, default=0,
                    help="block-mean window in frames (1 frame = 40 ps); 0 = raw only")
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for group, tag in (("0.1M", "0_1M"), ("1M", "1M")):
        if not (df["group"] == group).any():
            print(f"no rows for group {group}, skipping")
            continue
        fig = draw_group(df, group, args.smooth)
        p = outdir / f"figure_transport_stability_uma_agreement_{tag}.png"
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {p}")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
