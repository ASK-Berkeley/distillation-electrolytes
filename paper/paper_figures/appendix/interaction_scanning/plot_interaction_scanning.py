#!/usr/bin/env python3
"""Appendix figure: gas-phase ion-pair interaction scans, 2 rows x 3 columns.

Rows:    conservative axial force -dE/dd (top)  /  interaction energy (bottom)
Columns: Na+-OTf- (attraction) | OTf-...OTf- (repulsion) | Na+...Na+ (repulsion)

Self-contained: the cache loaders and the x-axis labels (from
source_scans/gas_phase_geoms.py) are inlined, so redrawing needs only numpy,
matplotlib and paths.py.

Inputs (all local):
  data/{system}__{model}.npz        per-model scan caches: D, E_raw, F_axial, model
  data/DFT/{system}_dft_energies.csv  wB97M-V/def2-TZVPD reference: d_angstrom, E_hartree

Every curve uses the CONSERVATIVE force -dE/dd (not the model's raw F_axial), so the
top row is exactly the negative slope of the bottom row and the three columns stay
directly comparable. Energies are referenced to the last (dissociated) scan point.

The scan drivers and DFT scan inputs that produced data/ are in source_scans/; they
are not needed to redraw this figure. The OPLS-AA curve is the paper's classical
baseline (source_scans/scan_driver_opls.py reads the production topologies).

    python -m paper.paper_figures.appendix.interaction_scanning.plot_interaction_scanning

(from the repo root; writes to paths.FIGURES_APPENDIX unless --out is given)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.observables import paths

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
HARTREE_EV = 27.211386245988

# ── model draw order + colours (labels come from each npz's 'model' field) ────
MODEL_ORDER = ["opls", "uma", "orb", "mace", "sevennet"]
COLOR = {"opls": "#ff7f0e", "uma": "#2ca02c", "orb": "#d62728", "mace": "#9467bd",
         "sevennet": "#8c564b"}
# models whose caches stay in data/ but are left off the figure
EXCLUDE = {"sevennet"}

SYSTEMS = ["na_otf", "otf_otf", "na_na"]
COL_TITLE = {
    "na_otf":  r"Na$^+$–OTf$^-$  (attraction)",
    "otf_otf": r"OTf$^-$···OTf$^-$  (repulsion)",
    "na_na":   r"Na$^+$···Na$^+$  (repulsion)",
}
XLABEL = {
    "na_otf":  r"Na–O$_3$(centroid) distance  $d$  (Å)",
    "otf_otf": r"O$_3$(A)–O$_3$(B) centroid distance  $d$  (Å)",
    "na_na":   r"Na$^+$–Na$^+$ distance  $d$  (Å)",
}
XLIM  = {"na_otf": (1.8, 12.0), "otf_otf": (2.5, 10.0), "na_na": (2.0, 12.0)}
FYLIM = {"na_otf": (-1.2, 1.2), "otf_otf": (-0.4, 3.2), "na_na": (-0.3, 2.0)}
EYLIM = {"na_otf": (-4.2, 5.0), "otf_otf": (-0.3, 3.8), "na_na": (-0.5, 5.0)}
DFT_LABEL = "ωB97M-V/def2-TZVPD (DFT ref.)"
DFT_STYLE = dict(color="k", ls="--", lw=2.0, marker="s", ms=3.0, zorder=6)

plt.rcParams.update({
    "font.size": 15,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 19,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


def load_models(system):
    """Ordered [(key, label, D, E_ref, F_conservative)] for a system from data/."""
    out = []
    for key in MODEL_ORDER:
        if key in EXCLUDE:
            continue
        f = DATA / f"{system}__{key}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        D = z["D"]; E = z["E_raw"] - z["E_raw"][-1]
        out.append((key, str(z["model"]), D, E, -np.gradient(E, D)))
    return out


def load_dft(system):
    """(d, E_ref, F) from data/DFT/{system}_dft_energies.csv, or None if absent."""
    csvf = DATA / "DFT" / f"{system}_dft_energies.csv"
    if not csvf.exists():
        return None
    dd, ee = [], []
    with open(csvf) as f:
        for row in csv.DictReader(f):
            e = str(row.get("E_hartree", row.get("E_eV", ""))).strip()
            if not e:
                continue
            dd.append(float(row["d_angstrom"])); ee.append(float(e))
    if len(ee) < 3:
        return None
    o = np.argsort(dd); d = np.array(dd)[o]; e = np.array(ee)[o]
    unit_ev = np.median(np.abs(e)) > 5000        # heuristic: hartree vs eV
    E = (e - e[-1]) * (1.0 if unit_ev else HARTREE_EV)
    return d, E, -np.gradient(E, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suptitle", action="store_true",
                    help="draw the overall title (omitted by default for the paper)")
    ap.add_argument("--out", default=str(paths.FIGURES_APPENDIX / "figure_interaction_scanning_2x3.png"))
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2))
    handles, labels = [], []          # one shared legend, deduplicated by label

    def _register(h, lab):
        if lab not in labels:
            labels.append(lab); handles.append(h)

    for col, sysname in enumerate(SYSTEMS):
        ax_f, ax_e = axes[0, col], axes[1, col]

        for key, label, D, E, F in load_models(sysname):
            label = label.replace(" MLIP", "")    # drop the "MLIP" tag from legend labels
            c = COLOR.get(key, "#555555")
            (lnf,) = ax_f.plot(D, F, "-", color=c, lw=2, marker="o", ms=2.5, label=label)
            ax_e.plot(D, E, "-", color=c, lw=2, marker="o", ms=2.5, label=label)
            _register(lnf, label)

        dft = load_dft(sysname)
        if dft is not None:
            d, E, F = dft
            (lnd,) = ax_f.plot(d, F, **DFT_STYLE, label=DFT_LABEL)
            ax_e.plot(d, E, **DFT_STYLE, label=DFT_LABEL)
            _register(lnd, DFT_LABEL)

        ax_f.set_title(COL_TITLE[sysname], fontsize=16, fontweight="bold")
        for ax, ylim in ((ax_f, FYLIM[sysname]), (ax_e, EYLIM[sysname])):
            ax.axhline(0, color="k", lw=0.8, alpha=0.5)
            ax.set_xlim(*XLIM[sysname]); ax.set_ylim(*ylim); ax.grid(alpha=0.25)
            ax.tick_params(labelsize=14, width=1.4)
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                tl.set_fontweight("bold")
        # extra labelpad drops the x-title, leaving room for the (d)-(f) labels
        # between the x-axis tick numbers and the title
        ax_e.set_xlabel(XLABEL[sysname], fontsize=15, fontweight="bold", labelpad=30)

    # panel labels (a)-(f) centered just below each subplot's x-axis; on the
    # bottom row the enlarged xlabel labelpad leaves them sitting between the
    # x-axis tick numbers and the x-axis title (matches the caption ordering)
    for col in range(3):
        for row, letter in ((0, "abc"[col]), (1, "def"[col])):
            axes[row, col].text(
                0.5, -0.075, f"({letter})", transform=axes[row, col].transAxes,
                fontsize=21, fontweight="bold", va="top", ha="center")

    # unit given with the "quantity / unit" convention so no parentheses are needed
    axes[0, 0].set_ylabel(r"axial force  /  eV Å$^{-1}$", fontsize=15, fontweight="bold")
    axes[1, 0].set_ylabel("interaction energy  /  eV", fontsize=15, fontweight="bold")

    # single shared legend across the bottom
    leg = fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=13,
                     frameon=True, bbox_to_anchor=(0.5, 0.0))
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    if args.suptitle:
        fig.suptitle("Gas-phase ion-pair scans vs ωB97M-V/def2-TZVPD DFT — "
                     "conservative force (top) & binding energy (bottom).",
                     fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.95 if args.suptitle else 0.99))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for p in (out,):
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print("saved", p)


if __name__ == "__main__":
    main()
