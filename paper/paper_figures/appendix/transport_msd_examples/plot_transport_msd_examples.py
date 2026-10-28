#!/usr/bin/env python
"""Appendix figure: example MSD curves and Fickian diffusivity fits.

Illustrates HOW the self-diffusivities are extracted: for the four solvent
families (both Na salts, spanning the fast-DME -> slow-TEGDME dynamic range)
at 1.0 M (top row) and 0.1 M (bottom row) we show the time-origin-averaged
center-of-mass MSD of the cation, anion, and solvent on log-log axes,
together with the linear Einstein fit over the automatically selected Fickian
window (the longest contiguous stretch with logarithmic slope
|beta - 1| <= 0.15; see Methods). The dashed segment spans exactly the fitted
window, so the reader sees both where the MSD is genuinely diffusive and
which part of the curve the reported D uses.

Reads the per-replica analysis output directly from
    data/painn_transport_observables_replicas/diffusivity/per_run/<system>/<rN>/
(msd.csv for the curves, summary.json for the Fickian fit windows) -- same
source as the SI transport tables (make_transport_tables.py).

    python plot_transport_msd_examples.py

Writes figure_transport_msd_examples.png here and copies it to
paper/paper_electrolytes/Figures/appendix/.

Replica choice: one representative replica per system, picked so that all
three species' Fickian fits converged (all_converged_fick in summary.json);
for NaOTf/TEGDME 1M that is replica 3 (the salt diffuses so slowly there that
some replicas never develop a clean beta~=1 window within 20 ns).

Env: any python with numpy/pandas/matplotlib.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.observables import paths

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PER_RUN = paths.OUT_PAINN_TRANSPORT_REPLICAS / "diffusivity" / "per_run"
FIGDIR = paths.FIGURES_APPENDIX

# (system_id, replica, panel title) -- columns ordered fast -> slow solvent
# dynamics; top row 1.0 M, bottom row the 0.1 M counterparts.
SYSTEMS = [
    ("napf6_dme_1M_298",       "r2", "NaPF$_6$ / DME (1.0 M)"),
    ("naotf_diglyme_1M_298",   "r3", "NaOTf / DEGDME (1.0 M)"),
    ("napf6_pc_1M_298",        "r1", "NaPF$_6$ / PC (1.0 M)"),
    ("naotf_tgdme_1M_298",     "r3", "NaOTf / TEGDME (1.0 M)"),
    ("napf6_dme_0_1M_298",     "r3", "NaPF$_6$ / DME (0.1 M)"),
    ("naotf_diglyme_0_1M_298", "r2", "NaOTf / DEGDME (0.1 M)"),
    ("napf6_pc_0_1M_298",      "r1", "NaPF$_6$ / PC (0.1 M)"),
    ("naotf_tgdme_0_1M_298",   "r1", "NaOTf / TEGDME (0.1 M)"),
]

# species palette (CVD-validated: all pairs pass protan/deutan/tritan checks)
SPECIES = [
    ("msd_cation",  "cation",  r"cation $D^{+}$",  "#2C6FBB"),
    ("msd_anion",   "anion",   r"anion $D^{-}$",   "#D9541E"),
    ("msd_solvent", "solvent", r"solvent $D^{0}$", "#159C8C"),
]
FIT_COLOR = "0.15"


def fit_segment(tau, msd, t_lo, t_hi):
    """Linear Einstein refit (slope+intercept) on the stored Fickian window."""
    m = (tau >= t_lo) & (tau <= t_hi)
    slope, icpt = np.polyfit(tau[m], msd[m], 1)
    return slope, icpt


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.2))
    for k, (ax, (sid, rep, title)) in enumerate(zip(axes.flat, SYSTEMS)):
        run = PER_RUN / sid / rep
        df = pd.read_csv(run / "msd.csv")
        summ = json.load(open(run / "summary.json"))["diffusivity"]
        tau = df["tau_ps"].to_numpy()
        pos = tau > 0

        for col, name, _label, color in SPECIES:
            msd = df[col].to_numpy()
            ax.plot(tau[pos], msd[pos], color=color, lw=1.8, alpha=0.95,
                    zorder=2)
            # dashed Einstein-fit segment over the stored Fickian window
            t_lo = summ[f"D_{name}_fick_t_lo_ps"]
            t_hi = summ[f"D_{name}_fick_t_hi_ps"]
            slope, icpt = fit_segment(tau, msd, t_lo, t_hi)
            tt = np.geomspace(t_lo, t_hi, 64)
            ax.plot(tt, slope * tt + icpt, ls="--", color=FIT_COLOR, lw=2.2,
                    zorder=3)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(tau[pos][0], tau[-1] * 1.3)
        ax.set_title(title, fontsize=15, pad=9)
        ax.grid(alpha=0.22, which="both")
        ax.tick_params(labelsize=15)
        if k >= 4:
            ax.set_xlabel(r"lag time $\tau$ (ps)", fontsize=19.5)
        ax.text(-0.20, 1.10, "abcdefgh"[k] + ")", transform=ax.transAxes,
                fontsize=17, fontweight="bold", va="top", ha="left")
    for row in (0, 1):
        axes[row, 0].set_ylabel(r"MSD ($\mathrm{\AA}^2$)", fontsize=19.5)

    # single shared legend: the three species colors + the fit line
    proxies = [plt.Line2D([], [], color=c, lw=2.6) for *_, c in SPECIES]
    labels = [lab for _, _, lab, _ in SPECIES]
    proxies.append(plt.Line2D([], [], ls="--", color=FIT_COLOR, lw=2.2))
    labels.append(r"Einstein fit, Fickian window ($|\beta-1|\leq0.15$)")
    fig.legend(proxies, labels, loc="lower center", ncol=4, fontsize=15,
               frameon=False, bbox_to_anchor=(0.5, -0.005),
               handlelength=2.4, columnspacing=1.8)

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "figure_transport_msd_examples.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    plt.close(fig)



if __name__ == "__main__":
    main()
