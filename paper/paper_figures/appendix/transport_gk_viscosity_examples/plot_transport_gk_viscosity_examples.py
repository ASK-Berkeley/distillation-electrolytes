#!/usr/bin/env python
"""Appendix figure: example Green-Kubo running viscosity integrals.

Illustrates HOW the shear viscosity is extracted: for the same eight
trajectories as the MSD appendix figure (four solvent families, 1.0 M top
row / 0.1 M bottom row) we show the running Green-Kubo integral eta(tau) of
the stress autocorrelation function, the statistical spread across the five
equal time blocks of the trajectory (the band used to weight and truncate
the time-decomposition fit), the Zhang-Maginn double-exponential fit itself
(dash-dotted), the noise-dominated tail excluded from that fit (shaded), and
the extrapolated infinite-time viscosity eta_inf that is reported (dashed).

The fit curve is recomputed with the SAME pipeline function
(src/analysis/observables/viscosity_gk.py::fit_eta_tdm) on the stored
0.5 ps-resolution running-integral curve; this reproduces each replica's
reported eta_inf to within ~2% (the production fit ran on the
full-resolution stress data). The dashed line and annotation keep the
reported summary.json value.

Reads the per-replica analysis output directly from
    data/painn_transport_observables_replicas/viscosity/per_run/<system>/<rN>/
(viscosity_gk_curve.csv for the running integral and block band, summary.json
for the fit window and the extrapolated eta_inf) -- same source as the SI
transport tables (make_transport_tables.py).

    python plot_transport_gk_viscosity_examples.py

Writes figure_transport_gk_viscosity_examples.png here and copies it to
paper/paper_electrolytes/Figures/appendix/.

The (system, replica) list matches plot_transport_msd_examples.py so the two
appendix figures describe the same four example trajectories.

Env: any python with numpy/pandas/matplotlib.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from src.analysis.observables import paths
PER_RUN = paths.OUT_PAINN_TRANSPORT_REPLICAS / "viscosity" / "per_run"
FIGDIR = paths.FIGURES_APPENDIX

from src.analysis.observables.viscosity_gk import fit_eta_tdm  # noqa: E402

# (system_id, replica, panel title, solvent color) -- same runs as the MSD
# figure (1.0 M top row, 0.1 M bottom row); colors follow the paper-wide
# solvent palette.
SYSTEMS = [
    ("napf6_dme_1M_298",       "r2", "NaPF$_6$ / DME (1.0 M)",     "#E8820C"),
    ("naotf_diglyme_1M_298",   "r3", "NaOTf / DEGDME (1.0 M)",     "#159C8C"),
    ("napf6_pc_1M_298",        "r1", "NaPF$_6$ / PC (1.0 M)",      "#E5177B"),
    ("naotf_tgdme_1M_298",     "r3", "NaOTf / TEGDME (1.0 M)",     "#2C6FBB"),
    ("napf6_dme_0_1M_298",     "r3", "NaPF$_6$ / DME (0.1 M)",     "#E8820C"),
    ("naotf_diglyme_0_1M_298", "r2", "NaOTf / DEGDME (0.1 M)",     "#159C8C"),
    ("napf6_pc_0_1M_298",      "r1", "NaPF$_6$ / PC (0.1 M)",      "#E5177B"),
    ("naotf_tgdme_0_1M_298",   "r1", "NaOTf / TEGDME (0.1 M)",     "#2C6FBB"),
]
FIT_COLOR = "0.15"       # double-exponential fit (dash-dotted)
ETA_COLOR = "#D62728"    # extrapolated eta_inf (dashed); CVD-checked against
                         # every solvent color (straight dashed line + same-color
                         # annotation carry the identity in the blue TGDME panels)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 8.6))
    for k, (ax, (sid, rep, title, color)) in enumerate(zip(axes.flat, SYSTEMS)):
        run = PER_RUN / sid / rep
        df = pd.read_csv(run / "viscosity_gk_curve.csv")
        summ = json.load(open(run / "summary.json"))["viscosity_gk"]
        tau = df["tau_ps"].to_numpy()
        eta = df["eta_running_cP"].to_numpy()
        band = df["band_cP"].to_numpy()
        eta_inf = summ["eta_gk_cP"]

        # block-spread band + running integral
        ax.fill_between(tau, eta - band, eta + band, color=color, alpha=0.20,
                        lw=0, zorder=1)
        ax.plot(tau, eta, color=color, lw=1.8, zorder=3)

        # noise-dominated tail excluded from the time-decomposition fit
        # (the fit window is [t_lo, t_hi]; t_lo is 1 ps, i.e. invisible here)
        t_hi = summ["t_hi_ps"]
        if t_hi < tau[-1]:
            ax.axvspan(t_hi, tau[-1], color="0.5", alpha=0.15, zorder=0)
            ax.axvline(t_hi, color="0.5", lw=1.0, zorder=1)

        # the Zhang-Maginn double-exponential fit, recomputed with the
        # pipeline's own fit_eta_tdm on the stored running-integral curve
        res = fit_eta_tdm(tau, eta, band)
        if res["fit_curve"] is not None:
            tf, yf = res["fit_curve"]
            ax.plot(tf, yf, ls="-.", color=FIT_COLOR, lw=2.4, zorder=4)
        drift = abs(res["est"] - eta_inf) / eta_inf
        if drift > 0.03:
            print(f"WARNING {sid}/{rep}: refit eta_inf {res['est']:.3f} vs "
                  f"stored {eta_inf:.3f} ({100 * drift:.1f}% apart)")

        ax.axhline(eta_inf, ls="--", color=ETA_COLOR, lw=2.0, zorder=2)
        ax.annotate(rf"$\eta_\infty$ = {eta_inf:.2f} cP",
                    xy=(0.975, eta_inf), xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points",
                    ha="right", va="bottom", fontsize=13, color=ETA_COLOR)

        ax.set_xlim(0, tau[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=15, pad=9)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=15)
        if k >= 4:
            ax.set_xlabel(r"integration time $\tau$ (ps)", fontsize=19.5)
        ax.text(-0.20, 1.10, "abcdefgh"[k] + ")", transform=ax.transAxes,
                fontsize=17, fontweight="bold", va="top", ha="left")
    for row in (0, 1):
        axes[row, 0].set_ylabel(r"running integral $\eta(\tau)$ (cP)",
                                fontsize=19.5)

    # shared legend (colors vary per panel, so use neutral proxies)
    proxies = [
        plt.Line2D([], [], color="0.35", lw=2.0),
        plt.Rectangle((0, 0), 1, 1, fc="0.35", alpha=0.25, lw=0),
        plt.Line2D([], [], ls="-.", color=FIT_COLOR, lw=2.4),
        plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.25, lw=0),
        plt.Line2D([], [], ls="--", color=ETA_COLOR, lw=2.0),
    ]
    labels = [
        r"running Green--Kubo integral $\eta(\tau)$",
        "spread across five time blocks",
        "double-exponential (TDM) fit",
        "noise-dominated tail (excluded from fit)",
        r"extrapolated $\eta_\infty$ (reported)",
    ]
    fig.legend(proxies, labels, loc="lower center", ncol=3, fontsize=15,
               frameon=False, bbox_to_anchor=(0.5, -0.005),
               handlelength=2.4, columnspacing=1.8)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "figure_transport_gk_viscosity_examples.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    plt.close(fig)



if __name__ == "__main__":
    main()
