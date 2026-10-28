#!/usr/bin/env python
"""Appendix figure: transport-property convergence vs trajectory length.

Self-contained -- reads ONLY `data_transport_convergence.csv` in this folder.
No trajectories, no pipeline imports.

    python plot_transport_convergence.py

Writes figure_transport_convergence_normalised.{png,pdf}

Panels a-d come from `data_transport_convergence.csv` (D and sigma); panel e is
the Green-Kubo viscosity from `data_viscosity_convergence.csv`. Both tables share
the same tidy schema and the same expanding-window convention.

WHAT THE FIGURE SHOWS
---------------------
Each panel is one transport observable. Every system's curve is divided by its
own value at the longest window (t_end = 20 ns), so the y-axis reads "how far
off is a shorter run from the 20 ns answer?".

IMPORTANT: because each system is normalised by its OWN final value, every curve
passes through 1.0 at t_end = 20 ns BY CONSTRUCTION. The endpoint carries no
information -- only the approach to the +-5% band does. Absolute values are in
the CSV (`value_mean`, `value_sd`, in the units named by `observable`).

METHOD (as recorded in the source run manifest)
-----------------------------------------------
  * Expanding window [t0 = 2 ns, t_end], t_end = 3, 4, ..., 20 ns. The windows
    are NESTED, so a single curve's scatter is NOT an error bar -- uncertainty
    comes from the spread across the 4 independent replicas.
  * Effective lag dt = 5 ps (trajectories saved at 1 ps/frame, subsampled x5).
  * sigma: byteff2 `onsager_calc`, MSD-slope fit over a FIXED 250-1000 ps lag
    window (= its hardcoded [50, 200) loaded-frame fit at 5 ps/frame).
  * D: molecular-COM MSD, slope/6, fitted over the SAME FIXED 250-1000 ps lag
    window. A fixed window (rather than a fit range that grows with t_end) is
    essential: with a percentage fit both the time-origin count and the fit
    range move at once and D appears never to converge.
  * sigma is undefined for t_end < 5 ns: `onsager_calc` needs > 401 loaded
    frames, which [2, 5] ns at 5 ps/frame is the first window to supply.

Env: any python with numpy/pandas/matplotlib.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from src.analysis.observables import paths

HERE = Path(__file__).resolve().parent
CSV = HERE / "data_transport_convergence.csv"
CSV_VISC = HERE / "data_viscosity_convergence.csv"      # merged in as panel e
TOL = 0.05                      # +-5% band

# ── styling: the same solvent palette as figures 3/4 and the other SI figures ──
# colour = solvent, marker = salt, fill = concentration
SOLVENT_COLOR = {
    "DME":     "#E8850E",
    "Diglyme": "#1ad73c",     # DEGDME
    "PC":      "#E6197D",
    "TGDME":   "#2E6FBF",     # TEGDME
}
SOLVENT_LABEL = {"DME": "DME", "Diglyme": "DEGDME", "PC": "PC", "TGDME": "TEGDME"}
SALT_MARKER = {"napf6": "o", "naotf": "s", "lipf6": "^"}
SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}
# 0.1 M open, 0.5 M half-filled, 1.0 M filled
FILLSTYLE = {0.1: "none", 0.5: "left", 1.0: "full"}
# only the 298 K LiPF6/DME system is shown (273 K and 323 K dropped)
LIPF6_TEMP_LS = {298.0: "-"}

# sigma_NE is deliberately not shown: it is built from the same self-diffusivities
# as the D panels (it assumes no ion-ion cross-correlation), so it adds no
# independent convergence information. Its values remain in the CSV.
PANELS = [
    ("D_cat_1e10_m2s",      r"Cation  $D^{+}$"),
    ("D_ani_1e10_m2s",      r"Anion  $D^{-}$"),
    ("D_sol_1e10_m2s",      r"Solvent  $D^{0}$"),
    ("sigma_onsager_mS_cm", r"Ionic conductivity  $\sigma_{\mathrm{Onsager}}$"),
    ("eta_inf_tdm_cP",      r"Dynamic viscosity  $\eta$  (Green-Kubo)"),
]

# The three diffusivity panels share one y-range so they can be compared directly
# by eye; sigma and eta keep their own autoscaled range (sigma in particular runs
# to ~3.2 at the shortest window and would flatten the D panels).
SHARED_D_YLIM = (0.5, 1.75)
YLIM = {obs: SHARED_D_YLIM for obs in
        ("D_cat_1e10_m2s", "D_ani_1e10_m2s", "D_sol_1e10_m2s")}


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    # transport (D, sigma) and viscosity share the same tidy schema and the same
    # expanding-window convention, so they concatenate directly
    frames = [pd.read_csv(CSV)]
    if CSV_VISC.exists():
        frames.append(pd.read_csv(CSV_VISC))
    df = pd.concat(frames, ignore_index=True)
    t0 = float(df["t0_ns"].iloc[0])

    meta = (df.drop_duplicates("system_id")
              .set_index("system_id")[["salt", "solvent", "concentration_M",
                                       "temperature_K"]])
    sids = sorted(df["system_id"].unique())
    # drop the LiPF6/DME 273 K and 323 K systems; keep only 298 K
    sids = [s for s in sids if not (meta.loc[s, "salt"] == "lipf6"
                                    and float(meta.loc[s, "temperature_K"]) not in LIPF6_TEMP_LS)]

    def style(sid):
        m = meta.loc[sid]
        salt, sol = m["salt"], m["solvent"]
        conc, temp = float(m["concentration_M"]), float(m["temperature_K"])
        color = SOLVENT_COLOR.get(sol, "gray")
        return dict(color=color, marker=SALT_MARKER.get(salt, "o"),
                    fillstyle=FILLSTYLE.get(conc, "full"),
                    markerfacecolor=("white" if conc == 0.1 else color),
                    markeredgecolor=color, markersize=4.2, markeredgewidth=1.1,
                    linestyle=(LIPF6_TEMP_LS.get(temp, "-") if salt == "lipf6" else "-"))

    # 2 rows: diffusivities on top, conductivity + viscosity below. The free
    # slot in row 2 holds the legend (as in the MLP transport-parity figure).
    fig, axes2 = plt.subplots(2, 3, figsize=(14.5, 8.4), squeeze=False)
    axs = list(axes2[0]) + list(axes2[1])
    leg_ax = axs[len(PANELS)]
    leg_ax.axis("off")
    for k, (ax, (obs, lab)) in enumerate(zip(axs, PANELS)):
        sub = df[df["observable"] == obs]
        for sid in sids:
            g = sub[sub["system_id"] == sid].sort_values("window_end_ns")
            v = g.set_index("window_end_ns")["value_mean"].dropna()
            if not len(v) or v.iloc[-1] == 0:
                continue
            ax.plot(v.index, v / v.iloc[-1], lw=1.3, alpha=0.9, **style(sid))
        ax.axhline(1.0, color="0.35", ls="--", lw=1.0, zorder=0)
        ax.axhspan(1 - TOL, 1 + TOL, color="k", alpha=0.10, zorder=0)
        if obs in YLIM:
            ax.set_ylim(*YLIM[obs])
        ax.set_title(lab, fontsize=12.5, pad=9)
        ax.set_xlabel(r"window end $t_{end}$ (ns)", fontsize=16)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.25)
        ax.text(-0.11, 1.07, "abcdefg"[k], transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top", ha="left")
    for ax in (axes2[0][0], axes2[1][0]):
        ax.set_ylabel("value / value at 20 ns", fontsize=16)

    # ── shared legend: one row per salt/solvent, showing the 0.1 M and 1 M
    #    fills side by side (as in the MLP appendix figure) ──────────────────
    def key(conc, salt, sol):
        c = SOLVENT_COLOR[sol]
        return Line2D([], [], color=c, marker=SALT_MARKER[salt], ls="none",
                      markersize=8.5, markeredgewidth=1.6, markeredgecolor=c,
                      markerfacecolor=("white" if conc == 0.1 else c),
                      fillstyle=FILLSTYLE[conc])

    handles, labels = [], []
    for sol in ["DME", "Diglyme", "PC", "TGDME"]:
        for salt in ["napf6", "naotf"]:
            present = meta[(meta.salt == salt) & (meta.solvent == sol)]
            if present.empty:
                continue
            handles.append(tuple(key(c, salt, sol) for c in (0.1, 1.0)))
            labels.append(f"{SALT_LABEL[salt]} in {SOLVENT_LABEL[sol]}")
    for temp, ls in LIPF6_TEMP_LS.items():
        if not (meta[(meta.salt == "lipf6") & (meta.temperature_K == temp)]).empty:
            c = SOLVENT_COLOR["DME"]
            handles.append(Line2D([], [], color=c, marker="^", ls=ls, lw=1.6,
                                  markersize=8.5, markeredgecolor=c,
                                  markerfacecolor=c, fillstyle="left"))
            labels.append("LiPF$_6$ in DME (0.5 M)")

    leg = leg_ax.legend(handles, labels, loc="center left",
                        bbox_to_anchor=(-0.02, 0.52), ncol=1, fontsize=12,
                        frameon=False, handlelength=2.6, labelspacing=0.85,
                        handletextpad=1.0, borderpad=0.0,
                        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.7)})
    leg.set_title("colour = solvent   ·   marker = salt\n"
                  "open = 0.1 M   filled = 1.0 M",
                  prop={"size": 12})
    leg.get_title().set_color("0.30")

    # No suptitle and no method caption: this figure carries its caption in the
    # manuscript. The method text lives in METHODS.md next to this script.
    fig.tight_layout()

    paths.FIGURES_APPENDIX.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES_APPENDIX / "figure_transport_convergence_normalised.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
