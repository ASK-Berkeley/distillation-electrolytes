#!/usr/bin/env python3
"""Appendix figure: experimental ionicity of the concentrated electrolytes.

A single log-log parity panel of the *experimental* Nernst--Einstein conductivity
against the *experimental* measured conductivity, for the concentrated systems.

- y-axis: sigma_NE built from the measured self-diffusivities (D_+ and D_-) via
  sigma_NE = e^2 N_A c (D_+ + D_-) / (k_B T)  [1:1 salts, z = +/-1], the identical
  Nernst--Einstein estimator used elsewhere (observables.conductivity).
- x-axis: the measured ionic conductivity, which IS the true total sigma (what the
  Onsager estimator targets); there is no separately measured "Onsager" number.
- The vertical distance above the 1:1 line is therefore the experimental ionicity
  (ionicity = sigma / sigma_NE = measured / NE): points on y=x are ideal
  (uncorrelated), points high above it are strongly ion-paired / correlation
  dominated. NaOTf in DME sits alone at the top -- ionicity ~= 0.03.

Data (both experimental):
  * measured D_+, D_-  : ground_truth/298p2K_exp_values.csv (PFG-NMR, 298.2 K)
  * measured sigma     : sigma_exp_mS_cm from the PaiNN transport master
                         (canonical measured conductivity, conc/T-matched)

The 1.0 M NaPF6/TEGDME system is omitted: the ground-truth table has no measured
cation diffusivity for it, so its sigma_NE cannot be formed. LiPF6/DME is included
at its 0.5 M point (the one non-1 M concentrated system).

    python plot_conductivity_ionicity.py
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.analysis.observables import paths

# No options: the parser only gives --help and rejects stray arguments, so a
# mistyped flag cannot silently regenerate the figure.
argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

EXP_VALUES = paths.GROUND_TRUTH / "298p2K_exp_values.csv"
PAINN_MASTER = paths.OUT_PAINN_TRANSPORT_REPLICAS / "conductivity" / "conductivity_master.csv"
FIGDIR = paths.FIGURES_APPENDIX

# Nernst--Einstein constants (SI), matching observables.conductivity._sigma_ne_mS_cm
E_CHARGE_C = 1.602176634e-19
N_A = 6.02214076e23
KB_J_PER_K = 1.380649e-23
T_K = 298.2
S_PER_M_TO_MS_PER_CM = 10.0

CONC_MIN = 0.15
T_TARGET = 298.0

SOLV_COLOR = {"DME": "#E8850E", "DIGLYME": "#1ad73c", "PC": "#E6197D", "TGDME": "#2E6FBF"}
SOLV_DISP  = {"DME": "DME", "DIGLYME": "DEGDME", "PC": "PC", "TGDME": "TEGDME"}
ANION_MARK = {"PF6": "o", "OTf": "s"}
SALT_DISP  = {("Na", "PF6"): "NaPF$_6$", ("Na", "OTf"): "NaOTf", ("Li", "PF6"): "LiPF$_6$"}

# Sample-string (298p2K_exp_values.csv) solvent token -> internal solvent key.
SOLV_FROM_SAMPLE = {"DME": "DME", "DEGDME": "DIGLYME", "TEGDME": "TGDME", "PC": "PC"}
# system_id solvent token (PaiNN master) -> internal solvent key.
SOLV_FROM_SYSID = {"dme": "DME", "diglyme": "DIGLYME", "tgdme": "TGDME", "pc": "PC"}


def sigma_ne_mS_cm(conc_M, d_plus_1e10, d_minus_1e10):
    """Nernst--Einstein sigma (mS/cm) for a 1:1 salt from measured D_+, D_-."""
    c = conc_M * 1000.0                      # mol/L -> mol/m^3
    d = (d_plus_1e10 + d_minus_1e10) * 1e-10  # 1e-10 m^2/s -> m^2/s
    sigma_si = E_CHARGE_C**2 * N_A * c * d / (KB_J_PER_K * T_K)  # S/m
    return sigma_si * S_PER_M_TO_MS_PER_CM


def parse_exp_diffusivities():
    """(cation, anion, solv_key, conc) -> (D_+, D_-) from the ground-truth table."""
    df = pd.read_csv(EXP_VALUES)
    out = {}
    for _, r in df.iterrows():
        s = str(r["Sample"]).strip()
        if " M " not in s or " in " not in s:
            continue
        conc_tok, rest = s.split(" M ", 1)
        salt_tok, solv_tok = rest.split(" in ", 1)
        try:
            conc = float(conc_tok)
        except ValueError:
            continue
        solv_key = SOLV_FROM_SAMPLE.get(solv_tok.strip())
        if solv_key is None:
            continue
        salt_tok = salt_tok.strip()
        cation = "Li" if salt_tok.startswith("Li") else "Na" if salt_tok.startswith("Na") else None
        anion = "OTf" if "OTf" in salt_tok else "PF6" if "PF6" in salt_tok else None
        if cation is None or anion is None:
            continue
        dp, dm = r["D*_+ (x 10^-10 m^2/s)"], r["D*_- (x 10^-10 m^2/s)"]
        try:
            dp, dm = float(dp), float(dm)
        except (TypeError, ValueError):
            continue
        if np.isnan(dp) or np.isnan(dm):
            continue
        out[(cation, anion, solv_key, round(conc, 2))] = (dp, dm)
    return out


def measured_sigma_by_system():
    """(cation, anion, solv_key, conc) -> measured sigma (mS/cm), concentrated set."""
    m = pd.read_csv(PAINN_MASTER)
    m = m[np.isclose(m["temperature_K"], T_TARGET) & (m["conc_M"] > CONC_MIN)]
    out = {}
    for sysid, g in m.groupby("system_id"):
        salt, solv_tok = sysid.split("_", 1)
        cation = "Li" if salt.startswith("li") else "Na"
        anion = "PF6" if "pf6" in salt else "OTf"
        solv_key = SOLV_FROM_SYSID[solv_tok]
        r0 = g.iloc[0]
        out[(cation, anion, solv_key, round(float(r0["conc_M"]), 2))] = float(r0["sigma_exp_mS_cm"])
    return out


exp_D = parse_exp_diffusivities()
meas = measured_sigma_by_system()

rows = []
for key, sigma_meas in meas.items():
    if key not in exp_D:            # e.g. NaPF6/TEGDME: no measured cation D
        continue
    dp, dm = exp_D[key]
    cation, anion, solv_key, conc = key
    sig_ne = sigma_ne_mS_cm(conc, dp, dm)
    rows.append(dict(cation=cation, anion=anion, solv_key=solv_key, conc=conc,
                     sigma_meas=sigma_meas, sigma_ne=sig_ne,
                     ionicity=sigma_meas / sig_ne))
data = pd.DataFrame(rows).sort_values("ionicity").reset_index(drop=True)
print(data.to_string(index=False))

dropped = [k for k in meas if k not in exp_D]
if dropped:
    print("omitted (no measured cation D):", dropped)

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 6.6))

lo = min(data["sigma_meas"].min(), data["sigma_ne"].min()) / 1.6
hi = max(data["sigma_meas"].max(), data["sigma_ne"].max()) * 1.6

# iso-ionicity guides: y = sigma_meas / ionicity (labelled along the top edge)
for ion, ls in [(1.0, "--"), (0.3, ":"), (0.1, ":"), (0.03, ":")]:
    xs = np.array([lo, hi])
    ax.plot(xs, xs / ion, ls=ls, color="0.6", lw=1.0, zorder=1)
    x_top = hi * ion                       # where the guide meets the top (y = hi)
    if lo * 1.05 < x_top < hi:
        lab = "ionicity = 1\n(ideal)" if ion == 1.0 else f"{ion:g}"
        ax.text(x_top, hi * 0.97, lab, color="0.45", fontsize=8.5,
                va="top", ha="center")

# marker encoding matches Figure 3: color = solvent, marker = anion, open = 0.1 M,
# half-filled pentagon = LiPF6 in DME (0.5 M)
for _, r in data.iterrows():
    col = SOLV_COLOR[r.solv_key]
    extra = {}
    if r.cation == "Li":
        mk, filled = "p", True
        extra = dict(fillstyle="left", markerfacecoloralt="none")
    else:
        mk = ANION_MARK[r.anion]; filled = r.conc > 0.5
    ax.plot(r.sigma_meas, r.sigma_ne, marker=mk, ls="none", ms=10.5,
            mfc=col if filled else "none", mec="0.4" if filled else col,
            mew=1.3, zorder=3, **extra)

# call out the extreme outlier with a red ring + short label
o = data.iloc[0]
ax.scatter(o.sigma_meas, o.sigma_ne, s=620, facecolor="none", edgecolor="#D62728",
           linewidth=1.8, zorder=4)
ax.text(o.sigma_meas * 1.28, o.sigma_ne * 0.90,
        f"$\\sigma_{{\\mathrm{{NE}}}}$ is {o.sigma_ne / o.sigma_meas:.0f}$\\times$\nthe measured $\\sigma$",
        color="#D62728", fontsize=10.5, ha="left", va="top")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.set_xlabel("Measured conductivity $\\sigma$  ($\\mathrm{mS\\,cm^{-1}}$)", fontsize=13)
ax.set_ylabel("Nernst--Einstein $\\sigma_{\\mathrm{NE}}$ from measured $D$  "
              "($\\mathrm{mS\\,cm^{-1}}$)", fontsize=13)
ax.tick_params(labelsize=11)

# legend: salt/solvent identity (lower-right, clear of the points)
# entries in the same order as the Figure 3 legend (solvent, then anion, LiPF6 last)
SOLV_ORDER = ["DME", "DIGLYME", "PC", "TGDME"]
leg_rows = data.assign(
    _o=[(r.cation == "Li", SOLV_ORDER.index(r.solv_key), r.anion != "PF6")
        for _, r in data.iterrows()]).sort_values("_o")
handles, labels = [], []
seen = []
for _, r in leg_rows.iterrows():
    key = (r.cation, r.anion, r.solv_key)
    if key in seen:
        continue
    seen.append(key)
    col = SOLV_COLOR[r.solv_key]
    if r.cation == "Li":
        handles.append(Line2D([], [], marker="p", color=col, mfc=col, mec="0.4",
                              ls="none", ms=11, fillstyle="left",
                              markerfacecoloralt="none"))
        labels.append(f"{SALT_DISP[(r.cation, r.anion)]} in {SOLV_DISP[r.solv_key]} (0.5 M)")
    else:
        handles.append(Line2D([], [], marker=ANION_MARK[r.anion], color=col, mfc=col,
                              mec="0.4", ls="none", ms=10))
        labels.append(f"{SALT_DISP[(r.cation, r.anion)]} in {SOLV_DISP[r.solv_key]}")
ax.legend(handles, labels, loc="lower right", fontsize=9.5, frameon=False,
          handlelength=1.6, labelspacing=0.6)

fig.tight_layout()

FIGDIR.mkdir(parents=True, exist_ok=True)
out = FIGDIR / "figure_conductivity_ionicity.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
plt.close(fig)
