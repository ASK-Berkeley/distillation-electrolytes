#!/usr/bin/env python3
"""MLP (teacher) transport parity, 1 x 5 row: D+ / D- / D0 / sigma_Onsager / eta.

Same visual language as main_paper/figure_3 (1x4 conductivity+viscosity parity)
and figure_4 (diffusivity parity), all panels from the CORRECT-MOLARITY MLP boxes:

  (a-c) self-diffusivity   meta_data/correct_molarity/MLP/merged/diffusivity_by_system.csv
                           (Yeh-Hummer finite-size correction built from the MLP's OWN
                            simulated Green-Kubo viscosity, i.e. the
                            D_*_corrected_simvisc_1e-10_m2s_* columns -- self-consistent
                            with panel (e), unlike figure 4 which used the experimental
                            viscosity)
  (d)   sigma_Onsager      meta_data/correct_molarity/MLP/merged/conductivity_by_system.csv
  (e)   eta (Green-Kubo)   meta_data/correct_molarity/MLP/viscosity/viscosity_by_system.csv

- 298 K only.
- Conductivity drops 0.1 M (as in figure 3); D and eta keep every concentration.
- Encoding: color = solvent family, marker = anion, triangle = LiPF6,
  fill = concentration: open = 0.1 M, half-open = 0.5 M, filled = 1.0 M.
- Log-log, equal aspect; the three D panels share one range, sigma and eta each
  carry their own (units differ).
- Stat box reports the log-log Pearson r, same statistic as figures 3 and 4.

The MLP runs' summary CSVs live outside this repository (``paths.MLP_BASELINE``).
When they are not reachable, the figure is drawn from the committed
``MLP_plotted_points.csv`` beside this script, which holds exactly the plotted
points; a run against the raw CSVs refreshes that file.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

from src.analysis.observables import paths

# No options: the parser only gives --help and rejects stray arguments, so a
# mistyped flag cannot silently regenerate the figure.
argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

MLP  = paths.MLP_BASELINE
OUT  = paths.FIGURES_APPENDIX
OUT.mkdir(parents=True, exist_ok=True)

DIFF = MLP / "merged" / "diffusivity_by_system.csv"
COND = MLP / "merged" / "conductivity_by_system.csv"
VISC = MLP / "viscosity" / "viscosity_by_system.csv"

T_TARGET = 298.0
CONC_MIN_COND = 0.15     # conductivity: keep 0.5 M and 1.0 M, drop 0.1 M
CONC_MIN_OTHER = 0.0     # D and eta: keep everything, 0.1 M included

MODEL_LABEL = "MLP"

# ── style maps (identical to figures 3 / 4) ─────────────────────────────────
SOLV_COLOR = {"DME": "#E8850E", "DIGLYME": "#1ad73c", "PC": "#E6197D", "TGDME": "#2E6FBF"}
SOLV_DISP  = {"DME": "DME", "DIGLYME": "DEGDME", "PC": "PC", "TGDME": "TEGDME"}
ANION_MARK = {"PF6": "o", "OTf": "s"}
SALT_DISP  = {("Na", "PF6"): "NaPF$_6$", ("Na", "OTf"): "NaOTf", ("Li", "PF6"): "LiPF$_6$"}

HALF_SIDE = "left"       # which half of the marker is inked for the 0.5 M systems

D_UNIT = "$10^{-10}\\,\\mathrm{m^2\\,s^{-1}}$"

# panel key -> (title, unit, axis-group)
PANELS = [
    ("Dcat", "Cation  $D^{+}$",                    D_UNIT,                     "diff"),
    ("Dani", "Anion  $D^{-}$",                     D_UNIT,                     "diff"),
    ("Dsol", "Solvent  $D^{0}$",                   D_UNIT,                     "diff"),
    ("cond", "Ionic conductivity  $\\sigma_{\\mathrm{Onsager}}$", "$\\mathrm{mS\\,cm^{-1}}$", "cond"),
    ("visc", "Dynamic viscosity  $\\eta$  (Green–Kubo)",          "cP",                       "visc"),
]
# row 1 = the three diffusivity panels, row 2 = conductivity + viscosity
ROW1 = ["Dcat", "Dani", "Dsol"]
ROW2 = ["cond", "visc"]
PANEL_INFO = {k: (t, u, g) for k, t, u, g in PANELS}
ROW1_HEADER = "Self-diffusivity (Yeh–Hummer, simulated $\\eta$)"


def fill_kw(conc, col):
    """Marker fill encoding of concentration: open / half-open / filled."""
    if conc < 0.3:
        return dict(mfc="none", mec=col, fillstyle="full")
    if conc < 0.75:
        return dict(mfc=col, mfcalt="none", mec=col, fillstyle=HALF_SIDE)
    return dict(mfc=col, mec="0.4", fillstyle="full")


def solv_key(token):
    """'dme' / 'DME' / 'Diglyme' / 'TGDME' -> canonical key."""
    return {"DME": "DME", "DIGLYME": "DIGLYME", "PC": "PC",
            "TGDME": "TGDME"}[str(token).strip().upper()]


def salt_parts(salt_token):
    """'lipf6' / 'naotf' -> (cation, anion)."""
    s = str(salt_token).lower()
    return ("Li" if s.startswith("li") else "Na", "PF6" if "pf6" in s else "OTf")


# ── parsers: each returns cols [cation, anion, solv_key, conc, exp, sim, sim_std] ──
def parse_diff(path, species):
    """system_id = '<salt>_<solvent>'; species in {cat, ani, sol}."""
    expcol = {"cat": "exp_D_cation_ref", "ani": "exp_D_anion_ref", "sol": "exp_D_solvent_ref"}[species]
    df = pd.read_csv(path)
    df = df[np.isclose(df["temperature_K"], T_TARGET)
            & (df["concentration_M"] > CONC_MIN_OTHER)]
    rows = []
    for _, r in df.iterrows():
        salt, solv_tok = r["system_id"].split("_", 1)
        cat, ani = salt_parts(salt)
        rows.append(dict(cation=cat, anion=ani, solv_key=solv_key(solv_tok),
                         conc=r["concentration_M"],
                         exp=r[expcol],
                         sim=r[f"D_{species}_corrected_simvisc_1e-10_m2s_mean"],
                         sim_std=r[f"D_{species}_corrected_simvisc_1e-10_m2s_std"]))
    return pd.DataFrame(rows)


def parse_cond(path):
    df = pd.read_csv(path)
    df = df[np.isclose(df["temperature_K"], T_TARGET)
            & (df["concentration_M"] > CONC_MIN_COND)]
    rows = []
    for _, r in df.iterrows():
        salt, solv_tok = r["system_id"].split("_", 1)
        cat, ani = salt_parts(salt)
        rows.append(dict(cation=cat, anion=ani, solv_key=solv_key(solv_tok),
                         conc=r["concentration_M"],
                         exp=r["exp_conductivity_mS_cm"],
                         sim=r["sigma_onsager_mS_cm_mean"],
                         sim_std=r["sigma_onsager_mS_cm_std"]))
    return pd.DataFrame(rows)


def parse_visc(path):
    df = pd.read_csv(path)
    df = df[np.isclose(df["temperature_K"], T_TARGET)
            & (df["concentration_M"] > CONC_MIN_OTHER)]
    rows = []
    for _, r in df.iterrows():
        rows.append(dict(cation=r["cation"], anion=r["anion"], solv_key=solv_key(r["solvent"]),
                         conc=r["concentration_M"],
                         exp=r["exp_dyn_viscosity_cP"],
                         sim=r["eta_tdm_mean_cP"],
                         sim_std=r["eta_tdm_std_cP"]))
    return pd.DataFrame(rows)


QUANT = {"Dcat": "D_cation_simvisc_1e-10_m2s", "Dani": "D_anion_simvisc_1e-10_m2s",
         "Dsol": "D_solvent_simvisc_1e-10_m2s", "cond": "sigma_onsager_mS_cm", "visc": "eta_cP"}
POINTS = Path(__file__).resolve().parent / "MLP_plotted_points.csv"

FROM_RAW = DIFF.is_file() and COND.is_file() and VISC.is_file()
if FROM_RAW:
    data = {
        "Dcat": parse_diff(DIFF, "cat"),
        "Dani": parse_diff(DIFF, "ani"),
        "Dsol": parse_diff(DIFF, "sol"),
        "cond": parse_cond(COND),
        "visc": parse_visc(VISC),
    }
else:
    print(f"{MLP} not reachable; drawing from the committed {POINTS.name}")
    pts = pd.read_csv(POINTS)
    data = {k: pts[pts["quantity"] == q]
                .drop(columns=["model", "quantity", "ratio_sim_over_exp"])
                .reset_index(drop=True)
            for k, q in QUANT.items()}


def finite(sub):
    if len(sub) == 0:
        return sub
    return sub[(sub.exp > 0) & sub.exp.notna() & (sub.sim > 0) & sub.sim.notna()]


def log_lims(vals, pad=3.0):
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0])
    return v.min() / pad, v.max() * pad


# axis range shared within each unit group
group_lims = {}
for grp in ("diff", "cond", "visc"):
    vals = []
    for key, _, _, g in PANELS:
        if g != grp:
            continue
        sub = finite(data[key])
        vals += list(sub.exp) + list(sub.sim)
    group_lims[grp] = log_lims(vals)


def logpearson(exp, sim):
    m = (exp > 0) & (sim > 0) & np.isfinite(exp) & np.isfinite(sim)
    if m.sum() < 2:
        return np.nan, int(m.sum())
    return np.corrcoef(np.log10(exp[m]), np.log10(sim[m]))[0, 1], int(m.sum())


def mard(exp, sim):
    """Mean absolute relative deviation, |sim/exp - 1| averaged, in %."""
    m = (exp > 0) & (sim > 0) & np.isfinite(exp) & np.isfinite(sim)
    if m.sum() == 0:
        return np.nan
    return 100.0 * np.mean(np.abs(sim[m] / exp[m] - 1.0))


stats = {}


def draw_panel(ax, key, grp, ms=8.5, statfs=8.0):
    lo, hi = group_lims[grp]
    ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2],
                    color="0.85", alpha=0.7, zorder=0, lw=0)
    ax.plot([lo, hi], [lo, hi], "--", color="0.35", lw=1.1, zorder=1)

    sub = finite(data[key])
    for _, row in sub.iterrows():
        col = SOLV_COLOR[row.solv_key]
        mk = "^" if row.cation == "Li" else ANION_MARK[row.anion]
        ax.errorbar(row.exp, row.sim, yerr=row.sim_std, fmt=mk, ms=ms,
                    mew=1.2, color=col, **fill_kw(row.conc, col),
                    ecolor="0.5", elinewidth=0.8, capsize=2.2, zorder=3)

    r, n = logpearson(sub.exp.values, sub.sim.values)
    stats[key] = (r, n, mard(sub.exp.values, sub.sim.values))
    ax.text(0.04, 0.955, f"log–log Pearson\nr = {r:.2f}  (n={n})",
            transform=ax.transAxes, va="top", ha="left", fontsize=statfs,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.95))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, which="both", ls=":", alpha=0.45)


SALT_ORDER = [("Na", "PF6", "DME"), ("Na", "OTf", "DME"),
              ("Na", "PF6", "DIGLYME"), ("Na", "OTf", "DIGLYME"),
              ("Na", "PF6", "PC"), ("Na", "OTf", "PC"),
              ("Na", "PF6", "TGDME"), ("Na", "OTf", "TGDME")]


def proxy(mk, col, conc, ms, mew=1.2):
    return Line2D([], [], marker=mk, color=col, mew=mew, ls="none", ms=ms,
                  **fill_kw(conc, col))


def legend_entries(ms=8.5):
    handles, labels = [], []
    for cat, ani, solv in SALT_ORDER:
        col = SOLV_COLOR[solv]; mk = ANION_MARK[ani]
        handles.append((proxy(mk, col, 0.1, ms), proxy(mk, col, 1.0, ms, mew=1.0)))
        labels.append(f"{SALT_DISP[(cat, ani)]} in {SOLV_DISP[solv]}")
    handles.append(proxy("^", SOLV_COLOR["DME"], 0.5, ms + 1))
    labels.append("LiPF$_6$ in DME (0.5 M)")
    handles.append(tuple(proxy("o", "0.35", c, ms) for c in (0.1, 0.5, 1.0)))
    labels.append("0.1 / 0.5 / 1.0 M")
    handles.append(Line2D([], [], ls="--", color="0.35", lw=1.1)); labels.append("1:1  ($y=x$)")
    handles.append(Patch(fc="0.85", ec="none")); labels.append("within ×2")
    handles.append(Line2D([], [], color="0.5", lw=0.9)); labels.append("±1 s.d. (replicas)")
    return handles, labels


LEGEND_NOTE_WRAPPED = ("open = 0.1 M    half-open = 0.5 M    filled = 1.0 M\n"
                       "0.1 M shown for $D$ and $\\eta$ only;\n"
                       "the conductivity panel is 0.5 / 1.0 M")


def make_2row():
    """Row 1 = D+/D-/D0, row 2 = sigma + eta with the legend filling the free slot."""
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 10.2))
    letters = iter("abcde")

    for ri, keys in enumerate((ROW1, ROW2)):
        for ci in range(3):
            ax = axes[ri, ci]
            if ci >= len(keys):
                ax.axis("off")           # free slot in row 2 -> legend goes here
                continue
            key = keys[ci]
            title, unit, grp = PANEL_INFO[key]
            draw_panel(ax, key, grp, ms=9.0, statfs=8.5)
            ax.text(-0.02, 1.035, next(letters), transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=15, fontweight="bold")
            ax.set_title(title, fontsize=12.5, pad=7)
            ax.set_xlabel(f"Experiment  ({unit})", fontsize=16)
            ax.set_ylabel(f"{MODEL_LABEL} simulation  ({unit})", fontsize=16)
            ax.tick_params(labelsize=12)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94], h_pad=3.0)

    # spanning header over the diffusivity row, placed from the final axes geometry
    b0 = axes[0, 0].get_position(); b1 = axes[0, 2].get_position()
    ytop = max(b0.y1, b1.y1)
    fig.text(0.5 * (b0.x0 + b1.x1), ytop + 0.052, ROW1_HEADER,
             ha="center", va="bottom", fontsize=14.5)
    fig.lines.append(plt.Line2D([b0.x0, b1.x1], [ytop + 0.045, ytop + 0.045],
                                transform=fig.transFigure, color="0.55", lw=1.0))

    handles, labels = legend_entries(ms=9)
    slot = axes[1, 2].get_position()
    leg = fig.legend(handles, labels, loc="center left",
                     bbox_to_anchor=(slot.x0 - 0.01, 0.5 * (slot.y0 + slot.y1)),
                     ncol=1, fontsize=12, frameon=False, handlelength=2.4,
                     handletextpad=0.7, labelspacing=0.85,
                     handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
                     title="fill = concentration", title_fontsize=12)
    leg.get_title().set_color("0.25")
    fig.text(0.5 * (slot.x0 + slot.x1), slot.y0 - 0.03, LEGEND_NOTE_WRAPPED,
             ha="center", va="top", fontsize=9, color="0.25")
    return fig, "MLP_diffusivity_conductivity_viscosity_parity_2row"


fig, stem = make_2row()
for ext in ("png",):
    p = OUT / f"{stem}.{ext}"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print("saved", p)
plt.close(fig)

# ── console summary + machine-readable dump of exactly what was plotted ──────
print("\npanel statistics (MLP, 298 K):")
for key, (r, n, md) in stats.items():
    print(f"  {key:5s}  n={n:2d}  log-log r={r:5.2f}  MARD={md:6.1f}%")

if FROM_RAW:
    rows = []
    for key, df in data.items():
        d = finite(df).copy()
        d.insert(0, "quantity", QUANT[key])
        d.insert(0, "model", MODEL_LABEL)
        rows.append(d)
    plotted = pd.concat(rows, ignore_index=True)
    plotted["ratio_sim_over_exp"] = plotted.sim / plotted.exp
    plotted.to_csv(POINTS, index=False)
    print("saved", POINTS)
