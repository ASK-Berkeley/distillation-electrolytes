#!/usr/bin/env python3
"""2-row x 3-col self-diffusivity parity: rows = Student / OPLS, cols = D+/D-/D0.

Adapted from general_analysis/plot/plot_diffusivity_parity_3models.py, restricted
to the two models of interest with the student model (PaiNN, labelled "Student"
in the figure) on the top row.
- **Fickian (beta~=1 window) diffusivities** (``D_*_fick_*`` columns, see
  ``FIT_FAMILY``): the per-species data-driven Einstein window, the production
  choice for the paper (the fixed 1-9 ns window sits in the noisy few-ion tail
  for the sparse boxes; ~3x larger replica scatter, same accuracy). Falls back
  to the fixed-window columns WITH A LOUD WARNING if a by-box file predates the
  dual-method drivers — rerun the driver to get the fickian columns.
- Yeh-Hummer corrected simulated D (mean over replicas, +/-1 s.d. error bars).
  The correction uses each model's own simulated Green-Kubo shear viscosity as
  the YH input (``D_{sp}_inf_sim_*``), so the correction is fully self-consistent
  within each model (no dependence on the experimental eta table).
- Experiment @ 298.2 K, systems restricted to 298 K.
- Color = solvent family, marker = anion, open/filled = 0.1 / 1.0 M (keyed in grey in
  the legend's last column), pentagon = LiPF6 0.5 M.
- Axes aligned across every panel.

Data comes from
``data/painn_transport_observables_replicas/diffusivity/diffusivity_by_box.csv``
(Student, the 4-replica set) and
``data/opls_transport_observables/diffusivity/diffusivity_by_box.csv`` (OPLS) --
both drivers share the same aggregation code
(``diffusivity_replicas.aggregate_by_box``), so the two files have identical
columns and are parsed with the same function here. Simulation points carry +/-1
s.d. error bars over the box's replicas and each panel shades the within-x2-of-
parity region in grey.
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
REPO = Path(__file__).resolve().parents[4]
DATA = paths.OBS_ROOT
OUT  = paths.FIGURES_MAIN
OUT.mkdir(parents=True, exist_ok=True)

DIFF_STUDENT = DATA / "painn_transport_observables_replicas" / "diffusivity" / "diffusivity_by_box.csv"
DIFF_OPLS    = DATA / "opls_transport_observables"           / "diffusivity" / "diffusivity_by_box.csv"

# ── style maps ───────────────────────────────────────────────────────────────
SOLV_COLOR = {"DME": "#E8850E", "DIGLYME": "#1ad73c", "PC": "#E6197D", "TGDME": "#2E6FBF"}
SOLV_DISP  = {"DME": "DME", "DIGLYME": "DEGDME", "PC": "PC", "TGDME": "TEGDME"}
ANION_MARK = {"PF6": "o", "OTf": "s"}
SALT_DISP  = {("Na", "PF6"): "NaPF$_6$", ("Na", "OTf"): "NaOTf", ("Li", "PF6"): "LiPF$_6$"}
SOLV_FROM_TOKEN = {"dme": "DME", "diglyme": "DIGLYME", "pc": "PC", "tgdme": "TGDME"}

# Which fit-window column family to plot: "fick_" = Fickian beta~=1 window (the
# production choice), "" = fixed 1-9 ns window (the cross-check).
FIT_FAMILY = "fick_"

PEARSON_MODE = "loglog"   # "loglog" (default) or "linear" -- see logpearson/linpearson below
SHOW_FIT_LINE = False     # toggle: draw the log-log best-fit line on the plot (slope is
                          # always computed and reported in the per-panel stat box)

# Layout toggle. True (default): the three metrics are pulled out of the panels
# into a compact table drawn BELOW both panel rows (above the legend) -- its three
# column groups line up under the D+/D-/D0 columns, and inside each group a Student
# value row sits above an OPLS-AA value row. False: the old per-panel stat box
# inside each axes.
SHOW_STATS_TABLE = True

PROPS = [("cation", r"$\mathbf{Cation}$ self-diffusivity $D^{+}$"),
         ("anion",  r"$\mathbf{Anion}$ self-diffusivity $D^{-}$"),
         ("solvent", r"$\mathbf{Solvent}$ self-diffusivity $D^{0}$")]

# ── type scale (matched to figure 2) ─────────────────────────────────────────
TICK_FS   = 14.0   # axis tick numbers
LABEL_FS  = 17.0   # axis labels
TITLE_FS  = 17.0   # column titles -- kept identical to LABEL_FS
STAT_FS   = 12.5   # per-panel stat box (log-log Pearson r etc.)
LETTER_FS = 18.0   # panel letters a..f
LEGEND_FS = 13.5   # right-margin system legend
LEG_TITLE_FS = 15.5  # legend title (the MdAPD acronym line)
TITLE_PAD = 34.0   # gap between column titles and the top-row panels/letters
ROW_GAP_IN = 0.75  # table mode: vertical gap (inches) between the two panel rows


def _cation_anion(salt_token):
    """'napf6' -> ('Na', 'PF6'); 'naotf' -> ('Na', 'OTf'); 'lipf6' -> ('Li', 'PF6')."""
    cation = "Li" if salt_token.startswith("li") else "Na"
    if "pf6" in salt_token:
        return cation, "PF6"
    if "otf" in salt_token:
        return cation, "OTf"
    raise ValueError(f"unknown salt {salt_token!r}")


EMPTY = pd.DataFrame(columns=["cation", "anion", "solv_key", "conc", "prop", "exp", "sim", "sim_std"])


def parse_by_box(path):
    """<root>/diffusivity/diffusivity_by_box.csv -> long-form rows, one per
    (box, species). Both PaiNN and OPLS drivers write this exact schema, so this
    one parser covers both models. Reads the ``FIT_FAMILY`` column family
    (Fickian by default); if the file predates the dual-method drivers it falls
    back to the fixed-window columns and warns loudly."""
    if not path.is_file():
        return EMPTY
    df = pd.read_csv(path)
    df = df[np.isclose(df["temperature_K"], 298.0)].copy()
    pre = FIT_FAMILY
    if pre and f"D_cation_{pre}inf_sim_mean_1e10" not in df.columns:
        print(f"[WARN] {path} has no '{pre}*' columns (predates the dual-method "
              f"driver) — FALLING BACK to the fixed-window columns. Rerun the "
              f"diffusivity driver to get the Fickian family.")
        pre = ""
    rows = []
    for _, r in df.iterrows():
        cation, anion = _cation_anion(r["salt"])
        solv_key = SOLV_FROM_TOKEN[r["solvent"]]
        for prop, _ in PROPS:
            rows.append(dict(cation=cation, anion=anion, solv_key=solv_key,
                             conc=r["conc_M"], prop=prop,
                             exp=r[f"D_{prop}_exp_1e10"],
                             sim=r[f"D_{prop}_{pre}inf_sim_mean_1e10"],
                             sim_std=r[f"D_{prop}_{pre}inf_sim_std_1e10"]))
    return pd.DataFrame(rows)


data = {
    "PaiNN": parse_by_box(DIFF_STUDENT),
    "OPLS":  parse_by_box(DIFF_OPLS),
}
MODELS = ["PaiNN", "OPLS"]          # PaiNN (displayed as "Student") on top row
ROW_LABEL = {"PaiNN": "Student", "OPLS": "OPLS-AA"}

# ── aligned limits (across both models) ─────────────────────────────────────
def log_lims(vals, pad=3.0):
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0])
    lo, hi = v.min(), v.max()
    return lo / pad, hi * pad

# single global x/y range shared by every panel (aligned across all cols & rows)
_all = []
for m in MODELS:
    sub = data[m]
    sub = sub[(sub.exp > 0) & sub.exp.notna() & (sub.sim > 0) & sub.sim.notna()]
    _all += list(sub.exp) + list(sub.sim)
GLOB = log_lims(_all)
col_lims = {prop: GLOB for prop, _ in PROPS}


def logpearson(exp, sim):
    m = (exp > 0) & (sim > 0) & np.isfinite(exp) & np.isfinite(sim)
    if m.sum() < 2:
        return np.nan, int(m.sum())
    r = np.corrcoef(np.log10(exp[m]), np.log10(sim[m]))[0, 1]
    return r, int(m.sum())


def linpearson(exp, sim):
    """Plain (linear-space) Pearson r -- more forgiving of multiplicative bias
    than logpearson since it's dominated by the largest-magnitude points."""
    m = (exp > 0) & (sim > 0) & np.isfinite(exp) & np.isfinite(sim)
    if m.sum() < 2:
        return np.nan, int(m.sum())
    r = np.corrcoef(exp[m], sim[m])[0, 1]
    return r, int(m.sum())


def pearson(exp, sim):
    """Dispatch to logpearson or linpearson per PEARSON_MODE."""
    fn = {"loglog": logpearson, "linear": linpearson}[PEARSON_MODE]
    return fn(exp, sim)


PEARSON_LABEL = {"loglog": "log–log Pearson r", "linear": "Pearson r"}


def mdapd(exp, sim):
    """Median absolute percentage deviation median(|sim-exp|/exp) -- unlike log-log
    Pearson r, this tracks actual distance from the y=x line drawn on the plot, not
    just whether sim/exp trend together. Median (not mean) so a single outlier
    doesn't dominate the summary."""
    m = (exp > 0) & (sim > 0) & np.isfinite(exp) & np.isfinite(sim)
    if m.sum() < 1:
        return np.nan, int(m.sum())
    return float(np.median(np.abs(sim[m] - exp[m]) / exp[m]) * 100.0), int(m.sum())


def fit_loglog(exp, sim):
    """Least-squares fit of log10(sim) = slope*log10(exp) + intercept.
    Returns (slope, intercept) or None if too few points."""
    m = (exp > 0) & (sim > 0) & np.isfinite(exp) & np.isfinite(sim)
    if m.sum() < 2:
        return None
    slope, intercept = np.polyfit(np.log10(exp[m]), np.log10(sim[m]), 1)
    return float(slope), float(intercept)


FIT_STYLE = dict(color="#4EA8DE", lw=1.4, alpha=0.75, ls="-")


# ── figure ──────────────────────────────────────────────────────────────────
NROW = len(MODELS)
# Legend lives below the panels (not in a right margin), so the full canvas
# width belongs to the plots: narrower figure, bigger panels. Extra height
# reserves room for the three legend rows underneath.
# In table mode the rows are pushed apart to make room for the stats strip that
# sits between them, so the canvas is taller.
# Wider canvas with the type scale held fixed => bigger panels relative to the
# labels/legend, i.e. less white space per unit of plot.
FIG_W = 13.0 if SHOW_STATS_TABLE else 11.8
FIG_H = 12.6 if SHOW_STATS_TABLE else 9.6
fig, axes = plt.subplots(NROW, 3, figsize=(FIG_W, FIG_H))
panel_letters = "abcdef"
panel_stats = {}   # (model, prop) -> dict(r=, mdapd=, slope=)

for ri, model in enumerate(MODELS):
    for ci, (prop, ptitle) in enumerate(PROPS):
        ax = axes[ri, ci]
        lo, hi = col_lims[prop]
        # within-x2-of-parity band (between y = x/2 and y = 2x), shaded grey
        ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2],
                        color="0.8", alpha=0.35, lw=0, zorder=0)
        ax.plot([lo, hi], [lo, hi], "--", color="0.35", lw=1.1, zorder=1)

        sub = data[model]
        sub = sub[(sub.prop == prop) & (sub.exp > 0) & sub.exp.notna()
                  & (sub.sim > 0) & sub.sim.notna()]
        for _, row in sub.iterrows():
            col = SOLV_COLOR[row.solv_key]
            extra = {}
            if row.cation == "Li":
                mk, filled = "p", True
                # LiPF6 in DME is 0.5 M -> half-filled pentagon
                extra = dict(fillstyle="left", markerfacecoloralt="none")
            else:
                mk = ANION_MARK[row.anion]
                filled = row.conc > 0.5
            mfc = col if filled else "none"
            mec = "0.4" if filled else col
            ax.errorbar(row.exp, row.sim, yerr=row.sim_std, fmt=mk, ms=9.5,
                        mfc=mfc, mec=mec, mew=1.2, color=col,
                        ecolor="0.5", elinewidth=0.8, capsize=2.2, zorder=3, **extra)

        r, n = pearson(sub.exp.values, sub.sim.values)
        mdapd_val, _ = mdapd(sub.exp.values, sub.sim.values)
        fit = fit_loglog(sub.exp.values, sub.sim.values)
        slope = fit[0] if fit is not None else None
        panel_stats[(model, prop)] = dict(r=r, mdapd=mdapd_val, slope=slope)
        if fit is not None and SHOW_FIT_LINE:
            xs = np.array([lo, hi])
            ys = 10 ** (fit[1] + slope * np.log10(xs))
            ax.plot(xs, ys, zorder=2, **FIT_STYLE)
        if not SHOW_STATS_TABLE:
            stat_txt = f"{PEARSON_LABEL[PEARSON_MODE]} = {r:.2f}\nMdAPD = {mdapd_val:.0f}%"
            if slope is not None:
                stat_txt += f"\nlog–log fit slope = {slope:.2f}"
            ax.text(0.04, 0.955, stat_txt,
                    transform=ax.transAxes, va="top", ha="left", fontsize=STAT_FS,
                    fontweight="bold")
        ax.text(-0.10, 1.02, f"{panel_letters[ri * 3 + ci]})", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=LETTER_FS, fontweight="bold")
        # model identifier inside the panel, top-left (matches figure 3)
        ax.text(0.045, 0.955, ROW_LABEL[model], transform=ax.transAxes,
                va="top", ha="left", fontsize=14.0, fontweight="bold", color="0.1")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(True, which="both", ls=":", alpha=0.45)
        ax.tick_params(which="major", labelsize=TICK_FS)
        # axes are aligned across every panel, so drop the redundant tick numbers:
        # x tick labels only on the bottom row, y tick labels only on the first column.
        if ri != NROW - 1:
            ax.tick_params(which="both", labelbottom=False)
        if ci != 0:
            ax.tick_params(which="both", labelleft=False)
        if ri == 0:
            ax.set_title(ptitle, fontsize=TITLE_FS, pad=TITLE_PAD)
        if ri == NROW - 1:
            ax.set_xlabel("Experiment  ($10^{-10}\\,\\mathrm{m^2\\,s^{-1}}$)", fontsize=LABEL_FS)
        if ci == 0:
            disp = ROW_LABEL[model].replace("-", "{-}")   # bold the model name only
            ax.set_ylabel(r"$\mathbf{%s\ simulation}$" % disp
                          + "\n($10^{-10}\\,\\mathrm{m^2\\,s^{-1}}$)", fontsize=LABEL_FS)

# ── legend (below the panels) ────────────────────────────────────────────────
salt_order = [("Na", "PF6", "DME"), ("Na", "OTf", "DME"),
              ("Na", "PF6", "DIGLYME"), ("Na", "OTf", "DIGLYME"),
              ("Na", "PF6", "PC"), ("Na", "OTf", "PC"),
              ("Na", "PF6", "TGDME"), ("Na", "OTf", "TGDME")]
handles, labels = [], []
for cat, ani, solv in salt_order:
    col = SOLV_COLOR[solv]; mk = ANION_MARK[ani]
    h_open = Line2D([], [], marker=mk, color=col, mfc="none", mec=col, mew=1.2, ls="none", ms=9)
    h_fill = Line2D([], [], marker=mk, color=col, mfc=col, mec="0.4", mew=1.0, ls="none", ms=9)
    handles.append((h_open, h_fill))
    labels.append(f"{SALT_DISP[(cat, ani)]} in {SOLV_DISP[solv]}")
# LiPF6 0.5 M pentagon
handles.append(Line2D([], [], marker="p", color=SOLV_COLOR["DME"], mfc=SOLV_COLOR["DME"],
                      mec="0.4", ls="none", ms=10, fillstyle="left",
                      markerfacecoloralt="none"))
labels.append("LiPF$_6$ in DME (0.5 M)")
# guide entries
handles.append(Line2D([], [], ls="--", color="0.35", lw=1.1)); labels.append("1:1  ($y=x$)")
handles.append(Patch(facecolor="0.8", alpha=0.6, edgecolor="none")); labels.append("within ×2 of parity")
if SHOW_FIT_LINE:
    handles.append(Line2D([], [], **FIT_STYLE)); labels.append("log–log best fit\n(slope shown per panel)")
# draw the ±1 s.d. key as a capped error bar (matching the on-plot error bars),
# not a plain line -- built from a dummy (nan) errorbar so nothing renders on the axis
_eb = axes[0, 0].errorbar([np.nan], [np.nan], yerr=[np.nan], fmt="none",
                          ecolor="0.5", elinewidth=0.8, capsize=2.2)
handles.append(_eb); labels.append("±1 s.d. (replicas)")
# concentration key in its own (5th) column: neutral-grey circle + square pairs,
# open = 0.1 M, filled = 1.0 M, whatever the anion (shape) or solvent (colour)
for filled, lab in ((False, "0.1 M"), (True, "1.0 M")):
    kw = dict(mfc="0.6", mec="0.4", mew=1.0) if filled else dict(mfc="none", mec="0.4", mew=1.2)
    handles.append(tuple(Line2D([], [], marker=mk, color="0.4", ls="none", ms=9, **kw)
                         for mk in ("o", "s")))
    labels.append(lab)
handles.append(Line2D([], [], ls="none")); labels.append("")

LEG_TITLE = "MdAPD = Median Absolute Percentage Deviation"


def draw_stats_table(fig, axes, panel_stats, y_top):
    """Table-mode: the stats strip drawn BELOW both panel rows (above the legend).

    Three column groups line up under the D+/D-/D0 panel columns; inside each
    group the three metrics sit in their own sub-column under a shared header.
    Each group carries a Student value row above an OPLS-AA value row. ``y_top``
    is the figure-fraction top of the block; returns the y of its lowest row so
    the legend can be placed just beneath it.
    """
    bot = [ax.get_position(original=False) for ax in axes[1]]
    cen = [(p.x0 + p.x1) / 2 for p in bot]
    dx = bot[0].width / 3.45                    # metric sub-column spacing
    x_lab_r = min(p.x0 for p in bot) - 0.018    # right edge of the row-label column

    # Row offsets in inches below the block top.
    O_HDR, O_IDEAL, O_RULE, O_V0, O_V1 = 0.11, 0.32, 0.50, 0.80, 1.20
    y_hdr, y_ideal, y_rule = (y_top - o / FIG_H for o in (O_HDR, O_IDEAL, O_RULE))
    y_val = [y_top - O_V0 / FIG_H, y_top - O_V1 / FIG_H]

    metrics = [("log\u2013log $r$", "ideal 1.00", "r"),
               ("MdAPD",           "ideal 0%",   "mdapd"),
               ("fit slope",       "ideal 1.00", "slope")]

    def fmt(key, v):
        if v is None or not np.isfinite(v):
            return "\u2014"
        return f"{v:.0f}%" if key == "mdapd" else f"{v:.2f}"

    for ci in range(3):
        for k, (name, ideal, _key) in enumerate(metrics):
            x = cen[ci] + (k - 1) * dx
            fig.text(x, y_hdr, name, ha="center", va="center",
                     fontsize=13.5, color="0.1", fontweight="bold")
            fig.text(x, y_ideal, ideal, ha="center", va="center",
                     fontsize=14.0, color="0.45")
        half = 1.45 * dx        # inset from the panel edges: the rule marks the
        fig.lines.append(plt.Line2D(   # table's own width, it is not a plot rule
            [cen[ci] - half, cen[ci] + half], [y_rule] * 2,
            transform=fig.transFigure, color="0.55", lw=1.0))

    for ri, model in enumerate(MODELS):
        fig.text(x_lab_r, y_val[ri], ROW_LABEL[model], ha="right", va="center",
                 fontsize=14.5, color="0.1", fontweight="bold")
        for ci, (prop, _t) in enumerate(PROPS):
            st = panel_stats.get((model, prop)) or {}
            for k, (_n, _i, key) in enumerate(metrics):
                fig.text(cen[ci] + (k - 1) * dx, y_val[ri], fmt(key, st.get(key)),
                         ha="center", va="center", fontsize=15.0, color="0.05")
    return y_val[-1]


# Reserve the bottom strip for the stats table + legend; panels use the full width.
fig.tight_layout(rect=[0.0, (0.30 if SHOW_STATS_TABLE else 0.12), 1.0, 1.0])
# Pull the two rows closer together (equal-aspect panels otherwise leave a wide
# vertical gap); tight_layout sets a default hspace we override here. A little
# extra room vs. squished so the d/e/f letters clear the top-row panels.
# wspace trimmed from the 0.2 default: the inter-column gap only has to clear the
# y tick labels, so tightening it grows the (width-limited, equal-aspect) panels.
fig.subplots_adjust(hspace=0.22, wspace=0.18)

leg_y = 0.115
if SHOW_STATS_TABLE:
    # The panels are square (equal aspect) and width-limited, so their drawn
    # boxes are smaller than the subplot slots. Measure them once, then pin both
    # rows explicitly with a small gap between them, close to the top of the
    # canvas, leaving the bottom for the stats table and the legend.
    fig.canvas.draw()
    drawn = [[ax.get_position(original=False) for ax in row] for row in axes]
    w, h = drawn[0][0].width, drawn[0][0].height
    gap = ROW_GAP_IN / FIG_H
    y_top0 = max(p.y1 for p in drawn[0]) - h
    y_bot0 = y_top0 - gap - h
    for ci in range(3):
        axes[0][ci].set_position([drawn[0][ci].x0, y_top0, w, h])
        axes[1][ci].set_position([drawn[1][ci].x0, y_bot0, w, h])
    # stats table below the bottom row, clearing its x tick labels + x label
    tbl_top = y_bot0 - 0.95 / FIG_H
    tbl_bottom = draw_stats_table(fig, axes, panel_stats, tbl_top)
    leg_y = tbl_bottom - 0.55 / FIG_H   # legend just beneath the table

leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, leg_y),
                 ncol=5, fontsize=LEGEND_FS, frameon=False, handlelength=2.2,
                 handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
                 title=LEG_TITLE, title_fontsize=LEG_TITLE_FS,
                 labelspacing=0.6, columnspacing=1.4)

# Tight-crop (keeps the bottom legend), then push only the right edge out to
# leave a little white space on the right of the last column of panels.
from matplotlib.transforms import Bbox

p = OUT / "figure4_diffusivity_parity_painn_opls.png"
fig.canvas.draw()
bb = fig.get_tightbbox(fig.canvas.get_renderer())
bb = Bbox.from_extents(bb.x0, bb.y0, bb.x1 + 0.45, bb.y1)   # +0.45 in on the right
fig.savefig(p, dpi=180, bbox_inches=bb)
print("saved", p)
plt.close(fig)
