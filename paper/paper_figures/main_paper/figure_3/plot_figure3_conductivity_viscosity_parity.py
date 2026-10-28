#!/usr/bin/env python3
"""Figure 3: 1 x 4 transport parity, (a, b) Onsager sigma and (c, d) shear viscosity,
each for the Student and OPLS-AA, with the stats table and legend beneath.
- Conductivity: Onsager sigma only (no Nernst-Einstein), 298 K, 0.5 M + 1.0 M
  systems only (0.1 M dropped).
- Viscosity: Green-Kubo eta_TDM vs experiment, 298 K, ALL concentrations
  including 0.1 M.
- Encoding: color = solvent family, marker = anion, pentagon = LiPF6 0.5 M,
  open = 0.1 M / filled = >= 0.5 M (open symbols occur in the eta column only),
  keyed in grey in the legend's last column.
- Axes are shared per column (units differ between columns), log-log, equal aspect.
- The stats table reports log-log Pearson r, MdAPD and log-log fit slope (as in figure 4).

Data comes from ``data/painn_transport_observables_replicas/`` (Student, the
4-replica set) and ``data/opls_transport_observables/`` (OPLS-AA) -- the
``*_by_box.csv`` outputs of
``conductivity_replicas.py``/``conductivity_opls_replicas.py`` and
``viscosity_gk_replicas.py``/``viscosity_gk_opls_replicas.py``. Both drivers of a
pair share the same aggregation code (``conductivity_common.aggregate`` /
``aggregate_by_box``), so the two files have identical columns and are parsed
with the same functions here. Simulation points carry +/-1 s.d. error bars over
the box's replicas and each panel shades the within-x2-of-parity region in grey.

    python -m paper.paper_figures.main_paper.figure_3.plot_figure3_conductivity_viscosity_parity
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.transforms import Bbox

from src.analysis.observables import paths

# No options: the parser only gives --help and rejects stray arguments, so a
# mistyped flag cannot silently regenerate the figure.
argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

DATA = paths.OBS_ROOT
OUT  = paths.FIGURES_MAIN
OUT.mkdir(parents=True, exist_ok=True)

COND_STUDENT = DATA / "painn_transport_observables_replicas" / "conductivity" / "conductivity_by_box.csv"
COND_OPLS    = DATA / "opls_transport_observables"           / "conductivity" / "conductivity_by_box.csv"
VISC_STUDENT = DATA / "painn_transport_observables_replicas" / "viscosity"    / "viscosity_gk_by_box.csv"
VISC_OPLS    = DATA / "opls_transport_observables"           / "viscosity"    / "viscosity_gk_by_box.csv"

CONC_MIN_COND = 0.15     # conductivity: keep 0.5 M and 1.0 M, drop 0.1 M
CONC_MIN_VISC = 0.0      # viscosity: keep everything, 0.1 M included
T_TARGET = 298.0

# ── style maps ───────────────────────────────────────────────────────────────
SOLV_COLOR = {"DME": "#E8850E", "DIGLYME": "#1ad73c", "PC": "#E6197D", "TGDME": "#2E6FBF"}
SOLV_DISP  = {"DME": "DME", "DIGLYME": "DEGDME", "PC": "PC", "TGDME": "TEGDME"}
ANION_MARK = {"PF6": "o", "OTf": "s"}
SALT_DISP  = {("Na", "PF6"): "NaPF$_6$", ("Na", "OTf"): "NaOTf", ("Li", "PF6"): "LiPF$_6$"}
SOLV_FROM_TOKEN = {"dme": "DME", "diglyme": "DIGLYME", "pc": "PC", "tgdme": "TGDME"}

PEARSON_MODE = "loglog"   # "loglog" (default) or "linear" -- see logpearson/linpearson below
SHOW_FIT_LINE = False     # toggle: draw the log-log best-fit line on the plot (slope is
                          # always computed and reported in the per-panel stat box)

# Layout toggle. False (default): the log-log Pearson r / MdAPD / fit-slope numbers
# sit in a stat box inside each panel. True: they are pulled out into a compact
# table beneath the four panels (one column per panel, grouped by property), with
# the value closer to ideal shaded so Student vs OPLS-AA is read at a glance.
SHOW_STATS_TABLE = True

MODELS = ["Student", "OPLS-AA"]                                  # top row, bottom row
COLS = [("cond", "Ionic conductivity $\\sigma_{\\mathrm{Onsager}}$", "$\\mathrm{mS\\,cm^{-1}}$"),
        ("visc", "Shear viscosity $\\eta$",                          "cP")]

# ── type scale (matched to figure 2) ─────────────────────────────────────────
TICK_FS   = 14.0   # axis tick numbers
LABEL_FS  = 17.0   # axis labels
TITLE_FS  = 17.0   # column titles -- kept identical to LABEL_FS
STAT_FS   = 12.5   # per-panel stat box (log-log Pearson r etc.)
LETTER_FS = 18.0   # panel letters a/b/c/d
LEGEND_FS = 13.5   # right-margin system legend
TITLE_PAD = 24.0   # gap between column titles and the top-row panels/letters


def _cation_anion(salt_token):
    """'napf6' -> ('Na', 'PF6'); 'naotf' -> ('Na', 'OTf'); 'lipf6' -> ('Li', 'PF6')."""
    cation = "Li" if salt_token.startswith("li") else "Na"
    if "pf6" in salt_token:
        return cation, "PF6"
    if "otf" in salt_token:
        return cation, "OTf"
    raise ValueError(f"unknown salt {salt_token!r}")


# ── parsers: every one returns cols [cation, anion, solv_key, conc, exp, sim, sim_std] ──
# Both drivers of a pair (PaiNN vs OPLS) share the same *_by_box.csv schema, so one
# parser each covers both models -- no per-source column-name translation needed.
def parse_cond(path):
    if not path.is_file():
        return EMPTY
    df = pd.read_csv(path)
    df = df[np.isclose(df["temperature_K"], T_TARGET) & (df["conc_M"] > CONC_MIN_COND)]
    rows = []
    for _, r in df.iterrows():
        cation, anion = _cation_anion(r["salt"])
        rows.append(dict(cation=cation, anion=anion,
                         solv_key=SOLV_FROM_TOKEN[r["solvent"]], conc=r["conc_M"],
                         exp=r["sigma_exp_mS_cm"],
                         sim=r["sigma_onsager_mS_cm_mean"],
                         sim_std=r["sigma_onsager_mS_cm_std"]))
    return pd.DataFrame(rows)


def parse_visc(path):
    """Green-Kubo eta_TDM table (<root>/viscosity/viscosity_gk_by_box.csv)."""
    if not path.is_file():
        return EMPTY
    df = pd.read_csv(path)
    df = df[np.isclose(df["temperature_K"], T_TARGET) & (df["conc_M"] > CONC_MIN_VISC)]
    rows = []
    for _, r in df.iterrows():
        cation, anion = _cation_anion(r["salt"])
        rows.append(dict(cation=cation, anion=anion,
                         solv_key=SOLV_FROM_TOKEN[r["solvent"]], conc=r["conc_M"],
                         exp=r["eta_exp_cP"],
                         sim=r["eta_gk_mean_cP"],
                         sim_std=r["eta_gk_std_cP"]))
    return pd.DataFrame(rows)


EMPTY = pd.DataFrame(columns=["cation", "anion", "solv_key", "conc", "exp", "sim", "sim_std"])

data = {
    ("Student", "cond"): parse_cond(COND_STUDENT),
    ("Student", "visc"): parse_visc(VISC_STUDENT),
    ("OPLS-AA", "cond"): parse_cond(COND_OPLS),
    ("OPLS-AA", "visc"): parse_visc(VISC_OPLS),
}
PLACEHOLDER = {k for k, v in data.items() if len(v) == 0}


def finite(sub):
    if len(sub) == 0:
        return sub
    return sub[(sub.exp > 0) & sub.exp.notna() & (sub.sim > 0) & sub.sim.notna()]


def log_lims(vals, pad=3.0):
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0])
    return v.min() / pad, v.max() * pad


# axis range shared by the two rows of a column (columns carry different units)
col_lims = {}
for short, *_ in COLS:
    vals = []
    for m in MODELS:
        sub = finite(data[(m, short)])
        vals += list(sub.exp) + list(sub.sim)
    col_lims[short] = log_lims(vals)


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


# ── figure (1 x 4) ────────────────────────────────────────────────────────────
# Panels are grouped by quantity so the two models sit side by side and can be
# compared directly: sigma(Student), sigma(OPLS), eta(Student), eta(OPLS). A
# spanning header labels each model pair and the system legend sits underneath.
ORDER = [("Student", "cond"), ("OPLS-AA", "cond"),
         ("Student", "visc"), ("OPLS-AA", "visc")]
unit_of  = {short: unit  for short, _, unit  in COLS}
title_of = {short: title for short, title, _ in COLS}

# type scale trimmed for the narrower 1x4 panels
TICK_FS, LABEL_FS, TITLE_FS = 12.0, 14.5, 15.0
STAT_FS, LETTER_FS, HDR_FS  = 9.5, 15.0, 17.0

# per-panel titles: spell "Student" out, keep OPLS-AA as-is
MODEL_DISP = {"Student": "Student Predictions", "OPLS-AA": "OPLS-AA"}
# short in-panel identifier, top-left inside each plot
MODEL_SHORT = {"Student": "Student", "OPLS-AA": "OPLS-AA"}

FIG_H = 9.0 if SHOW_STATS_TABLE else 5.25
FIG_DPI = 180
fig, axes = plt.subplots(1, 4, figsize=(15.4, FIG_H), dpi=FIG_DPI)
letters = "abcd"
panel_stats = {}   # (model, short) -> dict(r=, mdapd=, slope=)  or  None for placeholders

for i, (model, short) in enumerate(ORDER):
    ax = axes[i]
    unit = unit_of[short]
    lo, hi = col_lims[short]
    # within-x2-of-parity band (between y = x/2 and y = 2x), shaded grey
    ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2],
                    color="0.8", alpha=0.35, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], "--", color="0.35", lw=1.1, zorder=1)

    sub = finite(data[(model, short)])
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
        ax.errorbar(row.exp, row.sim, yerr=row.sim_std, fmt=mk, ms=8.5,
                    mfc=mfc, mec=mec, mew=1.2, color=col,
                    ecolor="0.5", elinewidth=0.8, capsize=2.2, zorder=3, **extra)

    if (model, short) in PLACEHOLDER:
        panel_stats[(model, short)] = None
        ax.text(0.5, 0.5, "PLACEHOLDER\n\n%s %s\nnot yet available" % (model, short),
                transform=ax.transAxes, va="center", ha="center",
                fontsize=11, color="0.45", style="italic",
                bbox=dict(boxstyle="round,pad=0.7", fc="white", ec="0.7",
                          ls="--", alpha=0.9), zorder=4)
    else:
        r, n = pearson(sub.exp.values, sub.sim.values)
        mdapd_val, _ = mdapd(sub.exp.values, sub.sim.values)
        fit = fit_loglog(sub.exp.values, sub.sim.values)
        slope = fit[0] if fit is not None else None
        panel_stats[(model, short)] = dict(r=r, mdapd=mdapd_val, slope=slope)
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
    ax.text(-0.10, 1.02, f"{letters[i]})", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=LETTER_FS, fontweight="bold")
    # model identifier inside the panel, top-left (same size as the stat-table labels)
    ax.text(0.045, 0.955, MODEL_SHORT[model], transform=ax.transAxes,
            va="top", ha="left", fontsize=14.0, fontweight="bold", color="0.1")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.tick_params(which="major", labelsize=TICK_FS)
    ax.set_xlabel(f"Experiment  ({unit})", fontsize=LABEL_FS)
    # (model identity is drawn as the bold header row of the stats table below)
    if short == "cond":
        # conductivity block keeps its y-axis on the left; only panel a is labelled
        if i == 0:
            ax.set_ylabel(f"Simulation  ({unit})", fontsize=LABEL_FS)
    else:
        # viscosity block mirrors its y-axis (ticks + label) to the right edge
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        if i == 3:
            ax.set_ylabel(f"Simulation  ({unit})", fontsize=LABEL_FS,
                          rotation=270, labelpad=18)

# ── legend (bottom) ───────────────────────────────────────────────────────────
salt_order = [("Na", "PF6", "DME"), ("Na", "OTf", "DME"),
              ("Na", "PF6", "DIGLYME"), ("Na", "OTf", "DIGLYME"),
              ("Na", "PF6", "PC"), ("Na", "OTf", "PC"),
              ("Na", "PF6", "TGDME"), ("Na", "OTf", "TGDME")]
handles, labels = [], []
for cat, ani, solv in salt_order:
    col = SOLV_COLOR[solv]; mk = ANION_MARK[ani]
    h_open = Line2D([], [], marker=mk, color=col, mfc="none", mec=col, mew=1.2, ls="none", ms=8.5)
    h_fill = Line2D([], [], marker=mk, color=col, mfc=col, mec="0.4", mew=1.0, ls="none", ms=8.5)
    handles.append((h_open, h_fill))
    labels.append(f"{SALT_DISP[(cat, ani)]} in {SOLV_DISP[solv]}")
h_li = Line2D([], [], marker="p", color=SOLV_COLOR["DME"], mfc=SOLV_COLOR["DME"],
              mec="0.4", ls="none", ms=9.5, fillstyle="left", markerfacecoloralt="none")
# draw the ±1 s.d. key as a capped error bar (matching the on-plot error bars),
# not a plain line -- built from a dummy (nan) errorbar so nothing renders on the axis
_eb = axes[0].errorbar([np.nan], [np.nan], yerr=[np.nan], fmt="none",
                       ecolor="0.5", elinewidth=0.8, capsize=2.2)


def conc_key(filled):
    """Neutral-grey circle + square pair: open = 0.1 M, filled = 1.0 M, whatever the
    anion (marker shape) or solvent (colour)."""
    kw = dict(mfc="0.6", mec="0.4", mew=1.0) if filled else dict(mfc="none", mec="0.4", mew=1.2)
    return tuple(Line2D([], [], marker=mk, color="0.4", ls="none", ms=8.5, **kw)
                 for mk in ("o", "s"))


# Row-major 5 x 3 grid: the eight Na systems fill columns 1-4 of rows 1-2, the
# concentration key (open / filled) takes column 5, and row 3 holds the LiPF6
# pentagon followed by the plot keys.
blank = (Line2D([], [], ls="none"), "")
grid = [handles[0:4] + [conc_key(False)], handles[4:8] + [conc_key(True)],
        [h_li, _eb, Line2D([], [], ls="--", color="0.35", lw=1.1),
         Patch(facecolor="0.8", alpha=0.6, edgecolor="none"), blank[0]]]
grid_lab = [labels[0:4] + ["0.1 M  ($\\eta$ only)"], labels[4:8] + ["1.0 M"],
            ["LiPF$_6$ in DME (0.5 M)", "±1 s.d. (replicas)", "1:1  ($y=x$)",
             "within ×2 of parity", ""]]
if SHOW_FIT_LINE:
    grid.append([Line2D([], [], **FIT_STYLE)] + [blank[0]] * 4)
    grid_lab.append(["log–log best fit\n(slope shown per panel)"] + [""] * 4)
handles = [h for row in grid for h in row]
labels = [l for row in grid_lab for l in row]

LEG_NCOL = 5


def rowmajor(handles, labels, ncol):
    """Reorder a row-major entry list into the column-major order matplotlib's
    legend consumes, padding short columns with blanks.

    ``fig.legend(..., ncol=n)`` fills top-to-bottom, column-by-column, so with 13
    entries and 5 columns the visual rows come out 5/4/4 with the trailing entries
    scattered. Laying the entries out on an explicit row-major grid and then
    transposing keeps each row as written above.
    """
    n = len(handles)
    nrow = int(np.ceil(n / ncol))
    H, L = [], []
    for j in range(ncol):
        for i in range(nrow):
            k = i * ncol + j
            if k < n:
                H.append(handles[k]); L.append(labels[k])
            else:                       # blank filler for the short last row
                H.append(Line2D([], [], ls="none")); L.append("")
    return H, L


handles, labels = rowmajor(handles, labels, LEG_NCOL)

def draw_stats_table(fig, axes, panel_stats):
    """Table-mode: a short, wide stats strip beneath the four panels.

    One value column sits directly under each panel (so every column lines up with
    the plot it summarises); the metric label + its ideal ("target") live in the
    left margin. A single rule separates the Student/OPLS-AA header from the values
    -- no framing top/bottom rules. Returns the caption y so the legend can be
    placed just below it.
    """
    poss = [ax.get_position() for ax in axes]
    y_pb = min(p.y0 for p in poss)
    c = [(p.x0 + p.x1) / 2 for p in poss]     # value columns aligned under panels a–d
    sw = 0.075                                # shaded-cell half-width
    x_lab_r = c[0] - sw - 0.02                # right edge of the metric-label column

    # ── row geometry, hung off the measured panel bottom ──
    # Columns line up under the panels, so the model labels are read from the plot
    # titles above -- no Student/OPLS-AA header row or rule here. Rows are packed
    # tight to keep the strip short.
    t = y_pb - 0.098
    rh = 0.048
    y_hdr   = t                                     # bold model-name row, top of the table
    y_main  = [t - 0.042 - rh * i for i in range(3)]
    y_ideal = [ym - 0.023 for ym in y_main]

    # dashed rule spanning the table, just above the bold model-name row (echoes
    # the vertical divider between the two property blocks)
    x_l = min(p.x0 for p in poss)
    x_r = max(p.x1 for p in poss)
    y_rule = y_hdr + 0.030
    fig.add_artist(Line2D([x_l, x_r], [y_rule, y_rule],
                          transform=fig.transFigure, color="0.55", lw=1.1,
                          ls=(0, (6, 5)), zorder=5))

    # bold model name over each value column (columns align under panels a–d)
    for j, (model, _short) in enumerate(ORDER):
        fig.text(c[j], y_hdr, MODEL_DISP[model], ha="center", va="center",
                 fontsize=TITLE_FS, color="0.05", fontweight="bold")

    metrics = [("log–log Pearson $r$", "ideal 1.00", "r"),
               ("MdAPD",              "ideal 0%",   "mdapd"),
               ("log–log fit slope",  "ideal 1.00", "slope")]

    def fmt(key, v):
        if v is None:
            return "—"
        return f"{v:.0f}%" if key == "mdapd" else f"{v:.2f}"

    for row, (name, ideal, key) in enumerate(metrics):
        ym = y_main[row]
        fig.text(x_lab_r, ym, name, ha="right", va="center", fontsize=14.0,
                 color="0.1", fontweight="normal")
        fig.text(x_lab_r, y_ideal[row], ideal, ha="right", va="center",
                 fontsize=11.5, color="0.45")
        vals = [(panel_stats.get(k) or {}).get(key) for k in ORDER]
        for j in range(len(ORDER)):
            fig.text(c[j], ym, fmt(key, vals[j]), ha="center", va="center",
                     fontsize=14.0, color="0.05", fontweight="normal")

    # (top, bottom) of the strip: top sits just above the first metric row, bottom
    # below the last row's "ideal" sublabel.
    # top of the strip is the horizontal rule itself, so the vertical divider drawn
    # later (spanning tbl_bottom..tbl_top) reaches up and meets it
    return y_rule, y_main[-1] - 0.035


rect = [0.0, 0.46, 1.0, 0.95] if SHOW_STATS_TABLE else [0.0, 0.235, 1.0, 0.91]
fig.tight_layout(rect=rect)

# Recenter the block of 4 equal-aspect panels horizontally: they are square and
# don't fill the figure width, so tight_layout leaves them off-centre relative to
# the full-width legend below. Shift every axis by the same delta so the block is
# symmetric in the figure (and thus in the bbox-tight png). In table mode also pin
# the panel tops so there is predictable room for the table + legend beneath.
fig.canvas.draw()
poss = [ax.get_position() for ax in axes]
shift = (1.0 - (min(p.x0 for p in poss) + max(p.x1 for p in poss))) / 2.0
vshift = (0.92 - max(p.y1 for p in poss)) if SHOW_STATS_TABLE else 0.0
for ax in axes:
    p = ax.get_position()
    ax.set_position([p.x0 + shift, p.y0 + vshift, p.width, p.height])

# spanning header over each model pair, placed from the final axes geometry
hs = 5.25 / FIG_H            # keep the header gap ~constant in inches across figure heights
for (i, j), short in (((0, 1), "cond"), ((2, 3), "visc")):
    b0 = axes[i].get_position(); b1 = axes[j].get_position()
    xc = 0.5 * (b0.x0 + b1.x1)
    ytop = max(b0.y1, b1.y1)
    fig.text(xc, ytop + 0.108 * hs, title_of[short], ha="center", va="bottom",
             fontsize=HDR_FS, fontweight="bold")
    fig.lines.append(plt.Line2D([b0.x0, b1.x1], [ytop + 0.096 * hs] * 2,
                                transform=fig.transFigure, color="0.55", lw=1.0))

# Centre the legend on the panel block. Earlier versions forced it to span exactly
# the block width with mode="expand"; because the entries' natural width exceeds
# that span, matplotlib solved it with *negative* column gaps and labels collided
# with the next column's marker. Letting the legend size itself and centring it on
# the same x as the panels keeps the columns apart and still leaves the saved
# (bbox-tight) png symmetric.
poss_final = [ax.get_position() for ax in axes]
leg_x0 = min(p.x0 for p in poss_final)
leg_x1 = max(p.x1 for p in poss_final)
leg_xc = 0.5 * (leg_x0 + leg_x1)

if SHOW_STATS_TABLE:
    tbl_top, tbl_bottom = draw_stats_table(fig, axes, panel_stats)
    leg_y = tbl_bottom - 0.012
    # key to the stats table, as the legend title (same as figure 4)
    leg_title = "MdAPD = Median Absolute Percentage Deviation"
else:
    leg_y = 0.175
    leg_title = ("log–log Pearson $r = \\mathrm{cov}(X,Y)/(\\sigma_X\\,\\sigma_Y)$,  "
                 "$X=\\log_{10}$exp,  $Y=\\log_{10}$sim\n"
                 "MdAPD = Median Absolute Percentage Deviation "
                 "$= \\mathrm{median}(|\\mathrm{sim}-\\mathrm{exp}|/\\mathrm{exp})$\n"
                 "log–log fit slope = $m$ from least-squares $Y=mX+b$  (ideal 1)")

leg = fig.legend(handles, labels, loc="upper center",
                 bbox_to_anchor=(leg_xc, leg_y),
                 ncol=LEG_NCOL, fontsize=LEGEND_FS, frameon=False, handlelength=1.9,
                 columnspacing=2.0, handletextpad=0.7, labelspacing=1.0,
                 handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
                 title=leg_title, title_fontsize=LEGEND_FS + 0.5)
if not SHOW_STATS_TABLE:
    leg.get_title().set_color("0.25")

# Force the four system columns to the same width. Column widths are otherwise set
# by their own longest entry, which leaves visibly ragged gaps between columns. The
# short concentration-key column (last) keeps its natural width. Widths live in
# display pixels, hence the figure is created at the save dpi.
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_cols = leg._legend_handle_box.get_children()[:-1]
_wmax = max(_k.get_bbox(_rend).width for _c in _cols for _k in _c.get_children())
for c in _cols:
    c.align = "left"        # 'baseline' (the legend default) ignores an explicit width
    c.set_width(_wmax)

# ── dashed divider between the sigma pair (a,b) and the eta pair (c,d) ────────
# Spans the stats table only, centred between its b and c value columns (which sit
# under the panel centres), so the numbers read as two property blocks.
if SHOW_STATS_TABLE:
    _pc = [(p.x0 + p.x1) / 2 for p in poss_final]
    x_div = 0.5 * (_pc[1] + _pc[2])
    # run the divider all the way up through the b)/c) panel gap to the header-rule
    # level, so it fully separates the conductivity and viscosity blocks
    y_div_top = max(p.y1 for p in poss_final) + 0.096 * hs
    fig.add_artist(Line2D([x_div, x_div], [tbl_bottom, y_div_top],
                          transform=fig.transFigure, color="0.55", lw=1.1,
                          ls=(0, (6, 5)), zorder=5))

p = OUT / "figure3_conductivity_viscosity_parity.png"
# tight bbox, but with a little extra breathing room on the right edge
fig.canvas.draw()
_bb = fig.get_tightbbox(fig.canvas.get_renderer())
_pad = 0.1
_bb = Bbox.from_extents(_bb.x0 - _pad, _bb.y0 - _pad, _bb.x1 + _pad + 0.45, _bb.y1 + _pad)
fig.savefig(p, dpi=FIG_DPI, bbox_inches=_bb)
print("saved", p)
plt.close(fig)
