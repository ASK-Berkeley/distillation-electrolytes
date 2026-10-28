"""Figure (structure + density + stability): a single 1x3 figure for the
distilled PaiNN models vs the UMA teacher and experiment, 298 K.

  Panel a : RDF overlay for one salt/solvent system (default NaOTf in TEGDME).
            PaiNN is the **4-replica mean** (solid, coloured by pair) with a
            **+/-1 s.d. band** (lighter shade); the UMA reference is **black**
            (dashed).
  Panel b : mass-density parity vs experiment -- UMA-OMol (black) and the
            PaiNN student (replica 0, single accent colour), all systems
            pooled; the R^2 annotations carry the per-model story, not
            per-system markers.
  Panel c : stability vs simulation time -- force MAE and stress MAE of the
            running PaiNN trajectory against UMA labels recomputed every 20 ps
            along one 20 ns trajectory.

RDFs come from 293 K, ~1 ns trajectories: UMA from
``data/uma_static_observables_1ns_NPT/rdfs_293K/<system>_<conc>/rdf.csv`` (a
single trajectory, ``compute_uma_1M_rdfs.py``), PaiNN from
``data/painn_static_observables_NPT/rdfs_293K/<system>_<conc>/rdf.csv`` -- which
is the *mean over 5 replicas*, written by ``compute_painn_replica_rdfs.py``.

That file also carries ``sd_<pair>`` columns (std across replicas, ddof=1) which
this figure does not yet use: both models are still drawn as plain lines (UMA
dashed black, PaiNN solid coloured). TODO: shade PaiNN +/-1 sigma from ``sd_``.
The per-replica curves behind the mean are in ``<system>_<conc>/replicas/``. PaiNN densities come from
``data/painn_static_observables_NPT/density_master.csv`` (replica 0 of exp01,
one 20 ns NPT trajectory per system); UMA + experimental densities from
``data/UMA_298K_1ns/densities_298K_1ns.csv`` (our own uma-s-1p1 1 ns NPT runs;
the earlier ``uma_static_observables_1ns_NPT/densities_298K_from_nitesh_paper.csv``
holds the previous, superseded UMA densities). Stability comes
from a per-frame ``uma_labels_every20ps.npz`` (uma_/ref_ forces and stress),
read from the trajectory directory on the machine that produced it and
otherwise from ``data/painn_static_observables_NPT/uma_error_for_fig2/``.

    python paper/paper_figures/main_paper/figure_2/plot_figure_rdf_density.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.analysis.observables import paths

REPO = Path(__file__).resolve().parents[4]
UMA_STATIC_DIR = paths.OUT_UMA_STATIC
PAINN_STATIC_DIR = paths.OUT_PAINN_STATIC
UMA_RDF_DIR = UMA_STATIC_DIR / "rdfs_293K"
PAINN_RDF_DIR = PAINN_STATIC_DIR / "rdfs_293K"
DENS_CSV = paths.OUT_UMA_298K_DENSITY / "densities_298K_1ns.csv"
DENS_MASTER = PAINN_STATIC_DIR / "density_master.csv"  # per-replica; we use replica 0
OUT_DIR = paths.FIGURES_MAIN

SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}
SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}
SALT_NAME = {"napf6": "NaPF6", "naotf": "NaOTf", "lipf6": "LiPF6"}
SOLV_NAME = {"dme": "DME", "pc": "PC", "diglyme": "DEGDME", "tgdme": "TEGDME"}

# ── type scale ────────────────────────────────────────────────────────────────
# One body size for tick labels, legends and annotation boxes (BASE_FS); axis
# labels step well up to LABEL_FS so they stay readable when the figure is
# printed at column width -- they are the first thing a reader needs off the
# axes, so they outrank the panel titles in size.
BASE_FS = 14.0       # ticks, legends, annotation boxes
LABEL_FS = 18.5      # axis labels
TITLE_FS = 16.5      # subplot titles (mini-panel + column headers)
CAPTION_FS = 23.0    # descriptive caption under each column
SUPTITLE_FS = CAPTION_FS  # figure title -- same size as the column captions, bold
LETTER_FS = CAPTION_FS    # panel letters a)/b)/c) -- matched to their caption

# Subplot-title toggles. The figure title and the a/b/c panel letters are not
# affected by either. Turning COLUMN titles off reclaims their strip for the
# axes; the MINI titles are the only thing naming panel a's four systems, so
# they stay on by default.
SHOW_COLUMN_TITLES = False   # "Na+ RDFs (1 ns)" / "Mass density" / "Stability ..."
SHOW_MINI_TITLES = True      # "NaPF6 / DME" etc. above each panel-a mini-panel
SHOW_CAPTIONS = True         # descriptive sentence under each column

# One descriptive caption per column, drawn *below* the axes so each panel says
# what it shows without the reader going to the figure caption. At CAPTION_FS
# only ~23 rendered characters fit across a column, so these are hard-wrapped
# to short lines and say only what the encoding doesn't already show. Panel a
# runs to three lines; the caption strip below is sized for that.
CAPTIONS = {
    "a": "Na$^+$ RDFs, 1 ns: student\n"
         "(5-replica mean) vs teacher",
    "b": "Mass density vs\n"
         "experiment",
    "c": "Student forces and stresses\n"
         "evaluated against teacher\n"
         "periodically",
}

# RDF pair colour keyed by the partner element (the cation is always Na).
PAIR_COLOR = {"O": "#C44E52", "F": "#55A868", "P": "#4C72B0", "S": "#8172B3", "N": "#937860"}

# panel a: the y-limit of each mini-panel is set so the tallest peak in it
# reaches this fraction of the panel height -- the remaining headroom is what
# the upper-right legend sits in. The first peak is always well left of the
# legend, so this can run tight without the two colliding.
RDF_PEAK_FRAC = 0.80
RDF_LW_PAINN = 1.77   # PaiNN (solid, coloured)
RDF_LW_UMA = 1.14     # UMA reference (black dashed)
RDF_BAND_ALPHA = 0.45  # PaiNN +/-1 sigma across replicas; matches the appendix figure

# panel b: one colour per model (no per-system encoding). Blue teacher / warm
# orange student -- coordinated with the RDF palette and colour-blind safe.
UMA_COLOR = "#4C72B0"    # teacher -- muted blue
PAINN_COLOR = "#DD8452"  # student -- warm orange

# the four systems shown as mini-panels inside panel a.
DEFAULT_RDF_SYSTEMS = ["napf6_dme", "naotf_pc", "napf6_diglyme", "naotf_tgdme"]
# system used for the stability-panel label.
DEFAULT_STABILITY_SYSTEM = "naotf_tgdme"

# per-frame UMA-vs-PaiNN labels for the stability panel (20 ns @ 20 ps).
# The npz is written next to the trajectory it came from, which only exists on
# the machine that ran it (perlmutter); elsewhere we read the copy mirrored
# into the observables tree.
STABILITY_NPZ_SOURCE = (paths.DATA_DIR / "experiments" / "exp01_mixed_noise_all_systems" /
                        "naotf_tgdme" / "npt_1M_298K_20ns_1ps_replicas" / "replica_0" /
                        "uma_labels_every20ps.npz")
STABILITY_NPZ_MIRROR = (paths.OBS_ROOT / "painn_static_observables_NPT" / "uma_error_for_fig2" /
                        "naotf_tgdme" / "uma_labels_every20ps.npz")


def default_stability_npz():
    """The trajectory-local npz when it's reachable, else the mirrored copy.

    At 85 MB this file is too large to track, so it is not in ``results/``: it
    comes with the deposited data rather than with a clone. Unpack it to
    STABILITY_NPZ_MIRROR, or pass ``--stability-npz``."""
    if STABILITY_NPZ_SOURCE.is_file():
        return STABILITY_NPZ_SOURCE
    if not STABILITY_NPZ_MIRROR.is_file():
        raise SystemExit(
            f"stability panel input not found:\n  {STABILITY_NPZ_MIRROR}\n"
            f"This 85 MB npz ships with the deposited data, not the repository. "
            f"Unpack it there or pass --stability-npz.")
    return STABILITY_NPZ_MIRROR


EV_A3_TO_BAR = 1602176.6269  # 1 eV/Å³ in bar

SAMPLE_RE = re.compile(r"^([\d.]+)\s*M\s+(\S+)\s+in\s+(\S+)$")


# ── RDF loading ────────────────────────────────────────────────────────────────
def _pairs_from_cols(cols):
    return [c[2:] for c in cols if c.startswith("g_")]


def _load_rdf(base_dir, system_id):
    """``(r, {pair: g}, {pair: sd})`` for one system, from
    ``base_dir/<system_id>_<conc>/rdf.csv`` (conc label varies: e.g. ``1M``,
    ``0.5M`` -- glob for whichever concentration this system has).

    ``g_<pair>`` is the curve to draw; ``sd_<pair>``, where present, is the std
    across replicas. The spread deliberately does NOT use a ``g_`` prefix --
    ``_pairs_from_cols`` would take it for another pair and plot it as a curve.
    ``sd`` is empty for a single-trajectory file (UMA), so callers draw no band.
    """
    matches = sorted(base_dir.glob(f"{system_id}_*/rdf.csv"))
    if not matches:
        raise FileNotFoundError(f"No rdf.csv for {system_id!r} under {base_dir}")
    df = pd.read_csv(matches[0])
    pairs = _pairs_from_cols(df.columns)
    g = {p: df[f"g_{p}"].values for p in pairs}
    # NaN when n=1 (e.g. lipf6_dme) -- treated as "no band", same as absent.
    sd = {p: df[f"sd_{p}"].values for p in pairs
          if f"sd_{p}" in df.columns and not df[f"sd_{p}"].isna().all()}
    return df["r"].values, g, sd


def load_uma_rdf(system_id):
    """UMA reference: a single 293 K trajectory, so no spread."""
    return _load_rdf(UMA_RDF_DIR, system_id)


def load_painn_rdf(system_id):
    """PaiNN: the mean over 5 replicas at 293 K, with the across-replica std,
    sampled/truncated with the same params as the UMA reference so the two are
    directly comparable. Written by ``compute_painn_replica_rdfs.py``."""
    return _load_rdf(PAINN_RDF_DIR, system_id)


def draw_rdf_panel(ax, system_id, r_max_plot=8.0, show_ylabel=False):
    salt, solv = system_id.split("_")[0], system_id.split("_")[1]
    r_u, g_u, _ = load_uma_rdf(system_id)
    r_p, g_p, sd_p = load_painn_rdf(system_id)

    handles = []
    for pair in g_p:
        partner = pair.split("-")[1]
        c = PAIR_COLOR.get(partner, "0.4")
        if pair in sd_p:
            ax.fill_between(r_p, g_p[pair] - sd_p[pair], g_p[pair] + sd_p[pair],
                            color=c, alpha=RDF_BAND_ALPHA, lw=0, zorder=2)
        ax.plot(r_p, g_p[pair], color=c, lw=2.2, zorder=3)
        ax.plot(r_u, g_u[pair], color="black", lw=1.32, ls="--", zorder=4)
        handles.append(Line2D([], [], color=c, lw=2.2, label=pair))
    handles.append(Line2D([], [], color="black", lw=1.32, ls="--", label="teacher"))

    ax.axhline(1.0, color="0.7", lw=0.7, ls=":", zorder=0)
    ax.set_xlim(1.2, r_max_plot)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$r$  (Å)")
    if show_ylabel:
        ax.set_ylabel(r"$g(r)$")
    ax.set_title(f"{SALT_LABEL[salt]} in {SOLV_LABEL[solv]}  (1 M)", fontsize=TITLE_FS,
                 fontweight="bold", pad=8)
    ax.legend(handles=handles, loc="upper right", fontsize=BASE_FS, framealpha=0.9,
              handlelength=1.6, labelspacing=0.3)
    ax.grid(True, ls=":", lw=0.6, alpha=0.45)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def draw_rdf_mini(ax, system_id, show_xlabel=False, show_ylabel=False):
    """Compact RDF mini-panel (one of the 2x2 inside panel a): PaiNN
    (coloured by pair, solid) over the black-dashed UMA reference -- both
    single 293 K trajectories."""
    salt, solv = system_id.split("_")[0], system_id.split("_")[1]
    r_u, g_u, _ = load_uma_rdf(system_id)
    r_p, g_p, sd_p = load_painn_rdf(system_id)

    handles = []
    for pair in g_p:
        c = PAIR_COLOR.get(pair.split("-")[1], "0.4")
        if pair in sd_p:
            ax.fill_between(r_p, g_p[pair] - sd_p[pair], g_p[pair] + sd_p[pair],
                            color=c, alpha=RDF_BAND_ALPHA, lw=0, zorder=2)
        ax.plot(r_p, g_p[pair], color=c, lw=RDF_LW_PAINN, zorder=3)
        ax.plot(r_u, g_u[pair], color="black", lw=RDF_LW_UMA, ls="--", zorder=4)
        handles.append(Line2D([], [], color=c, lw=RDF_LW_PAINN, label=pair))

    ax.axhline(1.0, color="0.7", lw=0.6, ls=":", zorder=0)
    ax.set_xlim(1.2, 8.0)
    # scale so the tallest peak fills RDF_PEAK_FRAC of the panel; the leftover
    # headroom is the legend's.
    # Include the top of the PaiNN band, not just the mean, so shading is not clipped.
    g_hi = {p: g_p[p] + sd_p[p] if p in sd_p else g_p[p] for p in g_p}
    gmax = max(float(np.nanmax(g[(r >= 1.2) & (r <= 8.0)]))
               for r, gs in ((r_p, g_hi), (r_u, g_u)) for g in gs.values())
    ax.set_ylim(0.0, gmax / RDF_PEAK_FRAC)
    # system label as a title (not an in-axes box) so the enlarged legend owns
    # the upper-right corner without colliding with it.
    if SHOW_MINI_TITLES:
        ax.set_title(f"{SALT_LABEL[salt]} / {SOLV_LABEL[solv]}", fontsize=TITLE_FS,
                     fontweight="bold", pad=4)
    ax.legend(handles=handles, loc="upper right", fontsize=BASE_FS, framealpha=0.85,
              handlelength=1.1, labelspacing=0.22, borderpad=0.3, handletextpad=0.45)
    ax.grid(True, ls=":", lw=0.5, alpha=0.4)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=BASE_FS)
    if show_xlabel:
        ax.set_xlabel(r"$r$ (Å)", fontsize=LABEL_FS)
    if show_ylabel:
        ax.set_ylabel(r"$g(r)$", fontsize=LABEL_FS)


# ── density parity ─────────────────────────────────────────────────────────────
def load_density_lookup(path):
    """(salt, solvent, conc) -> {'uma':, 'exp':} from the UMA densities.csv."""
    df = pd.read_csv(path)
    lookup = {}
    for _, r in df.iterrows():
        m = SAMPLE_RE.match(str(r["System"]).strip())
        if not m:
            continue  # pure-solvent rows
        conc, salt, solv = float(m.group(1)), m.group(2), m.group(3)
        lookup[(salt, solv, conc)] = {"uma": float(r["UMA-OMol"]), "exp": float(r["Expt"])}
    return lookup


def draw_density_parity(ax, painn, lookup, title="Mass density  (1 ns)"):
    """Density parity vs experiment, all systems pooled -- no per-system
    encoding, just UMA (red) vs PaiNN (green), equal-size markers with enough
    transparency that near-coincident points blend rather than fully occlude
    one another."""
    xe_l, yp_l, yu_l = [], [], []
    for _, row in painn.iterrows():
        salt_tok, solv_tok = row.system_id.split("_")[0], row.system_id.split("_")[1]
        key = (SALT_NAME[salt_tok], SOLV_NAME[solv_tok], float(row.concentration_M))
        rec = lookup.get(key)
        if rec is None or not np.isfinite(rec["exp"]):
            continue
        xe, yp, yu = rec["exp"], row.density_mean_g_cm3, rec["uma"]
        xe_l.append(xe); yp_l.append(yp); yu_l.append(yu)
        # teacher and student share the same experiment x, so a thin vertical
        # tie links the pair -- each such pair is one system + concentration.
        ax.plot([xe, xe], [yu, yp], color="0.62", lw=1.0, alpha=0.75, zorder=2.5)
        ax.scatter(xe, yu, s=150, marker="o", facecolor=UMA_COLOR, edgecolor="black",
                   linewidths=0.7, alpha=0.85, zorder=3)                        # teacher
        ax.scatter(xe, yp, s=150, marker="o", facecolor=PAINN_COLOR, edgecolor="black",
                   linewidths=0.7, alpha=0.85, zorder=3.6)                      # student

    def _r2(y):  # squared Pearson correlation between simulation and experiment
        x = np.asarray(xe_l, float); y = np.asarray(y, float)
        return np.corrcoef(x, y)[0, 1] ** 2

    allv = xe_l + yp_l + yu_l
    lo, hi = min(allv) * 0.97, max(allv) * 1.03
    ax.fill_between([lo, hi], [lo * 0.95, hi * 0.95], [lo * 1.05, hi * 1.05],
                    color="0.8", alpha=0.30, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.45", ls="--", lw=1.3, zorder=1)
    ax.text(0.05, 0.95,
            f"student  $R^2$ = {_r2(yp_l):.3f}\nteacher  $R^2$ = {_r2(yu_l):.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=BASE_FS,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7", alpha=0.85))
    ax.set(xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect("equal")
    ax.set_xlabel(r"Experimental density  ($\mathrm{g\,cm^{-3}}$)")
    ax.set_ylabel(r"Simulation density  ($\mathrm{g\,cm^{-3}}$)")
    if title:
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=8)
    leg = ax.legend(handles=[
        Line2D([], [], marker="o", mfc=UMA_COLOR, mec="black", mew=0.7, color="none",
               ms=12, label="teacher  (1 ns)"),
        Line2D([], [], marker="o", mfc=PAINN_COLOR, mec="black", mew=0.7, color="none",
               ms=12, label="student  (20 ns)"),
    ], loc="lower right", fontsize=BASE_FS, framealpha=0.9,
       title="each pair = one electrolyte", title_fontsize=BASE_FS)
    leg.get_title().set_color("0.35")
    ax.grid(True, ls=":", lw=0.7, alpha=0.5)
    ax.tick_params(which="both", direction="in", top=True, right=True)


# ── stability vs time ────────────────────────────────────────────────────────────
def _rolling_mean(y, w):
    """Centred rolling mean with edge-shrinking window (no NaNs, no phase shift)."""
    if w <= 1:
        return y
    k = np.ones(w)
    num = np.convolve(y, k, mode="same")
    den = np.convolve(np.ones_like(y), k, mode="same")
    return num / den


def draw_stability_panel(ax, npz_path, system_id, title="Stability vs simulation time  (20 ns)"):
    """Force MAE (left) and stress MAE (right) of the live PaiNN trajectory vs
    UMA labels recomputed every 20 ps along one 20 ns run. The point is that
    both stay flat -- no drift -- over the full trajectory."""
    d = np.load(npz_path)
    t_ns = d["time_ps"] / 1000.0
    fmae = np.abs(d["uma_forces"] - d["ref_forces"]).mean(axis=(1, 2)) * 1e3          # meV/Å
    smae = np.abs(d["uma_stress"] - d["ref_stress"]).mean(axis=(1, 2)) * EV_A3_TO_BAR  # bar
    w = max(1, len(t_ns) // 40)  # ~500 ps smoothing window

    cF, cS = "#3969AC", "#E73F74"
    ax.plot(t_ns, fmae, color=cF, lw=0.8, alpha=0.30, zorder=2)
    ax.plot(t_ns, _rolling_mean(fmae, w), color=cF, lw=2.3, zorder=3)
    ax.set_xlabel("Simulation time  (ns)")
    ax.set_ylabel(r"Force MAE vs teacher  (meV Å$^{-1}$)", color=cF)
    ax.tick_params(axis="y", which="both", colors=cF)
    ax.spines["left"].set_color(cF)
    ax.set_xlim(0, t_ns.max())
    ax.set_ylim(0, fmae.max() * 1.35)

    ax2 = ax.twinx()
    ax2.plot(t_ns, smae, color=cS, lw=0.8, alpha=0.30, zorder=2)
    ax2.plot(t_ns, _rolling_mean(smae, w), color=cS, lw=2.3, zorder=3)
    ax2.set_ylabel("Stress MAE vs teacher  (bar)", color=cS)
    ax2.tick_params(axis="y", which="both", colors=cS)
    ax2.spines["right"].set_color(cS)
    ax2.spines["left"].set_color(cF)
    ax2.set_ylim(0, smae.max() * 1.35)

    salt, solv = system_id.split("_")[0], system_id.split("_")[1]
    ax.text(0.04, 0.96,
            f"{SALT_LABEL[salt]} in {SOLV_LABEL[solv]}\n"
            rf"$\langle F\rangle$ = {fmae.mean():.1f} meV Å$^{{-1}}$" "\n"
            rf"$\langle \sigma\rangle$ = {smae.mean():.0f} bar",
            transform=ax.transAxes, ha="left", va="top", fontsize=BASE_FS,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7", alpha=0.9))
    if title:
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=8)
    ax.legend(handles=[
        Line2D([], [], color=cF, lw=2.3, label="Force MAE"),
        Line2D([], [], color=cS, lw=2.3, label="Stress MAE"),
    ], loc="upper right", fontsize=BASE_FS, framealpha=0.9)
    ax.grid(True, ls=":", lw=0.6, alpha=0.4)
    ax.tick_params(axis="x", which="both", direction="in", top=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rdf-systems", default=",".join(DEFAULT_RDF_SYSTEMS),
                    help="Comma list of 4 system_ids for the 2x2 RDF mini-panels (a).")
    ap.add_argument("--stability-npz", type=Path, default=None,
                    help="uma_labels_every20ps.npz for the stability panel (c). "
                         "Defaults to the trajectory-local copy if present, else the mirror.")
    ap.add_argument("--stability-system", default=DEFAULT_STABILITY_SYSTEM,
                    help="system_id used to label the stability panel (c).")
    ap.add_argument("--density-csv", type=Path, default=DENS_CSV,
                    help="UMA-OMol + experimental densities for panel b "
                         "(System,UMA-OMol,Expt). Defaults to our "
                         "UMA_298K_1ns/densities_298K_1ns.csv.")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "figure2_teacher_student_consistency.png")
    args = ap.parse_args(argv)

    stability_npz = args.stability_npz or default_stability_npz()
    rdf_systems = [s.strip() for s in args.rdf_systems.split(",") if s.strip()][:4]
    master = pd.read_csv(DENS_MASTER)
    painn = (master[master["replica"] == 0]
             .rename(columns={"mean_density_g_cm3": "density_mean_g_cm3"}))
    lookup = load_density_lookup(args.density_csv)

    plt.rcParams.update({
        "font.size": BASE_FS, "axes.titlesize": TITLE_FS, "axes.labelsize": LABEL_FS,
        "axes.linewidth": 0.9, "xtick.labelsize": BASE_FS, "ytick.labelsize": BASE_FS,
        "legend.fontsize": BASE_FS,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    # The captions need a strip of their own at the bottom; the figure grows by
    # exactly that strip so the axes keep their previous height rather than
    # being squeezed (all the fractions below are re-derived from FIG_H, so the
    # top/bottom margins stay the same *in inches* either way).
    FIG_H = 7.40 if SHOW_CAPTIONS else 5.78
    fig = plt.figure(figsize=(17.0, FIG_H))

    def _frac(inches_from_top):
        return 1.0 - inches_from_top / FIG_H

    # Shared vertical anatomy: all three panels use the same box (no external
    # legend below b anymore, so no need to stretch a/c to match one).
    # With the column titles hidden, the axes take over their strip -- but stop
    # short of the figure title so the letter row isn't crowded against it.
    # Everything below the suptitle sits 0.18 in lower than the axes-height
    # arithmetic strictly needs, which is the breathing room under the title;
    # FIG_H absorbs the same 0.18 in so the axes keep their height.
    TOP = _frac(1.30 if SHOW_COLUMN_TITLES else 1.14)
    BOT = (2.35 if SHOW_CAPTIONS else 0.73) / FIG_H   # ticks + xlabel (+ 3-line caption)
    TITLE_Y = _frac(1.09)                       # shared column-title baseline
    LETTER_Y = _frac(0.82 if SHOW_COLUMN_TITLES else 0.98)  # panel-letter baseline
    SUPTITLE_Y = _frac(0.14)
    CAPTION_Y = 1.51 / FIG_H                    # top of the caption block
    COLS = {"a": (0.072, 0.339), "b": (0.377, 0.644), "c": (0.682, 0.949)}  # equal-width (x0, x1)

    # a : 2x2 RDF mini-panels filling the shared box
    x0a, x1a = COLS["a"]
    gs_a = fig.add_gridspec(2, 2, left=x0a, right=x1a, top=TOP, bottom=BOT,
                            wspace=0.30, hspace=0.42)
    mini_axes = []
    for k, sysid in enumerate(rdf_systems):
        rr, cc = divmod(k, 2)
        axm = fig.add_subplot(gs_a[rr, cc])
        draw_rdf_mini(axm, sysid, show_xlabel=(rr == 1), show_ylabel=(cc == 0))
        mini_axes.append(axm)

    # b : density parity (square, centred in its box)
    x0b, x1b = COLS["b"]
    ax_b = fig.add_axes([x0b, BOT, x1b - x0b, TOP - BOT])
    draw_density_parity(ax_b, painn, lookup, title=None)

    # c : stability vs time
    x0c, x1c = COLS["c"]
    ax_c = fig.add_axes([x0c, BOT, x1c - x0c, TOP - BOT])
    draw_stability_panel(ax_c, stability_npz, args.stability_system, title=None)

    # shared titles (one baseline) and panel letters (one baseline)
    titles = {"a": r"Na$^+$ RDFs  (1 ns)", "b": "Mass density  (9 systems)",
              "c": "Stability vs simulation time  (20 ns)"}
    for key, (x0, x1) in COLS.items():
        if SHOW_COLUMN_TITLES:
            fig.text((x0 + x1) / 2, TITLE_Y, titles[key], ha="center", va="bottom",
                     fontsize=TITLE_FS, fontweight="bold")
        fig.text(x0 - 0.012, LETTER_Y, f"{key})", ha="left", va="bottom",
                 fontsize=LETTER_FS, fontweight="bold")
        if SHOW_CAPTIONS:
            fig.text((x0 + x1) / 2, CAPTION_Y, CAPTIONS[key], ha="center", va="top",
                     fontsize=CAPTION_FS, color="0.15", linespacing=1.30)

    fig.suptitle("Liquid structure, density and stability: distilled student vs teacher vs experiment",
                 fontsize=SUPTITLE_FS, fontweight="bold", y=SUPTITLE_Y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"saved -> {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
