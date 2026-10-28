"""Appendix figure: replica-averaged Na+ RDFs for all 8 salt/solvent systems
(2 salts x 4 solvents, 1 M, 293 K), distilled PaiNN vs the UMA teacher.

PaiNN is the mean over every surviving replica -- r0 plus the 1 ns reruns r1-r4,
which differ from it only in FeNNol's ``random_seed`` -- with a +/-1 sigma band
across replicas. (This supersedes an earlier single-trajectory version of this
figure that drew r0 alone with no band; the two were merged, since the replica
mean is strictly more informative and r0 is one of the replicas averaged here.)

Five of the original reruns evaporated (box expanded to ~450 Å, density collapsed
to ~4e-4 g/cm^3); those were rerun from a different seed and the healthy reruns
supersede them upstream, so all 8 systems are now n=5.

Those 5 (naotf_pc r2, napf6_pc r1-r4) do NOT share ``start.xyz`` with the rest of
their set -- they start from the 100 ps frame of the UMA 293 K / 1 M trajectory
(the same run this figure compares against) rather than the cubic packing box r0
uses. That box is UMA's own frame 0 straight out of Packmol at rho=1.205, and the
first ~100 ps of collapsing it to ~1.38 is what killed the PC systems; UMA's
100 ps frame is past the collapse but still at the initial Na-F pairing state
(CN 4.118, same as t=0), so it skips the transient without switching basins.

Panels containing those replicas are marked "†" so the mixed initial condition is
visible in the figure, not just in the metadata. It matters most for napf6_pc,
where 4 of 5 replicas come from the new seed. Both seeds are the same state point
(1 M, 293 K, 1 atm), the same lineage, and converge to the same density, so for an
equilibrium g(r) sampled after a 30 % equilibration drop this is defensible -- but
the band there mixes seed-to-seed and start-to-start variation.

CAVEAT ON WHAT THIS FIGURE SHOWS: UMA does not itself reach equilibrium within
1 ns for the PC systems (Na-F CN falls 4.1 -> 1.7 and is still dropping at the
end). The claim supported here is that PaiNN reproduces UMA, not that either is
the equilibrium liquid structure.

Input is the per-replica cache written by
``compute_painn_replica_rdfs.py`` (run that first):
    data/painn_static_observables_NPT/rdfs_293K/<system>_<conc>/replicas/rdf_r<N>.csv
(the replica mean of those same curves is the <system>_<conc>/rdf.csv one level up,
which figure 2 and the non-replica appendix figure read)
UMA reference is unchanged, from
    data/uma_static_observables_1ns_NPT/rdfs_293K/<system>_1M/rdf.csv

    python plot_figure_rdf_all_systems.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.analysis.observables import paths

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
# The figure is written into the paper submodule; figure_rdf_all_systems.png in
# this directory is a symlink pointing back at it.
OUT_DIR = paths.FIGURES_APPENDIX
UMA_RDF_DIR = paths.OUT_UMA_293K_RDF
PAINN_RDF_DIR = paths.OUT_PAINN_293K_RDF


def _painn_replica_dir(system_id):
    """``<system>_<conc>/replicas/`` -- the per-replica curves behind the mean in
    ``<system>_<conc>/rdf.csv``. Glob the concentration, which varies by system."""
    hits = sorted(PAINN_RDF_DIR.glob(f"{system_id}_*/replicas"))
    if not hits:
        raise FileNotFoundError(
            f"No {system_id}_*/replicas under {PAINN_RDF_DIR}. Run:\n"
            f"  python -m src.analysis.observables.structural"
            f".compute_painn_replica_rdfs")
    return hits[0]

SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf"}
SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}
SALTS = ["napf6", "naotf"]
SOLVENTS = ["dme", "diglyme", "pc", "tgdme"]
ALL_SYSTEMS = [f"{salt}_{solv}" for salt in SALTS for solv in SOLVENTS]

# RDF pair colour keyed by the partner element (the cation is always Na).
PAIR_COLOR = {"O": "#C44E52", "F": "#55A868", "P": "#4C72B0", "S": "#8172B3", "N": "#937860"}

# ── type scale + panel style (matched to figure 2's panel a) ────────────────────
BASE_FS = 14.0        # panel legends
TICK_FS = 16.8        # axis tick numbers (20% larger than BASE_FS)
LABEL_FS = 18.5       # axis labels
TITLE_FS = 16.5       # per-panel "Salt / Solvent" titles
RDF_PEAK_FRAC = 0.80  # tallest peak fills this fraction of the panel; rest is legend headroom
RDF_LW_PAINN = 1.77   # PaiNN replica mean (solid, coloured)
RDF_LW_UMA = 1.14     # UMA reference (black dashed)
RDF_BAND_ALPHA = 0.45  # PaiNN +/-1 sigma across replicas


def _pairs_from_cols(cols):
    return [c[2:] for c in cols if c.startswith("g_")]


def _load_uma(system_id):
    """(r, {pair: g}) for the single-trajectory UMA reference."""
    matches = sorted(UMA_RDF_DIR.glob(f"{system_id}_*/rdf.csv"))
    if not matches:
        raise FileNotFoundError(f"No UMA rdf.csv for {system_id!r} under {UMA_RDF_DIR}")
    df = pd.read_csv(matches[0])
    return df["r"].values, {p: df[f"g_{p}"].values for p in _pairs_from_cols(df.columns)}


def _load_painn_replicas(system_id, band="std"):
    """(r, {pair: (mean, spread)}, reps) over every cached replica of one system.

    Replicas are truncated to their shortest shared r grid before averaging (each
    run's r_max is capped at half its own smallest box width, so a denser replica
    can stop earlier).

    ``spread`` is computed per r bin across the n replica curves, with ``ddof=1``
    -- the replicas are a *sample* from the distribution of 1 ns runs (one per
    seed), not the whole population, so the variance needs Bessel's correction.

    ``band='std'`` gives the run-to-run scatter: how much a single 1 ns RDF
    varies. ``band='sem'`` divides by sqrt(n) to give the precision of the mean.
    std is the default because the UMA curve plotted against it is a *single*
    trajectory with no band -- the uncertainty on the PaiNN-UMA difference is
    sqrt(sigma^2/n + sigma_UMA^2), which the lone UMA run dominates, so a
    sqrt(n)-narrower band would overstate how well the comparison is resolved.

    Both are all-zero when only one replica survived (the caller draws no band).
    """
    sys_dir = _painn_replica_dir(system_id)
    files = sorted(sys_dir.glob("rdf_r*.csv"),
                   key=lambda p: int(re.search(r"rdf_r(\d+)\.csv", p.name).group(1)))
    if not files:
        raise FileNotFoundError(
            f"No rdf_r*.csv for {system_id!r} under {sys_dir}. Run:\n"
            f"  python -m src.analysis.observables.structural"
            f".compute_painn_replica_rdfs")

    reps = [int(re.search(r"rdf_r(\d+)\.csv", f.name).group(1)) for f in files]
    dfs = [pd.read_csv(f) for f in files]

    # Which replicas came from the UMA-100 ps reruns (different start.xyz). Recorded
    # by compute_painn_replica_rdfs.py; absent for older caches, hence the get().
    mixed = []
    for f, rep in zip(files, reps):
        meta = f.with_name(f"rdf_r{rep}_summary.json")
        if meta.exists():
            prov = json.loads(meta.read_text()).get("provenance", {})
            if not prov.get("shares_start_xyz_with_r0", True):
                mixed.append(rep)
    n = min(len(df) for df in dfs)
    r = dfs[0]["r"].values[:n]
    for df in dfs[1:]:
        if not np.allclose(df["r"].values[:n], r):
            raise ValueError(f"{system_id}: replica r grids disagree")

    pairs = _pairs_from_cols(dfs[0].columns)
    n_rep = len(dfs)
    out = {}
    for p in pairs:
        stack = np.vstack([df[f"g_{p}"].values[:n] for df in dfs])
        mean = stack.mean(axis=0)
        if n_rep > 1:
            spread = stack.std(axis=0, ddof=1)   # ddof=1 is undefined for n=1
            if band == "sem":
                spread = spread / np.sqrt(n_rep)
        else:
            spread = np.zeros_like(mean)
        out[p] = (mean, spread)
    return r, out, reps, mixed


def draw_rdf_panel(ax, system_id, band="std", show_xlabel=False, show_ylabel=False):
    """Compact RDF panel styled like figure 2's panel a: replica-mean PaiNN
    (coloured by pair, solid) with a +/-1 sigma band, over the black-dashed
    single-trajectory UMA reference. The system is named in a bold title above
    the panel and the y-limit reserves headroom for the upper-right legend."""
    salt, solv = system_id.split("_")[0], system_id.split("_")[1]
    r_u, g_u = _load_uma(system_id)
    r_p, g_p, reps, mixed = _load_painn_replicas(system_id, band)

    handles = []
    for pair, (mean, std) in g_p.items():
        c = PAIR_COLOR.get(pair.split("-")[1], "0.4")
        if len(reps) > 1:
            ax.fill_between(r_p, mean - std, mean + std, color=c,
                            alpha=RDF_BAND_ALPHA, lw=0, zorder=2)
        ax.plot(r_p, mean, color=c, lw=RDF_LW_PAINN, zorder=3)
        ax.plot(r_u, g_u[pair], color="black", lw=RDF_LW_UMA, ls="--", zorder=4)
        handles.append(Line2D([], [], color=c, lw=RDF_LW_PAINN, label=pair))

    ax.axhline(1.0, color="0.7", lw=0.6, ls=":", zorder=0)
    ax.set_xlim(1.2, 8.0)
    # scale so the tallest peak fills RDF_PEAK_FRAC of the panel; the leftover
    # headroom is the legend's. Include the top of the PaiNN band, not just the
    # mean, so the shading is not clipped.
    g_hi = {p: mean + std for p, (mean, std) in g_p.items()}
    gmax = max(float(np.nanmax(g[(r >= 1.2) & (r <= 8.0)]))
               for r, gs in ((r_p, g_hi), (r_u, g_u)) for g in gs.values())
    ax.set_ylim(0.0, gmax / RDF_PEAK_FRAC)

    ax.set_title(f"{SALT_LABEL[salt]} / {SOLV_LABEL[solv]}", fontsize=TITLE_FS,
                 fontweight="bold", pad=4)
    ax.legend(handles=handles, loc="upper right", fontsize=BASE_FS, framealpha=0.85,
              handlelength=1.1, labelspacing=0.22, borderpad=0.3, handletextpad=0.45)
    ax.grid(True, ls=":", lw=0.5, alpha=0.4)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=TICK_FS)
    if show_xlabel:
        ax.set_xlabel(r"$r$ (Å)", fontsize=LABEL_FS)
    if show_ylabel:
        ax.set_ylabel(r"$g(r)$", fontsize=LABEL_FS)


def report_agreement():
    """Print PaiNN-vs-UMA RMSE for r0 alone vs the replica mean, per system.

    This is the actual question the replicas were run to answer: averaging out
    single-trajectory noise should move the PaiNN curve *towards* UMA if the
    residual disagreement is sampling noise, and leave it put if the disagreement
    is a real force-field difference.
    """
    print(f"\n{'system':<16}{'n':>3}{'RMSE(r0)':>11}{'RMSE(mean)':>12}"
          f"{'change':>10}   per-pair mean sigma")
    print("-" * 78)
    for system_id in ALL_SYSTEMS:
        r_u, g_u = _load_uma(system_id)
        r_p, g_p, reps, _ = _load_painn_replicas(system_id)
        r0 = pd.read_csv(_painn_replica_dir(system_id) / "rdf_r0.csv")

        # Compare on the shared r grid, over the plotted window only.
        n = min(len(r_p), len(r0))
        win = (r_p[:n] >= 1.2) & (r_p[:n] <= 8.0)
        e0, em, sig = [], [], []
        for pair, (mean, std) in g_p.items():
            gu = np.interp(r_p[:n], r_u, g_u[pair])[win]
            e0.append((r0[f"g_{pair}"].values[:n][win] - gu) ** 2)
            em.append((mean[:n][win] - gu) ** 2)
            sig.append(std[:n][win].mean())
        rmse0 = float(np.sqrt(np.concatenate(e0).mean()))
        rmsem = float(np.sqrt(np.concatenate(em).mean()))
        delta = "--" if len(reps) == 1 else f"{100 * (rmsem - rmse0) / rmse0:+.1f}%"
        print(f"{system_id:<16}{len(reps):>3}{rmse0:>11.4f}{rmsem:>12.4f}{delta:>10}   "
              + "  ".join(f"{p}={s:.3f}" for p, s in zip(g_p, sig)))
    print("\nRMSE = root-mean-square g(r) difference vs UMA over 1.2-8.0 Å, all pairs pooled.")
    print("A negative change means replica-averaging moved PaiNN closer to UMA.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "figure_rdf_all_systems.png")
    ap.add_argument("--band", choices=["std", "sem"], default="std",
                    help="Band semantics: 'std' (default) = run-to-run scatter of a "
                         "single 1 ns RDF; 'sem' = std/sqrt(n), the precision of the "
                         "replica mean. See _load_painn_replicas for why std is the "
                         "default against a single-trajectory UMA reference.")
    ap.add_argument("--no-report", action="store_true",
                    help="Skip the printed r0-vs-replica-mean agreement table.")
    args = ap.parse_args(argv)

    plt.rcParams.update({
        "font.size": 11, "axes.linewidth": 0.9,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    # 4 rows (solvents) x 2 cols (salts), portrait, to fill a whole page.
    fig, axes = plt.subplots(len(SOLVENTS), len(SALTS), figsize=(11.0, 15.0),
                             sharex=True)

    for system_id in ALL_SYSTEMS:
        salt, solv = system_id.split("_")[0], system_id.split("_")[1]
        rr, cc = SOLVENTS.index(solv), SALTS.index(salt)
        draw_rdf_panel(axes[rr, cc], system_id, args.band,
                       show_xlabel=(rr == len(SOLVENTS) - 1), show_ylabel=(cc == 0))

    band_label = (r"PaiNN $\pm$1$\sigma$ across replicas" if args.band == "std"
                  else r"PaiNN $\pm$SEM ($\sigma/\sqrt{n}$) of the replica mean")
    fig.legend(handles=[
        Line2D([], [], color="0.3", lw=1.76, label="PaiNN replica mean (solid)"),
        Patch(facecolor="0.3", alpha=0.45, label=band_label),
        Line2D([], [], color="black", lw=1.21, ls="--", label="UMA (dashed)"),
    ], loc="upper center", ncol=3, fontsize=17.0, frameon=False, bbox_to_anchor=(0.5, 1.005))

    fig.suptitle(r"Na$^+$ RDFs, all 8 salt/solvent systems  (1 M, 293 K): "
                 r"replica-averaged PaiNN vs UMA",
                 fontsize=22.0, fontweight="bold", y=1.035)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"saved -> {args.out}")
    plt.close(fig)

    if not args.no_report:
        report_agreement()


if __name__ == "__main__":
    main()
