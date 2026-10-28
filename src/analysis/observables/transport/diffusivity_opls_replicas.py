#!/usr/bin/env python
"""Einstein self-diffusivity (Yeh-Hummer corrected) for the OPLS ``production_replicas`` set.

The OPLS twin of ``diffusivity_replicas.py`` — same estimator, same settings,
different input tree. Every replica's ``prod.xtc`` is streamed through the very
same ``observables.diffusivity.compute_diffusivity`` the PaiNN driver uses (MIC
molecule-COM unwrap in the *true system-COM frame*, FFT time-origin MSD per
species, Einstein fit D = slope/6), and the fit-window machinery is imported
from the PaiNN driver rather than re-implemented, so the two cannot drift apart.

Like the PaiNN driver, the trajectory is streamed ONCE per replica (eq cut 2 ns,
every frame) and BOTH fit methods are applied to the same MSD curve, so every
master row carries two column families in the identical format:
  * ``fixed``   -- fit [1000 ps, 0.5 x window], same window for all species
                   (primary ``D_*`` columns);
  * ``fickian`` -- per-species data-driven beta = dlnMSD/dlntau ≈ 1 window
                   (``D_*_fick_*`` columns, incl. beta + converged flags).

The finite-size (Yeh-Hummer) correction ``dD = kB T xi / (6 pi eta L)`` is applied
TWO ways, exactly as for PaiNN:
  * ``D_inf_exp`` -- eta = experimental shear viscosity;
  * ``D_inf_sim`` -- eta = this box's own OPLS GK viscosity, read from
                     ``<out-root>/viscosity/viscosity_gk_by_box.csv`` produced by
                     ``viscosity_gk_opls_replicas.py`` — run that first.

Two deliberate differences from the PaiNN driver, both forced by the input set:
  * molecule grouping comes from the **GROMACS topology** rather than this
    project's element-sequence patterns — LigParGen orders DME's atoms differently
    from the FeNNol boxes, so the pattern matcher cannot apply. Each residue's
    element multiset is validated against the same species dictionaries, and every
    downstream quantity is invariant to within-molecule atom order (see
    ``opls_runs.groups_from_topology``);
  * a per-box mean +/- s.d. over the 4 replicas is reported alongside the
    per-replica rows,
    and the parity figure uses the per-box means.

Outputs land under ``data/opls_transport_observables/diffusivity/`` (the OPLS run
tree itself is treated as read-only):
  * ``per_run/<box>/r<k>/{diffusivity.csv, msd.csv, diffusivity_msd.png, summary.json}``
  * ``diffusivity_master.csv``   -- one row per replica (64)
  * ``diffusivity_by_box.csv``   -- per-box mean +/- s.d. (16)
  * ``diffusivity_config.yaml``  -- settings + provenance sidecar
  * ``figure_transport_diff_msd.png``    -- MSD + fit grid, one panel per replica
  * ``figure_transport_diff_parity.png`` -- 2x3 parity (rows: YH exp-eta / YH sim-eta;
                                            cols: cation / anion / solvent)

CPU only; conda env ``fennol`` (needs ``mdtraj`` for the ``.xtc`` reader).
~20 s per replica. Run from the repo root as a module:

    python -m src.analysis.observables.transport.diffusivity_opls_replicas --dry-run
    python -m src.analysis.observables.transport.diffusivity_opls_replicas --jobs 8
"""
from __future__ import annotations

import os

# Keep per-process BLAS/FFT single-threaded so the ProcessPool doesn't oversubscribe.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml


from . import opls_runs as O
from ..paths import git_sha
from ..diffusivity import compute_diffusivity
from ..systems import SYSTEMS
from ..units import a2ps_to_m2s, yeh_hummer_correction_a2ps

# The PaiNN driver is the reference for the fit windows and every plotting
# convention here; import it rather than re-implement so the two stay in lockstep.
from . import diffusivity_replicas as PAINN

SETTINGS_FIXED = PAINN.SETTINGS_FIXED
SETTINGS_FICKIAN = PAINN.SETTINGS_FICKIAN
SPECIES = PAINN.SPECIES               # (name, latex label, colour, exp-CSV key)


# ── per-run compute + output ─────────────────────────────────────────────────
def _yh_dD_1e10(eta_cp, box_a, t_k):
    """Yeh-Hummer dD in 1e-10 m^2/s for a viscosity (cP), box edge (A) and T (K).

    Same expression as the PaiNN driver's ``_yh_dD_1e10``; taking T as an argument
    instead of the module-level 298 K is what lets the 273 / 323 K LiPF6 boxes be
    corrected with their own temperature.
    """
    if eta_cp is None or box_a is None:
        return np.nan
    if not (np.isfinite(eta_cp) and eta_cp > 0 and np.isfinite(box_a) and box_a > 0):
        return np.nan
    return a2ps_to_m2s(yeh_hummer_correction_a2ps(t_k, eta_cp, box_a)) * 1e10


def _merge_summary(obs_dir: Path, block: dict) -> None:
    p = obs_dir / "summary.json"
    summary = {}
    if p.is_file():
        try:
            summary = json.loads(p.read_text())
        except Exception:
            summary = {}
    summary["diffusivity"] = block
    p.write_text(json.dumps(summary, indent=2))


def load_sim_viscosity(out_root: Path, suffix: str = "") -> dict:
    """box name -> OPLS GK eta [cP] from ``viscosity_gk_by_box.csv``.
    Empty (with a warning) if the viscosity driver hasn't been run."""
    p = out_root / f"viscosity{suffix}" / "viscosity_gk_by_box.csv"
    if not p.is_file():
        print(f"[warn] no OPLS viscosity master at {p}; sim-eta YH column will be NaN. "
              f"Run viscosity_gk_opls_replicas.py first.", flush=True)
        return {}
    df = pd.read_csv(p)
    return {str(r["box"]): float(r["eta_gk_mean_cP"]) for _, r in df.iterrows()
            if np.isfinite(r["eta_gk_mean_cP"])}


def analyze_one(rec: dict, settings: dict, plots: bool) -> dict:
    """Compute MSD + Einstein D for one replica, write per-run outputs, return a
    light-weight result (scalar row + decimated MSD/fits for the aggregate figure)."""
    traj = rec["traj"]
    spec = SYSTEMS[rec["system_id"]]
    dt = O.xtc_dt_ps(traj)
    n_total = O.xtc_n_frames(traj)
    t_total_ns = n_total * dt / 1000.0
    t_k = rec["temperature_K"]

    dres = compute_diffusivity(
        traj, spec, dt,
        eq_time_ps=settings["eq_time_ps"], tau_min_fit_ps=settings["tau_min_fit_ps"],
        tau_max_fit_ps=settings["tau_max_fit_ps"], tau_max_fit_frac=settings["tau_max_fit_frac"],
        target_frames=settings["target_frames"], method=settings["msd_method"],
        yeh_hummer=False,
        frames_source=O.make_xtc_frame_source(rec["topology"]),
        groups_override=O.groups_from_topology(rec["topology"], spec),
        n_frames=n_total,
    )
    fits, fmeta = PAINN._compute_fits(dres, settings)                 # fixed window
    fick_fits, fick_meta = PAINN._compute_fits(dres, settings["fick"])  # beta~=1 window
    box_a = dres.box_length_a
    dD_sim = _yh_dD_1e10(rec["eta_sim"], box_a, t_k)
    dD_exp = _yh_dD_1e10(rec["eta_exp"], box_a, t_k)

    obs_dir = O.per_run_dir(settings["out_dir"], rec)

    D_pbc = {sp: (None if fits[sp] is None else a2ps_to_m2s(fits[sp][4]) * 1e10)
             for sp, _, _, _ in SPECIES}
    row = {
        "system_id": rec["system_id"], "salt": rec["salt"], "solvent": rec["solvent"],
        "box": rec["box"], "conc_label": rec["conc_label"], "conc_M": rec["conc_M"],
        "temperature_K": t_k, "replica": rec["replica"], "run_name": rec["run_name"],
        "fit_method": "fixed+fickian",
        "t_total_ns": round(t_total_ns, 3), "dt_ps": round(dt, 5), "dt_source": "xtc time",
        "frames_used": dres.frames_used, "effective_dt_ps": round(dres.effective_dt_ps, 5),
        "analysis_window_ps": round(dres.analysis_window_ps, 2),
        "eq_time_ps": settings["eq_time_ps"],
        "msd_method": dres.method,
        "box_L_A": round(box_a, 3),
        "eta_sim_cP": rec["eta_sim"], "eta_exp_cP": rec["eta_exp"],
        "yh_dD_sim_1e10": None if np.isnan(dD_sim) else round(dD_sim, 4),
        "yh_dD_exp_1e10": None if np.isnan(dD_exp) else round(dD_exp, 4),
    }
    all_conv = True
    for sp, _, _, expkey in SPECIES:
        d = D_pbc[sp]
        beta, conv = fmeta[sp]
        fit = fits[sp]
        row[f"D_{sp}_pbc_1e10"] = None if d is None else round(d, 4)
        row[f"D_{sp}_inf_sim_1e10"] = (None if d is None or np.isnan(dD_sim)
                                       else round(d + dD_sim, 4))
        row[f"D_{sp}_inf_exp_1e10"] = (None if d is None or np.isnan(dD_exp)
                                       else round(d + dD_exp, 4))
        row[f"D_{sp}_exp_1e10"] = rec[expkey]
        row[f"D_{sp}_t_lo_ps"] = None if fit is None else round(fit[0], 1)
        row[f"D_{sp}_t_hi_ps"] = None if fit is None else round(fit[1], 1)
        row[f"D_{sp}_beta"] = (None if beta is None or (isinstance(beta, float) and np.isnan(beta))
                               else round(beta, 3))
        row[f"D_{sp}_converged"] = None if conv is None else bool(conv)
        if conv is False:
            all_conv = False
    row["all_converged"] = all_conv

    # Fickian (beta~=1 window) column family, from the SAME MSD curve.
    all_conv_fick = True
    for sp, _, _, _ in SPECIES:
        fit = fick_fits[sp]
        beta, conv = fick_meta[sp]
        d = None if fit is None else a2ps_to_m2s(fit[4]) * 1e10
        row[f"D_{sp}_fick_pbc_1e10"] = None if d is None else round(d, 4)
        row[f"D_{sp}_fick_inf_sim_1e10"] = (None if d is None or np.isnan(dD_sim)
                                            else round(d + dD_sim, 4))
        row[f"D_{sp}_fick_inf_exp_1e10"] = (None if d is None or np.isnan(dD_exp)
                                            else round(d + dD_exp, 4))
        row[f"D_{sp}_fick_t_lo_ps"] = None if fit is None else round(fit[0], 1)
        row[f"D_{sp}_fick_t_hi_ps"] = None if fit is None else round(fit[1], 1)
        row[f"D_{sp}_fick_beta"] = (None if beta is None or (isinstance(beta, float) and np.isnan(beta))
                                    else round(beta, 3))
        row[f"D_{sp}_fick_converged"] = None if conv is None else bool(conv)
        if conv is False:
            all_conv_fick = False
    row["all_converged_fick"] = all_conv_fick

    pd.DataFrame([row]).to_csv(obs_dir / "diffusivity.csv", index=False)
    msd_out = {"tau_ps": dres.tau_ps, "msd_cation": dres.msd_cat}
    msd_out["msd_anion"] = dres.msd_anion if dres.msd_anion is not None else np.nan
    msd_out["msd_solvent"] = dres.msd_solvent if dres.msd_solvent is not None else np.nan
    pd.DataFrame(msd_out).to_csv(obs_dir / "msd.csv", index=False)
    _merge_summary(obs_dir, {**{k: row[k] for k in (
        "fit_method", "t_total_ns", "frames_used", "effective_dt_ps", "analysis_window_ps",
        "eq_time_ps", "msd_method", "box_L_A", "all_converged", "all_converged_fick",
        "eta_sim_cP", "eta_exp_cP", "yh_dD_sim_1e10", "yh_dD_exp_1e10")},
        **{k: row[k] for k in row if k.startswith("D_")},
        "code_version": git_sha(), "generated": settings["generated"]})

    msds = {"cation": dres.msd_cat, "anion": dres.msd_anion, "solvent": dres.msd_solvent}
    if plots:
        fig, ax = plt.subplots(figsize=(5, 4))
        PAINN._plot_msd_panel(ax, dres.tau_ps, msds, fits,
                              f"{rec['run_name']} ({round(t_total_ns)} ns)",
                              fits2=fick_fits)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout(); fig.savefig(obs_dir / "diffusivity_msd.png", dpi=130,
                                        facecolor="white")
        plt.close(fig)

    d2 = O.decimate(len(dres.tau_ps), 2000)
    plot_msds = {sp: (None if m is None else np.asarray(m)[d2]) for sp, m in msds.items()}
    return {"rec": rec, "row": row,
            "plot": {"tau": dres.tau_ps[d2], "msds": plot_msds, "fits": fits,
                     "fits2": fick_fits, "t_total_ns": t_total_ns}}


# ── aggregate figures ────────────────────────────────────────────────────────
def msd_grid(results: list, out: Path):
    """MSD + Einstein-fit grid: one row per box, one column per replica."""
    results = sorted(results, key=lambda r: (r["rec"]["salt"], r["rec"]["solvent"],
                                             r["rec"]["conc_M"], r["rec"]["temperature_K"],
                                             r["rec"]["replica"]))
    boxes = list(dict.fromkeys(r["rec"]["box"] for r in results))
    reps = sorted({r["rec"]["replica"] for r in results})
    ncol, nrow = max(len(reps), 1), len(boxes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.6 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for res in results:
        rec, pl = res["rec"], res["plot"]
        ax = axes[boxes.index(rec["box"]), reps.index(rec["replica"])]
        ax.axis("on")
        PAINN._plot_msd_panel(ax, pl["tau"], pl["msds"], pl["fits"],
                              f"{O.box_label(rec)} r{rec['replica']} "
                              f"({round(pl['t_total_ns'])} ns)",
                              fits2=pl.get("fits2"))
    handles = [Line2D([], [], color=c, lw=1.6, label=lab) for _, lab, c, _ in SPECIES]
    handles.append(Line2D([], [], color="0.15", lw=1.2, ls="--", label="fixed-window fit"))
    handles.append(Line2D([], [], color="#D95F02", lw=1.2, ls="-.", label=r"Fickian ($\beta$≈1) fit"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("OPLS self-diffusivity — time-origin MSD (log-log) with fixed-window Einstein "
                 "fit (black dashed, window grey) and Fickian $\\beta$≈1 fit (orange dash-dot)"
                 "  —  production_replicas, r0–r3",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.015, 1, 0.985])
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    print("saved ->", out, flush=True)


def parity_grid(bybox: pd.DataFrame, out: Path, fick: bool = False):
    """2x3 parity on the per-box replica means (error bar = s.d. over replicas):
    rows = YH with experimental eta / YH with the OPLS GK eta; columns =
    cation / anion / solvent. ``fick=True`` plots the Fickian (``D_*_fick_*``)
    column family instead of the fixed-window one."""
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                         "axes.facecolor": "white"})
    pre = "fick_" if fick else ""
    method_lab = r"Fickian $\beta$≈1 window" if fick else "fixed 1-9 ns window"
    ROWS = [("inf_exp", r"YH: experimental $\eta$"),
            ("inf_sim", r"YH: OPLS GK $\eta$")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    for i, (suffix, rowlab) in enumerate(ROWS):
        for j, (sp, splab, _, _) in enumerate(SPECIES):
            PAINN._parity_panel(axes[i, j], bybox, f"D_{sp}_exp_1e10",
                                f"D_{sp}_{pre}{suffix}_mean_1e10",
                                f"{splab}  ({rowlab})", conv_col=f"D_{sp}_{pre}converged",
                                err_col=f"D_{sp}_{pre}{suffix}_std_1e10")

    salts = [s for s in ("napf6", "naotf", "lipf6") if (bybox["salt"] == s).any()]
    solvs = [s for s in ("dme", "diglyme", "pc", "tgdme") if (bybox["solvent"] == s).any()]
    h = [Line2D([], [], marker=O.SALT_MARKER[s], color="0.3", ls="", ms=9, mfc="0.3",
                label=O.SALT_LABEL[s]) for s in salts]
    h += [Line2D([], [], marker="o", color=O.SOLV_COLOR[s], ls="", ms=9, label=O.SOLV_LABEL[s])
          for s in solvs]
    h += [Line2D([], [], marker="o", color="0.3", ls="", ms=9, mfc="white", label="0.1 M (hollow)"),
          Line2D([], [], marker="x", color="0.15", ls="", ms=9, label="not converged")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"OPLS self-diffusivity parity ({method_lab}) — Einstein MSD slope, Yeh-Hummer "
                 "finite-size correction with experimental η (top) vs OPLS GK η (bottom)\n"
                 "dD = kBT·ξ/(6πηL); per-box mean ± s.d. over 4 replicas  —  production_replicas",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print("saved ->", out, flush=True)


# Per-box aggregation (mean +/- s.d. over replicas) lives on the PAINN module as
# `aggregate_by_box` -- shared rather than duplicated so the two stay in lockstep.

# ── driver ───────────────────────────────────────────────────────────────────
MASTER_ORDER = [
    "system_id", "salt", "solvent", "box", "conc_label", "conc_M", "temperature_K",
    "replica", "run_name", "fit_method", "t_total_ns", "dt_ps", "dt_source",
    "frames_used", "effective_dt_ps", "analysis_window_ps", "eq_time_ps", "msd_method",
    "box_L_A", "eta_sim_cP", "eta_exp_cP", "yh_dD_sim_1e10", "yh_dD_exp_1e10",
    "all_converged",
]
MASTER_ORDER.append("all_converged_fick")
for _sp, _, _, _ in SPECIES:
    MASTER_ORDER += [f"D_{_sp}_pbc_1e10", f"D_{_sp}_inf_sim_1e10", f"D_{_sp}_inf_exp_1e10",
                     f"D_{_sp}_exp_1e10", f"D_{_sp}_t_lo_ps", f"D_{_sp}_t_hi_ps",
                     f"D_{_sp}_beta", f"D_{_sp}_converged"]
for _sp, _, _, _ in SPECIES:
    MASTER_ORDER += [f"D_{_sp}_fick_pbc_1e10", f"D_{_sp}_fick_inf_sim_1e10",
                     f"D_{_sp}_fick_inf_exp_1e10", f"D_{_sp}_fick_t_lo_ps",
                     f"D_{_sp}_fick_t_hi_ps", f"D_{_sp}_fick_beta", f"D_{_sp}_fick_converged"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=O.DEFAULT_ROOT,
                    help="OPLS production_replicas root.")
    ap.add_argument("--out-root", type=Path, default=O.DEFAULT_OUT_ROOT,
                    help="Base output dir; results go in <out-root>/diffusivity<suffix>.")
    ap.add_argument("--replicas", type=int, nargs="*", default=None,
                    help="Replica indices to analyse (default: all found).")
    ap.add_argument("--only", default=None, help="Substring filter on the box name.")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out-suffix", default="",
                    help="Suffix appended to the diffusivity output folder (e.g. '_fickian') "
                         "so a re-analysis doesn't clobber existing results.")
    ap.add_argument("--visc-suffix", default=None,
                    help="Suffix of the <out-root>/viscosity<...> folder to read the OPLS GK "
                         "viscosity from. Defaults to --out-suffix; viscosity is "
                         "fit-method-independent, so a '_fickian' run should usually pass ''.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    visc_suffix = args.out_suffix if args.visc_suffix is None else args.visc_suffix
    sim_eta = load_sim_viscosity(args.out_root, visc_suffix)
    runs, skipped = O.discover_runs(args.root, need="traj",
                                    replicas=args.replicas, only=args.only)
    for r in runs:
        r["eta_sim"] = sim_eta.get(r["box"])
    print(f"Discovered {len(runs)} replicas to analyze, {len(skipped)} skipped.")

    if args.dry_run:
        for r in sorted(runs, key=lambda x: (x["salt"], x["solvent"], x["conc_M"],
                                             x["temperature_K"], x["replica"])):
            e, es = r["eta_exp"], r["eta_sim"]
            print(f"  {r['box']:26s} r{r['replica']}  "
                  f"eta_exp={'--' if e is None else f'{e:.3f}'} "
                  f"eta_sim={'--' if es is None else f'{es:.3f}'} cP  "
                  f"D_exp(+/-/0)={r['D_exp_cation']}/{r['D_exp_anion']}/{r['D_exp_solvent']}")
        for r in skipped:
            print(f"  SKIP {r['run_name']}: {r['skip_reason']}")
        print(f"\nDRY RUN: {len(runs)} replicas, {len(skipped)} skipped. Nothing computed.")
        return

    out_dir = args.out_root / f"diffusivity{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # One MSD pass per replica (SETTINGS_FIXED's eq/stride); both fit methods are
    # applied to that curve — fixed fills the primary columns, fickian the
    # D_*_fick_* family (matches diffusivity_replicas.py).
    base = SETTINGS_FIXED
    settings = dict(base, fick=SETTINGS_FICKIAN, out_dir=str(out_dir),
                    generated=datetime.now().isoformat(timespec="seconds"))

    results, n = [], len(runs)
    print(f"[progress] launching {n} replicas on {args.jobs} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(analyze_one, r, settings, not args.no_plots): i
                for i, r in enumerate(runs)}
        for done, fut in enumerate(as_completed(futs), 1):
            r = runs[futs[fut]]
            tag = f"{r['box']} r{r['replica']}"
            try:
                res = fut.result()
                results.append(res)
                row = res["row"]
                print(f"[progress] {done}/{n}  {tag:30s} "
                      f"D_pbc(+/-/0)={row['D_cation_pbc_1e10']}/"
                      f"{row['D_anion_pbc_1e10']}/{row['D_solvent_pbc_1e10']}  "
                      f"({row['t_total_ns']:.1f} ns)", flush=True)
            except Exception as e:
                print(f"[progress] {done}/{n}  {tag:30s} ERROR: {e}", flush=True)

    if not results:
        print("No replicas analyzed; nothing to aggregate.")
        return

    master = pd.DataFrame([res["row"] for res in results])
    master = master.reindex(columns=[c for c in MASTER_ORDER if c in master.columns]
                            + [c for c in master.columns if c not in MASTER_ORDER])
    master = master.sort_values(["salt", "solvent", "conc_M", "temperature_K",
                                 "replica"]).reset_index(drop=True)
    master.to_csv(out_dir / "diffusivity_master.csv", index=False)
    bybox = PAINN.aggregate_by_box(master)
    bybox.to_csv(out_dir / "diffusivity_by_box.csv", index=False)

    cfg = dict(
        observable="diffusivity", forcefield="OPLS-AA: LigParGen solvents, CL&P anions, Aqvist cations (GROMACS)",
        estimator="einstein_fixed_window (primary D_* columns) + "
                  "einstein_fickian_beta1 (D_*_fick_* columns), same MSD curve",
        estimator_source="src/analysis/observables/diffusivity.py::compute_diffusivity "
                         "(same function the PaiNN driver uses)",
        painn_counterpart="transport/diffusivity_replicas.py",
        fit_method="fixed+fickian",
        grouping="GROMACS residues, element multiset validated against "
                 "systems.{cation,anion,solvent}_dict (see opls_runs.groups_from_topology)",
        yeh_hummer=True, yh_variants=["experimental_eta", "opls_gk_eta"],
        root=str(args.root), out_dir=str(out_dir),
        exp_source={str(k): str(v) for k, v in O.EXP_CSV_BY_T.items()},
        sim_viscosity_source=str(args.out_root / f"viscosity{visc_suffix}"
                                 / "viscosity_gk_by_box.csv"),
        settings={k: v for k, v in base.items()},
        fickian_settings={k: v for k, v in SETTINGS_FICKIAN.items()},
        n_replicas_analyzed=len(results), n_skipped=len(skipped), n_boxes=len(bybox),
        n_all_converged_fick=(int(master["all_converged_fick"].sum())
                              if "all_converged_fick" in master else None),
        skipped_runs=[f"{r['run_name']} ({r['skip_reason']})" for r in skipped],
        code_version=git_sha(), generated=settings["generated"],
    )
    (out_dir / "diffusivity_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if not args.no_plots:
        msd_grid(results, out_dir / "figure_transport_diff_msd.png")
        parity_grid(bybox, out_dir / "figure_transport_diff_parity.png")
        parity_grid(bybox, out_dir / "figure_transport_diff_parity_fickian.png", fick=True)

    print("\n=== OPLS self-diffusivity (Einstein MSD, 1e-10 m^2/s), per box ===")
    cols = ["box", "temperature_K", "box_L_A", "eta_sim_cP", "eta_exp_cP",
            "D_cation_pbc_mean_1e10", "D_cation_inf_exp_mean_1e10", "D_cation_exp_1e10",
            "D_solvent_pbc_mean_1e10", "D_solvent_exp_1e10"]
    print(bybox[[c for c in cols if c in bybox.columns]].to_string(index=False))
    print(f"\nDone: {len(results)} replicas -> {out_dir}/diffusivity_master.csv")
    print(f"  per-box means   -> {out_dir/'diffusivity_by_box.csv'}")
    print(f"  per-run outputs -> {out_dir}/per_run/<box>/r<k>/diffusivity*")
    print(f"  config          -> {out_dir/'diffusivity_config.yaml'}")


if __name__ == "__main__":
    main()
