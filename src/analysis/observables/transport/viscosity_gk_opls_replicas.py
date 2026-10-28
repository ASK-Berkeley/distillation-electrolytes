#!/usr/bin/env python
"""Green-Kubo shear viscosity for the OPLS/GROMACS ``production_replicas`` set.

The OPLS twin of ``viscosity_gk_replicas.py`` — same estimator, same settings,
different input tree. Both drivers call the identical
``observables.viscosity_gk.eta_running_curve`` + ``fit_eta_tdm`` pair (running
GK stress-ACF integral eta(tau) out to ``--max-lag-ps`` with a block-resampling
error band, then the Zhang-Maginn time-decomposition read-off, JCTC 2015), and
the OPLS ``.stress`` logs are written in exactly the FeNNol log format by
``edr2stress.py``, so nothing about the calculation changes.

The only deliberate differences from the PaiNN driver, both forced by the input set:
  * the GK prefactor uses **each box's own setpoint T** (273 / 298 / 323 K)
    instead of a hardcoded 298 K (override with ``--t-prefactor``);
  * a per-box mean +/- s.d. over the 4 replicas is reported alongside the
    per-replica rows.

Outputs land under ``data/opls_transport_observables/viscosity/`` (the OPLS run
tree itself is treated as read-only):
  * ``per_run/<box>/r<k>/viscosity_gk.csv``       -- one row per replica
  * ``per_run/<box>/r<k>/viscosity_gk_curve.csv`` -- decimated eta(tau) curve
  * ``per_run/<box>/r<k>/viscosity_gk.png``       -- per-replica convergence panel
  * ``per_run/<box>/r<k>/summary.json``
  * ``viscosity_gk_master.csv``  -- one row per replica (64)
  * ``viscosity_gk_by_box.csv``  -- per-box mean +/- s.d. (16); this is the file
                                    ``diffusivity_opls_replicas.py`` reads for the
                                    simulation-eta Yeh-Hummer variant
  * ``viscosity_gk_config.yaml`` -- settings + provenance sidecar
  * ``figure_transport_visc_convergence.png`` -- eta(tau) grid, one panel per replica
  * ``figure_transport_visc_parity.png``      -- per-box parity vs experiment

CPU only; conda env ``fennol``. ~40 s and ~3.5 GB RSS per replica (the stress logs
are 10M rows), so keep ``--jobs`` modest. Run from the repo root as a module:

    python -m src.analysis.observables.transport.viscosity_gk_opls_replicas --dry-run
    python -m src.analysis.observables.transport.viscosity_gk_opls_replicas --jobs 8
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
from ..viscosity_gk import eta_running_curve, fit_eta_tdm

# The PaiNN driver is the reference for every plotting/read-off convention here;
# import it rather than re-implement so the two stay in lockstep.
from . import viscosity_gk_replicas as PAINN

NBLOCK = PAINN.NBLOCK
CUTOFFS_PS = PAINN.CUTOFFS_PS       # fixed read-offs, for reference only


# ── per-run compute + output ─────────────────────────────────────────────────
def _merge_summary(obs_dir: Path, block: dict) -> None:
    p = obs_dir / "summary.json"
    summary = {}
    if p.is_file():
        try:
            summary = json.loads(p.read_text())
        except Exception:
            summary = {}
    summary["viscosity_gk"] = block
    p.write_text(json.dumps(summary, indent=2))


def analyze_one(rec: dict, settings: dict, plots: bool) -> dict:
    """Compute eta(tau) + TDM fit for one replica, write per-run outputs, return a
    light-weight result (scalar row + decimated curve for the aggregate figure)."""
    rd = Path(rec["run_dir"])
    t_pref = (settings["t_prefactor_k"] if settings["t_prefactor_k"] is not None
              else rec["temperature_K"])
    c = eta_running_curve(rd, max_lag_ps=settings["max_lag_ps"],
                          nblock=settings["nblock"], t_K=t_pref,
                          eq_time_ps=settings["eq_time_ps"])
    lags, full, band = c["lags_ps"], c["running_cp"], c["band"]
    p = fit_eta_tdm(lags, full, band)

    obs_dir = O.per_run_dir(settings["out_dir"], rec)

    def _cut(tc):
        i = min(len(full) - 1, int(round(tc / c["dt_ps"])))
        return float(full[i])
    eta_cut = {tc: _cut(tc) for tc in settings["cutoffs_ps"]}

    row = {
        "system_id": rec["system_id"], "salt": rec["salt"], "solvent": rec["solvent"],
        "box": rec["box"], "conc_label": rec["conc_label"], "conc_M": rec["conc_M"],
        "temperature_K": rec["temperature_K"], "replica": rec["replica"],
        "run_name": rec["run_name"],
        "eta_gk_cP": round(p["est"], 4), "eta_err_cP": round(p["err"], 4),
        "converged": bool(p["conv"]),
        "t_lo_ps": round(p["t_lo"], 2), "t_hi_ps": round(p["t_hi"], 2),
        "tau_slow_ps": round(p["tau_slow_ps"], 2), "frac_recovered": round(p["frac_recovered"], 4),
        "t_total_ns": round(c["t_total_ns"], 3), "T_mean_K": round(c["T_mean_K"], 2),
        "dt_fs": round(c["dt_ps"] * 1000.0, 3), "n_stress_samples": c["n_samples"],
        "eq_time_ps": settings["eq_time_ps"],
        "max_lag_ps": settings["max_lag_ps"], "T_prefactor_K": t_pref,
        "eta_exp_cP": rec["eta_exp"],
    }
    for tc in settings["cutoffs_ps"]:
        row[f"eta_at_{int(tc)}ps_cP"] = round(eta_cut[tc], 4)

    pd.DataFrame([row]).to_csv(obs_dir / "viscosity_gk.csv", index=False)
    d = O.decimate(len(lags), settings["curve_points"])
    pd.DataFrame({"tau_ps": lags[d], "eta_running_cP": full[d], "band_cP": band[d]}
                 ).to_csv(obs_dir / "viscosity_gk_curve.csv", index=False)
    _merge_summary(obs_dir, {**{k: row[k] for k in (
        "eta_gk_cP", "eta_err_cP", "converged", "t_lo_ps", "t_hi_ps", "tau_slow_ps",
        "frac_recovered", "t_total_ns", "eq_time_ps", "T_prefactor_K", "max_lag_ps")},
        "code_version": git_sha(),
        "generated": settings["generated"], "eta_exp_cP": rec["eta_exp"]})

    if plots:
        fig, ax = plt.subplots(figsize=(5, 4))
        flag = "OK" if p["conv"] else "NOT conv"
        PAINN._plot_panel(ax, lags, full, band, p, rec["eta_exp"],
                          f"{rec['run_name']} ({round(c['t_total_ns'])} ns)\n"
                          f"$\\eta$={p['est']:.2f} exp={rec['eta_exp'] or float('nan'):.2f} [{flag}]")
        fig.tight_layout(); fig.savefig(obs_dir / "viscosity_gk.png", dpi=130, facecolor="white")
        plt.close(fig)

    d2 = O.decimate(len(lags), 4000)
    return {"rec": rec, "row": row,
            "plot": {"lags": lags[d2], "full": full[d2], "band": band[d2],
                     "fit_curve": p["fit_curve"], "est": p["est"], "conv": p["conv"],
                     "t_lo": p["t_lo"], "t_hi": p["t_hi"], "t_total_ns": c["t_total_ns"]}}


# ── aggregate figures ────────────────────────────────────────────────────────
def convergence_grid(results: list, out: Path):
    """eta(tau) grid: one row per box, one column per replica."""
    results = sorted(results, key=lambda r: (r["rec"]["salt"], r["rec"]["solvent"],
                                             r["rec"]["conc_M"], r["rec"]["temperature_K"],
                                             r["rec"]["replica"]))
    boxes = list(dict.fromkeys(r["rec"]["box"] for r in results))   # sorted order, deduped
    reps = sorted({r["rec"]["replica"] for r in results})
    ncol, nrow = max(len(reps), 1), len(boxes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.6 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for res in results:
        rec, pl = res["rec"], res["plot"]
        i, j = boxes.index(rec["box"]), reps.index(rec["replica"])
        ax = axes[i, j]
        ax.axis("on")
        flag = "OK" if pl["conv"] else "NOT conv"
        e = rec["eta_exp"]
        p = {"t_lo": pl["t_lo"], "t_hi": pl["t_hi"], "fit_curve": pl["fit_curve"],
             "est": pl["est"], "conv": pl["conv"]}
        title = (f"{O.box_label(rec)} r{rec['replica']} ({round(pl['t_total_ns'])} ns)\n"
                 f"est={pl['est']:.2f} exp={e if e is not None else float('nan'):.2f} [{flag}]")
        PAINN._plot_panel(ax, pl["lags"], pl["full"], pl["band"], p, e, title)
        ax.get_lines()[0].set_color(O.SOLV_COLOR[rec["solvent"]])
        for coll in ax.collections:
            coll.set_color(O.SOLV_COLOR[rec["solvent"]]); coll.set_alpha(0.22)
    fig.suptitle("OPLS GK viscosity — running integral η(τ) to 1 ns, block band, "
                 "Zhang-Maginn TDM fit (black dashed) → η∞ (dotted), exp (red), "
                 "fit region (grey)  —  production_replicas, r0–r3",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    print("saved ->", out, flush=True)


def parity_plot(bybox: pd.DataFrame, out: Path):
    """Per-box replica-mean GK eta vs experiment, log-log. Same encoding as the
    PaiNN parity: solvent = colour, salt = marker, 0.1 M = hollow, factor-2 band;
    boxes where no replica converged are marked x and excluded from the MAPD."""
    fig, (ax, al) = plt.subplots(1, 2, figsize=(8.4, 6),
                                 gridspec_kw={"width_ratios": [1, 0.36]})
    vals = []
    for _, r in bybox.iterrows():
        for v in (_fin(r["eta_exp_cP"]), _fin(r["eta_gk_mean_cP"])):
            if np.isfinite(v) and v > 0:
                vals.append(v)
    if not vals:
        print("[warn] no (exp, sim) viscosity pairs to plot; skipping parity.")
        plt.close(fig)
        return
    lo, hi = min(vals) / 1.8, max(vals) * 1.8
    ax.fill_between([lo, hi], [lo/2, hi/2], [lo*2, hi*2], color="0.8", alpha=0.3, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.45", ls="--", lw=1.3, zorder=1)
    devs = []
    for _, r in bybox.iterrows():
        xe, ys, es = _fin(r["eta_exp_cP"]), _fin(r["eta_gk_mean_cP"]), _fin(r["eta_gk_std_cP"])
        if not (np.isfinite(xe) and np.isfinite(ys) and xe > 0 and ys > 0):
            continue
        c = O.SOLV_COLOR[r["solvent"]]
        filled = float(r["conc_M"]) != 0.1
        ax.errorbar(xe, ys, yerr=es if np.isfinite(es) else None, fmt="none", ecolor=c,
                    elinewidth=1.1, capsize=2, alpha=0.6, zorder=2)
        ax.scatter(xe, ys, s=130, marker=O.SALT_MARKER.get(r["salt"], "o"),
                   facecolor=c if filled else "white", edgecolor=c, linewidths=1.9, zorder=3)
        if int(r["n_converged"]) > 0:
            devs.append(abs(ys - xe) / xe)
        else:
            ax.scatter(xe, ys, s=70, marker="x", color="0.15", linewidths=1.6, zorder=4)
    txt = (f"MAPD = {100*np.mean(devs):.0f}%\n(boxes n = {len(devs)}/{len(bybox)})"
           if devs else f"none converged (n=0/{len(bybox)})")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7", alpha=0.85))
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi)); ax.set_aspect("equal")
    ax.set_xlabel(r"Experiment  (mPa$\cdot$s)"); ax.set_ylabel(r"OPLS GK  (cP)")
    ax.set_title(r"Shear viscosity $\eta$ (GK) — OPLS production_replicas"
                 "\n(mean ± s.d. over 4 replicas)")
    ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.55)
    ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.2)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    al.axis("off")
    salts = [s for s in ("napf6", "naotf", "lipf6") if (bybox["salt"] == s).any()]
    solvs = [s for s in ("dme", "diglyme", "pc", "tgdme") if (bybox["solvent"] == s).any()]
    h = [Line2D([], [], marker=O.SALT_MARKER[s], color="0.3", ls="", ms=9, mfc="0.3",
                label=O.SALT_LABEL[s]) for s in salts]
    h += [Line2D([], [], marker="o", color=O.SOLV_COLOR[s], ls="", ms=9, label=O.SOLV_LABEL[s])
          for s in solvs]
    h += [Line2D([], [], marker="o", color="0.3", ls="", ms=9, mfc="white", label="0.1 M (hollow)"),
          Line2D([], [], marker="x", color="0.15", ls="", ms=9, label="no replica converged")]
    al.legend(handles=h, loc="center", fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    print("saved ->", out, flush=True)


def _fin(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# Per-box aggregation (mean +/- s.d. over replicas) lives on the PAINN module as
# `aggregate_by_box` -- shared rather than duplicated so the two stay in lockstep.

# ── driver ───────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=O.DEFAULT_ROOT,
                    help="OPLS production_replicas root.")
    ap.add_argument("--out-root", type=Path, default=O.DEFAULT_OUT_ROOT,
                    help="Base output dir; results go in <out-root>/viscosity<suffix>.")
    ap.add_argument("--max-lag-ps", type=float, default=1000.0,
                    help="Running integral / block band computed out to this lag (default 1 ns).")
    ap.add_argument("--nblock", type=int, default=NBLOCK, help="Blocks for the error band.")
    ap.add_argument("--eq-time-ps", type=float, default=2000.0,
                    help="Discard stress rows before this time (default 2000 ps, matching "
                         "the PaiNN viscosity driver and the trajectory observables' "
                         "equilibration cut). 0 = use all.")
    ap.add_argument("--curve-points", type=int, default=2000,
                    help="Points kept in the saved per-run curve CSV (default 2000).")
    ap.add_argument("--t-prefactor", type=float, default=None,
                    help="Override the V/(kB T) prefactor temperature for every run. "
                         "Default: each box's own setpoint T from its folder name.")
    ap.add_argument("--replicas", type=int, nargs="*", default=None,
                    help="Replica indices to analyse (default: all found).")
    ap.add_argument("--only", default=None, help="Substring filter on the box name.")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                    help="Workers. Each holds a ~3.5 GB stress log in memory.")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix appended to the output folder so a re-analysis "
                         "doesn't clobber existing results.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    runs, skipped = O.discover_runs(args.root, need="stress",
                                    replicas=args.replicas, only=args.only)
    print(f"Discovered {len(runs)} replicas to analyze, {len(skipped)} skipped.")

    if args.dry_run:
        for r in sorted(runs, key=lambda x: (x["salt"], x["solvent"], x["conc_M"],
                                             x["temperature_K"], x["replica"])):
            e = r["eta_exp"]
            print(f"  {r['box']:26s} r{r['replica']}  T={r['temperature_K']:.0f}K  "
                  f"eta_exp={'--' if e is None else f'{e:.3f}'} mPa*s")
        for r in skipped:
            print(f"  SKIP {r['run_name']}: {r['skip_reason']}")
        print(f"\nDRY RUN: {len(runs)} replicas, {len(skipped)} skipped. Nothing computed.")
        return

    out_dir = args.out_root / f"viscosity{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = dict(
        max_lag_ps=args.max_lag_ps, nblock=args.nblock,
        t_prefactor_k=args.t_prefactor, cutoffs_ps=CUTOFFS_PS,
        eq_time_ps=args.eq_time_ps,
        curve_points=args.curve_points, out_dir=str(out_dir),
        generated=datetime.now().isoformat(timespec="seconds"),
    )

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
                print(f"[progress] {done}/{n}  {tag:30s} eta={row['eta_gk_cP']:.2f} cP "
                      f"±{row['eta_err_cP']:.2f}  conv={row['converged']}  "
                      f"tau_slow={row['tau_slow_ps']:.0f}ps  ({row['t_total_ns']:.1f} ns)",
                      flush=True)
            except Exception as e:
                print(f"[progress] {done}/{n}  {tag:30s} ERROR: {e}", flush=True)

    if not results:
        print("No replicas analyzed; nothing to aggregate.")
        return

    master = pd.DataFrame([res["row"] for res in results]).sort_values(
        ["salt", "solvent", "conc_M", "temperature_K", "replica"]).reset_index(drop=True)
    master.to_csv(out_dir / "viscosity_gk_master.csv", index=False)
    bybox = PAINN.aggregate_by_box(master)
    bybox.to_csv(out_dir / "viscosity_gk_by_box.csv", index=False)

    cfg = dict(
        observable="viscosity_gk", forcefield="OPLS-AA: LigParGen solvents, CL&P anions, Aqvist cations (GROMACS)",
        estimator="zhang_maginn_tdm",
        estimator_source="src/analysis/observables/viscosity_gk.py "
                         "(same module the PaiNN driver uses)",
        painn_counterpart="transport/viscosity_gk_replicas.py",
        root=str(args.root), out_dir=str(out_dir),
        exp_source={str(k): str(v) for k, v in O.EXP_CSV_BY_T.items()},
        t_prefactor="per-box setpoint T" if args.t_prefactor is None else args.t_prefactor,
        settings={k: (list(v) if isinstance(v, tuple) else v)
                  for k, v in settings.items() if k not in ("generated", "out_dir")},
        n_replicas_analyzed=len(results), n_skipped=len(skipped),
        n_converged=int(master["converged"].sum()), n_boxes=len(bybox),
        skipped_runs=[f"{r['run_name']} ({r['skip_reason']})" for r in skipped],
        code_version=git_sha(), generated=settings["generated"],
    )
    (out_dir / "viscosity_gk_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if not args.no_plots:
        convergence_grid(results, out_dir / "figure_transport_visc_convergence.png")
        parity_plot(bybox, out_dir / "figure_transport_visc_parity.png")

    print("\n=== OPLS GK viscosity (TDM fit), per box ===")
    cols = ["box", "temperature_K", "n_converged", "eta_gk_mean_cP", "eta_gk_std_cP",
            "eta_exp_cP", "sim_over_exp"]
    print(bybox[cols].to_string(index=False))
    print(f"\nDone: {len(results)} replicas "
          f"({int(master['converged'].sum())} converged) -> {out_dir}/viscosity_gk_master.csv")
    print(f"  per-box means   -> {out_dir/'viscosity_gk_by_box.csv'}")
    print(f"  per-run outputs -> {out_dir}/per_run/<box>/r<k>/viscosity_gk*")
    print(f"  config          -> {out_dir/'viscosity_gk_config.yaml'}")


if __name__ == "__main__":
    main()
