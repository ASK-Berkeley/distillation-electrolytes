#!/usr/bin/env python
"""Onsager + Nernst-Einstein ionic conductivity for the transport_* NVT runs.

Sibling of ``diffusivity_replicas.py`` and ``viscosity_gk_replicas.py``: it walks
the same transport runs (``iter_run_dirs``, imported from the diffusivity driver),
off the same trajectories, so all three transport observables come from one run set.
Default run tree is the 4-replica ``PAINN_NVT_REPLICAS`` set (298 K only;
off-temperature runs are skipped); the legacy exp04 ``r0`` layout still works via
``--experiment-root``.

OUTPUT (default): results go ONLY to the external ``--out-root``
(``data/painn_transport_observables_replicas`` by default), leaving the run tree
untouched. Pass ``--write-internal`` to ALSO write the per-run
``<run_dir>/observables/`` and aggregate ``<exp>/transport/`` outputs described below.

The estimator is ``observables.conductivity.compute_conductivity`` — the identical
function the OPLS driver (``conductivity_opls_replicas.py``) calls, and a native
re-implementation of the byteff2 ``onsager_calc`` kernel ``code_utils`` used.
Checked against that kernel on a real trajectory: agreement to **2.8e-7 relative**
on sigma_onsager, sigma_NE and every D_self.

Fit window
----------
``code_utils`` did not use one window for both backends — its GROMACS path fit
MSD lags 50-199 ps while its ASE path defaulted to 500-2000 ps — so a PaiNN/OPLS
head-to-head has to pick one. Both drivers here default to ``byteff2_ase``
(2 ns cut, 10 ps stride, 500-2000 ps fit), which reproduces ``code_utils``' ASE
default: the collective MSD is cleanly diffusive by ~0.5 ns, so fitting there
(rather than the early 50-199 ps) removes a systematic high-side bias in
sigma_onsager and lifts the vs-experiment log-log Pearson r from 0.56 to 0.65 on
the 0.5/1 M set. ``--window byteff2_gromacs`` recovers the legacy 200 ps cut /
50-199 ps fit the OPLS numbers were originally produced with, and
``--window transport`` uses this project's 2 ns cut / 1-9 ns convention (correct
for the self term, too noisy for the collective one). **Use the same
``--window`` on both sides when comparing** — the collective term is noisy and
the window genuinely moves sigma_onsager.

Per run it writes into ``<run_dir>/observables/conductivity_<N>ns<suffix>/``:
  * ``conductivity.csv``        -- one row
  * ``conductivity_curves.csv`` -- self MSDs + the three collective C_ij curves
  * ``conductivity.png``        -- self-MSD / collective-correlation diagnostic
  * merges a ``conductivity`` block into the run's ``summary.json``

Then it rolls up into ``<exp>/transport/conductivity<suffix>/``:
  * ``conductivity_master.csv``  -- one row per run
  * ``conductivity_by_system.csv``
  * ``conductivity_config.yaml``
  * ``figure_transport_cond_cross.png`` / ``figure_transport_cond_parity.png``

CPU only; conda env ``fennol``. Run from the repo root as a module (the
package uses relative imports):

    python -m src.analysis.observables.transport.conductivity_replicas --dry-run
    python -m src.analysis.observables.transport.conductivity_replicas --jobs 8
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


from . import conductivity_common as C
from .. import io
from ..paths import git_sha
from ..conductivity import compute_conductivity
from ..systems import SYSTEMS

# Discovery conventions are the diffusivity driver's; import rather than restate.
from . import diffusivity_replicas as PAINN
from . import time_presets as TP

DEFAULT_ROOT = PAINN.DEFAULT_ROOT
RUN_GLOB = PAINN.RUN_GLOB
RUN_RE = PAINN.RUN_RE
_SALT, _SOLVENT = PAINN._SALT, PAINN._SOLVENT
conc_to_molar = PAINN.conc_to_molar


def discover_runs(root: Path, exp_sigma_table: dict, only: str | None = None):
    """(runs, skipped) for every transport run under ``root`` — the same set
    ``diffusivity_replicas.py`` / ``viscosity_gk_replicas.py`` use, handling both the
    legacy exp04 (r0-only) layout and the 4-replica ``nvt_replica_<N>`` tree
    (discovery conventions imported from diffusivity_replicas.iter_run_dirs)."""
    root = Path(root)
    if not root.is_dir():
        C.paths.require(root, "PAINN_NVT_REPLICAS")
    runs, skipped = [], []
    for system_id, salt, solv, conc_tok, conc_m, temp_k, replica, rd in PAINN.iter_run_dirs(root):
        if system_id not in SYSTEMS:
            continue
        if only is not None and only not in rd.name:
            continue
        # 298 K pipeline: skip off-temperature runs (the lipf6_dme 0.5 M 273/323 K
        # series in replicas 1-3) rather than mislabelling them.
        if int(round(temp_k)) != 298:
            skipped.append(dict(system_id=system_id, conc_label=conc_tok, run_name=rd.name,
                                skip_reason=f"non-298K (T={int(round(temp_k))}K)"))
            continue
        rec = dict(
            system_id=system_id, salt=salt, solvent=solv,
            conc_label=conc_tok, conc_M=conc_m, temperature_K=temp_k,
            replica=replica, run_name=rd.name, run_dir=str(rd),
            box=f"{system_id}_{conc_tok}_{int(round(temp_k))}",
            sigma_exp=C.exp_sigma(exp_sigma_table, salt, solv, conc_m, temp_k),
        )
        try:
            rec["traj"] = str(io.resolve_traj(rd))
        except FileNotFoundError:
            rec["skip_reason"] = "no trajectory"
            skipped.append(rec)
        else:
            runs.append(rec)
    return runs, skipped


def _merge_summary(obs_dir: Path, block: dict) -> None:
    p = obs_dir / "summary.json"
    summary = {}
    if p.is_file():
        try:
            summary = json.loads(p.read_text())
        except Exception:
            summary = {}
    summary["conductivity"] = block
    p.write_text(json.dumps(summary, indent=2, default=str))


def _per_run_external_dir(settings: dict, rec: dict) -> Path:
    """External per-run target: OUT_ROOT/conductivity<suffix>/per_run/<box>/r<k>/
    (the OPLS-mirrored layout)."""
    return (settings["out_root"] / f"conductivity{settings['out_suffix']}"
            / "per_run" / rec["box"] / f"r{rec['replica']}")


def _mirror_per_run(obs_dir: Path, rec: dict, settings: dict) -> None:
    """Copy this run's per-run outputs (written internally into <run_dir>/observables/)
    out into the external per_run/<box>/r<k>/ mirror. Only used with --write-internal;
    without it the files are written straight into the external dir (no copy)."""
    mirror_dir = _per_run_external_dir(settings, rec)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for f in obs_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, mirror_dir / f.name)


def analyze_one(rec: dict, settings: dict, plots: bool) -> dict:
    """Onsager + NE conductivity for one run; writes per-run outputs."""
    rd = Path(rec["run_dir"])
    traj = rec["traj"]
    spec = SYSTEMS[rec["system_id"]]
    dt, dt_source = io.resolve_dt_ps(rd, traj)
    n_total = io.estimate_n_frames(traj)
    # --time 1ns/5ns: analyze only the first T of the run (convergence study);
    # the capped total is passed as n_frames so the estimator never reads past it.
    max_frames = TP.cap_frames(settings.get("time", TP.DEFAULT_TIME), dt)
    if max_frames is not None:
        n_total = min(n_total, max_frames)
    t_total_ns = n_total * dt / 1000.0
    w = settings["window"]

    res = compute_conductivity(
        traj, spec, dt, z_cation=settings["z_cation"], z_anion=settings["z_anion"],
        eq_time_ps=w["eq_time_ps"], tau_min_fit_ps=w["tau_min_fit_ps"],
        tau_max_fit_ps=w["tau_max_fit_ps"],
        tau_max_fit_frac=(w["tau_max_fit_frac"] if w["tau_max_fit_frac"] is not None else 0.5),
        target_frames=C.target_frames_for(w, n_total, dt),
        temperature_k=rec["temperature_K"], show_progress=False,
        n_frames=n_total,
    )

    row = C.build_row(res, rec, settings["window_name"], dt)
    row["dt_source"] = dt_source
    row["t_total_ns"] = round(t_total_ns, 3)

    # Per-run output target. With --write-internal: the run's own
    # observables/conductivity_<N>ns/ subfolder. Default: straight into the external
    # per_run/<box>/r<k>/ mirror, leaving the (read-only) run tree untouched.
    if settings["write_internal"]:
        obs_dir = rd / "observables" / f"conductivity_{round(t_total_ns)}ns{settings['out_suffix']}"
    else:
        obs_dir = _per_run_external_dir(settings, rec)
    obs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(obs_dir / "conductivity.csv", index=False)

    d = C.decimate(len(res.tau_ps), settings["curve_points"])
    pd.DataFrame({"tau_ps": res.tau_ps[d], "msd_cation": res.msd_cat[d],
                  "msd_anion": res.msd_anion[d], "msd_solvent": res.msd_solvent[d],
                  "cross_pp": res.cross_pp[d], "cross_pm": res.cross_pm[d],
                  "cross_mm": res.cross_mm[d]}
                 ).to_csv(obs_dir / "conductivity_curves.csv", index=False)
    _merge_summary(obs_dir, {**row, "code_version": git_sha(),
                             "generated": settings["generated"]})

    label = (f"{C.SALT_LABEL.get(rec['salt'], rec['salt'])} {C.SOLV_LABEL[rec['solvent']]} "
             f"{rec['conc_label'].replace('_', '.')} ({round(t_total_ns)} ns)")
    if plots:
        C.run_figure(res, label, obs_dir / "conductivity.png")

    if settings["write_internal"]:
        _mirror_per_run(obs_dir, rec, settings)
    # else obs_dir already *is* the external per_run target; nothing to copy.

    d2 = C.decimate(len(res.tau_ps), 3000)
    return {"rec": rec, "row": row, "label": label,
            "plot": {"tau": res.tau_ps[d2],
                     "cross": {"pp": res.cross_pp[d2], "pm": res.cross_pm[d2],
                               "mm": res.cross_mm[d2]}}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--time", choices=TP.TIME_CHOICES, default=TP.DEFAULT_TIME,
                    help="Analyze only the first 1ns/5ns of each run with the matching "
                         "scaled window preset (short-trajectory convergence study; see "
                         "time_presets.py). Default 20ns = full run, current behaviour.")
    ap.add_argument("--window", choices=sorted(C.WINDOWS), default=None,
                    help="Fit-window preset; must match the OPLS driver's for a "
                         f"head-to-head (default {C.DEFAULT_WINDOW}, or its _1ns/_5ns "
                         "variant when --time is set).")
    ap.add_argument("--z-cation", type=float, default=1.0)
    ap.add_argument("--z-anion", type=float, default=-1.0)
    ap.add_argument("--only", default=None, help="Substring filter on the run name.")
    ap.add_argument("--curve-points", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out-root", type=Path, default=None,
                    help="External output root (default data/painn_transport_observables_replicas"
                         ", suffixed _1ns/_5ns when --time is set). All results go here.")
    ap.add_argument("--write-internal", action="store_true",
                    help="ALSO write per-run outputs into each <run_dir>/observables/ and "
                         "the aggregate into <experiment-root>/transport/conductivity. Default: "
                         "write ONLY to --out-root, leaving the run tree untouched.")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix on the conductivity output folder (per-run + aggregate).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    # Resolve --time into the window preset / output root (explicit flags win).
    if args.window is None:
        args.window = TP.cond_window_name(args.time)
    if args.out_root is None:
        args.out_root = TP.out_root_for(args.time)

    exp = C.load_exp_conductivity()
    runs, skipped = discover_runs(args.experiment_root, exp, only=args.only)
    print(f"Discovered {len(runs)} runs to analyze, {len(skipped)} skipped.")

    if args.dry_run:
        for r in sorted(runs, key=lambda x: (x["salt"], x["solvent"], x["conc_M"])):
            s = r["sigma_exp"]
            print(f"  {r['system_id']:16s} {r['conc_label']:5s} T={r['temperature_K']:.0f}K  "
                  f"sigma_exp={'--' if s is None else f'{s:.4f}'} mS/cm   {r['run_dir']}")
        for r in skipped:
            print(f"  SKIP {r['run_name']}: {r['skip_reason']}")
        w = C.WINDOWS[args.window]
        hi = w["tau_max_fit_ps"] if w["tau_max_fit_ps"] is not None \
            else f"{w['tau_max_fit_frac']}*window"
        load = w["load_dt_ps"] if w["load_dt_ps"] is not None else "every frame"
        print(f"\nwindow '{args.window}': eq={w['eq_time_ps']} ps, "
              f"fit=[{w['tau_min_fit_ps']}, {hi}] ps, load_dt={load}")
        print(f"DRY RUN: {len(runs)} runs, {len(skipped)} skipped. Nothing computed.")
        return

    settings = dict(window=C.WINDOWS[args.window], window_name=args.window,
                    time=args.time,
                    z_cation=args.z_cation, z_anion=args.z_anion,
                    curve_points=args.curve_points, out_suffix=args.out_suffix,
                    out_root=args.out_root, write_internal=args.write_internal,
                    generated=datetime.now().isoformat(timespec="seconds"))

    results, n = [], len(runs)
    print(f"[progress] launching {n} runs on {args.jobs} workers "
          f"(window '{args.window}')", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(analyze_one, r, settings, not args.no_plots): i
                for i, r in enumerate(runs)}
        for done, fut in enumerate(as_completed(futs), 1):
            r = runs[futs[fut]]
            tag = f"{r['system_id']} {r['conc_label']}"
            try:
                res = fut.result()
                results.append(res)
                row = res["row"]
                print(f"[progress] {done}/{n}  {tag:22s} "
                      f"sigma_O={row['sigma_onsager_mS_cm']:.4f}  "
                      f"sigma_NE={row['sigma_NE_mS_cm']:.4f} mS/cm  "
                      f"ionicity={row['ionicity']:.3f}  "
                      f"({row['t_total_ns']:.1f} ns)", flush=True)
            except Exception as e:
                print(f"[progress] {done}/{n}  {tag:22s} ERROR: {e}", flush=True)

    if not results:
        print("No runs analyzed; nothing to aggregate.")
        return

    master = C.order_columns(pd.DataFrame([r["row"] for r in results]))
    master = master.sort_values(["salt", "solvent", "conc_M", "replica"]).reset_index(drop=True)
    exp_label = args.experiment_root.name.replace("_", " ")
    ordered = sorted(results, key=lambda r: (r["rec"]["salt"], r["rec"]["solvent"],
                                             r["rec"]["conc_M"], r["rec"]["replica"]))

    # box column keyed straight off (system, conc, T): run_name alone isn't unique
    # across systems ("nvt_transport_1M_298K_20ns_1ps" repeats per system), and there
    # are now 4 replicas per box to aggregate over.
    master_box = master.copy()
    box_col = (master_box["system_id"] + "_" + master_box["conc_label"] + "_"
               + master_box["temperature_K"].round().astype(int).astype(str))
    master_box.insert(3, "box", box_col)
    bybox = C.aggregate(master_box, "box")

    cfg = dict(
        observable="conductivity", forcefield="PaiNN/FeNNol",
        estimator="onsager + nernst_einstein",
        estimator_source="src/analysis/observables/conductivity.py::compute_conductivity",
        verified_against="byteff2 onsager_calc (code_utils/_vendored/onsager_conductivity.py); "
                         "agreement 2.8e-7 relative",
        opls_counterpart="transport/conductivity_opls_replicas.py",
        window=args.window, window_settings=C.WINDOWS[args.window],
        z_cation=args.z_cation, z_anion=args.z_anion,
        experiment_root=str(args.experiment_root),
        exp_source=str(C.GROUND_TRUTH / "conductivity_all.csv"),
        write_internal=args.write_internal,
        n_runs_analyzed=len(results), n_skipped=len(skipped),
        skipped_runs=[f"{r['run_name']} ({r['skip_reason']})" for r in skipped],
        code_version=git_sha(), generated=settings["generated"],
    )

    # ── internal aggregate (opt-in): <experiment-root>/transport/conductivity<suffix>/ ─
    if args.write_internal:
        out_dir = args.experiment_root / "transport" / f"conductivity{args.out_suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        master.to_csv(out_dir / "conductivity_master.csv", index=False)
        bybox.to_csv(out_dir / "conductivity_by_system.csv", index=False)
        (out_dir / "conductivity_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        if not args.no_plots:
            C.cross_grid(ordered, out_dir / "figure_transport_cond_cross.png",
                         "Ionic conductivity — ion-ion collective correlations "
                         f"$|C_{{ij}}(\\tau)|$, fit window shaded  —  {exp_label} "
                         f"(window '{args.window}')")
            C.parity_figure(bybox, out_dir / "figure_transport_cond_parity.png",
                            f"Ionic conductivity vs experiment  —  {exp_label} "
                            f"(window '{args.window}')",
                            ons_col="sigma_onsager_mS_cm_mean", ne_col="sigma_NE_mS_cm_mean")

    # ── external output (always): OUT_ROOT/conductivity<suffix>/, OPLS-mirrored layout ─
    mirror_dir = Path(args.out_root) / f"conductivity{args.out_suffix}"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    master_box.to_csv(mirror_dir / "conductivity_master.csv", index=False)
    bybox.to_csv(mirror_dir / "conductivity_by_box.csv", index=False)
    mirror_cfg = dict(cfg, out_dir=str(mirror_dir), n_boxes=len(bybox))
    (mirror_dir / "conductivity_config.yaml").write_text(yaml.safe_dump(mirror_cfg, sort_keys=False))
    if not args.no_plots:
        if args.write_internal:
            shutil.copy2(out_dir / "figure_transport_cond_cross.png",
                         mirror_dir / "figure_transport_cond_cross.png")
            shutil.copy2(out_dir / "figure_transport_cond_parity.png",
                         mirror_dir / "figure_transport_cond_parity.png")
        else:
            C.cross_grid(ordered, mirror_dir / "figure_transport_cond_cross.png",
                         "Ionic conductivity — ion-ion collective correlations "
                         f"$|C_{{ij}}(\\tau)|$, fit window shaded  —  {exp_label} "
                         f"(window '{args.window}')")
            C.parity_figure(bybox, mirror_dir / "figure_transport_cond_parity.png",
                            f"Ionic conductivity vs experiment  —  {exp_label} "
                            f"(window '{args.window}')",
                            ons_col="sigma_onsager_mS_cm_mean", ne_col="sigma_NE_mS_cm_mean")

    print(f"\n=== Ionic conductivity (window '{args.window}') ===")
    cols = ["system_id", "conc_label", "replica", "t_total_ns", "sigma_onsager_mS_cm",
            "sigma_NE_mS_cm", "sigma_exp_mS_cm", "ionicity", "t_cation_onsager"]
    print(master[[c for c in cols if c in master.columns]].to_string(index=False))
    print(f"\nDone: {len(results)} runs -> {mirror_dir}/conductivity_master.csv")
    print(f"  external output -> {mirror_dir}/ (per_run/<box>/r<k>, *_master.csv, *_by_box.csv)")
    if args.write_internal:
        print(f"  internal too    -> {out_dir}/ + <run_dir>/observables/conductivity_<N>ns/")


if __name__ == "__main__":
    main()
