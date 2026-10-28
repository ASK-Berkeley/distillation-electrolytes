#!/usr/bin/env python
"""Onsager + Nernst-Einstein ionic conductivity for the OPLS ``production_replicas`` set.

Functionally the same calculation ``code_utils`` ran for this set
(``conductivity/run_conductivity_cm_prod_replicas.py`` ->
``conductivity/compute.run_onsager_conductivity_gromacs`` -> byteff2
``onsager_calc``), but through this repo's native
``observables.conductivity.compute_conductivity`` so OPLS and PaiNN share one
code path. The two were checked against each other on a real replica and agree
to **2.8e-7 relative** on sigma_onsager, sigma_NE and every D_self — see
``conductivity_common`` for the fit-window presets and that verification note.

Fit window: the driver now defaults to ``byteff2_ase`` (2 ns cut, 10 ps stride,
500-2000 ps fit) — the collective MSD is cleanly diffusive by ~0.5 ns, so this
removes the systematic high-side bias the early 50-199 ps window put on
sigma_onsager (see ``conductivity_common`` and the PaiNN driver). To reproduce
the legacy byteff2 GROMACS numbers instead, pass ``--window byteff2_gromacs``:
the full 20 ns trajectory is loaded at its native 1 ps spacing, ``onsager_calc``
drops the first 200 frames and fits MSD lags 50..199 — i.e. a 200 ps
equilibration cut and a **50-199 ps** fit window, every frame. **Match the
``--window`` to whatever the PaiNN side used.** Ion charges are taken from the
summed ``.itp`` partial charges rounded to integers, as ``build_species_dicts`` did.

Differences from ``code_utils``, all deliberate:
  * MDAnalysis (``unwrap`` + ``NoJump``) is unavailable on Perlmutter, so frames
    come from **mdtraj** and the unwrap is this package's MIC molecule-COM route.
    Verified numerically equivalent (above).
  * molecule grouping from the GROMACS topology rather than element-sequence
    patterns (see ``opls_runs.groups_from_topology``).
  * only **298 K** boxes by default (``--temperature``); the 273/323 K LiPF6
    boxes have no PaiNN counterpart.
  * ``code_utils`` reported ``Dself_inf`` with a **placeholder eta = 1.0 cP**
    Yeh-Hummer correction, which is meaningless; the D columns here are the raw
    D_PBC (sigma never used the corrected values in either code).

Outputs under ``data/opls_transport_observables/conductivity/``:
  * ``per_run/<box>/r<k>/{conductivity.csv, conductivity_curves.csv,
    conductivity.png, summary.json}``
  * ``conductivity_master.csv``  -- one row per replica
  * ``conductivity_by_box.csv``  -- per-box mean +/- s.d. over replicas
  * ``conductivity_config.yaml`` -- settings + provenance sidecar
  * ``figure_transport_cond_cross.png``  -- collective-correlation grid
  * ``figure_transport_cond_parity.png`` -- Onsager + NE parity vs experiment

CPU only; conda env ``fennol`` (needs ``mdtraj``). ~25 s per replica. Run from
the repo root as a module:

    python -m src.analysis.observables.transport.conductivity_opls_replicas --dry-run
    python -m src.analysis.observables.transport.conductivity_opls_replicas --jobs 8
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


from . import conductivity_common as C
from . import opls_runs as O
from ..paths import git_sha
from ..conductivity import compute_conductivity
from ..systems import SYSTEMS


def analyze_one(rec: dict, settings: dict, plots: bool) -> dict:
    """Onsager + NE conductivity for one replica; writes per-run outputs."""
    traj = rec["traj"]
    spec = SYSTEMS[rec["system_id"]]
    dt = O.xtc_dt_ps(traj)
    n_total = O.xtc_n_frames(traj)
    w = settings["window"]
    z_cat, z_an = O.charges_from_itp(rec["run_dir"], spec)

    res = compute_conductivity(
        traj, spec, dt, z_cation=z_cat, z_anion=z_an,
        eq_time_ps=w["eq_time_ps"], tau_min_fit_ps=w["tau_min_fit_ps"],
        tau_max_fit_ps=w["tau_max_fit_ps"],
        tau_max_fit_frac=(w["tau_max_fit_frac"] if w["tau_max_fit_frac"] is not None else 0.5),
        target_frames=C.target_frames_for(w, n_total, dt),
        temperature_k=rec["temperature_K"], show_progress=False,
        frames_source=O.make_xtc_frame_source(rec["topology"]),
        groups_override=O.groups_from_topology(rec["topology"], spec),
        n_frames=n_total,
    )

    row = C.build_row(res, rec, settings["window_name"], dt)
    obs_dir = O.per_run_dir(settings["out_dir"], rec)
    pd.DataFrame([row]).to_csv(obs_dir / "conductivity.csv", index=False)

    d = C.decimate(len(res.tau_ps), settings["curve_points"])
    pd.DataFrame({"tau_ps": res.tau_ps[d], "msd_cation": res.msd_cat[d],
                  "msd_anion": res.msd_anion[d], "msd_solvent": res.msd_solvent[d],
                  "cross_pp": res.cross_pp[d], "cross_pm": res.cross_pm[d],
                  "cross_mm": res.cross_mm[d]}
                 ).to_csv(obs_dir / "conductivity_curves.csv", index=False)

    summary = {k: row[k] for k in row if k not in ("salt", "solvent", "system_id")}
    summary.update(code_version=git_sha(), generated=settings["generated"])
    (obs_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    label = f"{O.box_label(rec)} r{rec['replica']}"
    if plots:
        C.run_figure(res, label, obs_dir / "conductivity.png")

    d2 = C.decimate(len(res.tau_ps), 3000)
    return {"rec": rec, "row": row, "label": label,
            "plot": {"tau": res.tau_ps[d2],
                     "cross": {"pp": res.cross_pp[d2], "pm": res.cross_pm[d2],
                               "mm": res.cross_mm[d2]}}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=O.DEFAULT_ROOT)
    ap.add_argument("--out-root", type=Path, default=O.DEFAULT_OUT_ROOT)
    ap.add_argument("--window", choices=sorted(C.WINDOWS), default=C.DEFAULT_WINDOW,
                    help="Fit-window preset (default byteff2_gromacs = what code_utils "
                         "ran for this set: 200 ps cut, 50-199 ps fit, every frame).")
    ap.add_argument("--temperature", type=float, default=298.0,
                    help="Only analyse boxes at this setpoint T (default 298 K). "
                         "Use --all-temperatures for the full 273/298/323 K set.")
    ap.add_argument("--all-temperatures", action="store_true")
    ap.add_argument("--replicas", type=int, nargs="*", default=None)
    ap.add_argument("--only", default=None, help="Substring filter on the box name.")
    ap.add_argument("--curve-points", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out-suffix", default="",
                    help="Suffix on the output folder so a re-analysis (e.g. a second "
                         "--window) doesn't clobber the first.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    runs, skipped = O.discover_runs(args.root, need="traj",
                                    replicas=args.replicas, only=args.only)
    if not args.all_temperatures:
        kept = [r for r in runs if float(r["temperature_K"]) == args.temperature]
        dropped = len(runs) - len(kept)
        runs = kept
        if dropped:
            print(f"Temperature filter T = {args.temperature:.0f} K: dropped {dropped} replicas.")

    exp = C.load_exp_conductivity()
    for r in runs:
        r["sigma_exp"] = C.exp_sigma(exp, r["salt"], r["solvent"],
                                     r["conc_M"], r["temperature_K"])
    print(f"Discovered {len(runs)} replicas to analyze, {len(skipped)} skipped.")

    if args.dry_run:
        for r in sorted(runs, key=lambda x: (x["salt"], x["solvent"], x["conc_M"], x["replica"])):
            s = r["sigma_exp"]
            print(f"  {r['box']:26s} r{r['replica']}  T={r['temperature_K']:.0f}K  "
                  f"sigma_exp={'--' if s is None else f'{s:.4f}'} mS/cm")
        for r in skipped:
            print(f"  SKIP {r['run_name']}: {r['skip_reason']}")
        w = C.WINDOWS[args.window]
        hi = w["tau_max_fit_ps"] if w["tau_max_fit_ps"] is not None \
            else f"{w['tau_max_fit_frac']}*window"
        load = w["load_dt_ps"] if w["load_dt_ps"] is not None else "every frame"
        print(f"\nwindow '{args.window}': eq={w['eq_time_ps']} ps, "
              f"fit=[{w['tau_min_fit_ps']}, {hi}] ps, load_dt={load}")
        print(f"DRY RUN: {len(runs)} replicas, {len(skipped)} skipped. Nothing computed.")
        return

    out_dir = args.out_root / f"conductivity{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(window=C.WINDOWS[args.window], window_name=args.window,
                    curve_points=args.curve_points, out_dir=str(out_dir),
                    generated=datetime.now().isoformat(timespec="seconds"))

    results, n = [], len(runs)
    print(f"[progress] launching {n} replicas on {args.jobs} workers "
          f"(window '{args.window}')", flush=True)
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
                      f"sigma_O={row['sigma_onsager_mS_cm']:.4f}  "
                      f"sigma_NE={row['sigma_NE_mS_cm']:.4f} mS/cm  "
                      f"ionicity={row['ionicity']:.3f}", flush=True)
            except Exception as e:
                print(f"[progress] {done}/{n}  {tag:30s} ERROR: {e}", flush=True)

    if not results:
        print("No replicas analyzed; nothing to aggregate.")
        return

    master = pd.DataFrame([r["row"] for r in results])
    # carry the box name through explicitly (keyed on run_name, not row order)
    box_of = {r["row"]["run_name"]: r["rec"]["box"] for r in results}
    master.insert(3, "box", master["run_name"].map(box_of))
    master = C.order_columns(master)
    master = master.sort_values(["salt", "solvent", "conc_M", "temperature_K",
                                 "replica"]).reset_index(drop=True)
    master.to_csv(out_dir / "conductivity_master.csv", index=False)
    bybox = C.aggregate(master, "box")
    bybox.to_csv(out_dir / "conductivity_by_box.csv", index=False)

    cfg = dict(
        observable="conductivity", forcefield="OPLS-AA: LigParGen solvents, CL&P anions, Aqvist cations (GROMACS)",
        estimator="onsager + nernst_einstein",
        estimator_source="src/analysis/observables/conductivity.py::compute_conductivity",
        verified_against="byteff2 onsager_calc (code_utils/_vendored/onsager_conductivity.py); "
                         "agreement 2.8e-7 relative on napf6_dme_1M_298/r0",
        code_utils_counterpart="code_utils/conductivity/run_conductivity_cm_prod_replicas.py",
        painn_counterpart="transport/conductivity_replicas.py",
        window=args.window, window_settings=C.WINDOWS[args.window],
        temperature_filter=(None if args.all_temperatures else args.temperature),
        charges="summed .itp partial charges rounded to integer (as code_utils)",
        grouping="GROMACS residues validated against systems.{cation,anion,solvent}_dict",
        root=str(args.root), out_dir=str(out_dir),
        exp_source=str(C.GROUND_TRUTH / "conductivity_all.csv"),
        n_replicas_analyzed=len(results), n_skipped=len(skipped), n_boxes=len(bybox),
        code_version=git_sha(), generated=settings["generated"],
    )
    (out_dir / "conductivity_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if not args.no_plots:
        ordered = sorted(results, key=lambda r: (r["rec"]["salt"], r["rec"]["solvent"],
                                                 r["rec"]["conc_M"], r["rec"]["replica"]))
        C.cross_grid(ordered, out_dir / "figure_transport_cond_cross.png",
                     "OPLS ionic conductivity — ion-ion collective correlations "
                     f"$|C_{{ij}}(\\tau)|$, fit window shaded  —  window '{args.window}'")
        C.parity_figure(bybox, out_dir / "figure_transport_cond_parity.png",
                        "OPLS ionic conductivity vs experiment — per-box mean over replicas "
                        f"(window '{args.window}')",
                        ons_col="sigma_onsager_mS_cm_mean", ne_col="sigma_NE_mS_cm_mean",
                        ons_err="sigma_onsager_mS_cm_std", ne_err="sigma_NE_mS_cm_std")

    print(f"\n=== OPLS ionic conductivity (window '{args.window}'), per box ===")
    cols = ["box", "temperature_K", "n_replicas", "sigma_onsager_mS_cm_mean",
            "sigma_onsager_mS_cm_std", "sigma_NE_mS_cm_mean", "sigma_exp_mS_cm",
            "onsager_over_exp", "ionicity_mean"]
    print(bybox[[c for c in cols if c in bybox.columns]].to_string(index=False))
    print(f"\nDone: {len(results)} replicas -> {out_dir}/conductivity_master.csv")
    print(f"  per-box means   -> {out_dir/'conductivity_by_box.csv'}")
    print(f"  per-run outputs -> {out_dir}/per_run/<box>/r<k>/conductivity*")
    print(f"  config          -> {out_dir/'conductivity_config.yaml'}")


if __name__ == "__main__":
    main()
