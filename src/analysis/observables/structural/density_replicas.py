#!/usr/bin/env python
"""Density for every 298 K NPT replica simulation in an exp01-style replica tree.

Companion to :mod:`diffusivity_replicas`. Walks the same 4-replica trajectories
under ``experiments/<exp>/<system>/npt_<conc>_<temp>K_20ns_1ps_replicas/replica_{0..3}/``
(one frame / 1 ps, ~20 ns each) and computes mean +/- std density over the
production window: the whole run **after the first 0.5 ns** of equilibration,
evenly subsampled to ``--density-frames`` (~2000) frames. The start frame,
stride, and frame count are recorded.

To match ``diffusivity_replicas`` exactly, only the **298 K** replicas are
processed (that script analyses 298 K only — experimental viscosity is 298 K).

Per replica it writes ``<replica>/observables/density.txt`` and merges a
``density`` block into the existing ``summary.json`` (leaving any
``diffusivity``/``rdf`` blocks intact). Then it rolls every replica up into
``density_master.csv`` and a per-condition ``density_by_system.csv`` under the
output root.

``--max-time-ns N`` emulates a shorter run: each trajectory is truncated to its
first N ns *before* the same 0.5 ns equilibration cut (so ``--max-time-ns 1``
averages 0.5-1.0 ns, every 1 ps frame). Outputs get a ``_<N>ns`` suffix
(``density_master_1ns.csv``, ``density_by_system_1ns.csv``,
``density_config_1ns.json``, per-replica ``density_1ns.txt`` and a
``density_1ns`` block in ``summary.json``) so the full-run results are untouched.

(RDF computation used to live in this script too, for the 1 M systems only —
that's now a separate, purpose-built script:
``compute_painn_replica_rdfs.py``, which covers all 9 systems with params matched
to the UMA reference RDFs, and writes both the per-replica curves and the
replica mean the figures read.)

CPU only; conda env ``fennol``. Run from the repo root:

    python -m src.analysis.observables.structural.density_replicas --dry-run
    python -m src.analysis.observables.structural.density_replicas --jobs 16
    python -m src.analysis.observables.structural.density_replicas --jobs 16 --max-time-ns 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import io, paths
from ..paths import git_sha
from ..density import DensityAccumulator
from ..systems import SYSTEMS

_REPL_RE = re.compile(r"^npt_(?P<conc>.+?)_(?P<temp>\d+)K_20ns_1ps_replicas$")


def conc_to_molar(tok: str) -> float:
    """'0_1M'->0.1, '0_5M'->0.5, '1M'->1.0."""
    m = re.match(r"^(\d+)_(\d+)M$", tok)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.match(r"^(\d+)M$", tok)
    if m:
        return float(m.group(1))
    raise ValueError(f"Cannot parse concentration token {tok!r}")

# --- fixed analysis settings (see plan / approved decisions) -----------------
SETTINGS = dict(
    temperature_k=298,        # match diffusivity_replicas: 298 K replicas only
    # density: whole run after 0.5 ns equilibration, subsampled to ~2000 frames
    density_eq_time_ps=500.0,
    density_target_frames=2000,
)

DEFAULT_ROOT = paths.PAINN_NPT_REPLICAS_FOR_DENSITIES
OUT_ROOT = paths.OUT_PAINN_STATIC


# ── discovery ────────────────────────────────────────────────────────────────
def discover_runs(root: Path) -> list[dict]:
    """Every 298 K replica with a resolvable trajectory."""
    runs = []
    for sys_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        system_id = sys_dir.name
        if system_id not in SYSTEMS:
            continue
        for repl_dir in sorted(sys_dir.glob("npt_*_replicas")):
            m = _REPL_RE.match(repl_dir.name)
            if not m:
                continue
            temp_k = int(m.group("temp"))
            if temp_k != int(SETTINGS["temperature_k"]):
                continue
            conc_m = conc_to_molar(m.group("conc"))
            run_name = repl_dir.name[: -len("_replicas")]
            for rep_dir in sorted(repl_dir.glob("replica_*")):
                traj = rep_dir / f"{run_name}.traj"
                if not traj.is_file():
                    continue
                runs.append(dict(
                    system_id=system_id, concentration_M=conc_m, temperature_K=temp_k,
                    replica=int(rep_dir.name.split("_")[1]),
                    run_dir=str(rep_dir), traj=str(traj), run_name=run_name,
                ))
    return runs


# ── per-replica compute ────────────────────────────────────────────────────────
def _compute_density(traj_path, dt, total, max_time_ps=None):
    """Mean ± std density over the post-equilibration window, subsampled.

    ``max_time_ps`` truncates the trajectory first (emulates a shorter run)."""
    if max_time_ps is not None:
        total = min(total, int(round(max_time_ps / dt)))
    start = min(max(int(SETTINGS["density_eq_time_ps"] / dt), 0), max(total - 1, 0))
    avail = max(total - start, 1)
    stride = max(1, avail // SETTINGS["density_target_frames"])
    acc = DensityAccumulator()
    for atoms in io.iter_frames(traj_path, start=start, stride=stride,
                                max_count=min(SETTINGS["density_target_frames"],
                                              -(-avail // stride))):
        acc.update(atoms)
    dres = acc.finalize()
    sampling = dict(
        dt_ps=dt, eq_time_ps=SETTINGS["density_eq_time_ps"], start_frame=start,
        stride=stride, n_frames_total=int(total),
        n_frames_requested=SETTINGS["density_target_frames"],
        n_frames_used=dres.n_frames,
        window_ps=round((total - start) * dt, 1),
        max_time_ps=max_time_ps,
    )
    return dres, sampling


def _suffix(max_time_ns) -> str:
    return "" if max_time_ns is None else f"_{max_time_ns:g}ns"


def _worker(rec: dict, force: bool, max_time_ns=None, out_suffix: str = "") -> dict:
    """Compute density for one replica; write the per-replica files; return a
    row for the master CSV."""
    key = "density" + _suffix(max_time_ns) + out_suffix
    max_time_ps = None if max_time_ns is None else max_time_ns * 1000.0
    run_dir = Path(rec["run_dir"])
    obs_dir = run_dir / "observables"
    obs_dir.mkdir(parents=True, exist_ok=True)
    traj_path = Path(rec["traj"])

    summary_path = obs_dir / "summary.json"
    summary = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}

    if not force and key in summary:
        d = summary.get(key, {})
        s = d.get("sampling", {})
        return dict(rec, skipped=True, mean_g_cm3=d.get("mean_g_cm3"),
                    std_g_cm3=d.get("std_g_cm3"), n_frames_used=d.get("n_frames"),
                    start_frame=s.get("start_frame"), stride=s.get("stride"),
                    window_ps=s.get("window_ps"), dt_ps=s.get("dt_ps"))

    spec = SYSTEMS[rec["system_id"]]
    dt, dt_source = io.resolve_dt_ps(run_dir, traj_path)
    total = io.estimate_n_frames(traj_path)

    summary.setdefault("system_id", rec["system_id"])
    summary.update(dict(
        run_name=run_dir.name, cation=spec.cation, anion=spec.anion, solvent=spec.solvent,
        concentration_M=f"{rec['concentration_M']:g}M", temperature_K=f"{rec['temperature_K']}K",
    ))
    prov = summary.setdefault("provenance", {})
    prov.update(dict(traj_path=str(traj_path), dt_ps=dt, dt_source=dt_source,
                     n_frames_total=int(total), code_version=git_sha()))

    out = dict(rec, skipped=False, dt_ps=dt)

    dres, sampling = _compute_density(traj_path, dt, total, max_time_ps)
    (obs_dir / f"{key}.txt").write_text(
        f"frames_used: {dres.n_frames}\n"
        f"window: {dres.n_frames} frames, stride {sampling['stride']}, "
        f"start_frame {sampling['start_frame']} (after {SETTINGS['density_eq_time_ps']:.0f} ps eq), "
        f"{sampling['window_ps']:.0f} ps window\n"
        f"mean_density_g_cm3: {dres.mean_g_cm3:.4f}\n"
        f"std_density_g_cm3: {dres.std_g_cm3:.4f}\n"
    )
    summary[key] = dict(mean_g_cm3=dres.mean_g_cm3, std_g_cm3=dres.std_g_cm3,
                        n_frames=dres.n_frames, sampling=sampling)
    d = summary[key]
    s = d["sampling"]
    out.update(mean_g_cm3=d["mean_g_cm3"], std_g_cm3=d["std_g_cm3"],
               n_frames_used=d["n_frames"], start_frame=s["start_frame"],
               stride=s["stride"], window_ps=s["window_ps"], dt_ps=s["dt_ps"])

    summary_path.write_text(json.dumps(summary, indent=2))
    return out


# ── master / by-system CSVs ────────────────────────────────────────────────────
MASTER_FIELDS = [
    "system_id", "cation", "anion", "solvent", "concentration_M", "temperature_K",
    "replica", "run_name", "mean_density_g_cm3", "std_density_g_cm3",
    "n_frames_used", "start_frame", "stride", "window_ps", "dt_ps", "run_dir",
]


def _master_row(out: dict) -> dict:
    spec = SYSTEMS[out["system_id"]]
    return {
        "system_id": out["system_id"], "cation": spec.cation, "anion": spec.anion,
        "solvent": spec.solvent, "concentration_M": out["concentration_M"],
        "temperature_K": out["temperature_K"], "replica": out["replica"],
        "run_name": out["run_name"], "mean_density_g_cm3": out["mean_g_cm3"],
        "std_density_g_cm3": out["std_g_cm3"], "n_frames_used": out["n_frames_used"],
        "start_frame": out["start_frame"], "stride": out["stride"],
        "window_ps": out["window_ps"], "dt_ps": out["dt_ps"], "run_dir": out["run_dir"],
    }


def write_by_system(master_rows: list, out_path: Path) -> int:
    """Per-condition mean ± std of the replica mean densities."""
    groups: dict = {}
    for r in master_rows:
        groups.setdefault((r["system_id"], r["concentration_M"], r["temperature_K"]), []).append(r)
    fields = ["system_id", "cation", "anion", "solvent", "concentration_M",
              "temperature_K", "n_replicas", "density_mean_g_cm3", "density_std_g_cm3"]
    rows = []
    for (sid, conc, temp), grp in sorted(groups.items()):
        g0 = grp[0]
        dens = np.array([r["mean_density_g_cm3"] for r in grp
                         if r["mean_density_g_cm3"] is not None], dtype=float)
        rows.append({
            "system_id": sid, "cation": g0["cation"], "anion": g0["anion"],
            "solvent": g0["solvent"], "concentration_M": conc, "temperature_K": temp,
            "n_replicas": len(grp),
            "density_mean_g_cm3": float(np.mean(dens)) if dens.size else None,
            "density_std_g_cm3": float(np.std(dens)) if dens.size else None,
        })
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--master-out", type=Path, default=None,
                    help=f"Master CSV path (default: {OUT_ROOT}/density_master.csv).")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-time-ns", type=float, default=None,
                    help="Truncate each replica to its first N ns (emulate a shorter "
                         "run); outputs get a _<N>ns suffix. Default: full run.")
    ap.add_argument("--out-suffix", default="",
                    help="Appended to every output name (master/by-system CSVs, config, "
                         "per-replica files and summary.json key), so a rerun does not "
                         "overwrite the results the figures read.")
    args = ap.parse_args(argv)
    sfx = _suffix(args.max_time_ns) + args.out_suffix

    paths.require(args.experiment_root)
    runs = discover_runs(args.experiment_root)
    print(f"Discovered {len(runs)} replicas (298 K).")

    if args.dry_run:
        by_cond: dict = {}
        for r in runs:
            by_cond.setdefault((r["system_id"], r["concentration_M"]), []).append(r)
        for (sid, conc), grp in sorted(by_cond.items()):
            print(f"  {sid} {conc}M  reps={len(grp)}")
        print(f"\nDRY RUN: {len(runs)} replicas. Nothing computed.")
        return

    out_dir = args.master_out.parent if args.master_out else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    master_out = args.master_out or out_dir / f"density_master{sfx}.csv"

    results: dict = {}
    n = len(runs)
    print(f"[progress] launching {n} replicas on {args.jobs} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_worker, r, args.force, args.max_time_ns, args.out_suffix): i
                for i, r in enumerate(runs)}
        for done, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            r = runs[i]
            tag = f"{r['system_id']} {r['concentration_M']}M rep{r['replica']}"
            try:
                results[i] = out = fut.result()
                flag = "skip" if out.get("skipped") else "done"
                print(f"[progress] {done}/{n}  {tag}  [{flag}] "
                      f"rho={out['mean_g_cm3']:.4f}±{out['std_g_cm3']:.4f} g/cm^3"
                      f" ({out['n_frames_used']} frames)", flush=True)
            except Exception as e:
                print(f"[progress] {done}/{n}  {tag}  ERROR: {e}", flush=True)

    master_rows = [_master_row(results[i]) for i in sorted(results)]
    with open(master_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MASTER_FIELDS)
        w.writeheader()
        w.writerows(master_rows)
    by_system_out = master_out.parent / f"density_by_system{sfx}.csv"
    n_cond = write_by_system(master_rows, by_system_out)

    cfg = dict(
        settings=dict(SETTINGS, max_time_ns=args.max_time_ns), experiment_root=str(args.experiment_root),
        n_replicas=len(master_rows), n_conditions=n_cond,
        n_failed=len(runs) - len(results),
        code_version=git_sha(), generated=datetime.now().isoformat(timespec="seconds"),
    )
    (master_out.parent / f"density_config{sfx}.json").write_text(json.dumps(cfg, indent=2))

    print(f"\nDone: {len(master_rows)} replicas -> {master_out.name}; "
          f"{n_cond} conditions -> {by_system_out.name}; "
          f"{len(runs) - len(results)} failed.")
    print(f"  out dir -> {master_out.parent}")


if __name__ == "__main__":
    main()
