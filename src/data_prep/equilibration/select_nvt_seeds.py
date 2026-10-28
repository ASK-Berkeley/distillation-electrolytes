#!/usr/bin/env python
"""Pick per-system NVT-equilibration seed frames from the NPT replica runs.

For each system/concentration/temperature condition we take **replica 0**'s NPT
trajectory, look at its first ``--eq-window-ns`` (default 4 ns), and:

  1. average the mass density over the *density window* (default the first 3 ns),
     giving a target equilibrium density rho_bar;
  2. from the *selection window* (default the last 1 ns, i.e. 3-4 ns) pick the
     single frame whose instantaneous density is closest to rho_bar.

Because the mass is fixed, "closest to the mean density" == "closest to the mean
*volume*" — so the chosen frame is a real, un-strained equilibrium configuration
sitting at the mean box size. That frame (positions + cell) is written as an
extxyz seed compatible with ``fennol_npt.py --from-xyz`` for a subsequent NVT
relaxation → transport production run.

Seeds + a manifest land in ``<experiment-root>/<out-subdir>/`` (default
``nvt_equilibration_seeds/``). The manifest also reports the density averaged over
1-ns sub-windows so you can eyeball whether the density is still relaxing early in
the window (see ``--density-start-ns``).

CPU only; conda env ``fennol``. Run from the repo root:

    python -m src.data_prep.equilibration.select_nvt_seeds --dry-run
    python -m src.data_prep.equilibration.select_nvt_seeds --jobs 16
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory
from tqdm import tqdm

from src.analysis.observables.io import resolve_dt_ps
from src.analysis.observables.systems import SYSTEMS
from src.analysis.observables.units import density_g_cm3

DEFAULT_ROOT = Path(
    "/pscratch/sd/i/ishan_a/electrolytes_data/experiments/exp01_mixed_noise_all_systems"
)


@dataclass
class Condition:
    """One replica-0 NPT run to draw a seed from."""
    system_id: str
    run_name: str          # e.g. npt_1M_298K_20ns_1ps
    conc_tok: str          # e.g. 1M
    temp_k: int
    rep_dir: str           # .../replica_0
    traj: str              # .../<run_name>.traj


def discover_flat(root: Path, run_glob: str, run_prefix: str,
                  temperatures: set[int] | None = None) -> list[Condition]:
    """Flat single-run layout (exp03): ``<system>/<run>/<run>.traj`` — no replica
    folders, one NPT-equilibration run per system/concentration.

    The run name is ``<run_prefix>_<conc>_<temp>K_<rest>`` (e.g.
    ``npt_equil_1M_298K_1ns``); ``<conc>`` may contain an underscore
    (``0_1M`` = 0.1 M) so it is parsed non-greedily after the fixed prefix.
    ``rep_dir`` is the run directory itself (there is no replica nesting).
    """
    run_re = re.compile(rf"^{re.escape(run_prefix)}_(?P<conc>.+?)_(?P<temp>\d+)K(?:_.*)?$")
    conds: list[Condition] = []
    for sys_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if sys_dir.name not in SYSTEMS:
            continue
        for run_dir in sorted(sys_dir.glob(run_glob)):
            if not run_dir.is_dir():
                continue
            m = run_re.match(run_dir.name)
            if not m:
                continue
            conc_tok, temp_k = m.group("conc"), int(m.group("temp"))
            if temperatures is not None and temp_k not in temperatures:
                continue
            traj = run_dir / f"{run_dir.name}.traj"
            if not traj.is_file():
                continue
            conds.append(Condition(sys_dir.name, run_dir.name, conc_tok, temp_k,
                                    str(run_dir), str(traj)))
    return conds


def discover(root: Path, ensemble: str, replica: int,
             temperatures: set[int] | None = None) -> list[Condition]:
    """Every ``<system>/<ensemble>_*_replicas/replica_<replica>/<run>.traj`` under root.

    The run name is ``<ensemble>_<conc>_<temp>K_<...>``, where ``<conc>`` may itself
    contain an underscore (``0_1M`` = 0.1 M, ``0_5M`` = 0.5 M, ``1M`` = 1 M) — so it
    is parsed with a regex, not a naive split. ``temperatures`` (if given) keeps only
    those temperatures in K.
    """
    # <ensemble>_(conc)_(temp)K_<rest>  ; conc is non-greedy up to the _<digits>K token
    repl_re = re.compile(rf"^{ensemble}_(?P<conc>.+?)_(?P<temp>\d+)K(?:_.*)?$")
    conds: list[Condition] = []
    for sys_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if sys_dir.name not in SYSTEMS:
            continue
        for repl_dir in sorted(sys_dir.glob(f"{ensemble}_*_replicas")):
            run_name = repl_dir.name[: -len("_replicas")]
            m = repl_re.match(run_name)
            if not m:
                continue
            conc_tok, temp_k = m.group("conc"), int(m.group("temp"))
            if temperatures is not None and temp_k not in temperatures:
                continue
            rep_dir = repl_dir / f"replica_{replica}"
            traj = rep_dir / f"{run_name}.traj"
            if not traj.is_file():
                continue
            conds.append(Condition(sys_dir.name, run_name, conc_tok, temp_k,
                                    str(rep_dir), str(traj)))
    return conds


def _frame_volumes(traj_path: str, n_frames: int) -> tuple[np.ndarray, float]:
    """Per-frame cell volume (A^3) for the first ``n_frames`` frames, + total mass (amu)."""
    vols = np.empty(n_frames, dtype=float)
    mass_amu = None
    traj = Trajectory(traj_path, mode="r")
    try:
        for i in range(n_frames):
            atoms = traj[i]
            vols[i] = float(atoms.get_volume())
            if mass_amu is None:
                mass_amu = float(np.sum(atoms.get_masses()))
    finally:
        traj.close()
    return vols, mass_amu


def select_one(cond: Condition, args) -> dict:
    """Compute rho_bar over the density window and pick the closest frame in the
    selection window. Returns a manifest row (and writes the seed unless dry-run)."""
    dt_ps, dt_src = resolve_dt_ps(cond.rep_dir, cond.traj, override=args.dt_ps,
                                  fallback=args.dt_ps_fallback)

    n_avail = len(Trajectory(cond.traj, mode="r"))
    f_end = int(round(args.eq_window_ns * 1000.0 / dt_ps))
    if n_avail < f_end:
        return dict(cond=cond, status="skip",
                    reason=f"only {n_avail} frames ({n_avail * dt_ps / 1000:.2f} ns) "
                           f"< eq window {args.eq_window_ns} ns")

    dstart = int(round(args.density_start_ns * 1000.0 / dt_ps))
    dend = int(round(args.density_end_ns * 1000.0 / dt_ps))
    sstart = int(round(args.select_start_ns * 1000.0 / dt_ps))
    send = int(round(args.select_end_ns * 1000.0 / dt_ps))

    vols, mass_amu = _frame_volumes(cond.traj, f_end)
    rho = np.array([density_g_cm3(mass_amu, v) for v in vols])   # per-frame g/cm^3

    rho_bar = float(rho[dstart:dend].mean())
    sel = np.arange(sstart, send)
    chosen = int(sel[np.argmin(np.abs(rho[sel] - rho_bar))])

    # 1-ns sub-window densities across the eq window — to spot an early transient.
    subwin = {}
    for a_ns in range(int(args.eq_window_ns)):
        a = int(round(a_ns * 1000.0 / dt_ps))
        b = int(round((a_ns + 1) * 1000.0 / dt_ps))
        subwin[f"rho_{a_ns}_{a_ns + 1}ns"] = round(float(rho[a:b].mean()), 5)

    if args.layout == "flat":
        # exp03: write the seed into the run's own system folder, next to the
        # npt_equil run it was drawn from, named by molarity/temperature.
        out_name = args.seed_name_tmpl.format(conc=cond.conc_tok, temp=cond.temp_k)
        out_path = Path(args.experiment_root) / cond.system_id / out_name
    else:
        out_name = f"{cond.system_id}_{cond.conc_tok}_{cond.temp_k}K.xyz"
        out_path = Path(args.experiment_root) / args.out_subdir / out_name

    if not args.dry_run:
        atoms = Trajectory(cond.traj, mode="r")[chosen]
        # Clean copy: positions + cell only (drop forces/momenta/calculator) so the
        # seed matches what fennol_npt.py expects for a fresh --from-xyz start.
        clean = Atoms(symbols=atoms.get_chemical_symbols(),
                      positions=atoms.get_positions(),
                      cell=atoms.get_cell(), pbc=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ase_write(str(out_path), clean, format="extxyz")

    cell = np.asarray(Trajectory(cond.traj, mode="r")[chosen].get_cell()) if not args.dry_run else None
    row = dict(
        system_id=cond.system_id, concentration=cond.conc_tok, temperature_K=cond.temp_k,
        run_name=cond.run_name, replica=args.replica,
        dt_ps=round(dt_ps, 4), dt_source=dt_src, n_frames_avail=n_avail,
        density_window_ns=f"[{args.density_start_ns},{args.density_end_ns})",
        select_window_ns=f"[{args.select_start_ns},{args.select_end_ns})",
        target_density_g_cm3=round(rho_bar, 5),
        chosen_frame=chosen, chosen_time_ps=round(chosen * dt_ps, 3),
        chosen_density_g_cm3=round(float(rho[chosen]), 5),
        density_abs_err_g_cm3=round(abs(float(rho[chosen]) - rho_bar), 6),
        chosen_volume_A3=round(float(vols[chosen]), 3),
        L_avg_A=round(float(np.diag(cell).mean()), 4) if cell is not None else None,
        seed_path=str(out_path), status="ok",
    )
    row.update(subwin)
    return dict(cond=cond, status="ok", row=row)


MANIFEST_FIELDS = [
    "system_id", "concentration", "temperature_K", "run_name", "replica",
    "dt_ps", "dt_source", "n_frames_avail", "density_window_ns", "select_window_ns",
    "target_density_g_cm3", "chosen_frame", "chosen_time_ps", "chosen_density_g_cm3",
    "density_abs_err_g_cm3", "chosen_volume_A3", "L_avg_A", "seed_path",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--layout", choices=("replica", "flat"), default="replica",
                    help="run layout. 'replica' (default): exp01 "
                         "<sys>/<ens>_*_replicas/replica_N/. 'flat': exp03 "
                         "<sys>/<run>/<run>.traj single runs, seeds written per-system.")
    ap.add_argument("--run-glob", default="npt_equil_*",
                    help="[flat] glob for run dirs under each system (default: npt_equil_*)")
    ap.add_argument("--run-prefix", default="npt_equil",
                    help="[flat] run-name prefix before _<conc>_<temp>K (default: npt_equil)")
    ap.add_argument("--seed-name-tmpl", default="nvt_seed_{conc}_{temp}K.xyz",
                    help="[flat] per-system seed filename template (default: "
                         "nvt_seed_{conc}_{temp}K.xyz)")
    ap.add_argument("--ensemble", default="npt", help="replica folder prefix (default: npt)")
    ap.add_argument("--replica", type=int, default=0, help="which replica to seed from (default: 0)")
    ap.add_argument("--out-subdir", default="nvt_equilibration_seeds",
                    help="subfolder under the experiment root for seeds + manifest")
    ap.add_argument("--temperatures", default="298",
                    help="comma-separated temperatures (K) to keep, or 'all' (default: 298)")
    ap.add_argument("--eq-window-ns", type=float, default=4.0,
                    help="total window inspected from the start of each run (default: 4)")
    ap.add_argument("--density-start-ns", type=float, default=1.0,
                    help="start of the density-averaging window (default: 1; drops the "
                         "least-equilibrated first ns)")
    ap.add_argument("--density-end-ns", type=float, default=4.0,
                    help="end of the density-averaging window (default: 4; overlaps the "
                         "selection window so rho_bar is centered on the plateau)")
    ap.add_argument("--select-start-ns", type=float, default=3.0,
                    help="start of the frame-selection window (default: 3)")
    ap.add_argument("--select-end-ns", type=float, default=4.0,
                    help="end of the frame-selection window (default: 4)")
    ap.add_argument("--dt-ps", type=float, default=None,
                    help="override ps/frame (else auto-detected; these runs are 1 ps)")
    ap.add_argument("--dt-ps-fallback", type=float, default=1.0,
                    help="ps/frame used if auto-detection falls back (default: 1.0)")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--dry-run", action="store_true",
                    help="discover + compute selections but write no seeds/manifest")
    args = ap.parse_args(argv)

    temps = None if args.temperatures.strip().lower() == "all" else {
        int(t) for t in args.temperatures.split(",")}
    if args.layout == "flat":
        conds = discover_flat(args.experiment_root, args.run_glob, args.run_prefix, temps)
        print(f"Discovered {len(conds)} flat '{args.run_prefix}' run(s) under "
              f"{args.experiment_root}"
              + (f" (temperatures={sorted(temps)} K)" if temps else ""))
    else:
        conds = discover(args.experiment_root, args.ensemble, args.replica, temps)
        print(f"Discovered {len(conds)} replica-{args.replica} {args.ensemble} run(s) under "
              f"{args.experiment_root}"
              + (f" (temperatures={sorted(temps)} K)" if temps else ""))
    if not conds:
        return

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(select_one, c, args): c for c in conds}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="selecting seeds"):
            results.append(fut.result())

    ok = [r["row"] for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skip"]
    ok.sort(key=lambda r: (r["system_id"], r["concentration"], r["temperature_K"]))

    for r in ok:
        print(f"  {r['system_id']:>16} {r['concentration']:>4} {r['temperature_K']}K  "
              f"rho_bar={r['target_density_g_cm3']:.4f}  "
              f"-> frame {r['chosen_frame']} @ {r['chosen_time_ps']:.0f} ps "
              f"(rho={r['chosen_density_g_cm3']:.4f}, |drho|={r['density_abs_err_g_cm3']:.5f})")
    for r in skipped:
        print(f"  SKIP {r['cond'].system_id} {r['cond'].conc_tok} {r['cond'].temp_k}K: {r['reason']}")

    if args.dry_run:
        print(f"\nDRY RUN: {len(ok)} seeds would be written, {len(skipped)} skipped. "
              f"Nothing saved.")
        return

    out_dir = args.experiment_root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    # union of fixed fields + any sub-window columns present
    extra = [k for k in (ok[0].keys() if ok else []) if k not in MANIFEST_FIELDS]
    with open(out_dir / "seeds_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS + extra)
        w.writeheader()
        w.writerows(ok)

    meta = dict(
        generated=datetime.now().isoformat(timespec="seconds"),
        experiment_root=str(args.experiment_root), ensemble=args.ensemble,
        replica=args.replica, eq_window_ns=args.eq_window_ns,
        density_window_ns=[args.density_start_ns, args.density_end_ns],
        select_window_ns=[args.select_start_ns, args.select_end_ns],
        n_seeds=len(ok), n_skipped=len(skipped),
    )
    with open(out_dir / "seeds_config.txt", "w") as fh:
        for k, v in meta.items():
            fh.write(f"{k}: {v}\n")

    print(f"\nDone: {len(ok)} seeds -> {out_dir}/  (manifest: seeds_manifest.csv), "
          f"{len(skipped)} skipped.")


if __name__ == "__main__":
    main()
