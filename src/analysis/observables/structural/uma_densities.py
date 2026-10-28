#!/usr/bin/env python
"""UMA-OMol (teacher) mass densities from the 298 K / 1 ns NPT runs.

Reads one trajectory per system, ``<root>/<system>_<conc>/<system>_<conc>.traj``
(1 ps frames, written by ``src/simulate/fairchem_configs/launch_uma_298K_density.sh``),
and averages the density over the last ``--avg-ps`` of each run (default 500 ps,
i.e. the second half). Frames are taken from the end of the file, so a frame
duplicated by a requeue restart does not shift the window.

Writes ``densities_298K_1ns.csv`` with one row per system (``System, UMA-OMol,
Expt, std_g_cm3, n_frames_avg``; densities in g/cm^3) and a
``densities_298K_1ns_config.yaml`` sidecar. ``Expt`` is the measured density
from ``--exp-csv``. Figure 2b and the SI density table read this file.

CPU only; conda env ``fennol``. Run from the repo root:

    python -m src.analysis.observables.structural.uma_densities --dry-run
    python -m src.analysis.observables.structural.uma_densities --jobs 8
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml
from ase.io import read

from .. import paths
from ..density import DensityAccumulator
from ..systems import SYSTEMS
from .density_replicas import conc_to_molar

SALT_DISP = {"napf6": "NaPF6", "naotf": "NaOTf", "lipf6": "LiPF6"}
SOLV_DISP = {"dme": "DME", "pc": "PC", "diglyme": "DEGDME", "tgdme": "TEGDME"}

DEFAULT_EXP_CSV = paths.GROUND_TRUTH / "298p2K_exp_values.csv"


def find_runs(root: Path) -> list[tuple[str, float, Path]]:
    """(system_id, conc_M, traj) for every ``<system>_<conc>/`` run under ``root``,
    in registry order then by concentration."""
    runs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        system_id, _, conc_tok = d.name.partition("_")
        solvent, _, conc_tok = conc_tok.partition("_")
        system_id = f"{system_id}_{solvent}"
        if system_id not in SYSTEMS:
            continue
        traj = d / f"{d.name}.traj"
        if traj.is_file():
            runs.append((system_id, conc_to_molar(conc_tok), traj))
    order = list(SYSTEMS)
    return sorted(runs, key=lambda r: (order.index(r[0]), r[1]))


def label(system_id: str, conc_M: float) -> str:
    """'napf6_dme', 1.0 -> '1.0 M NaPF6 in DME' (the ground-truth CSV's naming)."""
    salt, solvent = system_id.split("_")
    return f"{conc_M:.1f} M {SALT_DISP[salt]} in {SOLV_DISP[solvent]}"


def load_exp(path: Path) -> dict[str, float]:
    with open(path, newline="") as fh:
        return {r["Sample"].strip(): float(r["Density (g/mL)"])
                for r in csv.DictReader(fh) if r["Density (g/mL)"].strip()}


def density_of(traj: Path, avg_ps: float) -> tuple[float, float, int, float]:
    """(mean, std, n_frames, frame_dt_ps) over the last ``avg_ps`` of ``traj``."""
    frames = read(traj, ":")
    dt_ps = 1.0   # launch_uma_298K_density.sh writes one frame per ps
    n = int(round(avg_ps / dt_ps))
    if len(frames) < n:
        raise ValueError(f"{traj}: {len(frames)} frames, need {n} for {avg_ps} ps")
    acc = DensityAccumulator()
    for atoms in frames[-n:]:
        acc.update(atoms)
    res = acc.finalize()
    return res.mean_g_cm3, res.std_g_cm3, res.n_frames, dt_ps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=paths.UMA_298K_1ns,
                    help="Folder holding one <system>_<conc>/ run per system.")
    ap.add_argument("--exp-csv", type=Path, default=DEFAULT_EXP_CSV,
                    help="Measured densities (column 'Density (g/mL)').")
    ap.add_argument("--temperature", type=float, default=298.0,
                    help="Simulation temperature in K (recorded in the sidecar).")
    ap.add_argument("--avg-ps", type=float, default=500.0,
                    help="Average over the last this many ps of each run.")
    ap.add_argument("--out-dir", type=Path, default=paths.OUT_UMA_298K_DENSITY)
    ap.add_argument("--out-suffix", default="",
                    help="Appended to the output file names.")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    paths.require(args.root, "UMA_298K_1ns")
    runs = find_runs(args.root)
    exp = load_exp(args.exp_csv)
    print(f"{len(runs)} runs under {args.root}")
    for system_id, conc, traj in runs:
        lab = label(system_id, conc)
        print(f"  {lab:24s} exp={exp.get(lab, float('nan')):.5f}  {traj}")
    if args.dry_run:
        return

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(density_of, [r[2] for r in runs],
                                [args.avg_ps] * len(runs)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"densities_298K_1ns{args.out_suffix}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["System", "UMA-OMol", "Expt", "std_g_cm3", "n_frames_avg"])
        for (system_id, conc, _), (mean, std, n, _) in zip(runs, results):
            lab = label(system_id, conc)
            w.writerow([lab, f"{mean:.4f}", exp.get(lab, ""), f"{std:.4f}", n])

    cfg = dict(
        script="src/analysis/observables/structural/uma_densities.py",
        code_version=paths.git_sha(),
        generated=datetime.now().isoformat(timespec="seconds"),
        root=str(args.root), exp_csv=str(args.exp_csv),
        temperature_K=args.temperature, avg_window_ps=args.avg_ps,
        frame_dt_ps=results[0][3] if results else None,
        model="uma-s-1p1 (OMol task)",
        runs=[str(r[2]) for r in runs],
    )
    (args.out_dir / f"densities_298K_1ns{args.out_suffix}_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
