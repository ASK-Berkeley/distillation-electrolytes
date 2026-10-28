"""Measure the equilibrated molarity of every electrolyte box before and after
the solute:solvent ratio correction, and write the small CSV the figure reads.

Two box sets, both measured the same way -- ion pairs divided by the volume the
box actually equilibrates to:

  * initial    -- exp01 boxes, packed by placing the ion pairs for the target
                  molarity into a 30 A cube and filling with solvent. These are
                  the boxes used for teacher sampling and for the student-vs-UMA
                  structural comparison. Read from the equilibrated NVT seed
                  frame (which is itself the final frame of an NPT run, so it is
                  an equilibrated cell despite the name).
  * production -- exp04 boxes from build_boxes_correct_M.py, which fixes the
                  solute:solvent ratio so the target molarity is reached at the
                  model's own equilibrium density. Read as the mean cell volume
                  over the 20 ns production trajectory.

Both values are single-cell measurements despite the wording above: the
production runs are NVT, so the cell is fixed and every frame gives the same
volume. The trajectory average is kept as a check rather than an estimate, and
the start-frame value is reported alongside it -- the two agree to every printed
digit on all 16 systems, which is what a constant cell should look like. If they
ever disagree, the run was not the fixed-cell NVT it is assumed to be.

Ion-pair counts come from counting cation atoms: one cation per formula unit,
and no solvent or anion in this set contains Na or Li.

    python compute_molarity_correction.py
    python compute_molarity_correction.py --stride 50   # denser volume average
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.io.trajectory import TrajectoryReader

from src.analysis.observables import paths

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DATA = paths.DATA_DIR / "experiments"
EXP01_SEEDS = DATA / "exp01_mixed_noise_all_systems" / "nvt_equilibration_seeds"
EXP04 = DATA / "exp04_correct_molarity_samebox"
OUT_CSV = HERE / "molarity_correction.csv"

# 1 mol/L expressed as particles per A^3.
MOL_PER_L_TO_PER_A3 = 6.02214076e-4

TEMPERATURE = 298
REP = 0

# (salt, solvent, conc_label) in the figure's display order: solvent-major
# within each salt, so the panels read the same way as the other appendix
# figures. lipf6_dme exists only at 0.5 M.
SYSTEMS = [
    ("napf6", "dme", "0_1M"), ("napf6", "diglyme", "0_1M"),
    ("napf6", "pc", "0_1M"),
    ("naotf", "dme", "0_1M"), ("naotf", "diglyme", "0_1M"),
    ("naotf", "pc", "0_1M"), ("naotf", "tgdme", "0_1M"),
    ("napf6", "dme", "1M"), ("napf6", "diglyme", "1M"),
    ("napf6", "pc", "1M"), ("napf6", "tgdme", "1M"),
    ("naotf", "dme", "1M"), ("naotf", "diglyme", "1M"),
    ("naotf", "pc", "1M"), ("naotf", "tgdme", "1M"),
    ("lipf6", "dme", "0_5M"),
]

TARGET_M = {"0_1M": 0.1, "1M": 1.0, "0_5M": 0.5}
# Which panel a system is drawn in. lipf6_dme is 0.5 M but groups with the
# concentrated set, matching the other appendix figures.
GROUP = {"0_1M": "0.1M", "1M": "1M", "0_5M": "1M"}


def n_ion_pairs(atoms) -> int:
    sym = atoms.get_chemical_symbols()
    return sym.count("Na") + sym.count("Li")


def molarity(n_pairs: int, volume_a3: float) -> float:
    return n_pairs / (volume_a3 * MOL_PER_L_TO_PER_A3)


def measure(salt: str, solvent: str, conc: str, stride: int) -> dict:
    system = f"{salt}_{solvent}"

    seed = EXP01_SEEDS / f"{system}_{conc}_{TEMPERATURE}K.xyz"
    initial = read(str(seed))
    n_pairs = n_ion_pairs(initial)
    m_initial = molarity(n_pairs, initial.get_volume())

    run = EXP04 / system / f"transport_{conc}_{TEMPERATURE}K_r{REP}"
    traj = TrajectoryReader(str(run / f"{run.name}.traj"))
    n_frames = len(traj)
    idx = range(0, n_frames, max(1, n_frames // stride))
    volumes = np.array([traj[i].get_volume() for i in idx])

    n_pairs_prod = n_ion_pairs(traj[0])
    if n_pairs_prod != n_pairs:
        # The correction changes the solvent count only; a differing ion-pair
        # count means the two boxes are not the same system.
        raise ValueError(f"{system} {conc}: ion pairs {n_pairs} -> {n_pairs_prod}")

    return {
        "salt": salt,
        "solvent": solvent,
        "conc": conc,
        "group": GROUP[conc],
        "target_M": TARGET_M[conc],
        "n_ion_pairs": n_pairs,
        "m_initial": m_initial,
        "m_production": molarity(n_pairs, volumes.mean()),
        # Reported so the frame-vs-mean inconsistency described above is visible.
        "m_production_start": molarity(n_pairs, traj[0].get_volume()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stride", type=int, default=200,
                    help="approx. number of frames to average the cell over")
    ap.add_argument("--output", default=str(OUT_CSV))
    args = ap.parse_args()

    rows, missing = [], []
    for salt, solvent, conc in SYSTEMS:
        try:
            row = measure(salt, solvent, conc, args.stride)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  SKIP {salt}_{solvent} {conc}: {exc}", file=sys.stderr)
            missing.append(f"{salt}_{solvent}:{conc}")
            continue
        rows.append(row)
        drift = abs(row["m_production"] - row["m_production_start"])
        print(f"  {salt}_{solvent:9s} {conc:5s} target {row['target_M']:.1f}  "
              f"{row['m_initial']:.3f} -> {row['m_production']:.3f} M  "
              f"(start-frame differs by {drift:.4f})")

    if not rows:
        print("nothing measured", file=sys.stderr)
        return 1

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.output}")
    if missing:
        print(f"MISSING {len(missing)}: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
