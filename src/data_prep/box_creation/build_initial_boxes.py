"""Rebuild the 16 initial electrolyte boxes with Packmol.

This reproduces the packing described in the appendix section "Initial Box
Setup and System Compositions": every system is a 30 x 30 x 30 A cube holding
the number of ion pairs that corresponds to the target molarity at that fixed
27,000 A^3 volume, with Packmol filling the remaining space with solvent.

    n_pairs = ceil(M_target * N_A / 1e27 * 27000 A^3)
            -> 2 pairs at 0.1 M, 9 at 0.5 M, 17 at 1 M

The counts below are transcribed from the *initial* column of the appendix
composition table rather than recomputed, so the output is byte-for-byte the
same set of systems the paper describes. ``--dry-run`` re-derives the ion-pair
count and the total atom count from the molecule files and checks both against
the table, so a bad molecule file or a mistyped row is caught before Packmol
runs.

These are the boxes that go on to OPLS NPT equilibration at 298 K; they are the
*uncorrected* boxes, which equilibrate over-concentrated (0.15-0.17 M and
1.19-1.32 M against nominal 0.1 M and 1 M). Correcting that is a separate step
and a separate script -- see ``build_boxes_correct_M.py``, which fixes the
solute-to-solvent ratio instead of the packed volume.

Packmol is not on PATH by default; point the PACKMOL env var at the binary
(on Delta: ``PACKMOL=~/miniconda3/envs/packmol/bin/packmol``).

Usage:
    # Show the composition table and verify it against the molecule files:
    python -m src.data_prep.box_creation.build_initial_boxes --dry-run

    # Build all 16 boxes:
    python -m src.data_prep.box_creation.build_initial_boxes

    # Rebuild just the TEGDME systems with a looser packing tolerance:
    python -m src.data_prep.box_creation.build_initial_boxes \
        --systems naotf_tgdme napf6_tgdme --tolerance 1.5 --overwrite
"""
import argparse
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from ase.io import read, write
from tqdm import tqdm

from src.analysis.observables.systems import lookup_system

REPO = Path(__file__).resolve().parents[3]

MOL_DIR = Path(__file__).resolve().parent / "molecules"
OUT_ROOT = REPO / "data" / "reproduced_nitesh_setup_boxes"

# The cube every initial box is packed into. 30 A on a side -> 27,000 A^3.
BOX_LENGTH = 30.0

# packmol binary (not on PATH by default; installed in a dedicated env).
PACKMOL_BIN = os.environ.get("PACKMOL", "packmol")

# molarity <-> number density: 1 mol/L = N_A / 1e27 particles per A^3
MOL_PER_L_TO_PER_A3 = 6.02214076e23 / 1.0e27      # = 6.02214076e-4

# systems.py species keys -> the molecule .xyz files in MOL_DIR
SOLVENT_FILE = {"DME": "dme.xyz", "Diglyme": "diglyme.xyz",
                "PC": "pc.xyz", "TGDME": "tegdme.xyz"}
CATION_FILE = {"Na": "na.xyz", "Li": "li.xyz"}
ANION_FILE = {"PF6": "pf6.xyz", "OTf": "otf.xyz"}

# conc label -> target molarity [mol/L]
CONC_TO_M = {"0_1M": 0.1, "0_5M": 0.5, "1M": 1.0}

# The 16 systems, transcribed from the *initial* column of the appendix
# composition table: (system_id, conc_label, n_pairs, n_solvent, n_atoms).
# n_solvent depends only on the solvent (DME 125, PC 164, DEGDME 94, TEGDME 62);
# n_atoms is the table's value and is re-derived as a check in _plan().
# The table names DEGDME/TEGDME where system_ids say diglyme/tgdme.
SYSTEMS = [
    ("napf6_dme",     "0_1M",  2, 125, 2016),
    ("napf6_dme",     "1M",   17, 125, 2136),
    ("napf6_pc",      "0_1M",  2, 164, 2148),
    ("napf6_pc",      "1M",   17, 164, 2268),
    ("napf6_diglyme", "0_1M",  2,  94, 2178),
    ("napf6_diglyme", "1M",   17,  94, 2298),
    ("napf6_tgdme",   "1M",   17,  62, 2430),
    ("naotf_dme",     "0_1M",  2, 125, 2018),
    ("naotf_dme",     "1M",   17, 125, 2153),
    ("naotf_pc",      "0_1M",  2, 164, 2150),
    ("naotf_pc",      "1M",   17, 164, 2285),
    ("naotf_diglyme", "0_1M",  2,  94, 2180),
    ("naotf_diglyme", "1M",   17,  94, 2315),
    ("naotf_tgdme",   "0_1M",  2,  62, 2312),
    ("naotf_tgdme",   "1M",   17,  62, 2447),
    ("lipf6_dme",     "0_5M",  9, 125, 2072),
]


def pairs_for_molarity(molarity, vol_a3=BOX_LENGTH ** 3):
    """Ion pairs needed for `molarity` at a fixed volume, rounded up.

    Rounding up is what the table's counts follow: 0.1 M -> 1.63 -> 2,
    0.5 M -> 8.13 -> 9, 1 M -> 16.26 -> 17.
    """
    return math.ceil(molarity * MOL_PER_L_TO_PER_A3 * vol_a3)


@lru_cache(maxsize=None)
def _mol_info(fname):
    """(total molar mass [amu], atoms per molecule) for a molecule .xyz."""
    atoms = read(str(MOL_DIR / fname))
    return float(atoms.get_masses().sum()), len(atoms)


def _plan(sys_id, conc, n_pairs, n_solvent, n_atoms_expected):
    """Re-derive the ion-pair count and atom count for one row and check them
    against the table. Returns a dict; `warnings` is empty when the row agrees."""
    spec = lookup_system(sys_id)
    m_target = CONC_TO_M[conc]

    m_cat, n_at_cat = _mol_info(CATION_FILE[spec.cation])
    m_an, n_at_an = _mol_info(ANION_FILE[spec.anion])
    m_solv, n_at_solv = _mol_info(SOLVENT_FILE[spec.solvent])

    n_atoms = n_solvent * n_at_solv + n_pairs * (n_at_cat + n_at_an)
    mass = n_solvent * m_solv + n_pairs * (m_cat + m_an)
    vol = BOX_LENGTH ** 3
    # amu/A^3 -> g/cm^3 is a factor of 1.6605 (= 1e24 * amu in grams)
    rho = mass * 1.66053906660 / vol
    packed_m = n_pairs / (vol * MOL_PER_L_TO_PER_A3)

    warnings = []
    n_pairs_derived = pairs_for_molarity(m_target)
    if n_pairs_derived != n_pairs:
        warnings.append(f"ion pairs: table {n_pairs}, ceil() gives {n_pairs_derived}")
    if n_atoms != n_atoms_expected:
        warnings.append(f"atoms: table {n_atoms_expected}, molecule files give {n_atoms}")

    return dict(spec=spec, n_pairs=n_pairs, n_solvent=n_solvent, n_atoms=n_atoms,
                m_target=m_target, packed_m=packed_m, rho=rho, warnings=warnings)


def packing_blocks(spec, n_solvent, n_pairs):
    """(molecule file, count) in the order Packmol writes them: solvent first,
    then cations, then anions. The order is what lets the output be split back
    into molecules for the contact check."""
    return [
        (MOL_DIR / SOLVENT_FILE[spec.solvent], n_solvent),
        (MOL_DIR / CATION_FILE[spec.cation], n_pairs),
        (MOL_DIR / ANION_FILE[spec.anion], n_pairs),
    ]


def write_packmol_input(inp_path, out_xyz, spec, n_solvent, n_pairs,
                        box_length=BOX_LENGTH, tolerance=2.0, seed=-1):
    """Packmol input packing n_solvent solvent + n_pairs cation/anion into the
    cube.

    Packs with Packmol's `pbc`, so the tolerance is enforced against periodic
    images too. Packing into a plain `inside box` region instead leaves the
    molecules near opposite faces overlapping through the boundary (contacts
    down to ~0.2 A), which the fixed-cell relaxation downstream would have to
    resolve before any dynamics could run."""
    lines = [f"tolerance {tolerance}", "filetype xyz", f"output {out_xyz}",
             f"seed {seed}",
             f"pbc 0. 0. 0. {box_length:.2f} {box_length:.2f} {box_length:.2f}", ""]
    for mol_file, number in packing_blocks(spec, n_solvent, n_pairs):
        lines += [f"structure {mol_file}", f"  number {number}",
                  "end structure", ""]
    inp_path.write_text("\n".join(lines))


def min_intermolecular_distance(atoms, blocks):
    """Shortest distance (A, minimum-image) between atoms of *different*
    molecules. Bonded pairs are excluded, so this is directly comparable to the
    Packmol tolerance."""
    sizes = []
    for mol_file, number in blocks:
        sizes += [_mol_info(mol_file.name)[1]] * number
    mol_id = np.repeat(np.arange(len(sizes)), sizes)
    d = atoms.get_all_distances(mic=True)
    d[mol_id[:, None] == mol_id[None, :]] = np.inf
    return float(d.min())


def _fmt(p, sys_id, conc):
    return (f"{sys_id + ' ' + conc:22s} pairs {p['n_pairs']:3d}  "
            f"solv {p['n_solvent']:4d}  atoms {p['n_atoms']:6d}  "
            f"M {p['m_target']:.2f} (packed {p['packed_m']:.3f})  "
            f"rho {p['rho']:.3f}")


def build_one(task):
    (sys_id, conc, n_pairs, n_solvent, n_atoms_expected,
     tolerance, seed, overwrite, dry_run, out_root) = task
    p = _plan(sys_id, conc, n_pairs, n_solvent, n_atoms_expected)
    info = _fmt(p, sys_id, conc)
    for w in p["warnings"]:
        info += f"\n{'':22s} ** MISMATCH {w}"
    if dry_run:
        return info
    if p["warnings"]:
        return f"{info}\n{'':22s} ** SKIPPED (table mismatch; fix before building)"

    out_dir = out_root / sys_id
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"initial_{conc}.xyz"
    if final_path.exists() and not overwrite:
        return f"{info}  SKIP (exists)"

    inp_path = out_dir / f"packmol_{conc}.inp"
    packed_path = out_dir / f"packed_{conc}.xyz"
    write_packmol_input(inp_path, packed_path, p["spec"], p["n_solvent"],
                        p["n_pairs"], tolerance=tolerance, seed=seed)

    with open(inp_path) as fh:
        result = subprocess.run([PACKMOL_BIN], stdin=fh, capture_output=True,
                                text=True, timeout=3600)
    if result.returncode != 0 or not packed_path.exists():
        return f"{info}  ** FAILED: {(result.stderr or result.stdout)[-300:]}"

    packed = read(str(packed_path))
    if len(packed) != p["n_atoms"]:
        return (f"{info}  ** FAILED: packmol wrote {len(packed)} atoms, "
                f"expected {p['n_atoms']}")
    packed.set_cell([BOX_LENGTH, BOX_LENGTH, BOX_LENGTH])
    packed.set_pbc(True)
    # `pbc` packing may leave a molecule a hair outside the cell; wrap it back in.
    packed.wrap()

    d_min = min_intermolecular_distance(
        packed, packing_blocks(p["spec"], p["n_solvent"], p["n_pairs"]))
    flag = "" if d_min >= 0.9 * tolerance else f"  ** CLOSE CONTACT {d_min:.2f} A"

    write(str(final_path), packed)
    # Packmol emits no cell; final_path is the same coords plus cell/pbc, so the
    # raw output is fully superseded once it is written.
    packed_path.unlink()
    return f"{info}  d_min {d_min:.2f}  OK -> {final_path}{flag}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT,
                    help=f"Where to write <system_id>/initial_<conc>.xyz (default: {OUT_ROOT})")
    ap.add_argument("--systems", nargs="+", default=None,
                    help="Only build these system_ids (e.g. naotf_dme napf6_dme)")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="Packmol min inter-molecular atom distance in A. Lower it "
                         "(e.g. 1.5) if a big solvent like TEGDME will not pack; the "
                         "downstream equilibration removes residual close contacts "
                         "(default: 2.0)")
    ap.add_argument("--seed", type=int, default=-1,
                    help="Packmol random seed; -1 lets Packmol pick one, so runs "
                         "differ. Set it for a bit-reproducible box (default: -1)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rebuild boxes that already exist (default: skip them)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the composition table and verify it against the "
                         "molecule files; build nothing")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel Packmol processes. The whole set takes well "
                         "under a minute serially, and forking several at once "
                         "fails on a busy login node (default: 1)")
    args = ap.parse_args()

    rows = SYSTEMS if not args.systems else [
        r for r in SYSTEMS if r[0] in args.systems]
    if not rows:
        sys.exit(f"No systems matched {args.systems}. "
                 f"Known: {sorted({r[0] for r in SYSTEMS})}")

    print(f"{'Dry run: ' if args.dry_run else ''}Building {len(rows)} initial boxes "
          f"in a {BOX_LENGTH:g} A cube ({BOX_LENGTH**3:.0f} A^3)")
    print(f"Packing: periodic, tolerance={args.tolerance} A, seed={args.seed}")
    print(f"Molecules -> {MOL_DIR}")
    print(f"Output    -> {args.out_root}\n")

    tasks = [(s, c, np_, ns, na, args.tolerance, args.seed,
              args.overwrite, args.dry_run, args.out_root)
             for s, c, np_, ns, na in rows]

    if args.dry_run or args.workers <= 1:
        results = [build_one(t) for t in tqdm(tasks, desc="Systems")]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(tqdm(ex.map(build_one, tasks), total=len(tasks),
                                desc="Building"))

    print()
    for r in results:
        print(r)

    n_bad = sum("MISMATCH" in r for r in results)
    if args.dry_run:
        print(f"\n{len(results) - n_bad}/{len(results)} rows match the "
              f"appendix composition table")
    else:
        n_ok = sum("OK ->" in r for r in results)
        n_fail = sum("FAILED" in r for r in results)
        n_close = sum("CLOSE CONTACT" in r for r in results)
        print(f"\nDone: {n_ok} succeeded, {n_fail} failed")
        if n_close:
            print(f"{n_close} box(es) have an inter-molecular contact well under "
                  f"the {args.tolerance} A tolerance -- rerun those with a "
                  f"different --seed")
    if n_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
