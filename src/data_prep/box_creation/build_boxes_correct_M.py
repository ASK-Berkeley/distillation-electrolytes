"""
Build correct-molarity simulation boxes for all electrolyte systems using Packmol.

The exp01 boxes came out over-concentrated (the nominal "1 M" boxes sit at
~1.2-1.3 M, "0.1 M" at ~0.16 M) because they held too few solvent molecules per
ion pair. This script fixes the *ratio*: for each system it reads the model's
equilibrium density from the exp01 NVT seed frame, then solves a mass balance so
that, at that same density, the molarity is exactly the target. The ion-pair
count is kept as is by default; ``--salt-factor`` scales it.

Per (system, concentration):
    n_salt   = salt_factor * n_salt_current            (default 1x)
    V_target = n_salt / (M_target * N_A_per_A3)         # A^3 at the target molarity
    mass_tot = density_from_seed * V_target             # amu the box must hold
    n_solvent = round((mass_tot - n_salt*m_pair) / m_solvent)

The packed box VOLUME is only a loose starting point (packed at
``--pack-density-frac`` = 80% of the target density by default, so Packmol and
the relaxation don't hit overlaps): the downstream pipeline is UMA fixed-cell
relaxation then an NPT equilibration, and the barostat sets the true box/density.
Getting the molecule *ratio* right is what makes the equilibrated molarity correct.

Outputs mirror the exp01 per-system layout:
    <exp04>/<system_id>/initial_<conc>.xyz

Usage:
    # Dry run -- show counts, densities, and achieved molarity, build nothing:
    python -m src.data_prep.box_creation.build_boxes_correct_M --dry-run

    # Build all 16 systems:
    python -m src.data_prep.box_creation.build_boxes_correct_M

    # Build only the DME systems, quintuple the salt:
    python -m src.data_prep.box_creation.build_boxes_correct_M --systems naotf_dme napf6_dme --salt-factor 5

    # Target the OPLS ground-truth densities instead of the exp01 seed densities
    # (outputs to exp05_opls_densities/):
    python -m src.data_prep.box_creation.build_boxes_correct_M --use-opls --dry-run
"""
import argparse
import csv
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from ase.io import read, write
from tqdm import tqdm

from src.analysis.observables.units import (density_g_cm3, AMU_TO_KG,
                                            ANGSTROM3_TO_M3)
from src.analysis.observables.systems import lookup_system
from src.analysis.observables import paths

REPO = Path(__file__).resolve().parents[3]

DATA_ROOT = paths.DATA_DIR
EXP01_ROOT = DATA_ROOT / "experiments" / "exp01_mixed_noise_all_systems"
SEED_DIR = EXP01_ROOT / "nvt_equilibration_seeds"
OUT_ROOT = DATA_ROOT / "experiments" / "exp04_correct_molarity_samebox"
# When targeting OPLS ground-truth densities instead of the exp01 seed densities.
OUT_ROOT_OPLS = DATA_ROOT / "experiments" / "exp05_opls_densities"
MOL_DIR = Path(__file__).resolve().parent / "molecules"

# OPLS ground-truth densities (used with --use-opls). This file has all 16 exp01
# systems and an explicit OPLS_density column.
OPLS_CSV = REPO / "src" / "analysis" / "observables" / "ground_truth" / \
    "opls_density_comparison.csv"
# CSVs name solvents differently from system_ids: 'tegdme'->'tgdme',
# 'degdme'->'diglyme'. Normalize to the system_id solvent token.
SOLVENT_ALIAS = {"tegdme": "tgdme", "degdme": "diglyme"}

# The 16 exp01 (system_id, conc_label) pairs -- matches launch_transport.sh.
SYSTEMS = [
    ("lipf6_dme", "0_5M"),
    ("naotf_diglyme", "0_1M"), ("naotf_diglyme", "1M"),
    ("naotf_dme", "0_1M"),     ("naotf_dme", "1M"),
    ("naotf_pc", "0_1M"),      ("naotf_pc", "1M"),
    ("naotf_tgdme", "0_1M"),   ("naotf_tgdme", "1M"),
    ("napf6_diglyme", "0_1M"), ("napf6_diglyme", "1M"),
    ("napf6_dme", "0_1M"),     ("napf6_dme", "1M"),
    ("napf6_pc", "0_1M"),      ("napf6_pc", "1M"),
    ("napf6_tgdme", "1M"),
]

# conc label -> target molarity [mol/L]
CONC_TO_M = {"0_1M": 0.1, "1M": 1.0, "0_5M": 0.5}


def _conc_label_from_M(conc_m):
    """Map a numeric concentration (e.g. the CSV's 0.12) to the nearest of our
    canonical conc labels (0_1M / 0_5M / 1M)."""
    return min(CONC_TO_M, key=lambda k: abs(CONC_TO_M[k] - conc_m))


def _opls_key(sys_id, conc):
    """(salt, solvent, conc_label) key for looking a system up in the OPLS CSV."""
    salt, solv = sys_id.split("_", 1)
    solv = solv.lower()
    return (salt.lower(), SOLVENT_ALIAS.get(solv, solv), conc)


def _col(row, *candidates):
    """First matching column value in a CSV row (tolerates the two OPLS CSV
    header schemes: 'salt'/'Salt', 'mean_density_g_cm3'/'OPLS_density (g/cm3)')."""
    for c in candidates:
        if c in row:
            return row[c]
    raise KeyError(f"none of {candidates} in CSV columns {list(row)}")


def load_opls_densities(csv_path, temperature=298):
    """Read an OPLS ground-truth CSV -> {(salt, solvent, conc_label): rho_g_cm3}
    for rows at the requested temperature."""
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if round(float(_col(row, "temperature_K", "T (K)"))) != round(temperature):
                continue
            salt = _col(row, "salt", "Salt").strip().lower()
            solv = _col(row, "solvent", "Solvent").strip().lower()
            solv = SOLVENT_ALIAS.get(solv, solv)
            label = _conc_label_from_M(float(_col(row, "concentration_M", "Conc (M)")))
            rho = float(_col(row, "mean_density_g_cm3", "OPLS_density (g/cm3)"))
            out[(salt, solv, label)] = rho
    return out

# systems.py species keys -> the molecule .xyz files in MOL_DIR
SOLVENT_FILE = {"DME": "dme.xyz", "Diglyme": "diglyme.xyz",
                "PC": "pc.xyz", "TGDME": "tegdme.xyz"}
CATION_FILE = {"Na": "na.xyz", "Li": "li.xyz"}
ANION_FILE = {"PF6": "pf6.xyz", "OTf": "otf.xyz"}

# packmol binary (not on PATH by default on Delta; installed in a dedicated env).
# Override with the PACKMOL env var, e.g. PACKMOL=/path/to/packmol.
PACKMOL_BIN = os.environ.get("PACKMOL", "packmol")

# molarity <-> number density: 1 mol/L = N_A / 1e27 particles per A^3
MOL_PER_L_TO_PER_A3 = 6.02214076e23 / 1.0e27      # = 6.02214076e-4


def mass_amu_from_density(rho_g_cm3, vol_a3):
    """Inverse of density_g_cm3: mass (amu) that fills vol_a3 at rho_g_cm3."""
    return rho_g_cm3 * vol_a3 * ANGSTROM3_TO_M3 * 1000.0 / AMU_TO_KG


def vol_a3_from_mass_density(mass_amu, rho_g_cm3):
    """Volume (A^3) occupied by mass_amu at density rho_g_cm3."""
    return mass_amu * AMU_TO_KG / (rho_g_cm3 * ANGSTROM3_TO_M3 * 1000.0)


def molarity(n_salt, vol_a3):
    return n_salt / (vol_a3 * MOL_PER_L_TO_PER_A3)


@lru_cache(maxsize=None)
def _mol_info(fname):
    """(total molar mass [amu], atoms per molecule) for a molecule .xyz."""
    atoms = read(str(MOL_DIR / fname))
    return float(atoms.get_masses().sum()), len(atoms)


def plan_box(sys_id, conc, salt_factor, rho_override=None):
    """Solve counts + a loose starting box for one (system, conc). No I/O beyond
    reading the seed frame and the molecule files.

    rho_override: if given (e.g. an OPLS ground-truth density), use it as the
    target density instead of the exp01 seed frame's equilibrium density. The
    seed frame is still read for the current ion-pair count."""
    spec = lookup_system(sys_id)
    seed = SEED_DIR / f"{sys_id}_{conc}_298K.xyz"
    atoms = read(str(seed))
    rho = (rho_override if rho_override is not None
           else density_g_cm3(atoms.get_masses().sum(), atoms.get_volume()))

    # current ion-pair count = number of cation atoms (1 cation per pair;
    # solvents are C/H/O only and anions carry no Na/Li).
    n_salt_cur = atoms.get_chemical_symbols().count(spec.cation)
    n_salt = max(1, round(n_salt_cur * salt_factor))

    m_cat, _ = _mol_info(CATION_FILE[spec.cation])
    m_an, _ = _mol_info(ANION_FILE[spec.anion])
    m_solv, n_at_solv = _mol_info(SOLVENT_FILE[spec.solvent])
    m_pair = m_cat + m_an

    M_target = CONC_TO_M[conc]
    v_target = n_salt / (M_target * MOL_PER_L_TO_PER_A3)
    mass_tot = mass_amu_from_density(rho, v_target)
    n_solvent = max(1, round((mass_tot - n_salt * m_pair) / m_solv))

    # achieved molarity once solvent count is rounded to an integer
    mass_actual = n_salt * m_pair + n_solvent * m_solv
    v_eq = vol_a3_from_mass_density(mass_actual, rho)
    m_actual = molarity(n_salt, v_eq)

    n_atoms = n_solvent * n_at_solv + n_salt * (1 + _mol_info(ANION_FILE[spec.anion])[1])
    return dict(spec=spec, rho=rho, n_salt_cur=n_salt_cur, n_salt=n_salt,
                n_solvent=n_solvent, M_target=M_target, M_actual=m_actual,
                v_eq=v_eq, n_atoms=n_atoms, m_solv=m_solv, m_pair=m_pair)


def write_packmol_input(inp_path, out_xyz, spec, n_solvent, n_salt, box_length,
                        tolerance=2.0):
    lo, hi = 2.0, box_length - 2.0
    blocks = [
        (MOL_DIR / SOLVENT_FILE[spec.solvent], n_solvent),
        (MOL_DIR / CATION_FILE[spec.cation], n_salt),
        (MOL_DIR / ANION_FILE[spec.anion], n_salt),
    ]
    lines = [f"tolerance {tolerance}", "filetype xyz", f"output {out_xyz}", ""]
    for mol_file, number in blocks:
        lines += [f"structure {mol_file}", f"  number {number}",
                  f"  inside box {lo:.1f} {lo:.1f} {lo:.1f} {hi:.1f} {hi:.1f} {hi:.1f}",
                  "end structure", ""]
    inp_path.write_text("\n".join(lines))


def _fmt(p, sys_id, conc):
    return (f"{sys_id+' '+conc:22s} rho={p['rho']:.4f}  "
            f"salt {p['n_salt_cur']:2d}->{p['n_salt']:3d}  solv {p['n_solvent']:4d}  "
            f"atoms {p['n_atoms']:6d}  M {p['M_target']:.2f}->{p['M_actual']:.3f}")


def build_one(task):
    (sys_id, conc, salt_factor, pack_density_frac, tolerance, dry_run,
     rho_override, out_root) = task
    p = plan_box(sys_id, conc, salt_factor, rho_override=rho_override)
    info = _fmt(p, sys_id, conc)
    if dry_run:
        return info

    # loose starting box: pack the actual molecules at pack_density_frac * rho
    box_vol = vol_a3_from_mass_density(
        p["n_salt"] * p["m_pair"] + p["n_solvent"] * p["m_solv"],
        p["rho"] * pack_density_frac)
    L = box_vol ** (1.0 / 3.0)

    out_dir = out_root / sys_id
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"initial_{conc}.xyz"
    if final_path.exists():
        return f"{info}  L={L:.1f}  SKIP (exists)"

    inp_path = out_dir / f"packmol_{conc}.inp"
    packed_path = out_dir / f"packed_{conc}.xyz"
    write_packmol_input(inp_path, packed_path, p["spec"], p["n_solvent"], p["n_salt"], L,
                        tolerance=tolerance)

    result = subprocess.run([PACKMOL_BIN], stdin=open(inp_path),
                            capture_output=True, text=True, timeout=3600)
    if result.returncode != 0 or not packed_path.exists():
        return f"{info}  L={L:.1f}  ** FAILED: {result.stderr[-200:]}"

    packed = read(str(packed_path))
    packed.set_cell([L, L, L])
    packed.set_pbc(True)
    write(str(final_path), packed)
    # Packmol emits no cell; final_path is the same coords + cell/pbc, so the raw
    # output is fully superseded once it is written.
    packed_path.unlink()
    return f"{info}  L={L:.1f}  OK -> {final_path}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salt-factor", type=float, default=1.0,
                    help="Multiply the current ion-pair count by this (default: 1)")
    ap.add_argument("--pack-density-frac", type=float, default=0.8,
                    help="Pack at this fraction of the target density so Packmol has "
                         "room; NPT sets the true density (default: 0.8)")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="Packmol min inter-molecular atom distance in A. Lower it "
                         "(e.g. 1.5) for hard-to-pack big molecules like TEGDME; the "
                         "downstream UMA relax removes residual close contacts (default: 2.0)")
    ap.add_argument("--systems", nargs="+", default=None,
                    help="Only build these system_ids (e.g. naotf_dme napf6_dme)")
    ap.add_argument("--use-opls", action="store_true",
                    help="Target the OPLS ground-truth densities from the CSV "
                         "instead of the exp01 NVT seed densities. Outputs to "
                         "exp05_opls_densities.")
    ap.add_argument("--opls-csv", type=Path, default=OPLS_CSV,
                    help=f"OPLS density CSV to read with --use-opls (default: {OPLS_CSV})")
    ap.add_argument("--opls-temperature", type=float, default=298,
                    help="Temperature (K) row to match in the OPLS CSV (default: 298)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    pairs = SYSTEMS if not args.systems else [
        (s, c) for s, c in SYSTEMS if s in args.systems]

    out_root = OUT_ROOT_OPLS if args.use_opls else OUT_ROOT

    opls = None
    if args.use_opls:
        opls = load_opls_densities(args.opls_csv, args.opls_temperature)

    src = (f"OPLS densities ({args.opls_csv.name} @ {args.opls_temperature:g}K)"
           if args.use_opls else "exp01 NVT seed densities")
    print(f"{'Dry run: ' if args.dry_run else ''}Building {len(pairs)} boxes "
          f"at salt_factor={args.salt_factor}, pack_density_frac={args.pack_density_frac}")
    print(f"Density source -> {src}")
    print(f"Output -> {out_root}\n")

    tasks, missing = [], []
    for s, c in pairs:
        rho_override = None
        if args.use_opls:
            key = _opls_key(s, c)
            if key not in opls:
                missing.append(f"{s} {c}")
                continue
            rho_override = opls[key]
        tasks.append((s, c, args.salt_factor, args.pack_density_frac,
                      args.tolerance, args.dry_run, rho_override, out_root))

    if missing:
        print(f"WARNING: no OPLS density found for {len(missing)} system(s), "
              f"skipping: {', '.join(missing)}\n")

    if args.dry_run or args.workers <= 1:
        results = [build_one(t) for t in tqdm(tasks, desc="Systems")]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(tqdm(ex.map(build_one, tasks), total=len(tasks),
                                desc="Building"))

    print()
    for r in results:
        print(r)
    if not args.dry_run:
        n_ok = sum("OK" in r for r in results)
        n_fail = sum("FAILED" in r for r in results)
        print(f"\nDone: {n_ok} succeeded, {n_fail} failed")


if __name__ == "__main__":
    main()
