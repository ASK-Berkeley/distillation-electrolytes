"""
Label extracted frames with MACE-Polar multipoles and energy decomposition.

Runs MACE-Polar inference on each frame, extracts per-atom charges and dipoles,
and per-system energy decomposition (electrostatic, electron, interaction).

REQUIRES GPU — run via salloc on ghx4-interactive.

USAGE
-----
  salloc --account=beye-dtai --partition=ghx4-interactive -n 1 --gpus=1 -t 01:00:00
  srun python src/analysis/labeling/label_with_mace_polar.py

CONDA ENV: fairchemV3
"""

import numpy as np
from pathlib import Path
from ase.io import read, write
from tqdm import tqdm

MODEL_PATH = "/projects/beye/iamin/distillation_project/models/MACE-POLAR-1-M.model"
DATA_DIR = Path("/projects/beye/iamin/distillation_project/mace_polar_distillation_data")

SYSTEMS = [
    "napf6_dme_1m",
    "naotf_tgdme_1m",
    "napf6_pc_1m",
]

TOTAL_CHARGE = 0
TOTAL_SPIN = 1  # singlet in MACE-Polar convention


def load_calculator():
    from mace.calculators import MACECalculator
    calc = MACECalculator(
        model_paths=MODEL_PATH,
        model_type="PolarMACE",
        device="cuda",
        default_dtype="float32",
    )
    return calc


def label_frames(frames, calc):
    labeled = []
    for atoms in tqdm(frames, desc="  labeling"):
        atoms.info["charge"] = TOTAL_CHARGE
        atoms.info["spin"] = TOTAL_SPIN
        atoms.calc = calc
        atoms.get_potential_energy()

        r = calc.results

        # Per-atom charges (monopoles) in elementary charges
        atoms.arrays["mace_charges"] = np.array(r["charges"], dtype=np.float64)

        # Per-atom dipoles: density_coefficients[:, 1:4] in e3nn Y_lm order (y,z,x)
        # Permute to Cartesian (x,y,z) via indices [3, 1, 2] on the l=1 block
        density = np.array(r["density_coefficients"], dtype=np.float64)
        dipoles_ylm = density[:, 1:4]  # (N, 3) in (y, z, x) order
        dipoles_xyz = dipoles_ylm[:, [2, 0, 1]]  # (N, 3) in (x, y, z) order
        atoms.arrays["mace_dipoles"] = dipoles_xyz

        # Energy decomposition (scalars, in eV)
        atoms.info["mace_energy"] = float(r["energy"])
        atoms.info["mace_electrostatic_energy"] = float(r["electrostatic_energy"])
        atoms.info["mace_electron_energy"] = float(r["electron_energy"])
        atoms.info["mace_interaction_energy"] = float(r["interaction_energy"])

        # Per-atom energies (E_local breakdown)
        atoms.arrays["mace_node_energy"] = np.array(r["node_energy"], dtype=np.float64)

        # Forces and stress
        atoms.arrays["mace_forces"] = np.array(r["forces"], dtype=np.float64)
        if "stress" in r:
            atoms.info["mace_stress"] = r["stress"].tolist()

        # Sanity check: charges should sum to ~0 for neutral systems
        q_sum = atoms.arrays["mace_charges"].sum()
        if abs(q_sum) > 0.1:
            print(f"  WARNING: charge sum = {q_sum:.4f} (expected ~0)")

        labeled.append(atoms)

    return labeled


def process_system(system, calc):
    sys_dir = DATA_DIR / system

    for split in ["train", "val"]:
        in_path = sys_dir / f"raw_{split}.xyz"
        out_path = sys_dir / f"labeled_{split}.xyz"

        if not in_path.exists():
            print(f"  SKIP {in_path} (not found)")
            continue

        print(f"\n  {split}: reading {in_path}")
        frames = read(str(in_path), index=":", format="extxyz")
        print(f"  {split}: {len(frames)} frames, {len(frames[0])} atoms/frame")

        labeled = label_frames(frames, calc)

        write(str(out_path), labeled, format="extxyz")
        print(f"  {split}: wrote {out_path}")

        # Print summary stats
        energies = [a.info["mace_electrostatic_energy"] for a in labeled]
        charges = [a.arrays["mace_charges"].sum() for a in labeled]
        print(f"  {split}: E_es range: [{min(energies):.3f}, {max(energies):.3f}] eV")
        print(f"  {split}: charge sum range: [{min(charges):.6f}, {max(charges):.6f}]")


def main():
    import sys
    if len(sys.argv) > 1:
        systems = [sys.argv[1]]
    else:
        systems = SYSTEMS

    print(f"Loading MACE-Polar model: {MODEL_PATH}")
    calc = load_calculator()
    print("Model loaded.\n")

    for system in systems:
        print(f"{'='*60}")
        print(f"System: {system}")
        process_system(system, calc)

    print(f"\nDone. Labeled data in {DATA_DIR}")


if __name__ == "__main__":
    main()
