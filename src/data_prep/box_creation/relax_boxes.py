"""
Relax packed initial boxes with UMA to remove bad contacts before NPT.

Layout (exp04): reads per-system folders
    <box-dir>/<system_id>/initial_<conc>.xyz
and writes the relaxed structure alongside as
    <box-dir>/<system_id>/initial_relaxed_<conc>.xyz

By default the cell is held FIXED (FIRE on the atoms only): the relax's only job
is to push apart overlapping atoms so the barostat doesn't explode on the first
NPT step. The true 298 K density is set later by NPT equilibration, NOT here.

--relax-cell enables variable-cell relaxation, but note two caveats and use it
only for experimentation:
  (a) it minimizes toward the 0 K enthalpy density, which is DENSER than the
      298 K liquid (no thermal expansion), so it over-compresses and is not a
      substitute for NPT; and
  (b) it requires the model to output a stress tensor -- the UMA 'omol' head may
      not provide one, in which case it will error.

Parallelized across GPUs: use --rank/--world-size, or --rank-from-env to read
SLURM_PROCID/SLURM_NTASKS.

Usage (GPU, fairchemV3 env):
    # Single GPU, all boxes:
    python -m src.data_prep.box_creation.relax_boxes --model /path/to/uma.pt

    # Multi-GPU via srun (4 GPUs):
    srun -n4 --gpus-per-task=1 python -m src.data_prep.box_creation.relax_boxes \
        --model /path/to/uma.pt --rank-from-env
"""
import argparse
import os
# Delta: use the GNU compilers, not the Cray wrappers, for the torch/e3nn JIT.
os.environ["CXX"] = "g++"
os.environ["CC"] = "gcc"
import time
from functools import partial
from pathlib import Path
print = partial(print, flush=True)

import numpy as np
from ase.io import read, write
from ase.optimize import FIRE
from tqdm import tqdm

DEFAULT_BOX_DIR = str(Path(__file__).resolve().parents[3] / "data" / "experiments"
                      / "exp04_correct_molarity_samebox")


def get_force_stats(atoms):
    forces = atoms.get_forces()
    per_atom = np.linalg.norm(forces, axis=1)
    return per_atom.mean(), per_atom.max()


def make_calc(model_path, device):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit, InferenceSettings

    settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=True,
        merge_mole=True,
        compile=False,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    predictor = load_predict_unit(
        model_path, device=device, inference_settings=settings,
        overrides={
            "_target_": "fairchem.core.models.base.HydraModel",
            "backbone": {"direct_forces": False},
        },
    )
    return FAIRChemCalculator(predictor, task_name="omol")


def _cell_filter(atoms):
    """Variable-cell wrapper for the optimizer (needs model stress)."""
    try:
        from ase.filters import FrechetCellFilter
        return FrechetCellFilter(atoms)
    except ImportError:
        from ase.constraints import ExpCellFilter
        return ExpCellFilter(atoms)


def relax_one(atoms, max_steps, fmax_target, relax_cell=False, label=""):
    mean_f, max_f = get_force_stats(atoms)
    tag = "  (variable cell)" if relax_cell else ""
    print(f"  {label} BEFORE: mean |F| = {mean_f:.3f} eV/A, max |F| = {max_f:.1f} eV/A{tag}")

    if max_f <= fmax_target and not relax_cell:
        print(f"  {label} Already below fmax={fmax_target}, skipping.")
        return mean_f, max_f, mean_f, max_f, 0

    target = _cell_filter(atoms) if relax_cell else atoms
    optimizer = FIRE(target, logfile=None)
    t0 = time.time()
    converged = optimizer.run(fmax=fmax_target, steps=max_steps)
    dt = time.time() - t0

    mean_f_after, max_f_after = get_force_stats(atoms)
    status = "CONVERGED" if converged else f"STOPPED after {max_steps} steps"
    print(f"  {label} AFTER:  mean |F| = {mean_f_after:.3f} eV/A, max |F| = {max_f_after:.1f} eV/A  "
          f"({status}, {dt:.0f}s)")
    return mean_f, max_f, mean_f_after, max_f_after, dt


def discover_boxes(box_dir, systems=None):
    """(system_dir, conc, initial_xyz) for every initial_<conc>.xyz under the
    per-system folders, skipping already-relaxed outputs."""
    work = []
    for sys_dir in sorted(d for d in box_dir.iterdir() if d.is_dir()):
        if systems and sys_dir.name not in systems:
            continue
        for xyz in sorted(sys_dir.glob("initial_*.xyz")):
            if "_relaxed_" in xyz.name:
                continue
            conc = xyz.stem[len("initial_"):]        # initial_1M.xyz -> "1M"
            work.append((sys_dir, conc, xyz))
    return work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to UMA .pt checkpoint")
    ap.add_argument("--box-dir", default=DEFAULT_BOX_DIR,
                    help="Root of the per-system box folders (default: exp04)")
    ap.add_argument("--systems", nargs="+", default=None,
                    help="Only relax these system_ids (e.g. naotf_dme napf6_pc)")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--fmax", type=float, default=0.5, help="Force convergence in eV/A")
    ap.add_argument("--relax-cell", action="store_true",
                    help="Variable-cell relax (0 K density, needs model stress -- see docstring)")
    ap.add_argument("--force", action="store_true",
                    help="Re-relax boxes even if initial_relaxed_<conc>.xyz already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the boxes that would be relaxed (no model, no GPU)")
    ap.add_argument("--rank", type=int, default=0, help="GPU rank (0-indexed)")
    ap.add_argument("--world-size", type=int, default=1, help="Total number of GPUs")
    ap.add_argument("--rank-from-env", action="store_true",
                    help="Read rank/world-size from SLURM_PROCID/SLURM_NTASKS")
    args = ap.parse_args()

    if args.rank_from_env:
        args.rank = int(os.environ.get("SLURM_PROCID", 0))
        args.world_size = int(os.environ.get("SLURM_NTASKS", 1))

    device = "cuda"
    box_dir = Path(args.box_dir)
    all_work = discover_boxes(box_dir, args.systems)
    my_work = all_work[args.rank::args.world_size]
    print(f"[Rank {args.rank}/{args.world_size}] device={device}, "
          f"handling {len(my_work)} boxes: {[f'{s.name}/{c}' for s, c, _ in my_work]}")
    if args.relax_cell:
        print("  ** --relax-cell ON: minimizes toward the 0 K density (denser than the "
              "298 K liquid) and requires model stress; NPT still sets the real density.")

    if args.dry_run:
        print(f"\n{'box':30s} {'N':>7s} {'L(A)':>7s}")
        for sys_dir, conc, xyz in my_work:
            a = read(str(xyz))
            print(f"{sys_dir.name+'/'+conc:30s} {len(a):7d} {a.get_cell().lengths()[0]:7.1f}")
        return

    results = []
    for sys_dir, conc, xyz in tqdm(my_work, desc=f"[Rank {args.rank}] Relaxing"):
        label = f"{sys_dir.name}/{conc}"
        out_path = sys_dir / f"initial_relaxed_{conc}.xyz"
        if out_path.exists() and not args.force:
            print(f"  {label}: relaxed output exists, skipping (use --force to redo).")
            continue

        atoms = read(str(xyz))
        atoms.set_pbc(True)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        # Fresh calc per box: merge_mole=True bakes the mixture-of-experts to ONE
        # element composition, so a single calc cannot be reused across systems
        # with different elements (S in OTf vs P in PF6 vs Li) -- doing so trips
        # UMA's "Compositions differ from merged model" assertion.
        atoms.calc = make_calc(args.model, device)

        print(f"\n{'='*60}")
        print(f"  {label}: {len(atoms)} atoms, L={atoms.get_cell().lengths()[0]:.1f} A")

        mean_b, max_b, mean_a, max_a, _dt = relax_one(
            atoms, args.max_steps, args.fmax, args.relax_cell, label)

        write(str(out_path), atoms)
        print(f"  Saved: {out_path}")
        results.append((label, len(atoms), mean_b, max_b, mean_a, max_a))

    print(f"\n{'='*60}\n[Rank {args.rank}] Summary:\n{'='*60}")
    print(f"{'box':30s} {'N':>6s} {'mean_bef':>9s} {'max_bef':>9s} {'mean_aft':>9s} {'max_aft':>9s}")
    for name, n, mb, mxb, ma, mxa in results:
        print(f"{name:30s} {n:6d} {mb:9.3f} {mxb:9.1f} {ma:9.3f} {mxa:9.1f}")


if __name__ == "__main__":
    main()
