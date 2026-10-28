"""
Evaluate a UMA model on a pkl dataset: force/stress errors + inference timing.

Usage (GPU required, fairchemV3 env):
    python src/benchmark/eval_uma.py MODEL.pt DATASET.pkl [--n-frames 50] [--split validation]

Energy comparison requires --linref pointing to linref.json (pkl energies are de-referenced).
"""
import argparse
import json
import os
import time

os.environ["CXX"] = "g++"
os.environ["CC"] = "gcc"

import pickle
import numpy as np
import torch
from ase import Atoms

EV_PER_A3_TO_BAR = 160.218e4


def load_frames(pkl_path, n_frames, split):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    frames = data[split]
    if n_frames and n_frames < len(frames):
        idxs = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        frames = [frames[i] for i in idxs]
    return frames


def load_linref(linref_path):
    with open(linref_path) as f:
        lr = json.load(f)
    coeff = np.zeros(100, dtype=np.float64)
    for z_str, val in lr["coeff_by_Z"].items():
        coeff[int(z_str)] = val
    return coeff


def evaluate(calc, frames, linref_coeff=None):
    force_errors = []
    stress_errors = []
    energy_errors = []
    timings = []

    for frame in frames:
        atoms = Atoms(
            numbers=frame["species"],
            positions=frame["coordinates"],
            cell=frame["cell"],
            pbc=True,
        )
        atoms.calc = calc

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        uma_forces = atoms.get_forces()
        uma_stress = atoms.get_stress(voigt=False).reshape(1, 3, 3)
        uma_energy = float(atoms.get_potential_energy())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        timings.append(dt)

        # Forces (eV/A)
        force_errors.append(uma_forces - frame["forces"])

        # Stress: pkl is eV/A^3 shape (1,3,3), ASE returns eV/A^3 — convert to bar
        stress_err_bar = (uma_stress - frame["stress"]) * EV_PER_A3_TO_BAR
        stress_errors.append(stress_err_bar[0])

        # Energy: pkl is de-referenced, UMA gives absolute
        if linref_coeff is not None:
            ref = float(np.dot(np.bincount(frame["species"], minlength=100), linref_coeff))
            stored_abs = frame["total_energy"] + ref
            energy_errors.append(uma_energy - stored_abs)

    result = {
        "force_errors": force_errors,
        "stress_errors": np.array(stress_errors),
        "timings": np.array(timings),
        "n_atoms": len(frames[0]["species"]),
    }
    if energy_errors:
        result["energy_errors"] = np.array(energy_errors)
    return result


def print_results(results):
    all_ferr = np.concatenate(results["force_errors"], axis=0)
    force_mae = np.abs(all_ferr).mean()
    force_rmse = np.sqrt((all_ferr ** 2).mean())
    per_atom_mae = np.array([np.abs(e).mean() for e in results["force_errors"]])

    print(f"\n  Forces ({results['n_atoms']} atoms/frame)")
    print(f"    MAE:  {force_mae:.6f} eV/A")
    print(f"    RMSE: {force_rmse:.6f} eV/A")
    print(f"    Per-frame MAE range: {per_atom_mae.min():.6f} - {per_atom_mae.max():.6f} eV/A")

    if "energy_errors" in results:
        eerr = results["energy_errors"]
        print(f"\n  Energy")
        print(f"    MAE:  {np.abs(eerr).mean():.6f} eV")
        print(f"    MAE/atom: {np.abs(eerr).mean() / results['n_atoms']:.8f} eV/atom")
        print(f"    Bias: {eerr.mean():+.6f} eV")

    serr = results["stress_errors"]  # (N, 3, 3) bar
    diag_err = serr[:, [0, 1, 2], [0, 1, 2]]
    P_err = -diag_err.mean(axis=1) / 3

    print(f"\n  Stress")
    print(f"    Diagonal MAE:  {np.abs(diag_err).mean():.1f} bar")
    print(f"    Full tensor MAE: {np.abs(serr).mean():.1f} bar")
    print(f"    Pressure bias: {P_err.mean():+.1f} bar")
    print(f"    Pressure MAE:  {np.abs(P_err).mean():.1f} bar")

    t = results["timings"]
    print(f"\n  Timing ({len(t)} frames)")
    print(f"    First call: {t[0]:.3f} s  (includes compile)")
    if len(t) > 1:
        t_rest = t[1:]
        print(f"    Mean (excl first): {t_rest.mean():.4f} s")
        print(f"    Throughput: {results['n_atoms'] / t_rest.mean():.0f} atoms/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to .pt UMA checkpoint")
    ap.add_argument("dataset", help="Path to .pkl dataset")
    ap.add_argument("--linref", default=None, help="Path to linref.json (needed for energy comparison)")
    ap.add_argument("--n-frames", type=int, default=None, help="Max frames to evaluate (default: all)")
    ap.add_argument("--split", default="validation", choices=["training", "validation"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit, InferenceSettings

    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}")

    settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=True,
        compile=True,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    predictor = load_predict_unit(
        args.model, device=args.device, inference_settings=settings,
        overrides={
            "_target_": "fairchem.core.models.base.HydraModel",
            "backbone": {"direct_forces": False},
        },
    )
    calc = FAIRChemCalculator(predictor, task_name="omol")
    print("Model loaded.")

    frames = load_frames(args.dataset, args.n_frames, args.split)
    print(f"Frames:  {len(frames)} ({args.split})")

    linref_coeff = load_linref(args.linref) if args.linref else None

    results = evaluate(calc, frames, linref_coeff)
    print_results(results)


if __name__ == "__main__":
    main()
