"""
Evaluate a FeNNol model on a pkl dataset: force/stress errors + inference timing.

Usage (GPU required):
    conda activate fennol
    python src/benchmark/eval_fennol.py MODEL.fnx DATASET.pkl [--n-frames 50] [--split validation]
"""
import argparse
import os
import time

os.environ.setdefault("CXX", "g++")
os.environ.setdefault("CC", "gcc")

import pickle
import numpy as np

EV_PER_A3_TO_BAR = 160.218e4  # 1 eV/A^3 = 160.218 GPa = 1.60218e6 bar


def load_frames(pkl_path, n_frames, split, n_atoms=None):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    frames = data[split]
    if n_atoms:
        # These datasets mix concentrations in one split, distinguishable only by
        # box size (e.g. napf6_pc: 2148 atoms = 0.1 M, 2268 = 1 M). Averaging the
        # two hides a per-concentration difference in accuracy.
        from collections import Counter
        hist = Counter(len(fr["species"]) for fr in frames)
        frames = [fr for fr in frames if len(fr["species"]) == n_atoms]
        if not frames:
            raise SystemExit(f"No {split} frames with {n_atoms} atoms. Present: {dict(hist)}")
        print(f"  Filtered to {n_atoms} atoms/frame: {len(frames)} frames "
              f"(split histogram: {dict(hist)})")
    if n_frames and n_frames < len(frames):
        idxs = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        frames = [frames[i] for i in idxs]
    return frames


def evaluate(model, frames):
    forces_pred_all = []
    forces_true_all = []
    stress_pred_all = []
    stress_true_all = []
    energy_pred_all = []
    energy_true_all = []
    timings = []

    for frame in frames:
        species = frame["species"].astype(np.int32)
        coords = frame["coordinates"].astype(np.float32)
        cell = frame["cell"].astype(np.float32)

        t0 = time.perf_counter()
        energy, forces, virial, _ = model.energy_and_forces_and_virial(
            species=species,
            coordinates=coords,
            cells=cell[None],
            gpu_preprocessing=True,
        )
        dt = time.perf_counter() - t0
        timings.append(dt)

        forces_pred_all.append(np.asarray(forces))
        forces_true_all.append(np.asarray(frame["forces"]))

        energy_pred_all.append(float(np.asarray(energy).item()))
        energy_true_all.append(float(frame["total_energy"]))

        volume = abs(np.linalg.det(cell))
        stress_pred_all.append(np.asarray(virial[0]) / volume * EV_PER_A3_TO_BAR)
        stress_true_all.append(np.asarray(frame["stress"][0]) * EV_PER_A3_TO_BAR)

    return {
        "forces_pred": forces_pred_all,
        "forces_true": forces_true_all,
        "stress_pred": np.array(stress_pred_all),
        "stress_true": np.array(stress_true_all),
        "energy_pred": np.array(energy_pred_all),
        "energy_true": np.array(energy_true_all),
        "timings": np.array(timings),
        "n_atoms": len(frames[0]["species"]),
    }


def print_results(results):
    fp = results["forces_pred"]
    ft = results["forces_true"]
    all_ferr = np.concatenate([p - t for p, t in zip(fp, ft)], axis=0)
    force_mae = np.abs(all_ferr).mean()
    force_rmse = np.sqrt((all_ferr ** 2).mean())
    per_frame_mae = np.array([np.abs(p - t).mean() for p, t in zip(fp, ft)])
    per_frame_max = np.array([np.linalg.norm(p - t, axis=-1).max() for p, t in zip(fp, ft)])

    print(f"\n  Forces ({results['n_atoms']} atoms/frame)")
    print(f"    MAE:  {force_mae:.4f} eV/A")
    print(f"    RMSE: {force_rmse:.4f} eV/A")
    print(f"    Per-frame MAE range: {per_frame_mae.min():.4f} - {per_frame_mae.max():.4f} eV/A")
    print(f"    Max |F_err| per frame:")
    print(f"      Worst:  {per_frame_max.max():.3f} eV/A")
    print(f"      95th %%: {np.percentile(per_frame_max, 95):.3f} eV/A")
    print(f"      Median: {np.median(per_frame_max):.3f} eV/A")

    ep = results["energy_pred"]
    et = results["energy_true"]
    eerr = ep - et
    print(f"\n  Energy")
    print(f"    MAE:  {np.abs(eerr).mean():.4f} eV")
    print(f"    MAE/atom: {np.abs(eerr).mean() / results['n_atoms']:.6f} eV/atom")
    print(f"    Bias: {eerr.mean():+.4f} eV")

    sp = results["stress_pred"]
    st = results["stress_true"]
    serr = sp - st  # (N, 3, 3) bar
    diag_err = serr[:, [0, 1, 2], [0, 1, 2]]
    P_err = -diag_err.mean(axis=1) / 3

    print(f"\n  Stress")
    print(f"    Diagonal MAE:  {np.abs(diag_err).mean():.1f} bar")
    print(f"    Full tensor MAE: {np.abs(serr).mean():.1f} bar")
    print(f"    Pressure bias: {P_err.mean():+.1f} bar")
    print(f"    Pressure MAE:  {np.abs(P_err).mean():.1f} bar")

    t = results["timings"]
    print(f"\n  Timing ({len(t)} frames)")
    print(f"    First call: {t[0]:.3f} s  (includes JIT)")
    if len(t) > 1:
        t_rest = t[1:]
        print(f"    Mean (excl first): {t_rest.mean():.4f} s")
        print(f"    Throughput: {results['n_atoms'] / t_rest.mean():.0f} atoms/s")


def save_results(results, path):
    np.savez(
        path,
        forces_pred=np.concatenate(results["forces_pred"], axis=0),
        forces_true=np.concatenate(results["forces_true"], axis=0),
        stress_pred=results["stress_pred"],
        stress_true=results["stress_true"],
        energy_pred=results["energy_pred"],
        energy_true=results["energy_true"],
        timings=results["timings"],
        n_atoms=results["n_atoms"],
    )
    print(f"\nSaved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to .fnx model")
    ap.add_argument("dataset", help="Path to .pkl dataset")
    ap.add_argument("--n-frames", type=int, default=None, help="Max frames to evaluate (default: all)")
    ap.add_argument("--split", default="validation", choices=["training", "validation"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save", default=None, help="Path to save results as .npz")
    ap.add_argument("--matmul-prec", default="highest", choices=["default", "high", "highest"])
    ap.add_argument("--n-atoms", type=int, default=None,
                    help="Keep only frames with this atom count. Splits mix concentrations "
                         "(e.g. napf6_pc: 2148=0.1M, 2268=1M); averaging them hides "
                         "per-concentration differences.")
    args = ap.parse_args()

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.replace("cuda:", "")

    import jax
    jax.config.update("jax_default_matmul_precision", args.matmul_prec)

    from fennol import FENNIX
    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}")
    model = FENNIX.load(args.model)

    frames = load_frames(args.dataset, args.n_frames, args.split, args.n_atoms)
    print(f"Frames:  {len(frames)} ({args.split})")

    results = evaluate(model, frames)
    print_results(results)

    if args.save:
        save_results(results, args.save)


if __name__ == "__main__":
    main()
