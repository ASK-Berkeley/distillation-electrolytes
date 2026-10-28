"""
Label ASE trajectory frames with UMA forces/stresses at a fixed time interval.

Usage (GPU, fairchemV3 env):
    python src/analysis/labeling/label_traj_with_uma.py TRAJ.traj --model MODEL.pt --every-ps 10

ALWAYS label with UMA-S-1p1, never 1p2. The whole distillation project is scored
against the 1p1 teacher, so every npz written here must come from 1p1 or the force
errors are not comparable across runs and machines. Checkpoint paths:
    Delta:      /projects/beye/iamin/distillation_project/models/uma-s-1p1.pt
    Perlmutter: /global/cfs/cdirs/m4558/distillation_project/models/uma-s-1p1.pt

How the label sets the paper reads were produced (all frames 1 ps apart, so
--save-interval-fs 1000):
  * uma_labels_every20ps.npz (Figure 2): every 298 K
    exp01_mixed_noise_all_systems/<system>/npt_*_298K_20ns_1ps_replicas/replica_*
    trajectory, --every-ps 20, one trajectory per GPU on Perlmutter.
  * uma_labels_every2ps.npz (no-noise ablation appendix): the four exp07 / exp04
    1 ns NPT runs, --every-ps 2, via launch_uma_labeling_interactive.sh on Delta.
"""
import os

# os.environ["CXX"] = "g++"
# os.environ["CC"] = "gcc"

import argparse
import logging
import time

import numpy as np
import torch
from ase.io.trajectory import TrajectoryReader
from tqdm import tqdm

EV_PER_A3_TO_BAR = 160.218e4


def _read_extxyz_at_indices(path, indices_set):
    """Read only frames at given indices by skipping lines for unwanted frames."""
    import re
    from ase import Atoms

    frames = []
    with open(path) as f:
        frame_idx = 0
        while True:
            header = f.readline()
            if not header:
                break
            n_at = int(header.strip())
            comment = f.readline()

            if frame_idx not in indices_set:
                for _ in range(n_at):
                    f.readline()
                frame_idx += 1
                continue

            m = re.search(r'Lattice="([^"]+)"', comment)
            lattice = np.array(list(map(float, m.group(1).split()))).reshape(3, 3) if m else None

            symbols = []
            positions = []
            for _ in range(n_at):
                parts = f.readline().split()
                symbols.append(parts[0])
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

            frames.append(Atoms(symbols=symbols, positions=positions,
                                cell=lattice, pbc=True))
            frame_idx += 1
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", help="Path to .traj file")
    ap.add_argument("--model", required=True, help="Path to UMA .pt checkpoint")
    ap.add_argument("--every-ps", type=float, default=10.0, help="Sampling interval in ps (default: 10)")
    ap.add_argument("--max-time-ps", type=float, default=None,
                     help="Only label frames up to this trajectory time in ps "
                          "(default: whole trajectory). e.g. 1000 labels the first ns")
    ap.add_argument("--save-interval-fs", type=float, default=100.0,
                     help="MD save interval in fs (default: 100, i.e. saved every 100 MD steps at 1 fs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=None, help="Output .npz path (default: auto)")
    ap.add_argument("--label", default="uma", help="Label prefix for output filename (default: uma)")
    ap.add_argument("--log", default=None, help="Log file path (default: auto)")
    ap.add_argument("--no-activation-checkpointing", action="store_true",
                     help="Disable activation checkpointing (needs ~80 GB GPU)")
    ap.add_argument("--no-compile", action="store_true",
                     help="Disable torch.compile. Use if inductor fails to build its C++ "
                          "kernels on a node (ld 'undefined reference to main'); output is "
                          "numerically identical, just slower.")
    args = ap.parse_args()

    if args.log:
        log_path = args.log
    else:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "slurm_logs",
                                f"uma_labeling_every{int(args.every_ps)}ps.log")
        log_path = os.path.normpath(log_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    log.info(f"Log: {log_path}")
    log.info(f"Trajectory: {args.traj}")
    log.info(f"Model: {args.model}")

    log.info("Importing fairchem...")
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit, InferenceSettings
    log.info("Imports done.")

    log.info("Opening trajectory...")
    is_extxyz = args.traj.endswith(".extxyz") or args.traj.endswith(".xyz")
    if is_extxyz:
        with open(args.traj) as f:
            n_atoms_file = int(f.readline().strip())
        lines_per_frame = n_atoms_file + 2
        with open(args.traj) as f:
            for _ in range(lines_per_frame):
                f.readline()
            bytes_per_frame = f.tell()
        file_size = os.path.getsize(args.traj)
        n_total = file_size // bytes_per_frame
        log.info(f"extxyz trajectory: ~{n_total} frames ({n_atoms_file} atoms/frame)")
    else:
        traj = TrajectoryReader(args.traj)
        n_total = len(traj)
    log.info(f"Trajectory opened: {n_total} frames")

    ps_per_frame = args.save_interval_fs / 1000.0
    stride = int(args.every_ps / ps_per_frame)
    indices = list(range(0, n_total, stride))

    if args.max_time_ps is not None:
        indices = [i for i in indices if i * ps_per_frame <= args.max_time_ps]
        log.info(f"  Capped to first {args.max_time_ps} ps -> {len(indices)} frames")

    if is_extxyz:
        log.info(f"Pre-reading {len(indices)} extxyz frames (stride {stride})...")
        extxyz_frames = _read_extxyz_at_indices(args.traj, set(indices))
        log.info(f"Read {len(extxyz_frames)} frames into memory")
    total_time_ns = n_total * ps_per_frame / 1000.0

    log.info(f"  {ps_per_frame} ps/frame, {total_time_ns:.2f} ns total")
    log.info(f"  Sampling every {args.every_ps} ps (stride={stride}) -> {len(indices)} frames")

    settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=True,
        compile=not args.no_compile,
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
    log.info("Model loaded.")

    uma_forces_list = []
    uma_stress_list = []
    uma_energy_list = []
    ref_forces_list = []
    ref_stress_list = []
    ref_energy_list = []
    time_ps_list = []
    timings = []

    for i, idx in enumerate(tqdm(indices, desc="Labeling with UMA")):
        if is_extxyz:
            atoms = extxyz_frames[i]
        else:
            atoms = traj[idx]
        time_ps = idx * ps_per_frame
        time_ps_list.append(time_ps)

        has_ref = atoms.calc is not None
        if has_ref:
            ref_forces_list.append(atoms.get_forces())
            try:
                ref_stress_list.append(atoms.get_stress(voigt=False))
            except Exception:
                if i == 0:
                    log.info("  Reference stress not available in trajectory, skipping")
            ref_energy_list.append(float(atoms.get_potential_energy()))

        atoms.calc = calc

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        uma_f = atoms.get_forces()
        uma_s = atoms.get_stress(voigt=False)
        uma_e = float(atoms.get_potential_energy())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        timings.append(dt)

        uma_forces_list.append(uma_f)
        uma_stress_list.append(uma_s)
        uma_energy_list.append(uma_e)

        if i == 0:
            log.info(f"  First frame: {dt:.2f}s (includes compile)")
        elif i == 1:
            log.info(f"  Second frame: {dt:.4f}s")
        elif i % 100 == 0:
            elapsed = sum(timings[1:])
            rate = (i - 1) / elapsed if elapsed > 0 else 0
            eta = (len(indices) - i) / rate if rate > 0 else 0
            log.info(f"  Frame {i}/{len(indices)} | {rate:.1f} frames/s | ETA {eta/60:.1f} min")

    if args.output:
        out_path = args.output
    else:
        traj_dir = os.path.dirname(os.path.abspath(args.traj))
        out_path = os.path.join(traj_dir, f"{args.label}_labels_every{int(args.every_ps)}ps.npz")

    save_dict = dict(
        time_ps=np.array(time_ps_list),
        uma_forces=np.array(uma_forces_list),
        uma_stress=np.array(uma_stress_list),
        uma_energy=np.array(uma_energy_list),
        timings=np.array(timings),
        n_atoms=uma_forces_list[0].shape[0],
        stride=stride,
        every_ps=args.every_ps,
    )
    if ref_forces_list:
        save_dict["ref_forces"] = np.array(ref_forces_list)
        save_dict["ref_stress"] = np.array(ref_stress_list)
        save_dict["ref_energy"] = np.array(ref_energy_list)
    np.savez(out_path, **save_dict)
    log.info(f"Saved {len(indices)} labeled frames to {out_path}")
    t_rest = np.array(timings[1:])
    n_at = uma_forces_list[0].shape[0]
    log.info(f"Throughput (excl first): {n_at / t_rest.mean():.0f} atoms/s, {t_rest.mean():.4f} s/frame")


if __name__ == "__main__":
    main()
