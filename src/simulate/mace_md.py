"""
Molecular Dynamics Simulation Script with MACE-POLAR Potential

Runs NPT molecular dynamics using MACE calculators. Supports single-GPU
and multi-GPU parallel execution (one trajectory per GPU via mp.spawn).
cuEquivariance acceleration is enabled by default.

USAGE:
    # Single trajectory
    python src/simulate/mace_md.py /path/to/traj_dir --models /path/to/model.model

    # Multiple trajectories across GPUs
    python src/simulate/mace_md.py /path/to/dir1 /path/to/dir2 \
        --models /path/to/model1.model /path/to/model2.model

    # With custom parameters
    python src/simulate/mace_md.py /path/to/traj_dir --models /path/to/model.model \
        --steps 2000000 --interval 50 --temperature 350

REQUIRED ARGUMENTS:
    trajectories: Path(s) to trajectory directories containing .traj files
    --models: Path(s) to MACE model files (.model)

OPTIONAL ARGUMENTS:
    --steps: Total target steps (default: 1000000)
    --interval: Trajectory write and log interval (default: 10)
    --temperature: Simulation temperature in K (default: 323)
    --dtype: Model dtype, float32 or float64 (default: float32)
    --no_cueq: Disable cuEquivariance acceleration

OUTPUT:
    - Appends to existing .traj files in trajectory directories
    - Creates/updates .log files with simulation status
    - Supports resuming interrupted simulations (reuses last-frame velocities when len > 1)
"""

import sys
import os
import time
import torch.multiprocessing as mp

from ase.io import Trajectory
from ase.md.npt import NPT
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator
import torch
import numpy as np
from copy import deepcopy


def worker(rank, trajectory_model_pairs, world_size, interval, total_target_steps,
           temperatures, dtype, enable_cueq):
    """Worker function that runs on each GPU."""
    if rank >= len(trajectory_model_pairs):
        print(f"Rank {rank}: No trajectory-model pair assigned")
        return

    traj_path, model_path = trajectory_model_pairs[rank]
    temperature = temperatures[rank] if rank < len(temperatures) else temperatures[0]
    print(f"Rank {rank}: trajectory={traj_path}, model={model_path}, "
          f"temperature={temperature} K")

    simulate(
        root_path=traj_path,
        rank=rank,
        interval=interval,
        total_target_steps=total_target_steps,
        model_path=model_path,
        temperature=temperature,
        dtype=dtype,
        enable_cueq=enable_cueq,
    )


def run_parallel_simulations(trajectory_model_pairs, world_size, interval, total_target_steps,
                              temperatures, dtype, enable_cueq):
    print(f"Starting parallel simulations on {world_size} GPUs")
    mp.spawn(
        worker,
        args=(trajectory_model_pairs, world_size, interval, total_target_steps,
              temperatures, dtype, enable_cueq),
        nprocs=world_size,
        join=True,
    )
    print("All parallel simulations completed successfully!")


def simulate(root_path, rank, interval, total_target_steps, model_path,
             temperature, dtype, enable_cueq):
    print(f"Rank {rank}: Starting simulation — traj={root_path}, model={model_path}, "
          f"T={temperature} K", flush=True)

    output_traj = f"{root_path}/{os.path.basename(root_path)}.traj"
    output_log  = f"{root_path}/{os.path.basename(root_path)}.log"

    if not os.path.exists(output_traj):
        raise FileNotFoundError(f"Trajectory not found: {output_traj}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MACE model not found: {model_path}")

    # === Load last frame ===
    print(f"Loading trajectory from: {output_traj}")
    with Trajectory(output_traj, 'r') as traj_reader:
        frame_count = len(traj_reader)
        if frame_count == 0:
            raise ValueError(f"Trajectory file {output_traj} is empty")
        print(f"Trajectory contains {frame_count} frames")
        base_structure = traj_reader[-1]
        existing_simulated_steps = (frame_count - 1) * interval

    base_structure.set_pbc([True, True, True])
    structure = deepcopy(base_structure)
    print(f"Loaded structure with {len(structure)} atoms")

    positions = structure.get_positions()
    if not np.isfinite(positions).all():
        raise ValueError(f"Structure from {output_traj} contains invalid positions (NaN or inf)")

    # === Initial velocities ===
    # Priority: use frame velocities if present and finite.
    # Single frame + no velocities: draw MB. Multiple frames + no velocities: error.
    v = structure.get_velocities()
    has_velocities = v is not None and np.isfinite(v).all()
    if has_velocities:
        print(f"Using velocities from last frame (no Maxwell–Boltzmann)", flush=True)
    elif frame_count > 1:
        raise ValueError(
            f"Cannot resume {output_traj}: {frame_count} frames but last frame has missing or "
            f"non-finite velocities — fix the trajectory or use a single-frame start."
        )
    else:
        MaxwellBoltzmannDistribution(structure, temperature_K=temperature)

    # PolarMACE requires charge and spin on the atoms object
    structure.info["charge"] = 0
    structure.info["spin"] = 1

    # === Device ===
    device = f"cuda:{rank}" if rank is not None else "cuda"
    print(f"Rank {rank}: Using device: {device}")

    # === CPU threading ===
    total_cpu_cores = os.cpu_count() or 1
    world_size_local = len([1])  # single process per rank
    cores_per_process = total_cpu_cores
    torch.set_num_threads(cores_per_process)

    # === Load MACE model ===
    print(f"Loading MACE model from: {model_path} (dtype={dtype}, cueq={enable_cueq})")
    kwargs = dict(model_paths=model_path, device=device, default_dtype=dtype)
    if enable_cueq:
        kwargs['enable_cueq'] = True
    structure.calc = MACECalculator(**kwargs)
    print("MACE model loaded successfully")

    # === NPT dynamics ===
    dyn = NPT(
        atoms=structure,
        timestep=1 * units.fs,
        temperature_K=temperature,
        externalstress=1.0 * units.bar,
        ttime=100 * units.fs,
        pfactor=0.1,
        mask=([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )

    traj = Trajectory(output_traj, "a", structure)
    dyn.attach(traj.write, interval=interval)

    log_fh = open(output_log, "a", buffering=1)

    _last_print_step = [None]
    _last_print_time = [None]

    def print_status(a=structure, fh=log_fh):
        epot = a.get_potential_energy()
        ekin = a.get_kinetic_energy()
        temp = a.get_temperature()
        vol  = a.get_volume()
        step = existing_simulated_steps + dyn.nsteps

        its_per_sec = None
        its_per_sec_str = ""
        if _last_print_step[0] is not None and _last_print_time[0] is not None:
            steps_since = step - _last_print_step[0]
            time_since  = time.time() - _last_print_time[0]
            if steps_since > 0 and time_since > 0:
                its_per_sec = steps_since / time_since
                its_per_sec_str = f" | {its_per_sec:6.2f} it/s"

        _last_print_step[0] = step
        _last_print_time[0] = time.time()

        remaining_steps = total_target_steps - step
        if its_per_sec is not None and its_per_sec > 0:
            remaining_seconds = remaining_steps / its_per_sec
            days    = int(remaining_seconds // 86400)
            hours   = int((remaining_seconds % 86400) // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            seconds = int(remaining_seconds % 60)
            remaining_time_str = f"{days}d {hours}h {minutes}m {seconds}s"
        else:
            remaining_time_str = "N/A"

        line = (f"Step {step:>8} | T={temp:6.1f} K | Epot={epot:10.3f} eV | "
                f"Ekin={ekin:10.3f} eV | Vol={vol:10.3f} Å³{its_per_sec_str} | "
                f"remaining steps: {remaining_steps} | remaining time: {remaining_time_str}")
        print(line, file=fh)

    dyn.attach(print_status, interval=interval)

    print(f"Starting NPT MD: target={total_target_steps} steps, "
          f"T={temperature} K, P=1.0 bar")
    start_time = time.time()
    dyn.run(steps=total_target_steps - existing_simulated_steps)
    elapsed = time.time() - start_time
    print(f"NPT run completed in {elapsed:.2f} seconds", file=log_fh)
    log_fh.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run MD simulations with MACE potential')
    parser.add_argument('trajectories', nargs='+',
                        help='Path(s) to trajectory directories')
    parser.add_argument('--models', nargs='+', required=True,
                        help='Path(s) to MACE model files (.model)')
    parser.add_argument('--steps', type=int, default=1000000,
                        help='Total target steps (default: 1000000)')
    parser.add_argument('--interval', type=int, default=10,
                        help='Trajectory write and log interval (default: 10)')
    parser.add_argument('--temperature', type=float, nargs='+', default=[323],
                        help='Simulation temperature(s) in K (default: 323)')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32',
                        help='Model dtype (default: float32)')
    parser.add_argument('--no_cueq', action='store_true',
                        help='Disable cuEquivariance acceleration')
    args = parser.parse_args()

    enable_cueq = not args.no_cueq
    trajectory_paths   = args.trajectories
    model_paths        = args.models
    temperatures       = args.temperature
    world_size = torch.cuda.device_count()

    if world_size == 0:
        raise RuntimeError("No CUDA devices available")

    print(f"Found {world_size} CUDA device(s)")
    print(f"Models: {model_paths}")
    print(f"Steps: {args.steps}, interval: {args.interval}")
    print(f"Trajectories: {trajectory_paths}")
    print(f"Temperatures: {temperatures}")
    print(f"dtype: {args.dtype}, cueq: {enable_cueq}")

    if len(trajectory_paths) != len(model_paths):
        raise ValueError(f"Number of trajectories ({len(trajectory_paths)}) must equal number of models ({len(model_paths)})")
    if len(temperatures) != len(trajectory_paths):
        raise ValueError(f"Number of temperatures ({len(temperatures)}) must equal number of trajectories ({len(trajectory_paths)})")
    if len(trajectory_paths) > 4:
        raise ValueError(f"Number of trajectory-model pairs ({len(trajectory_paths)}) cannot exceed 4")
    if len(trajectory_paths) > world_size:
        raise ValueError(f"More pairs ({len(trajectory_paths)}) than GPUs ({world_size})")

    for p in model_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model not found: {p}")
    for p in trajectory_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Trajectory directory not found: {p}")

    trajectory_model_pairs = list(zip(trajectory_paths, model_paths))

    if len(trajectory_model_pairs) == 1:
        traj_path, model_path = trajectory_model_pairs[0]
        print(f"Single GPU mode: {traj_path} + {model_path}")
        simulate(
            root_path=traj_path,
            rank=0,
            interval=args.interval,
            total_target_steps=args.steps,
            model_path=model_path,
            temperature=temperatures[0],
            dtype=args.dtype,
            enable_cueq=enable_cueq,
        )
    else:
        print(f"Multi-GPU mode: {len(trajectory_model_pairs)} pairs on {world_size} GPUs")
        run_parallel_simulations(
            trajectory_model_pairs=trajectory_model_pairs,
            world_size=min(world_size, len(trajectory_model_pairs)),
            interval=args.interval,
            total_target_steps=args.steps,
            temperatures=temperatures,
            dtype=args.dtype,
            enable_cueq=enable_cueq,
        )


if __name__ == "__main__":
    main()
