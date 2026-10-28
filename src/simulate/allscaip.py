"""
Molecular Dynamics Simulation Script with AllScAIP Potential

This script runs NPT (constant pressure and temperature) molecular dynamics simulations
using AllScAIP potentials. It supports both single GPU and multi-GPU parallel execution.

USAGE:
    # Single trajectory with single model (single GPU)
    python src/simulate/allscaip.py /path/to/trajectory --models /path/to/model.pt

    # Multiple trajectories with multiple models (multi-GPU)
    python src/simulate/allscaip.py /path/to/traj1 /path/to/traj2 /path/to/traj3 /path/to/traj4 \
        --models /path/to/model1.pt /path/to/model2.pt /path/to/model3.pt /path/to/model4.pt

    # With custom parameters
    python src/simulate/allscaip.py /path/to/trajectory --models /path/to/model.pt \
        --steps 2000000 --interval 50 --temperature 350 --max_atoms 2352

REQUIRED ARGUMENTS:
    trajectories: Path(s) to trajectory directories containing .traj files
    --models: Path(s) to AllScAIP model checkpoint files (.pt)
OPTIONAL ARGUMENTS:
    --steps: Total target steps for simulation (default: 1000000)
    --interval: Interval for trajectory writing and status printing (default: 10)
    --temperature: Simulation temperature in Kelvin (default: 323)
    --no_compile: Disable torch.compile (default: compile=True)
    --max_atoms: Override max atoms for padding (default: auto-detected from trajectory)

REQUIREMENTS:
    - CUDA-capable GPU(s)
    - Trajectory files (.traj) in specified directories
    - AllScAIP model checkpoint files (.pt)
    - Number of trajectory-model pairs cannot exceed 4
    - Number of trajectory-model pairs cannot exceed number of available GPUs

OUTPUT:
    - Appends to existing .traj files in trajectory directories
    - Creates/updates .log files with simulation status
    - Supports resuming interrupted simulations (reuses last-frame velocities when len > 1)

QUICK START (AllScAIP-OMol102M-md-cons):
    salloc --account=beye-dtai-gh --partition=ghx4-interactive --nodes=1 --ntasks-per-node=1 \
        --gpus-per-node=1 --cpus-per-task=12 --time=00:30:00

    srun bash -lc 'mamba run -n fairchemV3 python /u/iamin/electrolytes/src/simulate/allscaip.py \
        /path/to/traj_dir --models /projects/beye/iamin/distillation_project/models/AllScAIP-OMol102M-md-cons.pt \
        --max_atoms 2352 --temperature 298'
"""

import os
import time
import signal
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['CXX'] = 'g++'
os.environ['CC'] = 'gcc'

from ase.io import read, Trajectory
from ase.md.npt import NPT
from ase import units
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit, InferenceSettings
import torch
import numpy as np
from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def setup_distributed(rank, world_size):
    """Initialize torch.distributed for multi-GPU setup"""
    if world_size <= 1:
        print(f"Rank {rank}/{world_size}: No need for distributed setup with single GPU")
        return

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed processes"""
    if dist.is_initialized():
        dist.destroy_process_group()


def make_predictor(model_path, use_compile, max_atoms=None, device="cuda"):
    if use_compile and max_atoms is None:
        raise ValueError(
            "--max_atoms must be provided when using compile with AllScAIP."
        )
    inference_settings = InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        compile=use_compile,
        max_atoms=max_atoms if use_compile else None,
        predict_untrained_stress={'omol'},
    )
    predictor = load_predict_unit(model_path, device=device, inference_settings=inference_settings)

    # predict_untrained_stress registers the omol_stress task but never flips
    # backbone.regress_stress or inner head.regress_stress. Patch all three sites.
    m = predictor.model.module
    m.backbone.regress_stress = True
    for head in m.output_heads.values():
        if hasattr(head, 'regress_stress'):
            head.regress_stress = True
        if hasattr(head, 'head') and hasattr(head.head, 'regress_stress'):
            head.head.regress_stress = True

    return predictor


def worker(rank, trajectory_model_pairs, world_size, interval, total_target_steps, temperatures, use_compile, max_atoms):
    """Worker function that runs on each GPU"""
    try:
        if world_size > 1:
            setup_distributed(rank, world_size)
            print(f"Rank {rank}/{world_size}: Initialized distributed setup")

        if rank < len(trajectory_model_pairs):
            traj_path, model_checkpoint = trajectory_model_pairs[rank]
            temperature = temperatures[rank] if rank < len(temperatures) else temperatures[0]
            print(f"Rank {rank}: Assigned trajectory: {traj_path}, model: {model_checkpoint}, temperature: {temperature} K")
        else:
            print(f"Rank {rank}: No trajectory-model pair assigned")
            return

        if world_size > 1:
            dist.barrier()

        print(f"Rank {rank}: Starting simulation for {traj_path} with model {model_checkpoint}")
        try:
            simulate(
                root_path=traj_path,
                rank=rank,
                world_size=world_size,
                interval=interval,
                total_target_steps=total_target_steps,
                model_checkpoint=model_checkpoint,
                temperature=temperature,
                use_compile=use_compile,
                max_atoms=max_atoms,
            )
            print(f"Rank {rank}: Completed simulation for {traj_path}")
        except Exception as sim_error:
            print(f"Rank {rank}: Simulation failed for {traj_path}: {sim_error}")
            raise

    except Exception as e:
        print(f"Rank {rank}: Error in worker: {e}")
        raise
    finally:
        if world_size > 1 and dist.is_initialized():
            print(f"Rank {rank}: Waiting for all workers to complete before cleanup...")
            dist.barrier()
            print(f"Rank {rank}: All workers completed, proceeding with cleanup")

        if world_size > 1:
            if dist.is_initialized():
                cleanup_distributed()
                print(f"Rank {rank}: Cleaned up distributed setup")


def run_parallel_simulations(trajectory_model_pairs, world_size, interval=50, total_target_steps=1000000, temperatures=[323], use_compile=True, max_atoms=None):
    """Main function to run parallel simulations across multiple GPUs"""
    print(f"Starting parallel simulations on {world_size} GPUs")
    print(f"Total trajectory-model pairs: {len(trajectory_model_pairs)}")
    print(f"Temperatures: {temperatures}")

    mp.spawn(
        worker,
        args=(trajectory_model_pairs, world_size, interval, total_target_steps, temperatures, use_compile, max_atoms),
        nprocs=world_size,
        join=True
    )
    print("All parallel simulations completed successfully!")


def simulate(root_path, rank=None, world_size=None, interval=50, total_target_steps=1000000, model_checkpoint=None, temperature=323, use_compile=True, max_atoms=None):
    """Run MD simulation on a specific GPU rank."""
    print(f"Rank {rank}/{world_size}: dist.is_initialized(): {dist.is_initialized()}")
    print(f"Trajectory: {root_path}, Model: {model_checkpoint}, Temperature: {temperature}", flush=True)
    if rank is not None and world_size is not None and not dist.is_initialized():
        setup_distributed(rank, world_size)
        print(f"Rank {rank}/{world_size}: Initialized distributed setup")
    else:
        print(f"Not using distributed setup, single GPU mode")
    print(f"AllScAIP model checkpoint: {model_checkpoint}, Trajectory: {root_path}")

    model_path = model_checkpoint
    output_traj = f"{root_path}/{os.path.basename(root_path)}.traj"
    output_log = f"{root_path}/{os.path.basename(root_path)}.log"

    if not os.path.exists(output_traj):
        raise FileNotFoundError(f"Trajectory not found: {output_traj}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"AllScAIP model not found: {model_path}")

    # === Load last frame ===
    print(f"Loading trajectory from: {output_traj}")
    try:
        with Trajectory(output_traj, 'r') as traj_reader:
            frame_count = len(traj_reader)
            if frame_count == 0:
                raise ValueError(f"Trajectory file {output_traj} is empty")

            print(f"Trajectory contains {frame_count} frames")
            base_structure = traj_reader[-1]
            existing_simulated_steps = (frame_count - 1) * interval
    except Exception as e:
        print(f"Error loading trajectory {output_traj}: {e}")
        raise

    base_structure.set_pbc([True, True, True])
    structure = deepcopy(base_structure)
    print(f"Loaded structure with {len(structure)} atoms")

    if len(structure) == 0:
        raise ValueError(f"Structure from {output_traj} has no atoms")

    positions = structure.get_positions()
    if not np.isfinite(positions).all():
        raise ValueError(f"Structure from {output_traj} contains invalid positions (NaN or infinite)")

    print(f"Structure validation passed: {len(structure)} atoms, positions are finite")

    # === Initial velocity ===
    structure.info["charge"] = 0
    structure.info["spin"] = 1
    v = structure.get_velocities()
    has_velocities = v is not None and np.isfinite(v).all()
    if has_velocities:
        print(f"Using velocities from last frame (no Maxwell-Boltzmann)", flush=True)
    elif frame_count > 1:
        raise ValueError(
            f"Cannot resume {output_traj}: {frame_count} frames but last frame has missing or "
            f"non-finite velocities -- fix the trajectory or use a single-frame start."
        )
    else:
        MaxwellBoltzmannDistribution(structure, temperature_K=temperature)

    # === Set up model ===
    if rank is not None:
        device = f"cuda:{rank}"
        print(f"Rank {rank}: Using device: {device}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

    # === Configure CPU threading ===
    total_cpu_cores = os.cpu_count()
    if total_cpu_cores is None:
        total_cpu_cores = 1
        print("Warning: Could not detect CPU core count, defaulting to 1 thread")

    if world_size is not None and world_size > 1:
        cores_per_process = max(1, total_cpu_cores // world_size)
        print(f"Multi-GPU mode: Dividing {total_cpu_cores} CPU cores among {world_size} processes", flush=True)
        print(f"Rank {rank}: Allocated {cores_per_process} CPU cores per process", flush=True)
    else:
        cores_per_process = total_cpu_cores
        print(f"Single GPU mode: Using all {total_cpu_cores} CPU cores", flush=True)

    torch.set_num_threads(cores_per_process)

    if max_atoms is None:
        max_atoms = len(structure)
        print(f"Auto-detected max_atoms={max_atoms} from trajectory")

    print(f"Loading AllScAIP model from: {model_path}")
    predictor = make_predictor(model_path, use_compile=use_compile, max_atoms=max_atoms, device=device)
    structure.calc = FAIRChemCalculator(predictor, task_name="omol")
    print("AllScAIP model loaded successfully")

    # === Set up NPT dynamics ===
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
        vol = a.get_volume()
        step = existing_simulated_steps + dyn.nsteps

        its_per_sec_str = ""
        if _last_print_step[0] is not None and _last_print_time[0] is not None:
            steps_since = step - _last_print_step[0]
            time_since = time.time() - _last_print_time[0]
            if steps_since > 0 and time_since > 0:
                its_per_sec = steps_since / time_since
                its_per_sec_str = f" | {its_per_sec:6.2f} it/s"
        _last_print_step[0] = step
        _last_print_time[0] = time.time()
        remaining_steps = total_target_steps - step
        remaining_steps_str = f"{remaining_steps}"
        if 'its_per_sec' in locals() and its_per_sec > 0:
            remaining_seconds = remaining_steps / its_per_sec
            days = int(remaining_seconds // 86400)
            hours = int((remaining_seconds % 86400) // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            seconds = int(remaining_seconds % 60)
            remaining_time_str = f"{days}d {hours}h {minutes}m {seconds}s"
        else:
            remaining_time_str = "N/A"
        line = (f"Step {step:>8} | T={temp:6.1f} K | Epot={epot:10.3f} eV | "
                f"Ekin={ekin:10.3f} eV | Vol={vol:10.3f} A^3{its_per_sec_str} | remaining steps: {remaining_steps_str} | remaining time: {remaining_time_str}")
        print(line, file=fh)

    dyn.attach(print_status, interval=interval)

    print("Starting NPT molecular dynamics simulation...")
    print(f"Target: {total_target_steps} steps at 1 fs timestep")
    print(f"Simulation temperature: {temperature} K, Pressure: 1.0 bar")
    start_time = time.time()
    dyn.run(steps=total_target_steps - existing_simulated_steps)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"NPT run completed in {elapsed:.2f} seconds", file=log_fh)
    log_fh.close()

    if rank is not None and world_size is not None and world_size == 1:
        cleanup_distributed()
        print(f"Rank {rank}: Cleaned up distributed setup")


def main():
    """Main function that parses command line arguments and runs simulations"""
    import argparse

    parser = argparse.ArgumentParser(description='Run MD simulations with AllScAIP potential')
    parser.add_argument('trajectories', nargs='+',
                       help='Path(s) to trajectory directories (space-separated)')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Path(s) to AllScAIP model checkpoint files (space-separated)')
    parser.add_argument('--steps', type=int, default=1000000,
                       help='Total target steps for simulation (default: 1000000)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Interval for trajectory writing and status printing (default: 10)')
    parser.add_argument('--temperature', type=float, nargs='+', default=[323],
                       help='Temperature(s) for simulation (default: 323 K). Can specify multiple temperatures, one per trajectory.')
    parser.add_argument('--max_atoms', type=int, default=None,
                       help='Max atoms for padding (default: auto-detected from trajectory)')
    parser.add_argument('--no_compile', action='store_true',
                       help='Disable torch.compile (default: compile=True)')
    args = parser.parse_args()

    total_target_steps = args.steps
    interval = args.interval
    model_checkpoints = args.models
    trajectory_paths = args.trajectories
    temperatures = args.temperature
    use_compile = not args.no_compile
    max_atoms = args.max_atoms
    world_size = torch.cuda.device_count()

    if world_size == 0:
        raise RuntimeError("No CUDA devices available")

    print(f"Found {world_size} CUDA devices")
    print(f"Model checkpoints: {model_checkpoints}")
    print(f"Target steps: {total_target_steps}")
    print(f"Interval: {interval}")
    print(f"Trajectory paths: {trajectory_paths}")
    print(f"Temperatures: {temperatures}")
    print(f"Compile: {use_compile}, Max atoms: {max_atoms}")

    if len(trajectory_paths) != len(model_checkpoints):
        raise ValueError(f"Number of trajectories ({len(trajectory_paths)}) must equal number of models ({len(model_checkpoints)})")

    if len(temperatures) != len(trajectory_paths):
        raise ValueError(f"Number of temperatures ({len(temperatures)}) must equal number of trajectories ({len(trajectory_paths)})")

    if len(trajectory_paths) > 4:
        raise ValueError(f"Number of trajectory-model pairs ({len(trajectory_paths)}) cannot exceed 4")

    if len(trajectory_paths) > world_size:
        raise ValueError(f"Number of trajectory-model pairs ({len(trajectory_paths)}) is greater than the number of GPUs ({world_size})")

    for model_checkpoint in model_checkpoints:
        if not os.path.exists(model_checkpoint):
            raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")

    for traj_path in trajectory_paths:
        if not os.path.exists(traj_path):
            raise FileNotFoundError(f"Trajectory directory not found: {traj_path}")

    trajectory_model_pairs = list(zip(trajectory_paths, model_checkpoints))
    print(f"Created {len(trajectory_model_pairs)} trajectory-model pairs")

    if len(trajectory_model_pairs) == 1:
        traj_path, model_checkpoint = trajectory_model_pairs[0]
        temperature = temperatures[0]
        print(f"Running single trajectory-model pair on single GPU: {traj_path} + {model_checkpoint}")
        print(f"Temperature: {temperature} K")
        simulate(
            root_path=traj_path,
            rank=None,
            world_size=None,
            interval=interval,
            total_target_steps=total_target_steps,
            model_checkpoint=model_checkpoint,
            temperature=temperature,
            use_compile=use_compile,
            max_atoms=max_atoms,
        )
    else:
        print(f"Running {len(trajectory_model_pairs)} trajectory-model pairs on {world_size} GPUs")
        run_parallel_simulations(
            trajectory_model_pairs=trajectory_model_pairs,
            world_size=min(world_size, len(trajectory_model_pairs)),
            interval=interval,
            total_target_steps=total_target_steps,
            temperatures=temperatures,
            use_compile=use_compile,
            max_atoms=max_atoms,
        )


if __name__ == "__main__":
    main()
