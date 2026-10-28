"""
Molecular dynamics with FeNNol (JAX): NPT, NVT or NVE.

Runs on FeNNol's native MD engine (fennol_md); all integration happens in
JAX/XLA, with no Python MD loop overhead. NPT uses a Langevin thermostat and a
Langevin-piston barostat (the paper's 1 ns equilibration runs); the paper's NVT
transport runs use the global Bussi thermostat (--thermostat BUSSI --gamma-thz 0.1).

A run lives entirely inside one output directory (the run folder). There are
exactly two modes, distinguished by whether `--from-xyz` is given:

  Case 1 — FRESH START (--from-xyz <seed.xyz>):
      The output dir must NOT exist yet. It is created, the trajectory starts
      from the seed frame, and outputs are written inside it. Passing --from-xyz
      when the output dir already exists is an error (refuses to clobber a run).

  Case 2 — RESUME (no --from-xyz):
      The output dir must already exist and contain a .dyn.restart. The run
      continues from where it left off, appending to the existing trajectory.

USAGE:
    # Case 1 — fresh 1 ns NPT run, seeded from a dataset start frame
    #          (default --interval 1000 = a frame every 1 ps)
    python -m src.simulate.fennol_npt .../napf6_dme/npt_1M_293K_1ns_1ps \
        --from-xyz .../datasets/napf6_dme_rattled/simulation_start_1M.xyz \
        --model .../napf6_dme/model/best_model.fnx \
        --steps 1000000 --temperature 293

    # Case 2 — resume that run after a requeue (no --from-xyz)
    python -m src.simulate.fennol_npt .../napf6_dme/npt_1M_293K_1ns_1ps \
        --model .../napf6_dme/model/best_model.fnx --steps 1000000

ARGUMENTS:
    output_dir      Run directory for simulation I/O. Created fresh (Case 1) or
                    resumed (Case 2). Its basename is the default system name.
    --from-xyz      Seed structure (.xyz) for a fresh run; omit to resume.
    --model         Path to FeNNol .fnx model file
    --steps         Total target steps (default: 1000000)
    --interval      Steps between trajectory frame dumps (default: 1000 = 1 ps)
    --temperature   Temperature in K (default: 323)
    --pressure      Pressure in bar (default: 1.0)
    --timestep      Timestep in fs (default: 1.0)
    --traj-format   Output trajectory format (default: ase — a binary .traj with
                    O(1) random access; also extxyz/xyz/arc)
    --device        JAX device (default: cuda:0)
    --system-name   Override system name for output files

OUTPUT (inside output_dir):
    {system_name}.traj          — ASE trajectory (default; random-access)
    {system_name}.traj.extxyz   — trajectory (when --traj-format extxyz)
    {system_name}.dyn.restart   — binary restart file (drives Case 2 resume)
    {system_name}.fennol.yaml   — generated fennol_md config
    start.xyz                   — positions-only seed fennol_md reads for species
    stdout                      — energy, temperature, pressure, density logs
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml


def prepare_starting_structure(seed_xyz, output_dir):
    """Read a seed structure and write a positions-only FeNNol start.xyz.

    Writes ``<output_dir>/start.xyz`` from ``seed_xyz`` and returns
    (xyz_path, cell_flat, n_atoms). On a resume fennol_md overrides
    coordinates/cell/velocities from .dyn.restart, but still reads start.xyz for
    the atomic species — so this file is needed in both fresh-start and resume
    cases. (On resume, ``seed_xyz`` is the existing start.xyz, rewritten in place.)
    """
    from ase.io import read

    print(f"Seed structure: {seed_xyz}")
    atoms = read(str(seed_xyz), index=-1)
    atoms.set_pbc([True, True, True])
    n_atoms = len(atoms)
    cell = atoms.cell.array  # 3x3
    cell_flat = cell.flatten().tolist()  # row-major: [ax,ay,az, bx,by,bz, cx,cy,cz]
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    xyz_path = Path(output_dir) / "start.xyz"
    # Carry the cell in an extxyz Lattice comment so it survives a resume read.
    # fennol_md ignores this line (it takes the cell from the config), but on a
    # resume prepare_starting_structure re-reads start.xyz, and ASE needs the
    # Lattice to recover a non-singular cell. Atom lines stay positions-only, as
    # fennol_md's xyz reader expects.
    lattice = " ".join(f"{x:.8f}" for x in cell.flatten())
    with open(xyz_path, "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n')
        for sym, pos in zip(symbols, positions):
            f.write(f"{sym}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}\n")
    print(f"  Wrote {xyz_path} ({n_atoms} atoms)")

    return str(xyz_path), cell_flat, n_atoms


def build_formal_charges(output_dir, seed_xyz, system_id=None):
    """Per-atom mass-weighted formal charges for the ionic-conductivity current log.

    Cation z=+1, anion z=-1, solvent 0. The anion's total -1 is split across its atoms
    by mass (q_atom = z * m_atom / m_ion) so that sum_i q_i v_i = sum_ions z * v_COM,ion
    -- the *translational* ionic current (no intramolecular-vibration noise from
    dumping the whole -1 on a single atom). For the single-atom cation this reduces to
    +1. Electroneutrality (sum q = 0) is asserted. Returns (q, system_id) with q a
    numpy array of length nat in the seed structure's atom order (which is the order
    fennol_md keeps in system["vel"] and the trajectory).
    """
    from ase.io import read

    from src.analysis.observables.systems import lookup_system, system_id_from_parent_dir
    from src.analysis.observables.grouping import build_groups

    if system_id is None:
        # Layout: .../<system_id>/<run_name>; the parent folder names the system.
        system_id = system_id_from_parent_dir(output_dir)
    spec = lookup_system(system_id)

    atoms = read(str(seed_xyz), index=-1)
    symbols = atoms.get_chemical_symbols()
    masses = atoms.get_masses()
    groups = build_groups(symbols, spec)   # also asserts every atom is accounted for

    q = np.zeros(len(symbols), dtype=np.float64)
    for i in groups.cat_idx:                       # cations: single atoms, +1
        q[i] = +1.0
    for grp in groups.anion_groups:                # anions: -1 split by mass
        m = masses[grp]
        q[grp] = -1.0 * m / m.sum()
    # solvent atoms stay 0

    net = float(q.sum())
    if abs(net) > 1e-6:
        raise RuntimeError(
            f"Formal charges not electroneutral: sum(q)={net:.3e} for system "
            f"'{system_id}' ({len(groups.cat_idx)} cations, "
            f"{len(groups.anion_groups)} anions). Check the salt stoichiometry."
        )
    return q, system_id


def get_remaining_steps(root_path, system_name, total_target_steps, dt_fs):
    """Check for a restart file and compute remaining steps."""
    restart_file = Path(root_path) / f"{system_name}.dyn.restart"
    if restart_file.exists():
        with open(restart_file, "rb") as f:
            restart_data = pickle.load(f)
        sim_time_ps = restart_data.get("simulation_time_ps", 0.0)
        dt_ps = dt_fs * 1e-3
        completed_steps = int(round(sim_time_ps / dt_ps))
        remaining = total_target_steps - completed_steps
        print(f"Resume: {completed_steps} steps done ({sim_time_ps:.3f} ps), {remaining} remaining")
        if remaining <= 0:
            print("Simulation already complete!")
            sys.exit(0)
        return remaining
    else:
        print(f"Fresh start: {total_target_steps} steps")
        return total_target_steps


def build_config(args, xyz_path, cell_flat, remaining_steps):
    """Build the fennol_md YAML config dict with unit annotations."""
    dt_fs = args.timestep
    interval = args.interval
    tdump_ps = interval * dt_fs / 1000.0  # convert fs to ps

    config = {
        "device": args.device,
        "matmul_prec": args.matmul_prec,
        "system_name": args.system_name,
        "model_file": str(args.model),

        "xyz_input": {
            "file": str(xyz_path),
            "indexed": False,
            "has_comment_line": True,
        },

        "cell": cell_flat,
        "minimum_image": True,
        "wrap_box": True,
        "estimate_pressure": True,

        "nsteps": remaining_steps,
        "dt[fs]": dt_fs,

        # Target temperature — used by the thermostat (NPT/NVT) AND to draw the
        # initial Maxwell-Boltzmann velocities in NVE (where there is no
        # thermostat). Must be set for NVE or velocities default to 300 K.
        "temperature": args.temperature,

        # Output
        "traj_format": args.traj_format,
        "tdump[ps]": tdump_ps,
        "nprint": interval,
        "nsummary": interval * 100,
        "pressure_unit": "atm",
        "energy_unit": "eV",
        "per_atom_energy": True,

        # Neighbor list
        "nblist_skin": args.skin,

    }

    # Reproducible replicas: fixes initial velocities + thermostat/barostat noise.
    if args.seed is not None:
        config["random_seed"] = args.seed

    # Thermostat — used for NPT and NVT. Default LGV; a Green-Kubo transport run
    # should pass --thermostat BUSSI (a global velocity rescale) so the per-atom
    # Langevin friction doesn't damp the stress/current ACFs being integrated.
    if args.ensemble in ("NPT", "NVT"):
        thermostat = args.thermostat if args.thermostat is not None else "LGV"
        config["thermostat"] = thermostat
        config["temperature"] = args.temperature
        # Thermostat coupling rate (tau = 1/gamma). Default 1.0 THz (1 ps^-1). For a
        # combined-transport run use a *weak* coupling (e.g. 0.1 THz, tau=10 ps): far
        # slower than the sub-ps stress/current ACFs we integrate for viscosity /
        # conductivity (minimal bias) yet far faster than the ns-scale energy drift.
        config["gamma[THz]"] = args.gamma_thz
    elif args.ensemble == "NVE":
        # True NVE: FeNNol defaults the thermostat to LGV when unspecified, so we
        # must pin it to NVE explicitly, or an unrequested Langevin thermostat runs
        # (and would damp the stress ACF a Green-Kubo run depends on). Initial
        # velocities are still drawn at `temperature` (set in the base config).
        config["thermostat"] = "NVE"

    # Off-diagonal stress logging for equilibrium (Green-Kubo / Einstein-Helfand)
    # shear viscosity. Dumps the full symmetric pressure tensor every
    # stress_interval steps to <system>.stress (6 floats + volume + T per line).
    # Needs the pressure estimator on (already set above). Intended for NVE (or a
    # gentle global thermostat) production — NOT NPT (V must be fixed) and NOT a
    # Langevin thermostat (its per-atom friction damps the stress ACF).
    if args.stress_log:
        config["stress_log"] = True
        config["stress_interval"] = args.stress_interval
        config["estimate_pressure"] = True

    # Charge-current logging for equilibrium (Green-Kubo) ionic conductivity. Dumps
    # the collective current J = sum_i q_i v_i every current_interval steps to
    # <system>.current. The per-atom formal charges (config["formal_charges"]) are
    # NOT set here -- they need the seed structure, so simulate() injects them after
    # build_config (see build_formal_charges). Needs no pressure estimator.
    if args.current_log:
        config["current_log"] = True
        config["current_interval"] = args.current_interval

    # Barostat (Langevin piston) — used only for NPT
    if args.ensemble == "NPT":
        config["barostat"] = "LGV"
        config["target_pressure[atm]"] = args.pressure / 1.01325  # bar -> atm
        config["tau_piston[fs]"] = 200.0
        config["gamma_piston[THz]"] = 20.0
        if args.anisotropic:
            config["aniso_barostat"] = True
            config["aniso_mask"] = [True, True, True, False, False, False]

    return config


def simulate(args):
    """Run the FeNNol MD simulation — fresh start (Case 1) or resume (Case 2)."""
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    # System name: run-folder basename or override.
    if args.system_name is None:
        args.system_name = Path(output_dir).name

    restart_file = Path(output_dir) / f"{args.system_name}.dyn.restart"
    start_xyz = Path(output_dir) / "start.xyz"

    if args.from_xyz is not None:
        # ---- Case 1: fresh start from a seed structure ----
        if os.path.exists(output_dir):
            sys.exit(
                f"ERROR: output dir already exists: {output_dir}\n"
                f"Refusing to start a fresh run on top of it. Choose a new "
                f"output_dir, or omit --from-xyz to resume an existing run."
            )
        if not os.path.exists(args.from_xyz):
            raise FileNotFoundError(f"--from-xyz seed not found: {args.from_xyz}")
        os.makedirs(output_dir)
        print(f"Fresh start in {output_dir}\n  seed: {args.from_xyz}")
        seed_path = args.from_xyz
    else:
        # ---- Case 2: resume an existing run ----
        if not os.path.isdir(output_dir):
            sys.exit(
                f"ERROR: output dir does not exist: {output_dir}\n"
                f"For a fresh run, pass --from-xyz <seed.xyz>."
            )
        if not restart_file.exists():
            sys.exit(
                f"ERROR: no {restart_file.name} in {output_dir} to resume from.\n"
                f"For a fresh run, pass --from-xyz <seed.xyz>."
            )
        if not start_xyz.exists():
            sys.exit(f"ERROR: {start_xyz} is missing; cannot resume this run.")
        print(f"Resuming run in {output_dir}")
        seed_path = start_xyz

    # Prepare start.xyz: fennol_md reads it for species (and coords/cell on a
    # fresh start). On resume, coords/cell/velocities come from .dyn.restart.
    xyz_path, cell_flat, n_atoms = prepare_starting_structure(seed_path, output_dir)

    # Compute remaining steps (reads the restart file if present).
    remaining_steps = get_remaining_steps(
        output_dir, args.system_name, args.steps, args.timestep
    )

    print(f"System: {n_atoms} atoms")
    print(f"Model: {args.model}")
    print(f"Ensemble: {args.ensemble}")
    print(f"Temperature: {args.temperature} K")
    print(f"Pressure: {args.pressure} bar")
    print(f"Timestep: {args.timestep} fs")
    print(f"Steps: {remaining_steps}")

    # Build and write config
    config = build_config(args, xyz_path, cell_flat, remaining_steps)

    # Charge-current log needs per-atom formal charges from the seed structure; inject
    # them here (build_config has no atoms). Order matches system["vel"]/the traj.
    if args.current_log:
        q, sid = build_formal_charges(output_dir, seed_path, args.system_id)
        config["formal_charges"] = q.tolist()
        print(f"Current log ON (system '{sid}'): formal charges built, "
              f"sum={q.sum():.2e} over {len(q)} atoms")

    yaml_path = Path(output_dir) / f"{args.system_name}.fennol.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config: {yaml_path}")

    # Run fennol_md
    print("\n" + "=" * 60)
    print(f"  Starting FeNNol {args.ensemble} MD")
    print("=" * 60 + "\n", flush=True)

    # Change to working directory so output files land inside output_dir.
    original_dir = os.getcwd()
    os.chdir(output_dir)
    try:
        from fennol.md.dynamic import config_and_run_dynamic
        config_and_run_dynamic(yaml_path)
    finally:
        os.chdir(original_dir)

    print("\nSimulation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="MD simulation with FeNNol (JAX). Case 1: fresh start with "
                    "--from-xyz into a new output_dir. Case 2: resume an existing "
                    "output_dir (omit --from-xyz)."
    )
    parser.add_argument(
        "output_dir",
        help="Run directory for simulation I/O. Created fresh (Case 1, with "
             "--from-xyz) or resumed if it already exists (Case 2).",
    )
    parser.add_argument(
        "--from-xyz", default=None,
        help="Seed structure (.xyz) for a FRESH run. The output_dir must not "
             "exist yet — it is created and the trajectory starts from this "
             "frame. Omit to RESUME an existing output_dir from its .dyn.restart.",
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to FeNNol .fnx model file",
    )
    parser.add_argument(
        "--steps", type=int, default=1000000,
        help="Total target steps (default: 1000000)",
    )
    parser.add_argument(
        "--interval", type=int, default=1000,
        help="Steps between trajectory dumps (default: 1000 = 1 ps at a 1 fs "
             "timestep; ample for density/RDF/diffusivity, ~10x smaller files "
             "than 100 fs).",
    )
    parser.add_argument(
        "--temperature", type=float, required=True,
        help="Temperature in K (required; the paper uses 293 for RDF runs, 298 otherwise)",
    )
    parser.add_argument(
        "--pressure", type=float, default=1.0,
        help="Pressure in bar (default: 1.0)",
    )
    parser.add_argument(
        "--timestep", type=float, default=1.0,
        help="Timestep in fs (default: 1.0)",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="JAX device (default: cuda:0)",
    )
    parser.add_argument(
        "--system-name", default=None,
        help="Override system name for output files (default: output_dir basename)",
    )
    parser.add_argument(
        "--traj-format", choices=["ase", "extxyz", "xyz", "arc"], default="ase",
        help="MD trajectory output format (default: ase — a binary .traj with "
             "O(1) random access for analysis). extxyz writes .traj.extxyz text.",
    )
    parser.add_argument(
        "--ensemble", choices=["NPT", "NVT", "NVE"], default="NPT",
        help="Thermodynamic ensemble (default: NPT)",
    )
    parser.add_argument(
        "--anisotropic", action="store_true",
        help="Use anisotropic diagonal barostat instead of isotropic (NPT only; for crystals/interfaces)",
    )
    parser.add_argument(
        "--thermostat", choices=["LGV", "BUSSI", "NOSE"], default=None,
        help="Thermostat (NPT/NVT). Default LGV; a Green-Kubo transport run should "
             "pass BUSSI so the per-atom Langevin friction doesn't damp the "
             "stress/current ACFs being integrated.",
    )
    parser.add_argument(
        "--stress-log", action="store_true",
        help="Log the off-diagonal stress tensor every --stress-interval steps to "
             "<system>.stress, for equilibrium (Green-Kubo) shear viscosity. Run in "
             "NVE (recommended) or a gentle global thermostat (BUSSI/NOSE) — NOT NPT "
             "(volume must be fixed) and NOT Langevin (damps the stress ACF).",
    )
    parser.add_argument(
        "--stress-interval", type=int, default=2,
        help="Steps between stress-tensor dumps for --stress-log (default: 2 = 2 fs "
             "at dt=1 fs, matching the transport production runs). The stress ACF "
             "decorrelates in <1 ps, so keep this <=10 fs.",
    )
    parser.add_argument(
        "--gamma-thz", type=float, default=1.0,
        help="Thermostat coupling rate in THz (tau=1/gamma) for NVT/NPT (LGV/BUSSI). "
             "Default 1.0 (tau=1 ps). Use ~0.1 (tau=10 ps) for a combined-transport "
             "BUSSI run so the weak coupling barely perturbs the stress/current ACFs.",
    )
    parser.add_argument(
        "--current-log", action="store_true",
        help="Log the collective charge current J=sum_i q_i v_i every "
             "--current-interval steps to <system>.current, for equilibrium "
             "(Green-Kubo) ionic conductivity. Per-atom formal charges are derived "
             "from the system (parent folder or --system-id): cation +1, anion -1 "
             "(mass-weighted), solvent 0. Run in NVE or a gentle global thermostat.",
    )
    parser.add_argument(
        "--current-interval", type=int, default=2,
        help="Steps between charge-current dumps for --current-log (default: 2 = 2 fs "
             "at dt=1 fs). The current ACF decorrelates in <1 ps, so keep this small.",
    )
    parser.add_argument(
        "--system-id", default=None,
        help="Registered system id (e.g. napf6_dme) used to assign formal charges for "
             "--current-log. Default: inferred from the run's parent folder name.",
    )
    parser.add_argument(
        "--skin", type=float, default=2.0,
        help="Verlet neighbor-list skin in Å (default: 2.0; set 0 to rebuild every step)",
    )
    parser.add_argument(
        "--matmul-prec", choices=["default", "high", "highest"], default="highest",
        help="JAX matmul precision (default: highest = fp32). 'default' uses tf32 "
             "on Ampere/Hopper GPUs — faster, slightly lower precision.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random_seed for FeNNol MD (initial velocities + thermostat/barostat "
             "noise). Set distinct values for independent replicas; omit for a "
             "random seed.",
    )
    args = parser.parse_args()
    simulate(args)


if __name__ == "__main__":
    main()
