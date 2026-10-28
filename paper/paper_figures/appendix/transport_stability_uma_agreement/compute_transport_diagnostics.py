"""Extract per-frame diagnostics from the 16 production transport trajectories
into one small CSV, so the appendix figure can be redrawn without touching the
~60 GB of ``.traj`` files (which live on ``/work/nvme`` scratch and are treated
as reproducible-from-restart, not archived).

Two sources per system, sampled at the SAME 500 frames (every 40 ps of 20 ns):

  * ``<run_dir>/uma_labels_every40ps.npz`` -- written by
    ``src/analysis/labeling/label_traj_with_uma.py --every-ps 40``. Holds the UMA-s-1p2
    labels (``uma_energy/forces/stress``) and the PaiNN values that were
    already stored in the trajectory (``ref_energy/forces/stress``).
  * ``<run_dir>/<run>.traj`` -- re-read only for the momenta, which the .npz
    does not carry and which the temperature needs.

Columns written to ``REPO/data/experiments/exp04_correct_molarity_samebox/
transport_diagnostics.csv``:

  system, conc, group, time_ps, n_atoms
  energy_per_atom_eV   PaiNN potential energy / atom -- the model that drove
                       the dynamics, so this is the MD's own energy trace
  temperature_K        from the stored momenta
  pressure_bar         kinetic + configurational, see below
  force_mae_eV_A       mean |F_UMA - F_PaiNN| over all atoms and components
  max_force_err_eV_A   max over atoms of the per-atom error magnitude ||dF_i||
  stress_mae_bar       mean |sigma_UMA - sigma_PaiNN| over the 6 unique components

Pressure convention -- worth stating because the trajectory is ambiguous about
it. Each frame also carries an ``info['stress_voigt[atm]']`` string written by
FeNNol, but its diagonal mean tracks the TOTAL pressure with the opposite sign
to the ASE stress on the calculator, so reading it naively flips the sign. This
script therefore ignores that string and computes pressure explicitly from
quantities whose conventions are unambiguous:

    P = N k_B T / V  -  tr(sigma_ASE) / 3

i.e. the ideal-gas (kinetic) term plus the configurational virial term, with
sigma in ASE's eV/A^3 convention. On a spot-checked frame this agrees with the
FeNNol value to ~3% (-2156 vs -2216 bar); the residual is consistent with
FeNNol evaluating the kinetic term at a half-step of the integrator.

    python compute_transport_diagnostics.py
    python compute_transport_diagnostics.py --systems napf6_pc:1M   # one system
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase import units
from ase.io.trajectory import TrajectoryReader

from src.analysis.observables import paths

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
# REPO/data is a gitignored symlink to the scratch root, which is how every
# other figure script reaches run artifacts -- go through it rather than
# hardcoding /work/nvme, so the path follows if the symlink is repointed.
DATA_ROOT = paths.DATA_DIR
EXP_DIR = DATA_ROOT / "experiments" / "exp04_correct_molarity_samebox"
# The CSV lands beside the runs it summarizes, not in the figure folder: it is
# a run artifact, so it stays out of git with the rest of them.
OUT_CSV = EXP_DIR / "transport_diagnostics.csv"

EV_A3_TO_BAR = 1.602176634e6
TEMPERATURE = 298
REP = 0
EVERY_PS = 40

# "system_id:conc_label" -- the 16 runs submitted by src/simulate/launch_transport.sh,
# in that script's order. lipf6_dme only exists at 0.5 M.
SYSTEMS = [
    "lipf6_dme:0_5M",
    "naotf_diglyme:0_1M",
    "naotf_diglyme:1M",
    "naotf_dme:0_1M",
    "naotf_dme:1M",
    "naotf_pc:0_1M",
    "naotf_pc:1M",
    "naotf_tgdme:0_1M",
    "naotf_tgdme:1M",
    "napf6_diglyme:0_1M",
    "napf6_diglyme:1M",
    "napf6_dme:0_1M",
    "napf6_dme:1M",
    "napf6_pc:0_1M",
    "napf6_pc:1M",
    "napf6_tgdme:1M",
]

# Which figure a run belongs to. lipf6_dme is 0.5 M but goes with the
# concentrated set, per the figure's grouping.
GROUP = {"0_1M": "0.1M", "1M": "1M", "0_5M": "1M"}

# The 6 unique components of a symmetric 3x3 tensor.
_TRIU = np.triu_indices(3)


def run_dir(system: str, conc: str) -> Path:
    return EXP_DIR / system / f"transport_{conc}_{TEMPERATURE}K_r{REP}"


def diagnostics_for(system: str, conc: str) -> pd.DataFrame:
    d = run_dir(system, conc)
    run = d.name
    npz_path = d / f"uma_labels_every{EVERY_PS}ps.npz"
    traj_path = d / f"{run}.traj"
    for p in (npz_path, traj_path):
        if not p.exists():
            raise FileNotFoundError(p)

    z = np.load(npz_path)
    for key in ("uma_forces", "uma_stress", "ref_forces", "ref_stress", "ref_energy"):
        if key not in z:
            raise KeyError(f"{npz_path} is missing {key!r} -- relabel this run")

    time_ps = z["time_ps"]
    stride = int(z["stride"])
    n_frames = len(time_ps)
    # The labeller sampled traj[0::stride]; reproduce exactly that index set so
    # the momenta line up frame-for-frame with the labels.
    indices = np.arange(n_frames) * stride

    uma_f, ref_f = z["uma_forces"], z["ref_forces"]
    uma_s, ref_s = z["uma_stress"], z["ref_stress"]
    ref_e = z["ref_energy"]
    n_atoms = uma_f.shape[1]

    df_err = uma_f - ref_f                                   # (n, N, 3) eV/A
    force_mae = np.abs(df_err).mean(axis=(1, 2))
    max_force_err = np.linalg.norm(df_err, axis=2).max(axis=1)

    ds = np.abs(uma_s - ref_s)[:, _TRIU[0], _TRIU[1]]        # (n, 6) eV/A^3
    stress_mae_bar = ds.mean(axis=1) * EV_A3_TO_BAR

    # Temperature needs the momenta, which only the trajectory has.
    traj = TrajectoryReader(str(traj_path))
    temperature = np.empty(n_frames)
    volume = np.empty(n_frames)
    for i, idx in enumerate(indices):
        atoms = traj[int(idx)]
        temperature[i] = atoms.get_temperature()
        volume[i] = atoms.get_volume()

    p_kin = n_atoms * units.kB * temperature / volume
    p_cfg = -np.trace(ref_s, axis1=1, axis2=2) / 3.0
    pressure_bar = (p_kin + p_cfg) * EV_A3_TO_BAR

    return pd.DataFrame(
        {
            "system": system,
            "conc": conc,
            "group": GROUP[conc],
            "time_ps": time_ps,
            "n_atoms": n_atoms,
            "energy_per_atom_eV": ref_e / n_atoms,
            "temperature_K": temperature,
            "pressure_bar": pressure_bar,
            "force_mae_eV_A": force_mae,
            "max_force_err_eV_A": max_force_err,
            "stress_mae_bar": stress_mae_bar,
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems", nargs="*", default=None,
                    help="subset as system:conc (default: all 16)")
    ap.add_argument("--output", default=str(OUT_CSV))
    args = ap.parse_args()

    entries = args.systems if args.systems else SYSTEMS
    frames, missing = [], []
    for entry in entries:
        system, conc = entry.split(":")
        try:
            df = diagnostics_for(system, conc)
        except (FileNotFoundError, KeyError) as exc:
            print(f"  SKIP {system:16s} {conc:5s}  {exc}", file=sys.stderr)
            missing.append(entry)
            continue
        frames.append(df)
        print(f"  ok   {system:16s} {conc:5s}  {len(df):4d} frames  "
              f"T={df.temperature_K.mean():6.2f}K  P={df.pressure_bar.mean():9.1f}bar  "
              f"F_MAE={df.force_mae_eV_A.mean():.4f} eV/A")

    if not frames:
        print("nothing to write", file=sys.stderr)
        return 1

    out = pd.concat(frames, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nwrote {len(out)} rows for {out.system.nunique()} systems "
          f"({len(frames)} runs) -> {args.output}")
    if missing:
        print(f"MISSING {len(missing)}: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
