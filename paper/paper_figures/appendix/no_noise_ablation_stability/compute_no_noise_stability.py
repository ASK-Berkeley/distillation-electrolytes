"""Reduce the UMA-labelled NPT trajectories to the small CSVs the figure needs.

The two halves of the figure now live on two different machines, so this script
runs in one of two modes, each writing its own CSV:

  --mode no_noise   the no-noise ablation runs, on **Delta**
                    (REPO/data/experiments/exp07_no_noise/...), 1 ns each.
                    -> data/no_noise_stability_exp07.csv

  --mode mixed      the mixed-noise reference runs, on **Perlmutter**
                    (paths.PAINN_NPT_REPLICAS_FOR_DENSITIES, the exp01
                    all-systems NPT density replicas), one replica, first 1 ns.
                    -> data/mixed_noise_stability_exp01.csv

Splitting the reduction is deliberate: the mixed-noise curves used to come from
exp04's ``npt_equil`` runs, which start from a packed low-density box and so began
at a *different* density from the exp07 ablation — visible in the plot as a
mismatched starting density. They now come from the exp01 NPT density replicas,
which are seeded from equilibrated frames, so both pairs start from a matched
liquid density; but they live on Perlmutter, hence a separate mode that runs
there. The plot script concatenates whichever CSVs exist.

Reads, per run:
  <run>/uma_labels_every2ps.npz      UMA and FeNNol forces at 2 ps spacing
  <run>/*.traj                       the trajectory itself, for cell + momenta

Columns
  model                exp07 (no-noise ablation) or exp01 (mixed-noise reference).
                       These are the join keys the plot's colour/order/label dicts
                       are built on. exp01 is the mixed-noise production model; its
                       density replicas live under exp01_mixed_noise_all_systems.
  system, conc         napf6_tgdme / lipf6_dme, 1M / 0_5M
  time_ps              frame time; frames are 2 ps apart
  density_g_cm3        from the NPT cell -- the blow-up diagnostic
  temperature_K        kinetic, from the trajectory momenta
  force_mae_eV_A       mean |F_UMA - F_PaiNN| over all atoms and components
  max_force_err_eV_A   max over atoms of the per-atom error magnitude ||dF_i||

The two force definitions are copied verbatim from
``appendix/transport_stability_uma_agreement/compute_transport_diagnostics.py``
so the numbers in the two appendix figures mean the same thing.

    conda activate fennol   # ase + numpy + pandas; no GPU
    python compute_no_noise_stability.py --mode no_noise      # on Delta
    python compute_no_noise_stability.py --mode mixed         # on Perlmutter
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io.trajectory import TrajectoryReader

# paths.py is the single source of truth for where the Perlmutter density replicas
# live; import it so the mixed-mode root is never hardcoded in two places.
from src.analysis.observables import paths

HERE = Path(__file__).resolve().parent
# REPO/data is a gitignored symlink to scratch; the runs are far too large to commit,
# which is exactly why this script exists.
EXP = paths.DATA_DIR / "experiments"

AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660

NPZ_NAME = "uma_labels_every2ps.npz"

# The figure compares 1 ns of dynamics per run. The exp07 ablation runs are 1 ns;
# the Perlmutter density replicas are 20 ns, so mixed mode trims to the first
# nanosecond by default to keep both pairs on the same time axis.
MIXED_MAX_TIME_PS = 1000.0

MODE_CSV = {
    "no_noise": "no_noise_stability_exp07.csv",
    "mixed": "mixed_noise_stability_exp01.csv",
}


def no_noise_runs() -> list[tuple[str, str, str, Path]]:
    """(model tag, system, conc, run dir) for the no-noise ablation, on Delta."""
    base = EXP / "exp07_no_noise"
    return [
        ("exp07", "napf6_tgdme", "1M", base / "napf6_tgdme/npt_1M_298K_1ns"),
        ("exp07", "lipf6_dme", "0_5M", base / "lipf6_dme/npt_0_5M_298K_1ns"),
    ]


def mixed_runs(replica: int) -> list[tuple[str, str, str, Path]]:
    """(model tag, system, conc, run dir) for the mixed-noise reference, on Perlmutter.

    Layout (see density_replicas.py):
      <root>/<system>/npt_<conc>_298K_20ns_1ps_replicas/replica_<N>/
    The conc tags follow the same convention as the exp07 dirs (1M, 0_5M); adjust
    here if the Perlmutter tree names them differently.
    """
    root = paths.require(paths.PAINN_NPT_REPLICAS_FOR_DENSITIES)

    def run_dir(system: str, conc: str) -> Path:
        return root / system / f"npt_{conc}_298K_20ns_1ps_replicas" / f"replica_{replica}"

    return [
        ("exp01", "napf6_tgdme", "1M", run_dir("napf6_tgdme", "1M")),
        ("exp01", "lipf6_dme", "0_5M", run_dir("lipf6_dme", "0_5M")),
    ]


def diagnostics_for(model: str, system: str, conc: str, run: Path,
                    max_time_ps: float | None = None) -> pd.DataFrame:
    npz_path = run / NPZ_NAME
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} missing -- run src/analysis/labeling/pool_uma_labeling.sh first"
        )
    z = np.load(npz_path)
    if "ref_forces" not in z:
        raise KeyError(f"{npz_path} has no ref_forces; nothing to compare UMA against")

    time_ps = z["time_ps"]
    stride = int(z["stride"])

    # keep = indices into the npz rows we retain. With no cap this is every row;
    # with a cap it is the frames inside the first `max_time_ps`. Because row i of
    # the npz was frame i*stride of the trajectory, the traj frame for kept row j
    # is keep[j]*stride -- that is how the density loop stays aligned with the npz.
    if max_time_ps is not None:
        keep = np.where(time_ps <= max_time_ps)[0]
    else:
        keep = np.arange(len(time_ps))

    df_err = z["uma_forces"][keep] - z["ref_forces"][keep]      # (n, atoms, 3)
    force_mae = np.abs(df_err).mean(axis=(1, 2))
    max_force_err = np.linalg.norm(df_err, axis=2).max(axis=1)

    time_ps = time_ps[keep]
    n_frames = len(time_ps)

    trajs = sorted(run.glob("*.traj"))
    if not trajs:
        raise FileNotFoundError(f"no .traj in {run}")
    tr = TrajectoryReader(str(trajs[0]))

    indices = keep * stride
    density = np.empty(n_frames)
    temperature = np.empty(n_frames)
    for i, idx in enumerate(indices):
        atoms = tr[int(idx)]
        density[i] = atoms.get_masses().sum() / atoms.get_volume() * AMU_PER_A3_TO_G_PER_CM3
        temperature[i] = atoms.get_temperature()

    return pd.DataFrame(
        {
            "model": model,
            "system": system,
            "conc": conc,
            "time_ps": time_ps,
            "density_g_cm3": density,
            "temperature_K": temperature,
            "force_mae_eV_A": force_mae,
            "max_force_err_eV_A": max_force_err,
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=["no_noise", "mixed"],
                    help="no_noise = exp07 ablation (Delta); mixed = exp01 replicas (Perlmutter)")
    ap.add_argument("--replica", type=int, default=0,
                    help="which replica_<N> to reduce (mixed mode only; default 0)")
    ap.add_argument("--max-time-ps", type=float, default=None,
                    help="keep only frames up to this time (default: mixed=1000, no_noise=all)")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: data/<mode-specific name>)")
    args = ap.parse_args()

    if args.mode == "no_noise":
        runs = no_noise_runs()
        max_time_ps = args.max_time_ps
    else:
        runs = mixed_runs(args.replica)
        max_time_ps = MIXED_MAX_TIME_PS if args.max_time_ps is None else args.max_time_ps

    out = Path(args.out) if args.out else HERE / "data" / MODE_CSV[args.mode]

    if max_time_ps is not None:
        print(f"mode {args.mode}: keeping frames up to {max_time_ps:.0f} ps")

    frames = []
    for model, system, conc, run in runs:
        df = diagnostics_for(model, system, conc, run, max_time_ps=max_time_ps)
        frames.append(df)
        print(
            f"{model} {system:12s} {conc:5s} {len(df):4d} frames  "
            f"rho {df.density_g_cm3.iloc[0]:.3f} -> {df.density_g_cm3.iloc[-1]:.4f}  "
            f"F_MAE {df.force_mae_eV_A.mean():.4f} eV/A"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
