# Observables

Structural (RDF, density) and transport (self-diffusivity, shear viscosity,
ionic conductivity) observables from the MD trajectories, plus the experimental
reference values they are compared against. Everything here is CPU-only and
runs in the `fennol` conda env from the repo root.

## Layout

```
observables/
  paths.py          where every input trajectory set and output tree lives; git_sha()
  io.py             lazy frame iteration over ASE .traj / .extxyz, frame timestep resolution
  systems.py        the salt/solvent registry (SYSTEMS) and RDF pair defaults
  grouping.py       atoms -> cation / anion / solvent molecule groups
  units.py          constants, A^2/ps -> m^2/s, Yeh-Hummer correction, density
  rdf.py            RDFAccumulator (minimum-image g(r), streamed over frames)
  density.py        DensityAccumulator (mean +/- std rho over frames)
  diffusivity.py    FFT MSD, Einstein fit, compute_diffusivity()
  conductivity.py   Onsager (cross-MSD) and Nernst-Einstein conductivity
  viscosity_gk.py   Green-Kubo running eta(tau) and the Zhang-Maginn fit
  structural/       drivers for the RDF and density comparisons (see below)
  transport/        drivers for the transport observables, METHODS.md, OPLS runner
  ground_truth/     experimental reference tables
  tests/            test_msd_fft.py: FFT MSD == direct time-origin MSD
```

## Structural drivers (`structural/`)

| script | what it computes |
|---|---|
| `compute_uma_1M_rdfs.py` | RDFs of the 9 UMA 1 M / 293 K reference trajectories, one `rdf.csv` per system |
| `compute_painn_replica_rdfs.py` | PaiNN 293 K RDFs per replica plus the replica mean (`rdf.csv` with `sd_<pair>` columns) that every figure reads; sampling matched to the UMA RDFs |
| `density_replicas.py` | mean +/- std density of the 298 K NPT replicas, rolled up into `density_master.csv` / `density_by_system.csv` |

```bash
python -m src.analysis.observables.structural.compute_uma_1M_rdfs
python -m src.analysis.observables.structural.compute_painn_replica_rdfs
python -m src.analysis.observables.structural.density_replicas --jobs 16
```

## Transport drivers (`transport/`)

`diffusivity_replicas.py`, `viscosity_gk_replicas.py` and
`conductivity_replicas.py` walk the 4-replica PaiNN NVT transport tree
(`paths.PAINN_NVT_REPLICAS`) and write the OPLS-mirrored output layout under
`data/painn_transport_observables_replicas/`. The `*_opls_replicas.py` mirrors
push the GROMACS OPLS-AA trajectories through the same estimator functions
(`run_opls_transport_login.sh` runs the three OPLS drivers in the right order).
`conductivity_common.py`, `opls_runs.py` and `time_presets.py` are shared
helpers; `--time 1ns|5ns` runs the short-trajectory convergence study.

```bash
# order matters: diffusivity reads the per-box viscosity for the Yeh-Hummer correction
python -m src.analysis.observables.transport.viscosity_gk_replicas --jobs 8
python -m src.analysis.observables.transport.diffusivity_replicas  --jobs 8
python -m src.analysis.observables.transport.conductivity_replicas --jobs 8
```

Every driver takes `--dry-run` (preview run discovery) and `--out-suffix`
(keep re-analyses from clobbering existing outputs), and records the git SHA
and all settings in a `*_config.yaml` sidecar.
[`transport/METHODS.md`](transport/METHODS.md) documents the methods and every
setting needed to reproduce the numbers.

## Ground truth (`ground_truth/`)

| file | contents | read by |
|---|---|---|
| `293p2K_exp_values.csv`, `298p2K_exp_values.csv` | experimental density, viscosity and PFG-NMR self-diffusivities per system and concentration | diffusivity and viscosity drivers, `opls_runs.py` |
| `conductivity_all.csv` | experimental ionic conductivities alongside earlier MD estimates | `conductivity_common.py` |
| `opls_density_comparison.csv` | OPLS vs experimental densities per box | `src/data_prep/box_creation/build_boxes_correct_M.py` |
| `experimental_data.md` | the two `*_exp_values.csv` tables in readable form | people |

## Where the data lives

Input trajectories are pinned to the machine that holds them (Perlmutter CFS or
Delta `/work/nvme`); a driver run on the wrong machine exits with a message
saying so. Check with:

```bash
python -m src.analysis.observables.paths
```

## Test

```bash
python -m src.analysis.observables.tests.test_msd_fft
```
