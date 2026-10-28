# Appendix figure — transport-run stability and UMA-vs-PaiNN agreement

`python plot_transport_stability_uma_agreement.py` redraws both PNGs from one
CSV (numpy + pandas + matplotlib), without touching the trajectories.

The CSV is **not** in this folder. It is a derived run artifact, so it lives
with the runs it summarizes and stays out of git:

```
REPO/data/experiments/exp04_correct_molarity_samebox/transport_diagnostics.csv
```

`REPO/data` is a gitignored symlink to `/work/nvme/beye/iamin/electrolytes_data`.
Note that this is scratch — if it is purged, the CSV goes with the trajectories
needed to rebuild it. It is only 1.2 MB, so keep a copy elsewhere if these
figures need to be reproducible long after the runs are gone.

## Figures

Two figures, 5 panels each plus a legend, one per concentration group:

| file | systems |
|---|---|
| `figure_transport_stability_uma_agreement_0_1M.png` | the 7 dilute runs (0.1 M) |
| `figure_transport_stability_uma_agreement_1M.png` | the 8 concentrated runs (1 M) + LiPF₆/DME (0.5 M) |

The script writes them **into the paper submodule**, at
`paper/paper_electrolytes/Figures/appendix/`, matching the other figure
scripts. The two PNGs in this folder are relative symlinks back to those
files, so rerunning the script updates the manuscript with no copy step and
the version committed to the paper is always the one that was last built.

Panel layout in both:

```
  potential energy / atom     temperature        pressure      <- what the MD did
  force MAE                   max |dF|           (legend)      <- teacher vs student
```

The top row is the PaiNN simulation's own trace — PaiNN drove the dynamics, so
it is a stability / equilibration check. Temperature is purely kinetic and so is
identical for both models; it can only ever be an MD trace, never an error
metric. The bottom row is the frame-by-frame disagreement between the UMA-s-1p2
teacher and the PaiNN student evaluated on the **same** configurations, so it
measures accuracy with no dynamics feedback.

One line per system: **colour = solvent**, **dash pattern = salt**. Dashed grey
reference lines mark the 298 K setpoint and zero pressure.

## Data provenance

The 16 runs are exactly those submitted by `src/simulate/launch_transport.sh`
(20 ns, 298 K NVT, weak global Bussi thermostat, γ = 0.1 THz), one replica each
(`transport_<conc>_298K_r0`).

`compute_transport_diagnostics.py` builds the CSV (8000 rows = 16 runs × 500
frames) from two sources sampled at the same frames, every 40 ps:

* `<run_dir>/uma_labels_every40ps.npz` — UMA-s-1p2 labels plus the PaiNN values
  already stored in the trajectory. Written by
  `src/analysis/labeling/label_traj_with_uma.py --every-ps 40` (the launcher
  used for that run was not kept).
* `<run_dir>/<run>.traj` — re-read only for the momenta, which the `.npz` does
  not carry and which the temperature needs.

Rebuilding the CSV takes ~90 s and requires the trajectories on
`/work/nvme` scratch; redrawing the figures needs only the CSV.

## Pressure convention

Each frame also carries an `info['stress_voigt[atm]']` string written by FeNNol,
but its diagonal mean tracks the **total** pressure with the **opposite sign**
to the ASE stress on the calculator, so reading it naively flips the sign. The
compute script ignores that string and evaluates

```
P = N k_B T / V  -  tr(sigma_ASE) / 3
```

i.e. kinetic plus configurational virial, with sigma in ASE's eV/Å³ convention.
On a spot-checked frame this agrees with FeNNol's own value to ~3 % (−2156 vs
−2216 bar); the residual is consistent with FeNNol evaluating the kinetic term
at a half-step of the integrator.

## What the figures show

* All 16 runs are flat over the full 20 ns — no energy drift, no density or
  pressure runaway. Mean |P| stays under ~250 bar for every system.
* Mean temperature sits at **296.0–296.5 K against a 298 K setpoint**, i.e.
  ~1.5–2 K low in every run. Removing centre-of-mass degrees of freedom
  (3N vs 3N−3) accounts for only ~0.3 K of that, so the rest is a genuine
  offset of the weakly-coupled Bussi thermostat (τ = 10 ps), not a
  bookkeeping artefact. It is small but systematic and worth stating rather
  than rounding to "298 K".
* Teacher–student agreement is tight and, importantly, **does not degrade with
  time** — force MAE is flat from 0 to 20 ns in every system, so the student
  is not drifting into configurations the teacher disagrees about.
* PC systems are consistently the worst case (force MAE ≈ 0.009–0.010 eV/Å)
  and DME the best (≈ 0.0065 eV/Å); the ordering is set by solvent, not salt.

## Why stress is not plotted

The CSV still carries `stress_mae_bar`; the figure deliberately omits it.

These are NVT runs with no barostat, so stress never enters the equations of
motion and cannot affect a trajectory. Green-Kubo shear viscosity uses only the
off-diagonal stress ACF. Plotting a large-looking stress MAE alongside the
force panels would imply a defect the simulations do not have.

The full-tensor number is also misleading on its own. Decomposing the
UMA-minus-PaiNN stress into isotropic and traceless parts, over all 16 runs:

| | traceless (deviatoric) MAE | off-diagonal MAE | full-tensor MAE | ⟨ΔP⟩ = P_UMA − P_PaiNN | sd(ΔP) |
|---|---|---|---|---|---|
| 0.1 M | 17–21 bar | 15–20 bar | 219–364 bar | −422 to −711 bar | 17–21 bar |
| 1 M | 19–26 bar | 18–24 bar | 394–802 bar | −770 to −1581 bar | 23–33 bar |

The shear stress the dynamics and the viscosity actually depend on agrees to
15–26 bar. Everything above that is one near-constant isotropic offset: UMA
reports pressure several hundred to ~1600 bar below PaiNN, with a standard
deviation of only 17–33 bar, and the offset roughly doubles from 0.1 M to 1 M
for the same solvent. It is **not** a kinetic-term convention mismatch —
N k_B T / V is ~4100 bar in every one of these boxes while ΔP varies four-fold
between 422 and 1581 bar.

Two caveats worth carrying forward rather than forgetting:

* The absolute-pressure disagreement is real and unexplained. If the PaiNN
  validation stress MAE (~30 bar) was a full-tensor number, this offset is a
  genuine discrepancy on production liquid boxes, not a reporting artefact.
* `launch_npt_equilibration.sh` set these box volumes with **PaiNN's own**
  barostat at 1 bar, so PaiNN's pressure bias is baked into the densities the
  NVT runs then hold fixed. UMA would call the same boxes 400–1600 bar
  under-pressured. Density feeds D and η, so this is the one path — indirect,
  second-order — by which the isotropic error is not entirely harmless.

## Options

```
python plot_transport_stability_uma_agreement.py                # raw, as shipped
python plot_transport_stability_uma_agreement.py --smooth 10    # block-mean overlay
```

`--smooth N` keeps the raw trace faint underneath and overlays an N-frame
(N × 40 ps) centred rolling mean. The temperature and pressure panels are much
easier to read that way, because their frame-to-frame scatter (±10 K, ±4000 bar)
dwarfs the differences between systems; the shipped default is raw.
