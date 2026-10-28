# Appendix figure — training-noise ablation, NPT stability and UMA agreement

Self-contained bundle. `python plot_no_noise_stability.py` redraws
`figure_no_noise_ablation_stability.png` from the local `data/` alone
(numpy + pandas + matplotlib, no GPU).

Rebuilding `data/` from the trajectories needs `compute_no_noise_stability.py`
and the runs on scratch; the CSV is committed so the figure does not depend on
them surviving the purge.

## What it shows

Three panels, four runs, shared time axis:

| panel | quantity | scale |
|-------|----------|-------|
| Density | `rho` from the NPT cell | linear |
| Force MAE vs UMA | mean \|F_UMA − F_PaiNN\| over atoms and components | log |
| Max force error vs UMA | max over atoms of \|\|dF_i\|\| | log |

Two systems (NaPF6/TEGDME at 1 M, LiPF6/DME at 0.5 M) x two models. The models
share their training scheme, architecture, hyperparameters and seed frames — the
only difference is the dataset:

- **mixed noise** (`exp01`, the mixed-noise production model) — trained on
  rattled/noised geometries mixed with true MD points
- **no noise** (`exp07_no_noise`) — trained on true MD geometries only

So the gap between the curves is the noise ablation and nothing else.

## Result

| run | frames | evaporation onset | MAE liquid | MAE all | peak max\|dF\| | final rho |
|-----|-------:|------------------:|-----------:|--------:|---------------:|----------:|
| NaPF6/TEGDME — mixed noise | 500 | — | 0.0091 | 0.0092 | 1.3 | 1.177 |
| LiPF6/DME — mixed noise | 500 | — | 0.0077 | 0.0077 | 0.5 | 0.997 |
| NaPF6/TEGDME — no noise | 101 | **8 ps** | 0.0293 | 0.0165 | 40.2 | 0.0003 |
| LiPF6/DME — no noise | 500 | **424 ps** | 0.1210 | 0.3156 | 106.9 | 0.0003 |

Forces in eV/Ang, density in g/cm^3. "MAE liquid" is restricted to frames with
rho > 0.7 g/cm^3; for the no-noise NaPF6 run that is only 4 frames, so treat its
0.0293 as indicative rather than measured.

Both reference runs hold density for the full 1 ns and track UMA at 0.008-0.009
eV/Ang — essentially their training force MAE, sustained over a nanosecond. Both
ablation runs evaporate, by two different routes:

- **NaPF6/TEGDME** fails violently. One frame at 6 ps carries max \|dF\| = 19.3
  eV/Ang against a 0.17 baseline (~110x) with T = 614 K; by 8 ps the density is
  5e-6 g/cm^3 and T = 17580 K. The cell goes 30x27x28 Ang to 492x441x454 Ang.
- **LiPF6/DME** fails quietly. Temperature never leaves 290-310 K and the piston
  holds 1.0 +/- 0.3 atm, but the density walks monotonically downhill from 1.03
  and crosses 0.7 at 424 ps. The force MAE rises with it, smoothly, from the
  first tens of ps — the model's cohesive energy is too weak and the liquid
  sublimes rather than exploding.

The second mode is the more useful one for the argument: nothing about the
integrator or the thermostat is misbehaving, and a run that only logged
temperature and pressure would look healthy the whole way down.

## Conventions, and where they depart from the house style

**Colour is per curve, not per solvent, and it is the only encoding.** All four
colours come from the main-paper figure 3 palette (`SOLV_COLOR`: DME `#E8850E`,
DEGDME `#1FA187`, PC `#E6197D`, TEGDME `#2E6FBF`). The two reference runs keep
their own solvent's colour and so match figure 3; the two ablation runs borrow the
DEGDME and PC entries, which carry **no solvent meaning here**. Colouring by model
instead — two blues, two reds — made the four curves hard to tell apart.

All lines are solid. Four distinct colours are enough to separate four curves, and
dashing them only broke the traces up on the noisy log panels.

**The density panel starts slightly below zero**, at `-0.045 * top`. This is
deliberate and should not be "corrected" back to `ylim(0, ...)`. The evaporated
runs settle at ~3e-4 g/cm^3, which is zero at this scale; with the axis starting
exactly at 0 their traces are drawn under the bottom spine and disappear, so the
figure reads as two lines that stop rather than two boxes with nothing left in
them. The padding lifts them clear of the spine. Line width alone cannot fix it —
the line is centred on its data value, so half of it is always behind the spine.

**The time axis is linear.** The two failures are 60x apart in time (8 ps vs
424 ps), which argues for a log axis, but the 6 ps spike survives on a linear
axis as a visible needle and a log axis costs the t = 0 frame and makes the
1 ns plateau unreadable. Do not smooth the error panels if you want that spike:
`--smooth 5` averages the single 19.3 eV/Ang frame against its neighbours and
crushes it to 6.0. `--smooth` defaults to 0 for this reason.

**No onset rules.** An earlier draft drew a dotted vertical line at each
evaporation onset in all three panels; they added clutter without adding
information the curves do not already show. `EVAP_RHO` and `_evaporation_onset`
survive only to place the "evaporates at 8 ps" annotation and to compute the
liquid-only statistics quoted above.

If that onset is ever plotted again: it must be a crossing **from above**, not the
first frame below threshold. (Both current pairs are seeded from equilibrated
frames and start above the threshold, so this is the correct definition rather
than a workaround — earlier drafts sourced the mixed pair from exp04 `npt_equil`
runs that started at 0.59-0.70 g/cm^3, where a naive test reported an evaporation
at t = 0.)

**One run is truncated.** NaPF6/TEGDME — no noise stops at 200 ps, not 1 ns: it
was fully evaporated by 8 ps and the remaining wall time was not worth spending
on an ideal gas at fixed temperature. The filled dot at the end of that curve
marks the stop, in every panel.

## Provenance

Labels come from `src/analysis/labeling/label_traj_with_uma.py` (`omol` task) at
2 ps spacing, first 1 ns; each run directory holds the resulting
`uma_labels_every2ps.npz` with `uma_*` and `ref_*` (FeNNol) forces, stresses and
energies. The two halves are labeled on the two machines that hold their runs:

- **mixed noise (exp01)** — labeled on **Perlmutter** against `uma-s-1p1`, which
  is the only UMA checkpoint staged there (`/global/cfs/cdirs/m4558/.../uma-s-1p1.pt`).
- **no noise (exp07)** — labeled on **Delta** against `uma-s-1p1`, driven by
  `src/analysis/labeling/pool_uma_labeling.sh` — four GH200s, ~5 min wall clock.

Both pairs are therefore scored against the same teacher checkpoint.

The two force-error definitions are copied verbatim from
`appendix/transport_stability_uma_agreement/compute_transport_diagnostics.py`, so
the numbers mean the same thing in both appendix figures.

Run directories (on scratch, `REPO/data` is a gitignored symlink):

```
# mixed noise (exp01), on Perlmutter — first 1 ns of replica_0:
experiments/exp01_mixed_noise_all_systems/napf6_tgdme/npt_1M_298K_20ns_1ps_replicas/replica_0
experiments/exp01_mixed_noise_all_systems/lipf6_dme/npt_0_5M_298K_20ns_1ps_replicas/replica_0
# no noise (exp07), on Delta:
data/experiments/exp07_no_noise/napf6_tgdme/npt_1M_298K_1ns
data/experiments/exp07_no_noise/lipf6_dme/npt_0_5M_298K_1ns
```

The exp07 MD itself was run by `src/simulate/fennol_configs/pool_exp07_npt.sh`
(NPT, 298 K / 1 bar, dt 1 fs, LGV thermostat + LGV Langevin piston, fp32, seed 1)
seeded from equilibrated frames of the corresponding `version05_no_noise` dataset
— not from packed boxes, so the blow-ups are attributable to the model and not to
idealised builder geometry.
