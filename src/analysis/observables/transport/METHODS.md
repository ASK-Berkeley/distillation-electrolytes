# Transport observables — methods & reproduction guide

How the three transport observables (self-diffusivity, shear viscosity, ionic
conductivity) are computed from the PaiNN/FeNNol 20 ns NVT transport runs, with
every setting needed to reproduce the numbers. Estimator code lives in
`src/analysis/observables/` (`diffusivity.py`, `viscosity_gk.py`,
`conductivity.py`, `units.py`); the drivers in this directory
(`*_replicas.py`) walk the run tree, apply the settings below, and write the
CSV/figure outputs. The OPLS mirrors (`*_opls_replicas.py`) push GROMACS `.xtc`
trajectories through the *same* estimator functions and emit the identical
output format, so PaiNN-vs-OPLS comparisons use one method definition.

## 1. Input simulations

| Setting | Value |
|---|---|
| Run set | `paths.PAINN_NVT_REPLICAS` (the location is set in `paths.py`) |
| Layout | `nvt_replica_<1..4>/<salt>_<solvent>/nvt_transport_<conc>_298K_20ns_1ps/` |
| Replicas | 4 runs per box, all started from the same NPT-equilibrated seed frame; they differ only in the random seed (initial Maxwell–Boltzmann velocities and thermostat noise) |
| Ensemble | NVT, Bussi (stochastic velocity rescaling) thermostat, $T = 298$ K |
| Integrator | 1 fs timestep, 20 ns per run (FeNNol MD, per-system PaiNN model) |
| Cell | cubic, ~30.5–32.3 Å edge (fixed; set from the equilibrated NPT density) |
| Trajectory | ASE binary `.traj`, one frame per ps (20 000 frames) |
| Stress log | `<run>.stress`: time, 6 pressure-tensor components (**atm**), volume, T — every 2 fs |
| Off-temperature runs | the lipf6_dme 0.5 M 273/323 K series is present in the tree but **skipped** (everything below is a 298 K pipeline) |

Frame timing (`dt_ps`) is resolved per run from the `Time_ps` delta of the
first two frames, falling back to `tdump[ps]` in the sibling `.fennol.yaml`
(`io.resolve_dt_ps`).

## 2. Shared machinery (diffusivity + conductivity)

Both trajectory-based observables use one streaming pass
(`diffusivity._unwrap_com_frame`):

1. **Equilibration cut** — frames before $t_\mathrm{eq} = 2000$ ps are
   discarded; the analysis window is the final $T_w = 18$ ns.
2. **Species grouping** — atoms are grouped into cation / anion / solvent
   molecules by exact element-sequence match (`systems.py` dictionaries);
   a run aborts if any atom is left unassigned.
3. **Make molecules whole** — each molecule's atoms are folded to its first
   atom by the minimum-image convention, then the mass-weighted molecular
   center of mass (COM) is taken. (Assumes molecular extent $< L/2$. Verified
   empirically for the worst case, TEGDME in the ~30.8 Å boxes: misfolds affect
   ~1 in $10^4$ molecule-frames with ≤ 2 Å COM error that does not accumulate
   through the unwrap — negligible against fit-window MSD values of
   $10^2$–$10^3$ Å².)
4. **Time unwrap** — frame-to-frame COM displacements are MIC-folded and
   cumulatively summed (valid because COM motion per ps $\ll L/2$).
5. **System-COM frame** — the mass-weighted COM of all molecules is subtracted
   per frame. This matters: the thermostatted system's total momentum is not
   strictly conserved, so the unwrapped system COM random-walks over 20 ns and
   would otherwise inflate every MSD.
6. **MSD / cross-MSD by FFT** — displacement correlations averaged over all
   time origins $t$ and (for the self terms) all $M$ particles of the species,

   $$\mathrm{MSD}(\tau) \;=\; \Big\langle \big|\mathbf r(t+\tau)-\mathbf r(t)\big|^2 \Big\rangle_{t,\,\mathrm{particles}},$$

   computed with the Calandrini fast-correlation (FFT) algorithm in
   $O(T\log T)$ (`msd_fft`, `cross_msd_fft`); every 1 ps frame is used
   (stride 1), so the lag grid is uniform with $\Delta\tau = 1$ ps. Verified
   identical to the direct $O(T^2)$ loop (`tests/test_msd_fft.py`; cross-kernel
   checked to $2\times10^{-12}$).

## 3. Self-diffusivity (`diffusivity_replicas.py`)

**Estimator.** Einstein relation per species $s \in$ {cation, anion, solvent}:
in the diffusive regime $\mathrm{MSD}_s(\tau) = 6 D_s \tau + c$, so $D_s$ is
$1/6$ of the slope of an (unweighted) ordinary-least-squares line fitted to
$\mathrm{MSD}_s$ vs. $\tau$ over a fit window $[\tau_\mathrm{lo},
\tau_\mathrm{hi}]$. The cation MSD is per-atom; anion/solvent MSDs are
molecular-COM. Everything below is about choosing that window — the MSD curve
itself is computed once per run (settings of §2) and **both** window methods
are applied to it, so every master row carries two column families:

- **Fickian window** (`D_*_fick_*` columns) — **the production method**; a
  per-species, data-driven window restricted to where the transport is
  demonstrably diffusive. Reported in the paper.
- **Fixed window** (unsuffixed `D_*` columns) — one hard-coded window for all
  species; retained as a cross-check (the two agree to ~2 % for the solvent).

### 3.1 Fickian ($\beta \approx 1$) window — full specification

Motivation: an Einstein fit is only valid where the local power-law exponent

$$\beta(\tau) \;=\; \frac{d\ln \mathrm{MSD}}{d\ln \tau}$$

equals 1 (Fickian regime). Early lags are sub-diffusive (ballistic/caging,
$\beta \neq 1$); very long lags are unbiased but statistics-starved (few time
origins; only ~2 ion pairs in the 0.1 M boxes), so a fit there is
variance-dominated. The algorithm locates the longest lag range with
$\beta \approx 1$ and fits only there. Parameters (fixed for all runs):
$\tau_\mathrm{floor} = 5$ ps, $f_\mathrm{max} = 0.4$,
$\epsilon_\beta = 0.15$, $n_\mathrm{min} = 12$, $n_\mathrm{log} = 200$,
$w_\mathrm{sm} = 5$ samples.

Given one species' MSD on the uniform lag grid $\{\tau_i\}$ (window length
$T_w$):

1. **Search range.** Keep grid points with $\tau > 0$, $\mathrm{MSD} > 0$,
   $\tau \ge \tau_\mathrm{floor}$, and $\tau \le f_\mathrm{max}\,\tau_{max}$
   where $\tau_{max}$ is the last lag on the grid (so $\lesssim 7.2$ ns here;
   the zero-padded smoothing in step 3 makes the last three log samples unusable,
   so windows end by ~6.45 ns, about 36% of $\tau_{max}$, which is what the paper states);
   call the number of surviving points $n$. If $n < n_\mathrm{min}$, fit the
   whole surviving range and flag `converged = False`.
2. **Log-spaced resampling.** Build $\min(n_\mathrm{log}, n)$ geometrically
   spaced target lags between the first and last kept lag; map each to the
   nearest-above grid index (`searchsorted`) and de-duplicate. This yields a
   log-uniform subset $\{(\tau_k, y_k)\}$ so early and late decades carry
   comparable weight in the plateau search.
3. **Local exponent.** Smooth $\ln y_k$ with a centered $w_\mathrm{sm}$-sample
   moving average, then compute $\beta_k$ as the central-difference gradient of
   the smoothed $\ln y$ with respect to $\ln \tau_k$ (non-uniform two-sided
   differences, i.e. `numpy.gradient`).
4. **Plateau detection.** Mark sample $k$ *good* if
   $|\beta_k - 1| \le \epsilon_\beta$. Find the **longest contiguous run** of
   good samples. If its length is $\ge n_\mathrm{min}$, that run defines the
   window and `converged = True`. Contiguity is essential: it rejects isolated
   lucky crossings of $\beta = 1$ on a noisy curve.
5. **Fallback (non-converged).** If no contiguous run reaches
   $n_\mathrm{min}$: slide a width-$n_\mathrm{min}$ block across
   $|\beta_k - 1|$, pick the block with the smallest mean, use it as the
   window, and flag `converged = False`. A best-effort $D$ is still reported;
   downstream statistics (box means, parity MAPD) exclude non-converged fits.
6. **Einstein fit.** With $[\tau_\mathrm{lo}, \tau_\mathrm{hi}]$ = the chosen
   run's endpoint lags, do an unweighted OLS fit of $\mathrm{MSD}$ vs. $\tau$
   over **all original grid points** (not just the log-spaced subset) inside
   the window with $\mathrm{MSD} > 0$; $D = \mathrm{slope}/6$ (Å²/ps, reported
   $\times 10^{-10}$ m²/s). Record $\tau_\mathrm{lo}$, $\tau_\mathrm{hi}$, the
   mean $\beta$ inside the window, and the `converged` flag alongside $D$, so
   every fit is auditable from the CSV row alone.

On this dataset the method converges for 60/64 (cation), 61/64 (anion) and
64/64 (solvent) runs. Over the converged fits, ion windows start at a median
$\tau_\mathrm{lo}$ of 0.28 ns and end at a median $\tau_\mathrm{hi}$ of 3.5 ns
(cation) and 3.9 ns (anion), with a wide spread ($\tau_\mathrm{hi}$ interquartile
range about 1.6–6.45 ns; three anion windows close below 0.25 ns). Solvent
windows run from ~0.1 ns to the 6.45 ns cap in 62/64 runs. The OPLS-AA runs
(`opls_transport_observables/diffusivity/`) behave the same way: 58/64, 56/64 and
64/64 converged, median ion $\tau_\mathrm{hi}$ 3.1 ns (cation) and 3.0 ns (anion),
solvent at the cap in 61/64.

### 3.2 Fixed window (cross-check)

$[\tau_\mathrm{lo}, \tau_\mathrm{hi}] = [1000\ \mathrm{ps},\, 0.5\,T_w] =
[1, 9]$ ns for every species; same OLS fit. Solid for the many-molecule
solvent; for the sparse-box ions this window sits in the noisy long-lag tail
and inflates replica-to-replica scatter (CV ~45–48 % at 0.1 M vs ~13–14 % for
the Fickian window) without improving accuracy — which is why it is the
cross-check, not the production number.

### 3.3 Finite-size (Yeh–Hummer) correction

Periodic boundaries suppress self-diffusion hydrodynamically; the leading
correction to the infinite-box value is

$$D_\infty \;=\; D_\mathrm{PBC} \;+\; \frac{k_B T\, \xi}{6 \pi \eta L},
\qquad \xi = 2.837297 \ \text{(cubic box)},$$

with $L = \langle V\rangle^{1/3}$. The correction is species-independent and is
reported two ways for each column family:

- `D_*_inf_exp`: $\eta$ = experimental viscosity (298.2 K table);
- `D_*_inf_sim`: $\eta$ = this box's own simulated GK viscosity (per-box
  replica mean from `viscosity_gk_by_box.csv` — **run the viscosity driver
  first**). This variant is fully self-consistent within the model and is the
  one used in the paper figure.

**Experimental reference:** `ground_truth/298p2K_exp_values.csv` (PFG-NMR
$D^+_\mathrm{exp}$, $D^-_\mathrm{exp}$, $D^0_\mathrm{exp}$ and dynamic
viscosity at 298.2 K).

## 4. Shear viscosity (`viscosity_gk_replicas.py`)

**Estimator.** Green–Kubo from the equilibrium stress autocorrelation,

$$\eta \;=\; \frac{V}{k_B T} \int_0^{\infty}
\big\langle P_{\alpha\beta}(0)\, P_{\alpha\beta}(t) \big\rangle\, dt,$$

averaged over the five independent traceless components $P_{xy}$, $P_{xz}$,
$P_{yz}$, $\tfrac{1}{2}(P_{xx} - P_{yy})$, $\tfrac{1}{2}(P_{yy} - P_{zz})$.
Each component's ACF is computed by FFT after subtracting its mean, with the
unbiased normalization $1/(N - k)$ at lag $k$; the running integral
$\eta(\tau)$ is a cumulative trapezoid.

| Setting | Value |
|---|---|
| Stress samples | every 2 fs, from the run's `.stress` log (pressure converted from the header's unit — atm — to Pa) |
| Equilibration cut | rows with $t < 2000$ ps discarded (matches §2's cut) |
| Prefactor $T$ | nominal setpoint **298.0 K** (the `.stress` T column is an on-step KE quantity and not used) |
| Volume $V$ | run-mean of the logged volume (constant in NVT) |
| Max lag | $\eta(\tau)$ and the error band computed to $\tau = 1$ ns |
| Error band $b(\tau)$ | standard deviation of $\eta(\tau)$ over $n_b = 5$ equal time-blocks, each re-integrated and interpolated onto the full-run lag grid |

**Plateau read-off — Zhang–Maginn time decomposition** (Zhang, Otani & Maginn,
*JCTC* 2015).

*The model.* The running integral $\eta(\tau)$ rises and then plateaus, but
reading the plateau off by eye (or at a fixed cutoff) is arbitrary — the curve
becomes a random walk exactly where the plateau lives. The TDM approach
instead fits a physically-motivated functional form to the *rising* part and
takes the plateau from the fit. The stress ACF of a molecular liquid is well
approximated by two decaying exponentials — a fast, ps-scale collisional decay
and a slower structural-relaxation decay:
$\langle P(0)P(t)\rangle \approx a_1 e^{-t/\tau_1} + a_2 e^{-t/\tau_2}$.
Integrating that ACF (with the $V/k_BT$ prefactor) gives the running integral
the saturating form

$$\eta(t) \;=\; b_1\!\left(1 - e^{-t/\tau_1}\right) \;+\;
b_2\!\left(1 - e^{-t/\tau_2}\right),$$

where $b_i \propto a_i \tau_i$ is the total contribution of mode $i$ to the
plateau. As $t \to \infty$ both exponentials die, so the plateau value is
$\eta_\infty = b_1 + b_2$. **All four parameters
$(b_1, b_2, \tau_1, \tau_2)$ are free parameters determined by the fit** —
nothing about them is chosen by hand; the "initial guess" below is only the
optimizer's starting point and the bounds only keep it in a physical region.

*Fit procedure*, on the uniform lag grid (spacing $\Delta t$). Notation:
$\eta(\tau)$ is the running integral above and $b(\tau)$ the block error band
from the settings table (the s.d. of $\eta(\tau)$ over the $n_b = 5$
time-blocks); a subscript $s$ denotes the smoothed version of either.

1. **Trusted window.** Smooth both $b(\tau)$ and $|\eta(\tau)|$ with a
   5 ps centered moving average, giving $b_s(\tau)$ and $|\eta|_s(\tau)$. The
   cut lag $t_\mathrm{cut}$ is the first lag beyond $t_\mathrm{floor} = 1$ ps
   where the smoothed band exceeds 40 % of the smoothed value
   ($b_s > 0.4\,|\eta|_s$ — their "40 % rule"); if never exceeded,
   $t_\mathrm{cut}$ = the last lag. Only $[t_\mathrm{floor}, t_\mathrm{cut}]$
   enters the fit at all.
2. **Sampling.** Take up to 400 unique log-spaced lag indices in
   $[t_\mathrm{floor}, t_\mathrm{cut}]$, giving samples
   $(\tau_k, y_k = \eta(\tau_k))$. Log spacing keeps the ps-scale rise (which
   pins $\tau_1$) and the approach to the plateau (which pins $b_1 + b_2$)
   comparably represented.
3. **Weighted least squares.** Find $(b_1, b_2, \tau_1, \tau_2)$ minimizing

   $$\chi^2 \;=\; \sum_k
   \left[\frac{y_k - \eta(\tau_k;\, b_1, b_2, \tau_1, \tau_2)}{\sigma_k}\right]^2,
   \qquad
   \sigma_k = \max\!\big(b_s(\tau_k),\; 10^{-3} \max_k |y_k|\big),$$

   i.e. `scipy.optimize.curve_fit` with `sigma`. Here $\sigma_k$ is the
   *assumed noise* of point $k$, so each point's weight is $1/\sigma_k^2$:
   **noisier points count *less*, not more.** Since $b_s(\tau)$ grows with
   lag, this progressively de-weights the random-walk tail, which would
   otherwise dominate an unweighted fit simply because most of the lag axis
   lives there. The $10^{-3}\max|y_k|$ floor exists for the opposite edge:
   at the earliest lags the block band can be numerically ~0, and without a
   floor those few points would get near-infinite weight and pin the fit by
   themselves.
4. **Bounds & starting point** (optimizer inputs only, not results):
   $b_1, b_2 \ge 0$; $\tau_1 \in [0.05\ \mathrm{ps},\, t_\mathrm{cut}]$;
   $\tau_2 \in [1\ \mathrm{ps},\, 10\, t_\mathrm{cut}]$. Start from
   $b_1 = 0.5\,\eta(t_\mathrm{cut})$, $b_2 = 0.6\,\eta(t_\mathrm{cut})$,
   $\tau_1 = 2$ ps, $\tau_2 = \max(0.2\, t_\mathrm{cut}, 20\ \mathrm{ps})$ —
   i.e. "the plateau is near the end of the trusted window, split between a
   fast and a slow mode."
5. **Report.** $\eta_\infty = b_1 + b_2$ with uncertainty
   $\mathrm{err} = \sqrt{\mathrm{SE}^2 + b_s(\tau_p)^2}$, where SE is the fit
   standard error of $b_1 + b_2$ (from the covariance, Jacobian
   $[1,1,0,0]$) and $\tau_p = \min(t_\mathrm{cut},\, 3\,\tau_\mathrm{slow})$
   with $\tau_\mathrm{slow} = \max(\tau_1, \tau_2)$ — the fit's own SE plus
   the data noise at the lag where the plateau is effectively reached.
6. **Convergence.** `converged = True` iff the recovered fraction
   $\eta(t_\mathrm{cut})/\eta_\infty \ge 0.90$ (the data actually climb to
   ≥ 90 % of the reported plateau inside the trusted window, so $\eta_\infty$
   is a read-off, not an extrapolation) **and**
   $\mathrm{err}/\eta_\infty < 0.5$. If the nonlinear fit fails outright, the
   fallback estimate is $\eta(t_\mathrm{cut})$ with the band as its error and
   `converged = False`.

Fixed-cutoff read-offs $\eta(100/250/500/1000\ \mathrm{ps})$ are stored for
reference only. Sensitivity check (napf6_tgdme 1M and napf6_dme 1M, r1): the
$0 \to 2$ ns equilibration cut moves $\eta$ by far less than the block band, so
pre-cut results remain valid; the cut is kept for uniformity.

## 5. Ionic conductivity (`conductivity_replicas.py`)

**Estimator** (`observables.conductivity.compute_conductivity` — a native
re-implementation of the byteff2 `onsager_calc` kernel, verified to
$2.8\times10^{-7}$ relative agreement on $\sigma_\mathrm{Onsager}$,
$\sigma_\mathrm{NE}$, and every $D_\mathrm{self}$ on a shared test
trajectory):

- **Self term:** per-species $D_i$ from the same COM-frame FFT MSD + Einstein
  OLS fit as §3, but over this observable's own window (below).
- **Collective term:** per-species collective coordinates
  $\mathbf R_i(t) = \sum_{m \in i} \mathbf r_m(t)$ (molecule COMs, system-COM
  frame). The cross displacement correlations

  $$C_{ij}(\tau) \;=\; \Big\langle
  \big[\mathbf R_i(t+\tau) - \mathbf R_i(t)\big] \cdot
  \big[\mathbf R_j(t+\tau) - \mathbf R_j(t)\big] \Big\rangle_t$$

  (FFT, all time origins) are OLS-fitted over the same window, giving the
  Onsager matrix $L_{ij} = \mathrm{slope}/(6 N)$ with $N$ the total molecule
  count. The fixed-COM constraint is projected out
  ($L \to L - \tfrac{(L\mathbf m)\mathbf 1^\top + \mathbf 1 (L\mathbf m)^\top}{M}
  + \tfrac{\mathbf m^\top L \mathbf m}{M^2}$, $\mathbf m$ = molar masses,
  $M = \sum m_i$; byteff2 eq. A.6 — an exact no-op given COM-frame positions,
  kept for parity with byteff2).
- **Conductivities** ($z_+ = +1$, $z_- = -1$, $z_0 = 0$; $T = 298$ K;
  $V = \langle V\rangle$):

  $$\sigma_\mathrm{NE} \;=\; \frac{e^2}{V k_B T} \sum_i N_i z_i^2 D_i,
  \qquad
  \sigma_\mathrm{Onsager} \;=\; \frac{e^2 N}{V k_B T}\; \mathbf z^\top L\, \mathbf z
  \;=\; \frac{e^2}{6 V k_B T}\,
  \frac{d}{d\tau} \Big\langle \big| \textstyle\sum_i z_i\, \Delta\mathbf r_i(\tau) \big|^2 \Big\rangle .$$

  $\sigma_\mathrm{NE}$ uses the **PBC** (uncorrected) self-diffusivities,
  fitted over the same preset window as the collective term (below) — not the
  Fickian window of §3 — matching byteff2 convention. Also reported:
  transference numbers

  $$t_+^\mathrm{NE} = \frac{N_+ z_+^2 D_+}{N_+ z_+^2 D_+ + N_- z_-^2 D_-},
  \qquad
  t_+^\mathrm{Ons} = \frac{z_+^2 L_{++} + z_+ z_- L_{+-}}
  {z_+^2 L_{++} + 2 z_+ z_- L_{+-} + z_-^2 L_{--}},$$

  and the ionicity (inverse Haven ratio)
  $\sigma_\mathrm{Onsager}/\sigma_\mathrm{NE}$.

**Fit-window presets (`--window`)** — the collective term is noisy and the
window genuinely moves $\sigma_\mathrm{Onsager}$, so the preset is an explicit,
reported choice. **Use the same preset on both sides of any PaiNN-vs-OPLS
comparison.**

| Preset | eq cut | stride | fit window | note |
|---|---|---|---|---|
| `byteff2_ase` (**default**) | 2 ns | 10 ps | **500–2000 ps** | byteff2's ASE-path default; collective MSD is cleanly diffusive here |
| `byteff2_gromacs` | 200 ps | 1 ps | 50–199 ps | legacy window the original OPLS numbers used; sits in the sub-diffusive crossover |
| `transport` | 2 ns | 1 ps | 1–9 ns | matches §3.2; fine for the self term, too noisy for the collective one |

**Experimental reference:** `ground_truth/conductivity_all.csv`
(concentration- and temperature-matched experimental $\sigma$, mS/cm).

## 6. Replica aggregation & error bars

Per-run rows go to `*_master.csv`; per-box aggregates (`*_by_box.csv`) report
mean ± sample standard deviation (ddof = 1) over the 4 replicas of each box
(`<system>_<conc>_298`):

- **Diffusivity:** both column families aggregated independently; per species,
  the mean uses converged replicas when ≥ 1 converged (meaningful for the
  Fickian family; every replica counts as converged under `fixed`), else all
  finite replicas. Parity figures plot one point per box with the replica
  s.d. as the error bar; non-converged boxes are excluded from MAPD summaries.
- **Viscosity:** mean over converged replicas when ≥ 1 converged, else over all
  finite replicas (flagged via `n_converged`/`n_used`); SEM also reported.
- **Conductivity:** plain mean ± s.d. over replicas (no convergence flag).

The single-trajectory scatter of $\sigma_\mathrm{Onsager}$ is intrinsically
large (a synthetic ideal-ion test with matched geometry shows ~50 % relative
s.d. per trajectory), so per-box replica means — not single runs — are the
reportable numbers.

## 7. Reproduction commands

Run from the repo root in the `fennol` conda env (CPU-only; `--jobs 8` uses a
process pool). Order matters: viscosity first (diffusivity reads its per-box
$\eta$ for the Yeh–Hummer `D_inf_sim` columns).

```bash
python -m src.analysis.observables.transport.viscosity_gk_replicas --jobs 8
python -m src.analysis.observables.transport.diffusivity_replicas  --jobs 8
python -m src.analysis.observables.transport.conductivity_replicas --jobs 8
```

All results land in `paths.OUT_PAINN_TRANSPORT_REPLICAS`
(`data/painn_transport_observables_replicas/`)
(`{viscosity,diffusivity,conductivity}<suffix>/`: `per_run/<box>/r<k>/`,
`*_master.csv`, `*_by_box.csv`, `*_config.yaml`, figures). The run tree itself
is treated as read-only (`--write-internal` opts into writing there). Every
aggregate ships a `*_config.yaml` sidecar recording the exact settings, git
SHA, and skipped runs; `--dry-run` previews run discovery, `--out-suffix`
keeps re-analyses from clobbering existing outputs.

## 8. Validation summary

- FFT MSD ≡ direct time-origin MSD (regression test `tests/test_msd_fft.py`);
  FFT cross-MSD ≡ direct to $2\times10^{-12}$.
- Conductivity kernel ≡ byteff2 `onsager_calc` to $2.8\times10^{-7}$ relative
  on a real trajectory.
- Unit chains verified against analytic ground truth on synthetic data
  (2026-08): Einstein $D$ on Brownian motion, GK $\eta$ on an
  Ornstein–Uhlenbeck stress process with known ACF integral,
  $\sigma_\mathrm{NE}$/$\sigma_\mathrm{Onsager}$ on independent ideal ions,
  Yeh–Hummer term against a hand-computed value.
- Fixed vs Fickian windows agree to ~2 % on the solvent $D$ everywhere;
  MAPD vs the PFG-NMR references is equal or slightly better under Fickian
  while replica scatter on the ions drops ~3× (the basis for making Fickian
  the production method).
- MIC make-whole assumption checked against a sequential bond-walk reference
  on a real TEGDME run (worst-case molecule/box): negligible (§2).

## 9. Short-trajectory convergence study (1 ns / 5 ns)

For the appendix showing how each observable converges with trajectory length,
all three estimators are rerun on only the **first 1 ns and first 5 ns** of the
same 20 ns runs (truncation before any analysis, so a 1 ns analysis is exactly
what one would have obtained from a 1 ns simulation). Every time-dimensioned
analysis parameter is scaled **proportionally to the trajectory length**,
$f = T / 20\,\mathrm{ns}$, with three floors where linear scaling would break
physics or statistics:

1. the Fickian window search's lower bounds ($t_\mathrm{floor} = 5$ ps and the
   5 ps minimum fit lag) stay absolute — the ballistic/caging crossover is a
   property of the liquid, not of how long we simulated;
2. the GK viscosity lag cutoff is capped at one error-estimation block,
   $\tau_\mathrm{max} = \min(1000\ \mathrm{ps},\ 0.2\,(T - t_\mathrm{eq}))$, so
   the 5-block error band stays defined;
3. the conductivity loading stride ($10\,f$ ps) would fall below the 1 ps dump
   interval, so the short presets simply use every frame.

All upper fit bounds that are defined as *fractions* of the lag axis (the fixed
$0.5\times$ window, the Fickian $t_\mathrm{max\,frac}=0.4$ search bound) scale
automatically and are unchanged. Resolved values:

| Parameter | rule | 1 ns | 5 ns | 20 ns |
|---|---|---|---|---|
| Diffusivity $t_\mathrm{eq}$ [ps] | $2000f$ | 100 | 500 | 2000 |
| Diffusivity fixed $\tau_\mathrm{min}$ [ps] | $1000f$ | 50 | 250 | 1000 |
| Viscosity $t_\mathrm{eq}$ [ps] | $2000f$ | 100 | 500 | 2000 |
| Viscosity $\tau_\mathrm{max\,lag}$ [ps] | $\min(1000, 0.2(T-t_\mathrm{eq}))$ | 180 | 900 | 1000 |
| Conductivity $t_\mathrm{eq}$ [ps] | $2000f$ | 100 | 500 | 2000 |
| Conductivity fit window [ps] | $[500, 2000]\,f$ | [25, 100] | [125, 500] | [500, 2000] |
| Conductivity stride [ps] | $10f$, floored at dump interval | 1 | 1 | 10 |

Everything else (Zhang–Maginn fitting and its 40 % noise rule, the Fickian
$\beta$ criteria, $n_\mathrm{block} = 5$, YH corrections, aggregation) is
identical to the production 20 ns analysis. Convergence failures at short
lengths are reported by the same `converged` flags, not patched over — they are
part of the result.

Implementation: the presets live in `time_presets.py` and are selected with a
single flag on each driver, e.g.

```bash
python -m src.analysis.observables.transport.viscosity_gk_replicas --time 1ns --jobs 8
python -m src.analysis.observables.transport.diffusivity_replicas  --time 1ns --jobs 8
python -m src.analysis.observables.transport.conductivity_replicas --time 1ns --jobs 8
```

(same viscosity-before-diffusivity order as §7 — the 1 ns diffusivity's YH
`D_inf_sim` uses the 1 ns GK viscosity, keeping each timescale
self-consistent). Outputs land in sibling roots
`painn_transport_observables_replicas_1ns/` and `..._5ns/` with the identical
internal layout, so every downstream reader works by swapping the root.
`--time 20ns` is the default and reproduces the production analysis
bit-identically (verified on a full run: every master-row field equal).
