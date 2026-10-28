# Transport-property convergence vs trajectory length — methods

**Figure:** `figure_transport_convergence_normalised.{png,pdf}`
**Reproduce:** `python plot_transport_convergence.py`
**Inputs:** `data_transport_convergence.csv` (panels a–d) and
`data_viscosity_convergence.csv` (panel e) — nothing else. No trajectories, no
`.stress` logs, no pipeline imports.

| Panel | Observable | Source CSV |
|---|---|---|
| a | Cation self-diffusivity `D⁺` | transport |
| b | Anion self-diffusivity `D⁻` | transport |
| c | Solvent self-diffusivity `D⁰` | transport |
| d | Ionic conductivity `σ_Onsager` | transport |
| e | Dynamic viscosity `η` (Green-Kubo) | viscosity |

---

## 1. Simulation data

| | |
|---|---|
| Model | PAINN, FP32 |
| Set | `In_distribution_exp/correct_molar/painn_nvt` |
| Ensemble | NVT, Bussi thermostat |
| Replicas | `nvt_replica_1` … `nvt_replica_4` (independent) |
| Systems | 18 = {NaOTf, NaPF6} × {DME, DEGDME, PC, TEGDME} at 0.1 M and 1 M, plus LiPF6/DME at 0.5 M and 273/298/323 K |
| Runs | 70 (replica 4 lacks the LiPF6/DME 273 K and 323 K runs) |
| Integration step | 1 fs |
| Saved-frame spacing | **1 ps** (`tdump[ps]=1.0`) → `dt_fs = 1000` |
| Length | 20 ns (20 000 frames) |

`dt_fs` was read from each run's `.fennol.yaml`, **not** from the directory name.
(The sibling `NVT_langevin` set uses `tdump = 0.1 ps` → `dt_fs = 100`; mixing them
up silently rescales every diffusivity by 10×.)

Diffusivity and conductivity come from the `.traj` files. **Viscosity does not
read the trajectory at all** — it reads the separate `.stress` log written
alongside it:

```
# time_ps  Pxx Pyy Pzz Pxy Pxz Pyz  volume[Ang^3]  T[Kelvin]
```

logged every 2 steps = **0.002 ps**, i.e. ~10⁷ samples and ~1.1 GB of ASCII per
run. Logs are parsed with `viscosity/viscosity_gk.py::_read_stress`, which
tolerates a partially-written final line and discards replayed rows after a
requeue by keeping only strictly-increasing timestamps.

## 2. The expanding window

The window **start is pinned** at `t0 = 2 ns` — the project-standard equilibration
cut, discarding the first 2 ns of every trajectory. The window **end is swept**:

```
t_end = 3, 4, 5, …, 20 ns      →  windows [2,3], [2,4], …, [2,20] ns
```

Every observable is recomputed from scratch on each window. The question the
figure answers is: *how long must a trajectory be before this observable stops
changing?*

**The windows are nested.** Consecutive windows share ~90 % of their data, so a
single curve's point-to-point scatter is **not** a statistical error bar and must
not be read as one. All uncertainty in this analysis comes from the spread across
the 4 independent replicas.

Each trajectory (and each `.stress` log) is read **once** over `[2, 20] ns`; every
window is then a plain array slice of that single load — one read per run, not 18.

## 3. Diffusivity and conductivity (panels a–d)

Frames are subsampled by a stride of 5 to an **effective lag dt of 5 ps**.

**Ionic conductivity** — `byteff2.md_utils.onsager_conductivity.onsager_calc`,
which yields both σ_Onsager (collective / Einstein–Helfand, including ion–ion
cross-correlations) and σ_NE (Nernst–Einstein, self-diffusivities only). It
discards the first 200 loaded frames as extra equilibration and fits its MSD slope
over lags `[50, 200)` loaded frames — at 5 ps/frame a **fixed 250–1000 ps lag
window**, identical for every `t_end`. Slopes are corrected for the actual frame
spacing (`byteff2` internally assumes 1 ps/frame). Volume is the mean over each
window's own frames (constant here, since NVT).

> **σ is undefined for `t_end` < 5 ns.** `onsager_calc` requires more than
> `200 + 200 = 401` loaded frames; `[2, 5] ns` at 5 ps/frame is the first window
> that supplies them. The 3 and 4 ns points are genuinely absent, not dropped.

σ_NE is **not plotted**: it is built from the same self-diffusivities as panels
a–c (it is the no-cross-correlation estimate), so it carries no convergence
information independent of them. Its values remain in the CSV.

**Self-diffusivity** — molecular **centre-of-mass** trajectories per species are
built from the same unwrapped, reordered atom positions used by `onsager_calc`, so
σ and D see identical coordinates. MSD is time-origin averaged over all origins
and all molecules of a species, computed with the FFT (Fast Correlation Algorithm)
estimator — mathematically identical to the O(T²) double loop over time origins
(verified to a maximum relative deviation of ~10⁻¹³) but O(T log T).
`D = slope / 6`, converted Å²/ps → m²/s.

### The D fit range is FIXED, and this matters

The fit is taken over a **fixed 250–1000 ps lag window** for every `t_end` — the
same lag range `onsager_calc` uses, so σ and D converge for the same reason and
are directly comparable.

This is a deliberate departure from the default convention (`fit_pct × τ_max`, an
upper bound that is a *percentage of the window*). With a percentage fit, D changes
for two reasons at once — more time origins **and** a longer fit range — and D
appears never to converge. Switching to the fixed window on this data:

| | percentage fit | fixed 250–1000 ps fit |
|---|---|---|
| D_anion converged by | 19 ns | 13.5 ns |
| D_anion replica scatter | ~45 % | ~9 % |
| systems never settling | 2–6 of 18 | 0 of 18 |

Most of the apparent non-convergence of diffusivity was the moving fit range, not
the physics.

No Yeh–Hummer finite-size correction is applied; these are PBC values. The
correction is species-independent and would shift all systems together, leaving
the convergence behaviour studied here unchanged.

## 4. Viscosity (panel e)

$$\eta = \frac{V}{k_B T}\int_0^{\tau_{cut}} \langle \sigma_{ab}(0)\,\sigma_{ab}(t)\rangle\,dt$$

with the autocorrelation averaged over the **5 independent traceless stress
components** to reduce noise:

```
Pxy ,  Pxz ,  Pyz ,  (Pxx - Pyy)/2 ,  (Pyy - Pzz)/2
```

Pressures are converted atm → Pa; the ACF is computed by FFT (unbiased — each lag
divided by its number of overlapping pairs); the running integral `η(τ)` is
accumulated so the plateau can be inspected rather than assumed. Volume is the
mean over each window's own samples.

> **The prefactor uses the NOMINAL setpoint temperature** parsed from the run
> name, not the `T` column of the `.stress` log — that column is a known-incorrect
> on-step kinetic-energy quantity.

**The integration cutoff `τ_cut` is held FIXED while `t_end` is swept.** GK
viscosity has two independent times — the integration cutoff (a *lag*) and the
trajectory length used to average the ACF — and only the second is the subject of
this study. Letting both move at once produces an uninterpretable curve, the same
trap the percentage-based fit range creates for diffusivity. η was recorded at
cutoffs of 10, 20, 50 and 100 ps in the source data.

### The plotted quantity is η_∞ (Zhang-Maginn), not a fixed cutoff

The running integral is fitted to the form implied by a two-exponential stress ACF

$$\eta(t) = b_1\left(1-e^{-t/\tau_1}\right) + b_2\left(1-e^{-t/\tau_2}\right),
\qquad \eta_\infty = b_1 + b_2$$

with the fit range chosen from a block-resampling noise band (5 equal time blocks
within the window, each re-integrated and interpolated onto the full lag grid).

This extrapolation is not cosmetic: **for the viscous systems the ACF has not
plateaued by 200 ps**, so any fixed-cutoff η is truncation-biased low.

| System | η@20 ps | η@50 ps | η@100 ps | η_∞ (TDM) | η(100)/η(50) |
|---|---|---|---|---|---|
| NaOTf/PC 1 M | 9.3 | 14.5 | 18.4 | 18.2 | **1.27** |
| NaOTf/TEGDME 1 M | 8.0 | 12.1 | 15.3 | 16.4 | **1.26** |
| NaPF6/PC 1 M | 8.0 | 11.9 | 14.4 | 14.3 | **1.21** |
| NaPF6/DME 0.1 M | 0.77 | 0.77 | 0.77 | 0.78 | 0.99 |
| LiPF6/DME 0.5 M, 298 K | 0.79 | 0.81 | 0.83 | 0.80 | 1.02 |

Eight systems — all the PC and TEGDME ones plus NaPF6/DEGDME at 1 M — rise more
than 10 % between the 50 and 100 ps cutoffs. The low-viscosity DME systems plateau
by ~20 ps and their fixed-cutoff and extrapolated values agree to ~2 %. The TDM
fit was flagged converged in 100 % of runs at the 20 ns window.

## 5. Averaging over replicas

For each (system, observable, `t_end`):

```
value_mean = mean over the replicas that have that system
value_sd   = sample standard deviation over the same replicas   (ddof = 1)
n_replicas = 4, or 3 for LiPF6/DME at 273 K and 323 K
```

Replicas are averaged **only at the very end** — each replica is swept over all
windows independently first. `value_mean`, `value_sd` and `n_replicas` are all in
both CSVs.

## 6. Normalisation in the figure

Each panel plots, per system,

```
y(t_end) = value_mean(t_end) / value_mean(t_end = 20 ns)
```

so the y-axis reads "how far off is a shorter run from the 20 ns answer?".

> **Every curve passes through 1.0 at 20 ns by construction.** Systems are *not*
> converging to a common physical value — each is divided by its own endpoint.
> The endpoint carries no information; only the *approach* to the ±5 % band does.
> Absolute values (units implied by `observable`) are in the CSVs.

The figure draws the replica **mean** only. `value_sd` is in the CSVs if bands are
wanted.

**Encoding** (matched to the MLP transport-parity appendix figure): colour =
solvent, marker = salt, fill = concentration (open 0.1 M, half 0.5 M, filled
1.0 M). The three LiPF6/DME temperatures share a marker and are separated by line
style.

## 7. Convergence criterion quoted in the text

"Converged by `t_end`" = the earliest window from which the replica mean stays
within ±5 % of its 20 ns value for **all** later windows. A single late excursion
therefore pushes the answer out — a deliberately strict criterion that reads later
than eyeballing the curves.

## 8. Result summary (18 systems)

| Observable | Converged by | Never settles | Replica scatter | Drift over last 5 ns |
|---|---|---|---|---|
| D_solvent | 4 ns | 0/18 | 2.0 % | — |
| D_cation | 10 ns | 0/18 | 8.0 % | 7.4 % |
| η_∞ (Green-Kubo) | 11 ns | 0/18 | 6.6 % | — |
| σ_NE (not plotted) | 12 ns | 0/18 | 7.4 % | 2.5 % |
| D_anion | 13.5 ns | 0/18 | 9.3 % | 9.5 % |
| **σ_Onsager** | **18 ns** | **4/18** | **24.7 %** | 8.6 % |

Collective Onsager conductivity is the one transport property that 20 ns does not
pin down: ~3× the replica scatter of every other observable, and 4 of 18 systems
never settle. σ_NE, which omits cross-correlations, converges fine — so the
bottleneck is specifically the ion–ion cross-correlation term.

Viscosity is among the **best**-converged properties here, but with a caveat: the
replica spread stops shrinking past ~8–10 ns, staying at roughly ±10–20 % for the
viscous systems. Beyond ~10 ns, more trajectory buys very little; reducing the
uncertainty on η requires **more replicas**, not longer runs.

Physically the viscosities are internally consistent: DME < DEGDME < PC ≈ TEGDME,
1 M above 0.1 M for every salt/solvent pair, and the three LiPF6/DME temperatures
ordering correctly (0.45 cP at 323 K → 1.09 cP at 273 K).

## 9. Provenance

| | |
|---|---|
| Source (a–d) | `.../correct_molar/painn_nvt/convergence/transport_expanding_window/` |
| Source (e) | `.../correct_molar/painn_nvt/convergence/viscosity_expanding_window/` |
| Compute (a–d) | `observable_scripts/run_expanding_window_transport_painn_indist_nvt.py` |
| Compute (e) | `observable_scripts/run_expanding_window_viscosity_painn_correct_molar.py` |
| GK kernel | `observable_scripts/viscosity/viscosity_gk.py` (`_read_stress`, `_acf_fft`, `_running_integral`, `fit_eta_tdm`) |
| Plot pipelines | `observable_scripts/plot_expanding_window_transport_painn_indist_nvt.py --fixed-fit-ns 0.25 1.0` and `plot_expanding_window_viscosity.py` |
| Batch | `observable_scripts/submit_expanding_window_correct_molar.sh {transport,viscosity}` |
| Parameter records | `*/run_manifest.json` in each source dir |
| Raw per-replica rows | `expanding_window_transport_all.csv` and `expanding_window_viscosity_all.csv` (1260 rows each) |
| Raw curves | `intermediate/*.npz` in each source dir — MSD curves (transport) and running `η(τ)` curves (viscosity), so D can be refitted over any lag range and η re-read at any cutoff with **no** recompute |

The two CSVs in this folder are the aggregated (system × observable × window)
tables actually plotted: 1620 rows (transport) and 324 rows (viscosity).
