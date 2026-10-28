# UMA trajectory coverage vs. experimental systems (293 K)

Which of the experimental salt + solvent systems have a UMA simulation
trajectory, at which concentrations.

- **Experimental data:** `src/analysis/observables/ground_truth/experimental_data.md`
  (15 salt + solvent systems, measured at 293.2 K and 298.2 K).
- **Trajectories:** a shared project directory on Perlmutter
  (`.../distillation_project/data/raw_data_from_UMA_simulation/293K/`), organized
  into three concentration folders: `UMA_trajs_293K_0_1M_100ps`,
  `UMA_trajs_293K_0_5M_100ps`, `UMA_trajs_293K_1M_1ns`.

Name mapping: `diglyme`/`deg` = DEGDME, `tgdme` = TEGDME,
`propylene_carbonate` = PC.

> **Folder labels are correct.** Concentrations were verified by computing
> molarity from the first frame of each trajectory as
> (number of cations) / (N_A × box volume): the `0_1M` folder really is ~0.1 M
> (0.12–0.15 M), `0_5M` is ~0.5 M (0.64 M), and `1M` is ~1 M (1.0–1.27 M). The
> `..._1m_...` token inside some `0_1M` subfolder names is a stale label — those
> systems are 0.1 M. (Spread within a tier is just per-solvent box-volume
> differences; values trend slightly high because the first frame is
> pre-equilibration.)

## Coverage by system

| System (salt + solvent) | Exp. concs | Traj concs available | Status |
|---|---|---|---|
| CsPF6 / DME | 0.05 M | 1 M | ⚠️ traj exists but **wrong conc** (1 M, not 0.05 M) |
| KPF6 / DME | 0.1, 1 M | — | ❌ **no trajectory at all** |
| LiPF6 / DME | 0.1, 0.5 M | 0.5 M, 1 M | ⚠️ 0.5 M ✓; 0.1 M missing (1 M traj has no exp) |
| NaOTf / DEGDME | 0.1, 1 M | 0.1, 1 M | ✅ |
| NaOTf / DME | 0.1, 1 M | 0.1, 1 M | ✅ |
| NaOTf / PC | 0.1, 1 M | 0.1, 1 M | ✅ |
| NaOTf / TEGDME | 0.1, 1 M | 0.1, 1 M | ✅ |
| NaPF6 / DEGDME | 0.1, 1 M | 0.1, 1 M | ✅ |
| NaPF6 / DME | 0.1, 1 M | 0.1, 1 M | ✅ |
| NaPF6 / PC | 0.1, 1 M | 0.1, 1 M | ✅ |
| NaPF6 / TEGDME | 0.1, 1 M | 1 M | ⚠️ 1 M ✓; 0.1 M missing |

## Summary

- **Fully covered (7):** all four NaOTf systems, plus NaPF6/DEGDME, NaPF6/DME,
  NaPF6/PC.
- **No trajectory at all (1):** KPF6/DME.
- **Conc mismatch / partial (3):** CsPF6/DME (only 1 M, exp is 0.05 M),
  LiPF6/DME (missing 0.1 M), NaPF6/TEGDME (missing 0.1 M).
- **Extra trajectories with _no_ experimental data:** NaPF6 in **DMC**, **EC**,
  **THF** (all at 1 M), plus the off-experiment concentrations noted above
  (CsPF6/DME 1 M, LiPF6/DME 1 M, NaPF6/DME 0.5 M).

## Datasets built — `version01_mixed_noise`

The 9 systems with both experimental data and a trajectory were turned into
training datasets (`data/datasets/version01_mixed_noise/<system>/`) by
`src/data_prep/dataset_creation/create_rattled_dataset.py` (run as
`python -m src.data_prep.dataset_creation.create_rattled_dataset`). Each concentration contributes 4000
frames over the first 100 ps — 2000 unnoised MD frames (every 50 fs) interleaved
with 2000 σ=0.1 Å rattled frames (every 50 fs, offset 20 fs) — all UMA-labeled
(uma-s-1p1). Each pool is split 90/10 train/val *within* every concentration, so
a 2-concentration system holds 8000 frames and a 1-concentration system 4000.

Only off-experiment concentrations were dropped: CsPF6/DME (only a 1 M traj, exp
is 0.05 M → excluded entirely), NaPF6/DME 0.5 M, and LiPF6/DME 1 M.

| Dataset (system) | Concs used | MD | noised | train | val | total |
|---|---|---:|---:|---:|---:|---:|
| lipf6_dme | 0.5 M | 2000 | 2000 | 3600 | 400 | 4000 |
| naotf_diglyme | 0.1 + 1 M | 4000 | 4000 | 7200 | 800 | 8000 |
| naotf_dme | 0.1 + 1 M | 4000 | 4000 | 7200 | 800 | 8000 |
| naotf_pc | 0.1 + 1 M | 4000 | 4000 | 7200 | 800 | 8000 |
| naotf_tgdme | 0.1 + 1 M | 4000 | 4000 | 7200 | 800 | 8000 |
| napf6_diglyme | 0.1 + 1 M | 4000 | 4000 | 7200 | 800 | 8000 |
| napf6_dme | 0.1 + 1 M | 4000 | 4000 | 7200 | 800 | 8000 |
| napf6_pc | 0.1 + 1 M | 4000 | 4000 | 7200 | 800 | 8000 |
| napf6_tgdme | 1 M | 2000 | 2000 | 3600 | 400 | 4000 |
| **Total** | | **32000** | **32000** | **57600** | **6400** | **64000** |

Each dataset folder also has `linref.json` (per-element energy reference +
force/stress RMS normalizers used by training) and one
`simulation_start_<conc>.xyz` per concentration (the 100 ps frame, to seed MD).

Training config: `src/train/fennol_configs/exp01_mixed_noise.yaml` (fan out with
`src/train/launch_fennol_experiment.py --nersc`).
