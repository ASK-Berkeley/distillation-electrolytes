# electrolytes: notes for agents

Distillation of UMA-S into per-system PaiNN students, 20 ns transport MD with
the students, and the paper that reports it. **Read `README.md` first**: it is
the source of truth for the pipeline, the experiment ids, the data layout, and
which script makes which figure. This file holds only what an agent needs
beyond that.

## Where things are

| what | where |
|---|---|
| repo | `/u/iamin/electrolytes` (Delta); mirror at `/global/homes/i/ishan_a/electrolytes` (Perlmutter) |
| data root | `data -> /work/nvme/beye/iamin/electrolytes_data` (scratch, not archived; `.traj` files are reproducible from `.dyn.restart`) |
| what lives on which machine | `python -m src.analysis.observables.paths` prints the manifest with a reachability check; `paths.py` is the single source of truth, never hardcode a data path in a script |
| UMA-S teacher (`uma-s-1p1`, the only teacher; 1p2 numbers are not comparable) | Delta `/projects/beye/iamin/distillation_project/models/uma-s-1p1.pt`, Perlmutter `/global/cfs/cdirs/m4558/distillation_project/models/uma-s-1p1.pt` |
| trained students | `data/experiments/<exp>/<system>/model/best_model.fnx` (no EMA; `best_model.fnx` is the production model) |
| committed analysis outputs | `results/` (~180 MB): the summary CSVs, `*_config.yaml` sidecars, RDF curves and the transport `per_run/` curves (CSV/JSON) the figures read. `paths.OBS_ROOT` prefers `data/` and falls back to `results/`, so figures draw from a clean checkout |
| released inputs | `paper_data/` (44 MB): the 9 students and the initial/corrected boxes, with `MANIFEST.csv`; refresh with `python -m src.export_paper_data` (`--check`) |
| submodules | `fennol/` (FeNNol fork) only; `git submodule update --init` after a fresh clone |
| paper | `paper/paper_electrolytes` is its own git repo, gitignored here and not part of the released code; figure scripts in `paper/paper_figures/<figure>/` write into `paths.FIGURES` |
| logs | scripts write their own log files (under `slurm_logs/` or next to their outputs); never rely on piped stdout |

## Environments and scheduling

* `fennol` (`fennol_env.yaml` + editable install of the FeNNol fork; the
  `fennol/` submodule pins the paper's commit, the env on Delta is installed
  from the working checkout `/u/iamin/FeNNol`): FeNNol/JAX training
  and MD, all observables analysis, benchmarks. `fairchemV3`
  (`fairchemV3_env.yaml`; the `fairchemV3-setup` skill has the install order):
  UMA/AllScAIP/MACE MD, dataset creation, UMA labeling. `mace_polar`: only the
  gas-phase MACE-Polar scans. Always mamba, never conda.
* Delta: account `beye-dtai-gh`, partitions `ghx4` / `ghx4-interactive` (2 h
  cap), login nodes have no GPU. Use the `delta-slurm` skill for interactive
  allocations; hold a `salloc` open in tmux and `srun` into it rather than
  chaining `salloc` and `srun`. Prefer `--nodes=N --gpus-per-node=1` over one
  multi-GPU node for queue time. Perlmutter: account `m5250_g`, `-C gpu`.
* Long GPU jobs that must survive the 2 h interactive cap follow the
  detached-tmux + `salloc` cycle pattern in
  `src/simulate/fennol_configs/launch_exp07_npt_interactive.sh` and
  `src/analysis/labeling/launch_uma_labeling_interactive.sh` (`--dry-run`,
  `--stop`, cycle-safe `.rc` files). Batch production MD goes through
  `src/simulate/run_fennol_sim.sh` (self-requeue, resume from `.dyn.restart`).
* torch.compile on Delta needs `CXX=g++ CC=gcc` to bypass the Cray wrapper.

## Conventions

* Run everything from the repo root as a module: `python -m src.analysis.observables.transport.diffusivity_replicas`. No `sys.path` hacks; packages use relative imports.
* Physical inputs (temperature, viscosity, concentration) are explicit CLI flags, never parsed from run-folder names.
* Pressure and stress in **bar**, never GPa. Diffusivities in 1e-10 m^2/s, conductivity in mS/cm, viscosity in cP.
* FeNNol training pickles must carry the cell under the key `cell` (singular); FeNNol silently fakes a 1000 A box otherwise.
* Training recipe: AdamW + MAE + one-cycle schedule, no `ema_decay` (see `src/train/fennol_configs/exp01_mixed_noise.yaml`).
* Every analysis driver records the git SHA and all settings in a `*_config.yaml` sidecar (`paths.git_sha()`); supports `--dry-run` and `--out-suffix`.
* After rerunning an analysis driver, refresh the committed copy: `python -m src.analysis.observables.export_results`, then commit `results/`. `--check` prints what is stale and exits nonzero (nothing breaks if you skip it: local figures read `data/` directly, only the committed copy goes behind). A genuinely new output *type* needs a line in that script's `PATTERNS`; reruns of existing observables are picked up automatically.
* Prose for the paper and READMEs must not read as AI-written: minimal em dashes, no stock phrasing. Multi-panel paper figures are wide 4-column rows; PNG output only.
* Commit only when asked. Do not touch `data/`; it is gitignored scratch.

## Experiment ids

`version01_mixed_noise` (datasets) -> `exp01_mixed_noise` (students, NPT
replicas, Figure 2, NVT seeds) -> `exp04_correct_molarity_samebox`
(corrected-molarity boxes and the 4 x 20 ns NVT transport runs; every transport
number). `version05_no_noise` -> `exp07_no_noise` is the noise ablation.
`exp00`, `exp02`, `exp03`, `exp05` are pilots or superseded layouts.

## Code map

* `src/data_prep/`: `box_creation/build_initial_boxes.py` (the 16 initial 30 A boxes, packed periodically with `pbc`), `box_creation/build_boxes_correct_M.py` (molarity-corrected production boxes), `box_creation/relax_boxes.py`, `box_creation/molecules/` (the solute/solvent `.xyz` both scripts pack, in-repo), `dataset_creation/create_rattled_dataset.py`, `equilibration/select_nvt_seeds.py`.
* `src/train/launch_fennol_experiment.py` fans one YAML template over its `systems:` list and submits `fennol_train`.
* `src/simulate/fennol_npt.py` is the student MD driver (fresh start with `--from-xyz`, resume without); `launch_transport.sh` is the production launcher; `uma_npt.py`, `allscaip.py`, `mace_md.py` are the teacher-class drivers.
* `src/analysis/observables/`: estimators at the top (`diffusivity.py`, `conductivity.py`, `viscosity_gk.py`, `rdf.py`, `density.py`), drivers in `structural/` and `transport/`. `transport/METHODS.md` is the paper-grade method description; `tests/test_msd_fft.py` is the one regression test.
* `src/analysis/labeling/label_traj_with_uma.py` writes `uma_labels_every<n>ps.npz`; 2 ps for the noise ablation, 20 ps for Figure 2, 40 ps for the SI stability figure.
* `src/benchmark/eval_*.py`: single-point accuracy and timing on a dataset pickle.

## Paper

* Main text is limited to 3,500 words (Nature Computational Science Article);
  the `paper-wordcount` skill counts Introduction + Results + Discussion.
* Figures are made by the scripts in `paper/paper_figures/`; the SI transport
  tables come from `appendix/transport_data_tables/make_transport_tables.py`
  and are pasted into `sections_appendix/_7_transport_data.tex` verbatim. Other SI tables are hand-maintained.
* `paper/*.md` are working notes, not manuscript text.

## Reference docs

| topic | file |
|---|---|
| transport estimators, settings, validation | `src/analysis/observables/transport/METHODS.md` |
| observables package layout | `src/analysis/observables/README.md` |
| FeNNol physics modules (FIDs, params) | `fennol/README.md` (the fork, as a submodule) |
| which systems have 293 K UMA trajectories | `docs/uma_trajectory_coverage_293K.md` |
| data roots, model checkpoints, figure output | `src/analysis/observables/paths.py` |
| environment pitfalls (CUDA wheel, pins, toolchain) | the header comments in `fairchemV3_env.yaml` |
