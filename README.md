# electrolytes

Code behind *Distilling Large-Scale Machine Learning Interatomic Potentials to
Predict Long-Timescale Electrolyte Transport Properties*. We distill the
UMA-S foundation potential (OMol task) into a small PaiNN student per
salt/solvent pair, run 20 ns molecular dynamics with the student, and compute
self-diffusivities, ionic conductivities and shear viscosities of sixteen Na-
and Li-ion electrolytes.

```
packed boxes (Packmol)  ->  teacher MD (UMA-S)  ->  noise-augmented dataset
        ->  PaiNN student (FeNNol)  ->  20 ns NVT production runs
        ->  D, sigma, eta  ->  paper figures
```

Everything is run from the repository root. The Slurm wrappers (`*.sh`) carry
the account and partition names of the clusters we used; edit those headers for
your own machine, or call the Python drivers directly.

**Names used in the code.** Systems are `<salt>_<solvent>`, with `diglyme` =
DEGDME and `tgdme` = TEGDME; "painn" is the student. Experiment ids name the
`data/` folders: `version01_mixed_noise` (distillation datasets) ->
`exp01_mixed_noise_all_systems` (the students and their NPT runs) ->
`exp04_correct_molarity_samebox` (corrected boxes and the transport runs);
`version05_no_noise` -> `exp07_no_noise` is the noise ablation.

## 1. Environments

Two conda environments, one per model family:

| env | yaml | used for |
|---|---|---|
| `fairchemV3` | `fairchemV3_env.yaml` | the UMA-S teacher: teacher MD, dataset creation, UMA labeling |
| `fennol` | `fennol_env.yaml` | the PaiNN student: training, production MD, and all analysis |

```bash
mamba env create -f fairchemV3_env.yaml    # then reinstall the CUDA torch wheel, see the yaml header
mamba env create -f fennol_env.yaml
```

All UMA-S runs in the paper (teacher MD, dataset labeling) used an editable
install of fairchem from
[FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem) at commit
`28292f090743f36c1151002964140b04d4b2c6af` (2026-04-24, `main`), with one local
patch: `.contiguous()` added to `cell_col` in
`src/fairchem/core/calculate/torchsim_interface.py`. To reproduce exactly:

```bash
git clone https://github.com/FAIR-Chem/fairchem && cd fairchem
git checkout 28292f090743f36c1151002964140b04d4b2c6af
pip install -e packages/fairchem-core
```

The student is trained and run with **our fork of FeNNol**,
[ishanthewizard/FeNNol-fork](https://github.com/ishanthewizard/FeNNol-fork),
which adds the stress and charge-current logging the transport analysis
depends on. The paper used commit `5d55122e1d31f4cda7d615f1588b8a25b2236001`
(2026-07-16), to which the `fennol/` submodule is pinned. `fennol_env.yaml`
installs FeNNol's JAX dependencies but not FeNNol itself, so install it into the env:

```bash
git clone --recurse-submodules <this repo>      # or: git submodule update --init
mamba activate fennol && pip install -e fennol
# without the submodule:
# git clone https://github.com/ishanthewizard/FeNNol-fork fennol && git -C fennol checkout 5d55122e1d31f4cda7d615f1588b8a25b2236001
```

Both environments need a CUDA GPU for MD and training; the analysis (section 6)
is CPU-only. Two tools are installed separately: Packmol for box building
(`mamba install -c conda-forge packmol`) and GROMACS 2025 for the OPLS-AA
baseline (see `src/baselines/OPLS/README.md`). The SI interaction scans use
further models with their own environments, listed in
`paper/paper_figures/appendix/interaction_scanning/README.md`.

## 2. Building the initial boxes

Each system starts as a 30 x 30 x 30 A cube holding the number of ion pairs
that corresponds to the target molarity at that fixed 27,000 A^3 volume (2
pairs at 0.1 M, 9 at 0.5 M, 17 at 1 M), with Packmol filling the rest with
solvent. `build_initial_boxes.py` rebuilds all 16 of them:

```bash
mamba activate fennol
PACKMOL=/path/to/packmol python -m src.data_prep.box_creation.build_initial_boxes
```

The molecules are in `src/data_prep/box_creation/molecules/` and the boxes land
in `data/reproduced_nitesh_setup_boxes/<system_id>/initial_<conc>.xyz`;
`--dry-run` prints the composition table without packing.

Before the teacher runs, the packed boxes were energy-minimized and
equilibrated for 1 ns of NPT at 298 K with OPLS-AA, following the protocol of
the structural study the boxes come from (Kumar et al.). Our OPLS-AA/GROMACS
pipeline is in `src/baselines/OPLS/` (see its [README](src/baselines/OPLS/README.md)):
its build, minimization and NPT steps perform this pre-equilibration, and the
same code runs the OPLS-AA transport baseline of the paper.

These are the *initial* boxes, and they equilibrate over-concentrated: molarity
is a property of the equilibrated cell, not the packed one, so fixing the
volume at 27,000 A^3 leaves the nominal 1 M boxes near 1.2-1.3 M. The
production transport runs use corrected boxes that fix the solute-to-solvent
*ratio* at the student's own equilibrium density; section 5.1 builds them.

## 3. Simulating the UMA-S teacher

Reference data comes from 1 ns NPT trajectories of each system with UMA-S
(checkpoint `uma-s-1p1`, OMol head) at 293 K and 1 bar, 1 fs timestep,
Nose-Hoover thermostat and barostat. `src/simulate/uma_npt.py` runs this through
ASE and `FAIRChemCalculator`, one trajectory per GPU, and resumes if the `.traj`
already exists:

```bash
mamba activate fairchemV3
ase convert box.xyz <run_dir>/<run_dir>.traj      # the driver reads <run_dir>/<basename>.traj
python -m src.simulate.uma_npt <run_dir> --models uma-s-1p1.pt --temperature 293 --steps 1000000 --interval 10
```

The run appends to that `.traj` and resumes from its last frame. The 298 K
teacher densities of Fig. 2b come from 1 ns runs of the same driver, launched by
`src/simulate/fairchem_configs/launch_uma_298K_density.sh`.

## 4. Training the student

### 4.1 Build the distillation dataset

`src/data_prep/dataset_creation/create_rattled_dataset.py` turns the teacher
trajectories of one salt/solvent pair (one per concentration) into the training
set: from the first 100 ps it takes unperturbed frames every 50 fs and, on a
second grid offset by 20 fs, frames rattled by Gaussian noise of 0.1 A that it
re-labels with UMA-S. Each grid is split 90/10 and the concentrations are pooled.

```bash
mamba activate fairchemV3
python -m src.data_prep.dataset_creation.create_rattled_dataset \
    --traj napf6_dme_0.1M.traj napf6_dme_1M.traj --concentrations 0.1M 1M \
    --uma-checkpoint uma-s-1p1.pt --output-dir data/datasets/version01_mixed_noise/napf6_dme
```

This writes `training_all.pkl` (what training reads), `linref.json`
(per-element energy reference and force/stress normalizers) and one
`simulation_start_<conc>.xyz` seed frame per concentration.
`run_create_rattled_dataset.sh` in the same directory is the batch wrapper that
built all of ours; `run_create_no_noise_dataset.sh` (`--no-noise`) built the
ablation datasets.

### 4.2 Train

`src/train/fennol_configs/exp01_mixed_noise.yaml` is the exact recipe behind
every student in the paper. Its `base:` block is the model and training scheme
(PaiNN, 128 features, 3 message-passing layers, 5.1 A cutoff, trainable ZBL
repulsion; equal-weight energy/force/stress loss normalized by the force RMS;
AdamW with a one-cycle cosine schedule peaking at 8e-4; 275 epochs, batch 4, no
EMA) and its `systems:` list names the salt/solvent pairs. One model is trained
per pair, shared across its concentrations.

`fennol_train` takes a single complete YAML, so
`src/train/launch_fennol_experiment.py` fans the template out: for each system
it fills in the dataset path, output directory and loss normalizers from that
dataset's `linref.json`, writes `data/experiments/exp01_mixed_noise_all_systems/<system>/model/config.yaml`,
and submits it.

```bash
mamba activate fennol
python src/train/launch_fennol_experiment.py --experiment src/train/fennol_configs/exp01_mixed_noise.yaml --dry-run
python src/train/launch_fennol_experiment.py --experiment src/train/fennol_configs/exp01_mixed_noise.yaml --no-submit   # just write the configs
python src/train/launch_fennol_experiment.py --experiment src/train/fennol_configs/exp01_mixed_noise.yaml               # submit via run_fennol_train_delta.sh
```

Set `experiment.data_root` in the YAML to your data directory. With
`--no-submit` you can run `fennol_train <system>/model/config.yaml` yourself on
any GPU; a model takes 9 to 22 h on one GH200. The result is
`<system>/model/best_model.fnx`. Training logs to Weights & Biases (project
name in the YAML). The ablation students use `exp07_no_noise.yaml` the same way.

## 5. Running student simulations

Production MD uses `src/simulate/fennol_npt.py`, a driver around FeNNol's JAX
MD engine. Point it at a seed frame and a new run directory to start; point it
at an existing run directory to resume from its `.dyn.restart`:

```bash
mamba activate fennol
python -m src.simulate.fennol_npt <run_dir> --from-xyz seed.xyz --model best_model.fnx \
    --ensemble NVT --thermostat BUSSI --gamma-thz 0.1 --temperature 298 \
    --steps 20000000 --interval 1000 --stress-log --stress-interval 2 --current-log --current-interval 2 --seed 1
```

Those are the paper's transport settings: NVT at 298 K with a weak global
Bussi thermostat (10 ps), 1 fs timestep, a frame every 1 ps, and the stress
tensor and charge current logged every 2 fs to `<run>.stress` and
`<run>.current` (the analysis uses only the stress). `--ensemble NPT` gives the
Langevin NPT runs used for equilibration and densities.

On a cluster, `src/simulate/run_fennol_sim.sh` is the Slurm job for one run:
it takes every setting above as an environment variable, requeues itself
before the wall limit and resumes from `.dyn.restart`, so one submission runs
to completion. `src/simulate/launch_transport.sh` is what we ran for the paper:
it submits `run_fennol_sim.sh` for all sixteen systems, four replicas each
(`NREPLICAS=4`), with independent seeds. `src/simulate/check_transport_progress.sh`
reports how many ns each run has completed.

### 5.1 Corrected-molarity production boxes (`exp04_correct_molarity_samebox`)

The transport runs start from boxes rebuilt so that each equilibrates at its
target molarity. The chain below produced every exp04 box and seed in the
paper; the defaults point at `data/experiments/`, and `--dry-run` on each Python
step previews what it would do.

```bash
mamba activate fennol
E01=data/experiments/exp01_mixed_noise_all_systems
E04=data/experiments/exp04_correct_molarity_samebox

# 1. Equilibrium density of each exp01 student on its initial box: from replica 0
#    of its 298 K NPT runs (E01/<system>/npt_<conc>_298K_20ns_1ps_replicas/, launched
#    by src/simulate/launch_exp01_production.sh; these also give the Fig. 2b densities),
#    average the density over 1-4 ns and keep the 3-4 ns frame closest to it.
python -m src.data_prep.equilibration.select_nvt_seeds --experiment-root $E01

# 2. Repack each system with the ion-pair count unchanged and the solvent count
#    that gives the target molarity at that density -> E04/<system>/initial_<conc>.xyz
PACKMOL=/path/to/packmol python -m src.data_prep.box_creation.build_boxes_correct_M

# 3. The students are unchanged: copy each exp01 model next to its new boxes.
for d in $E04/*/initial_*.xyz; do s=$(basename "$(dirname "$d")")
  mkdir -p $E04/$s/model && cp $E01/$s/model/best_model.fnx $E04/$s/model/; done

# 4. Fixed-cell UMA-S relaxation to remove packing overlaps (GPU, fairchemV3 env)
#    -> E04/<system>/initial_relaxed_<conc>.xyz
mamba activate fairchemV3
python -m src.data_prep.box_creation.relax_boxes --model uma-s-1p1.pt
mamba activate fennol

# 5. 1 ns NPT at 298 K and 1 bar with each student (Slurm) -> E04/<system>/npt_equil_<conc>_298K_1ns/
src/simulate/fennol_configs/launch_npt_equilibration.sh

# 6. Seed frame: mean density over the last 500 ps, closest frame in the last 300 ps
#    -> E04/<system>/nvt_seed_<conc>_298K.xyz
python -m src.data_prep.equilibration.select_nvt_seeds --experiment-root $E04 --layout flat \
    --eq-window-ns 1 --density-start-ns 0.5 --density-end-ns 1.0 \
    --select-start-ns 0.7 --select-end-ns 1.0

# 7. Four 20 ns NVT replicas per system (Slurm) -> E04/<system>/transport_<conc>_298K_r<k>/
src/simulate/launch_transport.sh
```

Step 6 with these flags reproduces the recorded seed frames of all sixteen
systems, and step 2 reproduces the atom counts of every exp04 box.

### 5.2 Other runs in the paper

| Run | Launcher |
|---|---|
| Student 293 K NPT, 5 x 1 ns per system (RDFs, Fig. 2a) | `src/simulate/launch_painn_1ns_replicas.sh` (r1-r4; r0 is an earlier run with the same settings) |
| Student 298 K NPT, 4 x 20 ns per box (densities, Fig. 2b; step 1 above) | `src/simulate/launch_exp01_production.sh` |
| No-noise ablation students, 1 ns NPT | `src/simulate/fennol_configs/launch_exp07_npt_interactive.sh` |
| UMA-S relabeling of student frames (Fig. 2c, SI stability and ablation) | `src/analysis/labeling/label_traj_with_uma.py` (batch: `launch_uma_labeling_interactive.sh`) |
| OPLS-AA baseline | `src/baselines/OPLS/` (see its README) |

## 6. Analysis

`src/analysis/observables/` holds the estimators and the driver scripts that
produced every number in the paper. Run them as modules from the repository
root in the `fennol` env; all are CPU-only and take `--dry-run` to preview
which runs they found.

**Structural checks against the teacher** (`structural/`): RDFs of the
teacher and student trajectories with identical sampling, and student
densities.

```bash
python -m src.analysis.observables.structural.compute_uma_1M_rdfs
python -m src.analysis.observables.structural.compute_painn_replica_rdfs
python -m src.analysis.observables.structural.density_replicas --jobs 16
python -m src.analysis.observables.structural.uma_densities --jobs 8     # teacher densities, 298 K / 1 ns runs
```

**Transport coefficients** (`transport/`): per replica after discarding the
first 2 ns, reported as mean and standard deviation over replicas.
Self-diffusivity from the Einstein relation with an FFT mean-squared
displacement and the Yeh-Hummer finite-size correction, ionic conductivity
from the Onsager (collective) and Nernst-Einstein estimators, shear viscosity
from the Green-Kubo stress autocorrelation with a time-decomposition fit.
Viscosity runs first because diffusivity reads it for the Yeh-Hummer
correction.

```bash
python -m src.analysis.observables.transport.viscosity_gk_replicas --jobs 8
python -m src.analysis.observables.transport.diffusivity_replicas  --jobs 8
python -m src.analysis.observables.transport.conductivity_replicas --jobs 8
```

The `*_opls_replicas.py` scripts push GROMACS `.xtc` trajectories through the
same estimators for the OPLS-AA baseline. Each driver writes `per_run/<system>/r<k>/`
results, `*_master.csv` (one row per replica), `*_by_box.csv` (mean and s.d.)
and a `*_config.yaml` recording every setting. Experimental reference values
are in `ground_truth/`. `transport/METHODS.md` documents every estimator,
window and validation check in detail, and `src/analysis/observables/README.md`
describes the package.

Input and output locations for all of these are defined once in
`src/analysis/observables/paths.py`; edit that file to point at your own data
(`python -m src.analysis.observables.paths` prints the manifest).

## 7. Paper figures

Each figure has a directory under `paper/paper_figures/` with the script that
draws it, run as a module from the repository root, e.g.
`python -m paper.paper_figures.main_paper.figure_4.plot_figure4_diffusivity_parity_painn_opls`.

| Figure | Directory |
|---|---|
| Fig. 1 (workflow schematic; HTML rendered by `render.sh`) | `main_paper/figure_1/` |
| Figs. 2, 3, 4 | `main_paper/figure_{2,3,4}/` |
| SI: molarity correction | `appendix/molarity_correction/` |
| SI: noise ablation; stability and UMA-S agreement | `appendix/no_noise_ablation_stability/`, `appendix/transport_stability_uma_agreement/` |
| SI: all RDFs | `appendix/figure_rdf_all_systems/` |
| SI: transport tables | `appendix/transport_data_tables/` |
| SI: MSD and Green-Kubo examples | `appendix/transport_msd_examples/`, `appendix/transport_gk_viscosity_examples/` |
| SI: ionicity; interaction scans | `appendix/conductivity_ionicity/`, `appendix/interaction_scanning/` |
| SI: expanding-window convergence (the two error-analysis figures beside it have no script here) | `appendix/transport_convergence_normalised/` |
| SI: MLP baseline | `appendix/MLP_results/` |

Every input and output location comes from `src/analysis/observables/paths.py`,
including the `Figures/` tree the scripts write into, which they create on
first run. Point `paths.FIGURES` (or the `ELECTROLYTES_FIGURES` environment
variable) somewhere else to redirect them; the manuscript sources live in a
separate repository and are not part of this one.

The analysis outputs the figures read are committed under `results/` (about
180 MB): the summary CSVs, their `*_config.yaml` sidecars, the RDF curves, and
the `per_run/` trees of per-replica MSD, Green-Kubo and collective-conductivity
curves that every transport fit was made on. `paths.py` reads `data/` when it is
there and falls back to `results/` when it is not, so every figure except one
redraws from a fresh checkout. After recomputing an observable, refresh the
tracked copy with `python -m src.analysis.observables.export_results` (`--check`
reports staleness). Two folder names are historical: `uma_static_observables_1ns_NPT/`
holds the 293 K teacher RDFs and `UMA_298K_1ns/` the 298 K teacher densities.

The exception is the figure-2 stability panel (c), which reads an 85 MB npz of
per-frame UMA-S labels along one 20 ns student trajectory. It is not released;
`label_traj_with_uma.py` regenerates it from that trajectory, and the script
says so when the file is missing.

## Data availability

Everything released is in this repository. The trained students and the
simulation boxes are in `paper_data/` (44 MB), copied from the working tree by
`python -m src.export_paper_data`, which also writes `paper_data/MANIFEST.csv`
with a SHA-256 for every file:

| Contents | Path |
|---|---|
| The 9 trained students, one per salt/solvent system (`best_model.fnx`), with their FeNNol training configs | `paper_data/models/<system>/` |
| Initial boxes: the teacher frame the first student runs started from, and the student NPT-equilibrated frame whose density sets the molarity correction | `paper_data/boxes/initial/<system>/` |
| Corrected-molarity boxes (as packed, with their Packmol inputs, and after UMA-S relaxation) and the seed frames the 20 ns production runs start from | `paper_data/boxes/corrected/<system>/` |
| Every reported number, and the per-replica curves behind the transport coefficients | `results/` (see section 7) |

The trajectories (teacher and student MD, the 64 student and 64 OPLS-AA 20 ns
production runs) and the distillation datasets are too large to host and are
not released. They are reproducible: the datasets from the teacher runs
(section 4.1), the production runs from the released students and seed frames
(sections 5 and 5.1). MD is chaotic, so regenerated trajectories are not
bit-identical to ours; the transport coefficients reproduce within the replica
spread reported in the paper.

## Repository layout

```
fairchemV3_env.yaml, fennol_env.yaml   the two environments
src/
  simulate/       fennol_npt.py (student MD), uma_npt.py (teacher MD), run_fennol_sim.sh, launch_transport.sh
  data_prep/      box_creation/ (build_initial_boxes.py, build_boxes_correct_M.py,
                  relax_boxes.py, molecules/), dataset_creation/, equilibration/
  train/          launch_fennol_experiment.py, fennol_configs/exp01_mixed_noise.yaml, Slurm training scripts
  analysis/
    observables/  estimators, structural/ and transport/ drivers, ground_truth/, transport/METHODS.md
    labeling/     label_traj_with_uma.py: re-label student trajectories with UMA-S (SI fidelity checks)
  benchmark/      eval_fennol.py / eval_uma.py / eval_mace.py: accuracy and timing on a dataset pickle
  baselines/OPLS/ OPLS-AA/GROMACS pipeline: box pre-equilibration and the classical transport baseline
paper/
  paper_figures/        one directory per figure: the plotting script and its PNG
results/                committed analysis outputs (~180 MB): summary CSVs, *_config.yaml
                        provenance sidecars, RDF curves and the per_run/ transport curves
paper_data/             released inputs (44 MB): the 9 trained students and the simulation boxes
fennol/                 our FeNNol fork (submodule), see section 1
LICENSE                 BSD 3-Clause (LBNL variant)
```

## License

BSD 3-Clause (Lawrence Berkeley National Laboratory variant), see
[LICENSE](LICENSE). The FeNNol fork in `fennol/` is a separate repository under
the LGPL-3.0 of upstream FeNNol; this repository uses it as a library and does
not include any of its source.
