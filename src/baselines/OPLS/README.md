# OPLS-AA / GROMACS baseline

The classical force-field baseline for the electrolyte-transport paper. It takes
the Packmol-packed boxes, assigns OPLS-AA parameters, equilibrates, and runs the
same four-replica 20 ns NVT transport protocol used for the students, so the
resulting trajectories go through the identical analysis pipeline.

The MD settings here are the ones described in the paper Methods (Classical
force-field baseline): steepest-descent minimization, 1 ns NPT at the target
temperature and 1 bar (CSVR thermostat 0.1 ps + C-rescale barostat 2 ps), then
20 ns NVT at the equilibrated volume (weak global CSVR thermostat 10 ps), 1 fs
timestep, no bond constraints, PME with 1.2 nm real-space and van der Waals
cutoffs, long-range dispersion corrections, and a Verlet pair-list buffer
tolerance of 1e-3 kJ/mol/ps.

## Layout

```
build_opls_boxes.py     build the OPLS/GROMACS boxes from packed .xyz
check_topologies.py     verify every molecule topology (run after any change to data/ or lib/)
repack_boxes.py         second pass of the molarity correction: repack at this force field's density
run_equilibration.py    EM + NPT equilibration            (+ run_equilibration.sh)
prepare_production.py   pick the production seed frame from each NPT run
run_production.py       four 20 ns NVT replicas + stress logs  (+ run_production.sh)
config.sh               all paths and toolchain in one place
PARAMETERS.md           every force-field parameter and its citation
PF6_TOPOLOGY_BUG.md     the 2026-10 PF6- exclusion bug: what, evidence, impact
topology_check.txt      last output of check_topologies.py (committed)
lib/                    imported library code
  build_system_lpg.py     .xyz -> .gro/.top/.itp; defines the ion atom types
  edr2stress.py           GROMACS .edr -> Green-Kubo stress log
  provenance.py           gmx version + topology hashes + git SHA for the run metadata
tools/
  analyze_equilibrated.py packed vs equilibrated molarity/density check (optional)
slurm/                  Perlmutter job scripts: perlmutter_prep.sh (two-pass
                        equilibration + seeds), perlmutter_prod.sh (production array)
data/                   read-only reference inputs
  molecules/             ion topologies: na, li (Aqvist), pf6, otf (CL&P)
  molecules_lpg/         LigParGen OPLS-AA/CM1A solvents: DME, DEGDME, TEGDME, PC
  superseded/            inputs no longer used, kept for the record
```

## Force field

Solvents are OPLS-AA with 1.14*CM1A charges from LigParGen. Both anions
(PF6-, OTf-) are CL&P (Canongia Lopes and Pádua), which is built on the
OPLS-AA functional form. Na+ and Li+ are the Åqvist ions as distributed with
GROMACS. All ions carry full formal charges. Every number and its citation is
in `PARAMETERS.md`; the sources are also quoted in each `.itp` header.

Run `python check_topologies.py` (needs `GMXBIN`) after changing anything under
`data/` or in `lib/build_system_lpg.py`. It builds each molecule alone in
vacuum, runs grompp with no warnings allowed, and checks exclusions, 1-4 pairs,
net charge, masses and the minimised geometry against the topology's own
parameters. Its output is committed as `topology_check.txt`.

## Toolchain

Set your installations in `config.sh` (or export the same variables):

* `GMXBIN` / `GMXDIR` / `GMXRC`: GROMACS 2025.x with the CUDA build.
* `OPLS_PYTHON`: a Python with `ase`, `numpy`, `networkx`, `matplotlib`.

`build_opls_boxes.py` needs no GROMACS; every other stage does.

## Input: the boxes root

`build_opls_boxes.py` reads Packmol-packed boxes from a *boxes root*: one
subdirectory per system, each holding a plain-xyz initial configuration per
concentration, named `initial_<conc>.xyz`. The `<conc>` token becomes part of
the box name (`0_1M`, `0_5M`, `1M`).

```
<boxes_root>/
├── lipf6_dme/
│   └── initial_0_5M.xyz
├── naotf_dme/
│   ├── initial_0_1M.xyz
│   └── initial_1M.xyz
├── napf6_pc/
│   ├── initial_0_1M.xyz
│   └── initial_1M.xyz
└── ...                       # one dir per salt_solvent system
```

Only `initial_*.xyz` files are read; any other files in a system dir (for
example the `packmol_*.inp` decks that produced them) are ignored. Each `.xyz`
is a bare coordinate file whose cell is taken from `build_boxes_correct_M.py`;
system names starting with `lipf6` are expanded to three temperature replicas
(273/298/323 K), everything else to 298 K.

These boxes are produced by the shared box builder
`src/data_prep/box_creation/build_boxes_correct_M.py --use-opls`, which fixes the
solute-to-solvent ratio at OPLS-AA's own equilibrium density (see the paper
Methods and SI). Point the builder at the boxes root with `--src`
(or `$OPLS_XYZ_ROOT`); with no flag it looks for `./boxes_root`.

## Molarity correction after a force-field change

The packed boxes fix the solute:solvent ratio so that the molarity equals its
label at the force field's own equilibrium density. That density depends on the
parameters, so after changing them the ratio is recomputed in two passes:

```bash
# pass 1: density-determination run on the existing packed boxes (6 ns NPT)
CM_BOX_ROOT=pass1/opls_ready_boxes CM_EQ_ROOT=pass1/equilibrated \
    python run_equilibration.py --npt-ns 6
# new solvent counts from the pass-1 density, packed with Packmol
python repack_boxes.py --eq-root pass1/equilibrated --out pass2/boxes_root --window-frac 0.3334
# pass 2: the normal pipeline below, starting from pass2/boxes_root
```

`repack_boxes.py` reports, per box, the density, its drift across the averaging
window (a box still drifting by more than 0.5% is flagged), the old and new
solvent counts and the predicted molarity. `repack_boxes.py --validate` checks
that its mass balance reproduces the compositions of the existing boxes.
`slurm/perlmutter_prep.sh` runs both passes and falls back to pass-1 seeds if
the repack fails.

## Run

```bash
source config.sh

# 0. verify the topologies (once per change to data/ or lib/)
python check_topologies.py

# 1. build boxes: <boxes_root>/ -> opls_ready_boxes/<box>/
#    (--lipf6-temps 298 builds the 16-box paper set; the default also builds
#    LiPF6/DME at 273 and 323 K)
python build_opls_boxes.py --src /path/to/boxes_root --lipf6-temps 298

# 2. energy minimization + NPT equilibration -> runs/equilibrated/<box>/
./run_equilibration.sh --gpus 0,1
python tools/analyze_equilibrated.py        # optional density/molarity check

# 3. pick one production seed frame per box -> runs/production_ready/<box>/
python prepare_production.py

# 4. four 20 ns NVT replicas -> runs/production_replicas/<box>/r{0..3}/
./run_production.sh --gpus 0,1,2,3
```

Each driver takes `--dry-run` (write dirs and mdps, call no gmx), `--sys` to
filter boxes, and `--shard I/N` to split the set across SLURM jobs. All output
paths and protocol knobs are the `CM_*` variables in `config.sh`.
`CM_TEGDME_PME=gpu` puts PME on the GPU for the tetraglyme boxes too (the
default keeps it on the CPU, a workaround for one workstation; it is a
performance setting and does not change the trajectory physics).

Every stage is idempotent and resumable: a finished stage is skipped and an
interrupted MD run resumes from its checkpoint, so re-running continues rather
than restarting. Never delete outputs to retry.

grompp runs with `-maxwarn 0` in every stage: a GROMACS warning stops the run
and has to be fixed at its cause, never waved through. `eq_meta.json` and
`rep_meta.json` record the GROMACS version and build, the SHA-256 of every
topology file in the run directory, and the code commit, so any run can be
matched to the exact force-field files that produced it.

## Output

`run_production.py` writes, per replica, a `<box>_r<k>.stress` Green-Kubo stress
log (pressure tensor every 2 fs) and a `prod.part*.xtc` trajectory (every 1 ps).
These feed the transport analysis in `src/analysis/observables/`.
