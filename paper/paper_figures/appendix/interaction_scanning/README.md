# Appendix figure: gas-phase ion-pair interaction scans

`python -m paper.paper_figures.appendix.interaction_scanning.plot_interaction_scanning`
(from the repo root) redraws `figure_interaction_scanning_2x3.png` from the local
`data/` alone (numpy + matplotlib only) and writes it to `paths.FIGURES_APPENDIX`.

## Figure

2 rows x 3 columns. Rows: conservative axial force `-dE/dd` (top), interaction energy
referenced to the dissociated scan point (bottom). Columns: `Na+-OTf-` (attraction),
`OTf-...OTf-` (repulsion), `Na+...Na+` (repulsion).

Every curve, MLIP and classical alike, is differentiated from its own energy
(`-np.gradient(E, D)`), not taken from the model's analytic `F_axial`. That makes the
top row exactly the negative slope of the bottom row and keeps the three columns
comparable. Four models plus the wB97M-V/def2-TZVPD DFT reference.

The OPLS-AA curve is the classical baseline of the paper: CL&P triflate and the Aqvist
Na+, full charges, geometric combination rules. `source_scans/scan_driver_opls.py`
reads the charges and Lennard-Jones parameters from the topology files of the
production runs in `src/baselines/OPLS`, so the scan cannot drift from the baseline.
Each `*__opls.npz` records the parameters it used and the SHA-256 of the two `.itp`
files in its `forcefield` field.

SevenNet-Omni is left off the figure. Its caches are still in `data/`; the `EXCLUDE`
set at the top of the script controls this, so clearing it brings the curves back
with no other edit.

`--suptitle` restores the overall title (off by default for the paper).

## Contents

| Path | What |
|------|------|
| `plot_interaction_scanning.py` | the figure script (only thing needed to redraw) |
| `data/{system}__{model}.npz` | per-model scan caches: `D`, `E_raw`, `F_axial`, `model` (42 points each) |
| `data/DFT/{system}_dft_energies.csv` | wB97M-V/def2-TZVPD reference, `d_angstrom,E_hartree` |
| `source_scans/` | the drivers that write `data/`, plus the DFT scan geometries and their READMEs |

`system` is one of `na_otf`, `otf_otf`, `na_na`; `model` is one of `opls`, `uma`,
`orb`, `mace`, `sevennet` (the last excluded from the figure). Legend labels are read
from each npz's `model` field.

## Rerunning the scans

`source_scans/gas_phase_geoms.py` holds the rigid geometries, scan grids and total
charges every driver shares. Run the drivers from the repo root, one per environment:

```bash
P=paper.paper_figures.appendix.interaction_scanning.source_scans
python -m $P.scan_driver_opls         # OPLS-AA, analytic; any env with numpy + ase (fennol)
python -m $P.scan_driver_fairchem     # UMA uma-s-1p1 + OrbMol-v2
python -m $P.scan_driver_mace_polar   # MACE-Polar polar-1-m (env mace_polar)
python -m $P.scan_driver_sevennet     # SevenNet-Omni
```

Each driver overwrites its own `data/{system}__{model}.npz`. The MLIP drivers use a GPU
if one is visible. There is no DFT code here: the ORCA single points were run
separately from the geometries in `source_scans/dft_scan*/`, and each of those folders
has a `README_DFT_scan.md` describing the calculation.
