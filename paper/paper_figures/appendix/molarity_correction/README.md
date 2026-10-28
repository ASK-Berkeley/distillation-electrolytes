# Appendix figure — molarity before and after the ratio correction

Shows that the initial electrolyte boxes were systematically over-concentrated
and that the corrected boxes hit their nominal targets.

```
python compute_molarity_correction.py    # measure both box sets -> CSV
python plot_molarity_correction.py       # CSV -> PNG
```

## Files

| file | what |
|---|---|
| `compute_molarity_correction.py` | reads both box sets from disk, writes the CSV |
| `molarity_correction.csv` | 16 rows: ion pairs, target, initial and corrected molarity |
| `plot_molarity_correction.py` | CSV → the figure |
| `figure_molarity_correction.png` | symlink into the paper submodule |

Unlike the `transport_stability_uma_agreement` bundle, the CSV is kept **here**
rather than under `REPO/data`: it is 1.4 KB, so committing it costs nothing and
keeps the figure redrawable even if the scratch trajectories are purged.

The PNG is written straight into
`paper/paper_electrolytes/Figures/appendix/` and symlinked back, like the other
figure scripts, so rerunning the plot updates the manuscript with no copy step.

## What is measured

Both numbers are ion pairs divided by the volume the box actually equilibrates
to — molarity is a property of the equilibrated cell, not the packed one.

* **initial** — exp01 boxes, packed by placing the ion pairs for the target
  molarity into a 30 Å cube and filling with solvent. Read from
  `exp01_.../nvt_equilibration_seeds/<system>_<conc>_298K.xyz`, which despite
  the name is the equilibrated final cell of an NPT run. These are the boxes
  used for teacher sampling and the student-vs-UMA structural comparison.
* **production** — exp04 boxes from `build_boxes_correct_M.py`, which fixes the
  solute:solvent ratio so the target is reached at the model's own equilibrium
  density. Read from the 20 ns production trajectory.

The production runs are NVT, so the cell is constant and every frame gives the
same volume. The script averages over the trajectory anyway and prints the
start-frame value beside it as a check — they agree to every printed digit on
all 16 systems, which is what a fixed cell should look like. A disagreement
would mean the run was not the fixed-cell NVT it is assumed to be.

Ion-pair counts come from counting cation atoms (one cation per formula unit;
no solvent or anion here contains Na or Li), and the script asserts they are
unchanged between the two box sets — the correction alters only the solvent
count.

## Result

| group | initial | corrected | target |
|---|---|---|---|
| 0.1 M (7 systems) | 0.153–0.166 M | 0.099–0.100 M | 0.1 M |
| 1 M (8 systems) | 1.185–1.318 M | 0.962–0.994 M | 1.0 M |
| LiPF₆/DME | 0.733 M | 0.486 M | 0.5 M |

The initial boxes miss in a consistent direction — always over-concentrated,
by 50–66% at 0.1 M and 19–32% at 1 M — because they held too few solvent
molecules per ion pair. LiPF₆/DME is the largest relative miss in both
directions and is drawn in the 1 M panel (matching the other appendix figures)
with its own 0.5 M reference segment.
