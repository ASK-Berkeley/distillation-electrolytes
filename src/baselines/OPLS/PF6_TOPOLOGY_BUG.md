# The PF6⁻ topology bug in the OPLS-AA baseline

*Audit date: 2026-10-06.*

> **Status: FIXED, and the runs have been redone.** `pf6.itp` is now the CL&P
> hexafluorophosphate topology (`nrexcl 3`, no `[ pairs ]`, cited bonded terms).
> `check_topologies.py` verifies it (49 exclusion entries; P-F minimises to within
> 0.004 A of b0), and P-F averages 1.606 A in the new liquid runs. All 16 boxes were
> rerun on 2026-10-06 with the corrected force field; `paths.OPLS_20ns_298K` points at
> them. Everything below describes the OLD topology and the OLD runs
> (`paths.OPLS_20ns_298K_OLD`) and is kept as the record of what was wrong.
> Current parameters: `PARAMETERS.md`.

This note explains, for someone who works with MLIPs rather than classical
force fields, what went wrong in the OPLS-AA baseline, how it was confirmed,
what it affects, and what else turned up during the audit.

**Summary.** The hand-written topology for the PF6⁻ anion tells GROMACS to
keep the six fluorine atoms interacting with each other through the normal
nonbonded potential, and then adds a second, half-strength copy of that same
interaction on top. In a correct OPLS topology the fluorines inside one ion do
not see each other at all. The result is 1.5x the physical F–F repulsion inside
every anion, which blows the ion up: the P–F bond sits at 1.74 Å in every
production frame instead of the 1.58 Å the force field asks for. All eight PF6
boxes (seven NaPF6 plus LiPF6/DME) are affected. The NaOTf boxes are not.

---

## 1. Background: how a classical force field decides who interacts with whom

With an MLIP you hand the model positions and atomic numbers, and it decides
everything. A classical force field like OPLS-AA does not. You have to tell it
the *topology*: which atoms are bonded, which triples form angles, which
quadruples form dihedrals, and, crucially, which atom pairs are **excluded**
from the nonbonded (Lennard-Jones + Coulomb) sum.

The reason exclusions exist: atoms that are bonded, or one atom apart, are
already held in place by the bond and angle springs. If you also let them feel
LJ and Coulomb, you double count. Two fluorines on the same phosphorus both
carry -0.39 e and sit 2.2 Å apart; the Coulomb repulsion alone would be huge
and the angle spring was never fitted with it present.

GROMACS encodes this with two things in each molecule's `.itp` file:

- **`nrexcl`** in `[ moleculetype ]`: "exclude every pair within this many
  bonds". `nrexcl 3` is the OPLS standard. It excludes 1-2 pairs (bonded),
  1-3 pairs (two bonds apart, the two ends of an angle) and 1-4 pairs (three
  bonds apart, the two ends of a dihedral).
- **`[ pairs ]`**: a list of atom pairs that get a special *reduced* nonbonded
  interaction back. In OPLS this is used for the 1-4 pairs: they are excluded
  by `nrexcl 3`, and the `[ pairs ]` list re-adds LJ and Coulomb between them
  scaled by 0.5 (`fudgeLJ = fudgeQQ = 0.5` in the `[ defaults ]` block). The
  pair term is **added**; it does not replace anything. It only makes sense
  for a pair that is already excluded.

So the standard OPLS pattern is: `nrexcl 3`, and `[ pairs ]` lists exactly
the 1-4 pairs. LigParGen writes this automatically for the solvents and for
OTf⁻, and those files are fine.

### What PF6⁻ should look like

PF6⁻ is one phosphorus with six fluorines bonded to it. Every atom is within
two bonds of every other atom (F–P is 1-2, F–P–F is 1-3). There are no 1-4
pairs. So with `nrexcl 3` (or even `nrexcl 2`) every pair in the ion is
excluded, the `[ pairs ]` block should be empty, and the ion's internal
structure is held entirely by the six P–F bond springs and the fifteen F–P–F
angle springs. That is how published OPLS-compatible PF6⁻ topologies are
written.

---

## 2. What the committed file does instead

`src/baselines/OPLS/data/molecules/pf6.itp` has

```
[ moleculetype ]
; name  nrexcl
PF6     1
```

and a `[ pairs ]` block listing all 15 F–F pairs.

- `nrexcl 1` excludes only the six P–F bonds. The fifteen F–F pairs are
  **not excluded**, so they get the full LJ + Coulomb interaction, as if they
  were on different molecules.
- The `[ pairs ]` block then adds a 0.5-scaled LJ + Coulomb on each of those
  same fifteen pairs, on top of the full interaction.

Net result: every F–F pair inside one anion interacts at **1.5x** the normal
nonbonded strength, where it should be at 0x.

The ion responds the only way it can: the fluorines push outwards against the
P–F bond springs until the spring force balances the extra repulsion. A one
dimensional model of the ion (six harmonic P–F bonds at the file's b0 and kb,
plus F–F LJ and Coulomb at the file's parameters, with cis pairs at r√2 and
trans pairs at 2r) gives the equilibrium bond length under each scheme:

| exclusion scheme | predicted P–F (Å) | spurious intra-ion energy (kJ/mol per ion) |
|---|---|---|
| `nrexcl 1` + full F–F `[ pairs ]` (**as run**) | 1.741 | ~2050 |
| `nrexcl 1`, no pairs | 1.709 | ~1430 |
| `nrexcl 3` + F–F pairs | 1.665 | ~780 |
| `nrexcl 3`, no pairs (**correct**) | 1.580 | 0 |

The spurious energy is a constant offset per ion and does not matter by
itself. The geometry does: the fluorine shell is ~10% further from the
phosphorus, so the anion is ~10% larger in radius and ~30% larger in volume
than the force field intends.

---

## 3. How it was confirmed in the production runs

Reading the file is suggestive but not proof, because GROMACS could in
principle have been given a different file at run time. Four independent
checks were made on the production tree

```
/global/cfs/cdirs/m5250/distillation_proj/simulation_results/OPLS/corrected_molarity/correct_parameter
```

1. **The topology on disk in each run directory.** Every `r*/pf6.itp` is
   byte-identical to the repo copy, and each box's `.top` includes it.

2. **What grompp said when building the run.** `grompp` is the GROMACS
   preprocessor that turns `.mdp` + `.gro` + `.top` into the binary `.tpr`
   that `mdrun` executes. Its output is saved in `r0/prod.log`:

   ```
   Excluding 3 bonded neighbours molecule type 'DME'
   Excluding 1 bonded neighbours molecule type 'NA'
   Excluding 1 bonded neighbours molecule type 'PF6'
   ```

   DME and OTf get 3, as they should. PF6 gets 1. The same lines appear for
   napf6_dme_1M_298, lipf6_dme_0_5M_298 and napf6_tgdme_1M_298.

3. **The exclusion list inside the binary run input.** `gmx dump -s prod.tpr`
   prints exactly what `mdrun` used. For the PF6 molecule type in both the
   NaPF6/DME and LiPF6/DME boxes:

   ```
   excls[0][num=7]={0, 1, 2, 3, 4, 5, 6}    <- P excludes everything
   excls[1][num=2]={0, 1}                   <- each F excludes only P and itself
   excls[2][num=2]={0, 2}
   ...
   ```

   That is 7 + 6x2 = **19** exclusion entries. A fully excluded 7-atom ion has
   7x7 = **49**. (There is no `gmx` on the Perlmutter login nodes; a copy was
   installed into the session scratchpad with
   `mamba create -p <dir> -c conda-forge gromacs` to run this.)

4. **The geometry in the trajectory.** P–F distances measured directly from
   the saved `.gro` frames (seed frame and segments 1, 5, 10 of replica r0):

   | box | P–F mean (Å) | spread (Å) | cis F–P–F | trans F–P–F |
   |---|---|---|---|---|
   | napf6_dme_1M_298 | 1.739 to 1.744 | 1.67 to 1.81 | 90.0° | ~178° |
   | lipf6_dme_0_5M_298 | 1.733 to 1.745 | 1.68 to 1.80 | 90.0° | ~178° |
   | napf6_pc_1M_298 | 1.739 to 1.744 | 1.67 to 1.83 | 90.0° | ~178° |

   The target b0 in the file is 1.580 Å. The measured 1.74 Å matches the
   1.741 Å predicted for the 1.5x scheme in the table above to three digits,
   so there is no competing explanation.

### The `wrong_parameter` tree

The sibling tree `corrected_molarity/wrong_parameter` has the **same**
`pf6.itp`. It differs from `correct_parameter` only in the ion LJ table in
`forcefield_lpg.itp`:

| parameter | wrong_parameter | correct_parameter |
|---|---|---|
| Li+ sigma | 1.402 Å | 2.126 Å |
| Na+ epsilon | 0.1159 kJ/mol | 0.01160 kJ/mol |

So the earlier "fix" repaired the cation LJ typos and never touched PF6. Both
trees have the stretched anion.

---

## 4. What it affects

- **Every OPLS number for a PF6 system**: self-diffusivities (cation, anion,
  solvent), Green–Kubo viscosity, Onsager conductivity, for the seven NaPF6
  boxes and lipf6_dme_0_5M_298 (and its 273 K / 323 K siblings if those are
  used). The anion is the wrong size, so solvation shells, ion pairing
  distances and hydrodynamic radius are all shifted. The direction is
  plausibly lower anion D and higher viscosity, but the size of the effect
  cannot be guessed; it has to be rerun.
- **The OPLS densities that fed the molarity correction.** The
  `--use-opls` path of `build_boxes_correct_M.py` targets OPLS densities from
  the same buggy anion. With at most 17 PF6 per ~2700 to 3700 atom box the
  volume error is of order 1%, which is comparable to the current 1.4% mean
  molarity deviation. Reusing the packed boxes is defensible; check the seed
  molarity report after re-equilibration and only repack if it gets worse.
- **Not affected**: all NaOTf boxes (OTf⁻ is a LigParGen topology with
  `nrexcl 3`), every UMA and PaiNN result, and the experimental data. The
  initial 30 Å boxes for the teacher runs were pre-equilibrated with OPLS
  (SI, system compositions), but the MLIPs relax PF6 within femtoseconds, so a
  stretched starting anion has no lasting effect there.

---

## 5. The fix

*Done on 2026-10-06 (see the status note at the top). Two things went beyond the plan
below: OTf- was moved to CL&P as well, so all 16 boxes were rerun, and the boxes were
repacked at the new force field's density (`repack_boxes.py`).*

In `pf6.itp`:

1. Change `nrexcl 1` to `nrexcl 3`.
2. Delete the whole `[ pairs ]` block.
3. Fix the header comment (see issue A below).

Then rebuild only the PF6 boxes with `build_opls_boxes.py` from the same
packed `initial_*.xyz` inputs, run `run_equilibration.py`,
`prepare_production.py` and `run_production.py` (4 x 20 ns) into a **new**
tree next to `correct_parameter`, point `paths.OPLS_20ns_298K` at it, rerun
the three `*_opls_replicas` drivers and `export_results`. That is 8 boxes x 4
replicas x 20 ns, about 640 ns of ~3000 atom GROMACS, roughly a day on a few
GPUs. A quick sanity check after the first equilibration: P–F in `npt.gro`
should be 1.58 ± 0.03 Å.

---

## 6. Other issues found

*Status: A, B, C, D, F and G are fixed in the repository and the paper. E needed no action.*

None of these is in the same class as the exclusion bug, but they are worth
fixing or knowing about while the files are open.

**A. The trans-pair comment in `pf6.itp` is wrong.** The header says "F1
opposite F6, F2 opposite F5, F3 opposite F4". The `[ angles ]` block actually
puts the 180° angles on F1/F6, F2/F4 and F3/F5. The angle block is internally
consistent (each F appears in exactly one 180° angle), so the physics is
fine; only the comment lies. Fix it so the next reader is not misled.

**B. The PF6 bonded parameters are uncited and hand-chosen.** The file uses
b0 = 1.58 Å, kb = 3.0e5 kJ/mol/nm², ktheta = 1046 kJ/mol/rad². GROMACS's own
`oplsaa.ff/ffbonded.itp` has no P–F bond type or F–P–F angle type at all, so
these did not come from there. They are in the right ballpark of the
Canongia Lopes and Pádua ionic-liquid PF6⁻ parameters usually paired with
OPLS-AA, but not equal to them. When rewriting the file, take the values
from that paper and cite it in the header.

**C. The hand-copied ion LJ parameters differ slightly from `oplsaa.ff`.**

| atom type | quantity | in `build_system_lpg.py` | in GROMACS `oplsaa.ff` |
|---|---|---|---|
| opls_786 (F in PF6) | epsilon | 0.25104 kJ/mol | 0.25522 kJ/mol |
| opls_407 (Na+) | sigma | 3.3284 Å | 3.3305 Å |

Both are 1 to 2% or less and will not change any result, but since the paper
calls these "standard OPLS-AA values" they should match the reference file.
Li+ (opls_406) and P (opls_785) match exactly.

**D. A real grompp warning was silenced by `-maxwarn 5`.** Every grompp
call in `run_equilibration.py` and `run_production.py` passes `-maxwarn 5`,
which lets the run proceed through warnings. The production grompp issued:

```
WARNING 1 [file prod.mdp]:
  With a verlet-buffer-tolerance of 0.005 kJ/mol/ps, a reference
  temperature of 298 and a tau_t of 10, your temperature might be off by up
  to 0.5%. To ensure the error is below 0.1%, decrease
  verlet-buffer-tolerance to 1e-03 or decrease tau_t.
```

This is about the neighbour-list buffer: with a weak thermostat, pair
interactions that drift out of the list between rebuilds show up as a
temperature offset. Measured over all 10 million stress frames of
napf6_dme_1M_298 r0 the mean temperature is 297.83 K against a 298 K target,
and the last-segment log average is 297.47 K, so the effect is real but
small (~0.1 to 0.2%). Add `verlet-buffer-tolerance = 1e-3` to the production
`.mdp` template for the rerun, and prefer `-maxwarn 0` or `1` so new
warnings are not hidden.

**E. Duplicated frames at segment boundaries (handled, no action).** The
20 ns runs are written as ten 2 ns segments with `mdrun -noappend`. Each
`prod.partNNNN.xtc` after the first repeats the previous segment's last frame
as its own first frame (step 2000000 at t = 2000 ps appears in both part 1
and part 2). The analysis reader in
`src/analysis/observables/transport/opls_runs.py` detects and drops these by
timestamp, and `edr2stress.py` does the same for the stress log (verified:
one row at t = 2000.000 ps, uniform 2 fs spacing). Nothing to fix; just do
not read the `.xtc` segments with another tool without de-duplicating.

**F. The methods text overstates LigParGen's role.** The paper says the
OPLS-AA parameters are "generated with LigParGen". That is true for the
solvents (DME, DEGDME, TEGDME, PC) and for OTf⁻. Na+, Li+ and PF6⁻ are
hand-set OPLS-AA ion parameters in `build_system_lpg.py` and
`data/molecules/`. The sentence should say so, and after the fix it should
cite the PF6 parameter source from issue B.

**G. Dead file.** `data/mdp/em.mdp` is copied into each box directory by
`build_opls_boxes.py` but never used: `run_equilibration.py` writes its own
`em.mdp` from the `EM_MDP` template (emtol 100, not the 500 in the data
file). Harmless, but remove it or make the driver use it so the two cannot
diverge.

---

## 7. Glossary for the MLIP reader

| term | meaning |
|---|---|
| `.itp` | "include topology": the per-molecule definition (atoms, charges, bonds, angles, dihedrals, exclusions). |
| `.top` | the system topology: includes the force-field defaults and the molecule `.itp`s, then lists how many of each molecule are in the box. |
| `.gro` | coordinates + box vectors, one frame. |
| `.mdp` | run parameters (integrator, thermostat, cutoffs, output frequency). |
| `grompp` | preprocessor: `.mdp` + `.gro` + `.top` -> `.tpr`. Its warnings are the only place topology mistakes get flagged. |
| `.tpr` | the binary run input `mdrun` actually executes. `gmx dump -s x.tpr` prints everything in it, including the exclusion lists. |
| `nrexcl` | "exclude nonbonded interactions between atoms this many bonds apart or closer". OPLS uses 3. |
| `[ pairs ]` | pairs that get a scaled (0.5x in OPLS) LJ + Coulomb added back; meant for 1-4 pairs that `nrexcl 3` excluded. |
| `[ defaults ]` | force-field-wide settings: combination rule, whether pair parameters are generated automatically (`gen-pairs`), and the 1-4 scale factors (`fudgeLJ`, `fudgeQQ`). |
| 1-2 / 1-3 / 1-4 | atom pairs separated by one, two or three bonds. |
| `-maxwarn N` | lets grompp continue despite up to N warnings. Convenient, dangerous. |
