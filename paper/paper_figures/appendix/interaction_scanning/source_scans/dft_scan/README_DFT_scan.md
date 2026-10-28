# Gas-phase Na⁺···OTf⁻ rigid scan — DFT benchmark package

**Goal.** Reproduce, at **ωB97M-V / def2-TZVPD** DFT, the same rigid gas-phase
approach curve we already have for two OPLS charge sets and the UMA (uma-s-1p1)
MLIP, so the DFT curve can be overlaid as the reference in
`conductivity_gas_phase_Na_OTf_force_curve.png`.

You are on the machine that **has ORCA** (or another QM code). Everything you
need to build the DFT jobs is in this folder. Nothing here calls ORCA for you —
transform the geometry into your DFT inputs, run them, then produce **one CSV**
(format in §5) for `interaction_scanning/data/DFT/`.

---

## 1. The physical system

A single **Na⁺ + OTf⁻ (triflate, CF₃SO₃⁻) contact-ion-pair**, scanned in vacuum.

- The **OTf⁻ fragment is rigid** — its 8 atoms are frozen at a geometry taken from
  one frame of a classical MD run for every point.
- **Na⁺ is moved along one axis**: the line from the triflate **C atom through the
  centroid of the three SO₃ oxygens** (the "SO₃-face" approach). Na sits *outside*
  the O₃ face, i.e. `Na = O₃_centroid + d · axis`, `axis = (O₃_centroid − C)/|…|`.
- The scan coordinate **`d` is the Na–O₃(centroid) distance in Å**, over a
  **42-point grid**: dense `linspace(1.6, 6.0, 34)` + sparse `linspace(6.4, 12.0, 8)`.
  The 12 Å point is the dissociated reference.

**Charge / multiplicity of every point: total charge = 0, multiplicity = 1.**
(Na⁺ is +1, OTf⁻ is −1 → neutral pair; 84 electrons, even → closed-shell singlet.)

---

## 2. Files in this package

| file | what it is |
|---|---|
| `na_otf_scan_geometries.xyz` | **42-frame XYZ trajectory** — one frame per `d`. Each 9-atom frame = 8 rigid OTf atoms (C,S,F,F,F,O,O,O) + Na. Comment line carries `d`, `charge`, `mult`. **This is the geometry source of truth.** |
| `otf_anion_rigid.xyz` | the rigid OTf⁻ fragment alone (8 atoms), for reference/inspection. |
| `na_otf_pair_contact.mol2` | one representative contact pair (`d ≈ 2.53 Å`) **with OTf bond topology** (C–S, 3×C–F, 3×S–O; Na is a free ion). Helps a tool identify connectivity/atom types. The charge column holds MM partial charges and is **not** used by DFT. |
| `README_DFT_scan.md` | this file. |

Atom order is identical in every frame: `C S F F F O O O Na` (indices 1–9).

---

## 3. What to compute

A **single-point energy at each of the 42 frames** — a rigid (non-relaxed) scan.
Do **not** optimize geometry; keep the coordinates exactly as given so the DFT
points line up 1:1 with the existing curves.

**Method: ωB97M-V / def2-TZVPD**, charge 0, multiplicity 1, in vacuum (no solvent).

Forces are **not** required. The plot derives the axial force as the conservative
`−dE/dd` (finite difference) from the sampled energies — identically for every
curve — so energies alone are enough. (If you happen to also produce analytic
gradients that's fine, but the CSV only needs energies.)

---

## 4. Suggested ORCA setup (guidance — adapt as you like)

ωB97M-V is a range-separated hybrid meta-GGA with **VV10 nonlocal correlation**
(included automatically by the keyword). A reasonable ORCA 5/6 header:

```
! wB97M-V def2-TZVPD def2/J RIJCOSX DEFGRID3 TightSCF
%pal nprocs 8 end
%maxcore 3000
* xyz 0 1
  ... 9 atoms of the frame ...
*
```

Notes:
- `DEFGRID3` is recommended for VV10 functionals; `RIJCOSX` + `def2/J` speeds the
  hybrid up with negligible error. Drop `RIJCOSX`/`def2/J` for a slower exact run.
- `TightSCF` keeps the curve smooth enough for a clean `−dE/dd`.
- Run one SP **per frame** of `na_otf_scan_geometries.xyz` (split it into 42
  geometries, or use your code's trajectory/compound loop). Take ORCA's
  **`FINAL SINGLE POINT ENERGY`** (Hartree) from each output.
- BSSE/counterpoise is optional and **not** expected — every curve here is
  referenced to its own 12 Å value, so a smooth basis-set offset largely cancels.
  If you do a CP correction, just keep it consistent across all 42 points.

---

## 5. Output format to send back  ← **the important part**

Produce a plain CSV named **`na_otf_dft_energies.csv`** and put it in
`interaction_scanning/data/DFT/`, where the plot script looks for it.

**Columns (header required):**

```
idx,d_angstrom,E_hartree
0,1.6000,-1123.4567890123
1,1.7333,-1123.5678901234
...
41,12.0000,-1123.4001234567
```

- **`idx`** — frame index 0…41 (order doesn't matter; the plotter matches on `d`).
- **`d_angstrom`** — the Na–O₃(centroid) distance, copied from the frame comment.
  Must match the scan grid to ±1e-3 Å (values are listed in §6).
- **`E_hartree`** — the converged **total electronic energy** for that frame,
  in **Hartree** (ORCA `FINAL SINGLE POINT ENERGY`). Absolute values are fine;
  only differences are plotted.

Accepted variations (handled by `load_dft()` in `plot_interaction_scanning.py`):
- Energies in **eV** instead of Hartree → name the column `E_eV`;
  units are detected by magnitude (median |E| above 5000 is taken as eV).
- A **partial** run is OK: include only the frames you finished (≥3 rows). Leave
  `E` blank for any failed point. **Include the `d = 12.000 Å` frame if you can** —
  it is the zero reference; otherwise the largest `d` present is used as the ref.

---

## 6. The 42 distances `d` (Å)

```
1.6000 1.7333 1.8667 2.0000 2.1333 2.2667 2.4000 2.5333 2.6667 2.8000 2.9333
3.0667 3.2000 3.3333 3.4667 3.6000 3.7333 3.8667 4.0000 4.1333 4.2667 4.4000
4.5333 4.6667 4.8000 4.9333 5.0667 5.2000 5.3333 5.4667 5.6000 5.7333 5.8667
6.0000 6.4000 7.2000 8.0000 8.8000 9.6000 10.4000 11.2000 12.0000
```

---

## 7. When the CSV is back: plotting

Put the CSV in `interaction_scanning/data/DFT/` (the committed copy is already there)
and redraw the appendix figure from the repo root:

```
python -m paper.paper_figures.appendix.interaction_scanning.plot_interaction_scanning
```

`load_dft()` in that script reads `data/DFT/na_otf_dft_energies.csv` next to the
cached model curves in `data/na_otf__{model}.npz`. Every curve, DFT included, is
referenced to its largest-`d` value and the force panel is `-dE/dd` of that energy, so
DFT (energies only) is treated on the same footing as the models.

---

## 8. Reference: cached model curves (`data/na_otf__{model}.npz`)

The DFT results slot alongside these arrays (all length 42, aligned on `D`):

| key | meaning | units |
|---|---|---|
| `D` | Na–O₃(centroid) distance grid | Å |
| `E_raw` | total pair energy from the model | eV |
| `F_axial` | the model's axial force on Na (not plotted) | eV/Å |
| `model` | legend label | |

`charge`, `spin` and `system` record the scan setup; the OPLS file also carries a
`forcefield` field (parameters and `.itp` checksums). In the plot every energy is
shifted to its own 12 Å value (`E − E[-1]`) and the plotted force is the conservative
`−np.gradient(E_ref, D)`, not `F_axial`, so DFT (energies only) is treated on the same
footing.
