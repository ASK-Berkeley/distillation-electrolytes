# Gas-phase OTf⁻···OTf⁻ rigid scan — DFT benchmark package

**Goal.** Reproduce, at **ωB97M-V / def2-TZVPD** DFT, the same rigid gas-phase
SO₃-face···SO₃-face approach curve we already have for the OPLS charge sets and
the MLIPs, so the DFT curve can be overlaid as the reference in
`conductivity_gas_phase_OTf_OTf_force_curve.png`.

You are on the machine that **has ORCA** (or another QM code). Everything you
need to build the DFT jobs is in this folder. Nothing here calls ORCA for you —
transform the geometry into your DFT inputs, run them, then produce **one CSV**
(format in §5) for `interaction_scanning/data/DFT/`.

---

## 1. The physical system

Two **OTf⁻ (triflate, CF₃SO₃⁻) anions**, SO₃-face to SO₃-face, scanned in vacuum —
the closest inter-anion contact seen in the 1 M NaOTf/DME anion–anion RDF
(O···O peak ~3.6 Å, Na⁺-bridged). This is the **like-charge (−/−) repulsion**
counterpart to the Na⁺–OTf⁻ attraction scan.

- **Both OTf⁻ fragments are rigid** — each is the same 8-atom geometry, taken from one classical MD frame.
- **Anion A is fixed**; **anion B** is a rigid copy reflected through the plane whose
  normal is the SO₃-face axis (so its SO₃ face points back at A's), then translated so
  the two O₃ centroids are separated by `d` along A's SO₃-face axis
  (`axis = (O₃_centroid − C)/|…|`).
- The scan coordinate **`d` is the O₃(A)–O₃(B) centroid distance in Å**, over a
  **42-point grid**: dense `linspace(2.5, 7.0, 34)` + sparse `linspace(7.6, 14.0, 8)`.
  The 14 Å point is the dissociated reference.

**Charge / multiplicity of every point: total charge = −2, multiplicity = 1.**
(Each OTf⁻ is −1 → total −2; 148 electrons, even → closed-shell singlet.)

---

## 2. Files in this package

| file | what it is |
|---|---|
| `otf_otf_scan_geometries.xyz` | **42-frame XYZ trajectory** — one frame per `d`. Each 16-atom frame = 8 rigid OTf(A) atoms (C,S,F,F,F,O,O,O) + 8 rigid OTf(B) atoms in the same order. Comment line carries `d`, `charge`, `mult`. **This is the geometry source of truth.** |
| `otf_anion_rigid.xyz` | the rigid OTf⁻ fragment alone (8 atoms), for reference/inspection. |
| `README_DFT_scan.md` | this file. |

Atom order is identical in every frame: `C S F F F O O O | C S F F F O O O` (A then B, indices 1–16).

---

## 3. What to compute

A **single-point energy at each of the 42 frames** — a rigid (non-relaxed) scan.
Do **not** optimize geometry; keep the coordinates exactly as given so the DFT
points line up 1:1 with the existing curves.

**Method: ωB97M-V / def2-TZVPD**, charge −2, multiplicity 1, in vacuum (no solvent).

Forces are **not** required. The plot derives the axial force as the conservative
`−dE/dd` (finite difference) from the sampled energies — identically for every
curve — so energies alone are enough.

> **Note on a −2 anion in vacuum.** A dianion at large separation is close to the
> basis-set stability limit; def2-TZVPD (with diffuse functions) is the right choice.
> Keep `TightSCF`; if any far-separation point struggles to converge, still record its
> `FINAL SINGLE POINT ENERGY` (the curve is referenced to 14 Å, so a smooth offset cancels).

---

## 4. Suggested ORCA setup (guidance — adapt as you like)

ωB97M-V is a range-separated hybrid meta-GGA with **VV10 nonlocal correlation**
(included automatically by the keyword). A reasonable ORCA 5/6 header:

```
! wB97M-V def2-TZVPD def2/J RIJCOSX DEFGRID3 TightSCF
%pal nprocs 8 end
%maxcore 3000
* xyz -2 1
  ... 16 atoms of the frame ...
*
```

Notes:
- `DEFGRID3` is recommended for VV10 functionals; `RIJCOSX` + `def2/J` speeds the
  hybrid up with negligible error. Drop `RIJCOSX`/`def2/J` for a slower exact run.
- Run one SP **per frame** of `otf_otf_scan_geometries.xyz` (split it into 42
  geometries, or use your code's trajectory/compound loop). Take ORCA's
  **`FINAL SINGLE POINT ENERGY`** (Hartree) from each output.
- BSSE/counterpoise is optional and **not** expected — every curve is referenced to
  its own 14 Å value, so a smooth basis-set offset largely cancels. If you do a CP
  correction, keep it consistent across all 42 points.

---

## 5. Output format to send back  ← **the important part**

Produce a plain CSV named **`otf_otf_dft_energies.csv`** and put it in
`interaction_scanning/data/DFT/`, where the plot script looks for it.

**Columns (header required):**

```
idx,d_angstrom,E_hartree
0,2.5000,-2247.1234567890
1,2.6364,-2247.2345678901
...
41,14.0000,-2247.0987654321
```

- **`idx`** — frame index 0…41 (order doesn't matter; the plotter matches on `d`).
- **`d_angstrom`** — the O₃(A)–O₃(B) centroid distance, copied from the frame comment.
  Must match the scan grid to ±1e-3 Å (values in §6).
- **`E_hartree`** — the converged **total electronic energy** for that frame, in
  **Hartree** (ORCA `FINAL SINGLE POINT ENERGY`). Absolute values are fine; only
  differences are plotted. **Include the `d = 14.000 Å` frame** — it is the zero
  reference; otherwise the largest `d` present is used.

Energies in **eV** are also accepted — name the column `E_eV`; a **partial** run
(≥3 rows, blanks for failed points) is fine.

---

## 6. The 42 distances `d` (Å)

```
2.5000 2.6364 2.7727 2.9091 3.0455 3.1818 3.3182 3.4545 3.5909 3.7273 3.8636
4.0000 4.1364 4.2727 4.4091 4.5455 4.6818 4.8182 4.9545 5.0909 5.2273 5.3636
5.5000 5.6364 5.7727 5.9091 6.0455 6.1818 6.3182 6.4545 6.5909 6.7273 6.8636
7.0000 7.6000 8.5143 9.4286 10.3429 11.2571 12.1714 13.0857 14.0000
```

---

## 7. When the CSV is back: plotting

Put the CSV in `interaction_scanning/data/DFT/` (the committed copy is already there)
and redraw the appendix figure from the repo root:

```
python -m paper.paper_figures.appendix.interaction_scanning.plot_interaction_scanning
```

`load_dft()` in that script reads `data/DFT/otf_otf_dft_energies.csv` next to the
cached model curves in `data/otf_otf__{model}.npz`. Every curve, DFT included, is
referenced to its largest-`d` value and the force panel is `-dE/dd` of that energy, so
DFT (energies only) is treated on the same footing as the models.
