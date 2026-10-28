# Gas-phase Na⁺···Na⁺ rigid scan — DFT benchmark package

**Goal.** Reproduce, at **ωB97M-V / def2-TZVPD** DFT, the same gas-phase Na⁺–Na⁺
approach curve we already have for OPLS and the MLIPs, so the DFT curve can be
overlaid as the reference in `conductivity_gas_phase_Na_Na_force_curve.png`.

You are on the machine that **has ORCA** (or another QM code). Everything you
need to build the DFT jobs is in this folder. Nothing here calls ORCA for you —
transform the geometry into your DFT inputs, run them, then produce **one CSV**
(format in §5) for `interaction_scanning/data/DFT/`.

---

## 1. The physical system

Two **bare Na⁺ point cations**, scanned in vacuum along the line joining them —
the **like-charge (+/+) repulsion** counterpart to the Na⁺–OTf⁻ attraction and the
OTf⁻–OTf⁻ repulsion scans. There is no internal geometry: Na(A) sits at the origin,
Na(B) at `d · x̂`.

- The scan coordinate **`d` is the Na–Na distance in Å**, over a **42-point grid**:
  dense `linspace(2.0, 7.0, 34)` + sparse `linspace(7.6, 14.0, 8)`.
  The 14 Å point is the dissociated reference.

**Charge / multiplicity of every point: total charge = +2, multiplicity = 1.**
(Each Na⁺ has 10 electrons → 20 total, even → closed-shell singlet.)

---

## 2. Files in this package

| file | what it is |
|---|---|
| `na_na_scan_geometries.xyz` | **42-frame XYZ trajectory** — one frame per `d`. Each 2-atom frame = `Na Na` with Na(A) at the origin and Na(B) at `d` along +x. Comment line carries `d`, `charge`, `mult`. **This is the geometry source of truth.** |
| `README_DFT_scan.md` | this file. |

---

## 3. What to compute

A **single-point energy at each of the 42 frames** — a rigid scan.
**Method: ωB97M-V / def2-TZVPD**, charge +2, multiplicity 1, in vacuum (no solvent).

Forces are **not** required. The plot derives the axial force as the conservative
`−dE/dd` (finite difference) from the sampled energies — identically for every curve.

> **Note.** Two closed-shell Na⁺ ions have no bonding; the curve is pure Coulomb
> repulsion (+1/r tail) plus short-range Pauli repulsion. It should be smooth and
> monotonically decreasing with `d`; no SCF difficulty is expected. `TightSCF` keeps
> `−dE/dd` clean.

---

## 4. Suggested ORCA setup (guidance — adapt as you like)

```
! wB97M-V def2-TZVPD def2/J RIJCOSX DEFGRID3 TightSCF
%pal nprocs 8 end
%maxcore 3000
* xyz 2 1
  Na   x   y   z
  Na   x   y   z
*
```

- `DEFGRID3` is recommended for the VV10 functional; `RIJCOSX` + `def2/J` speed up the
  hybrid with negligible error.
- Run one SP **per frame** of `na_na_scan_geometries.xyz`. Take ORCA's
  **`FINAL SINGLE POINT ENERGY`** (Hartree) from each output.
- BSSE/counterpoise is optional and **not** expected (referenced to 14 Å; a smooth
  offset cancels). If you do it, keep it consistent across all 42 points.

---

## 5. Output format to send back  ← **the important part**

Produce a plain CSV named **`na_na_dft_energies.csv`** and put it in
`interaction_scanning/data/DFT/`, where the plot script looks for it.

**Columns (header required):**

```
idx,d_angstrom,E_hartree
0,2.0000,-324.1234567890
1,2.1515,-324.2345678901
...
41,14.0000,-324.0987654321
```

- **`idx`** — frame index 0…41 (order doesn't matter; the plotter matches on `d`).
- **`d_angstrom`** — the Na–Na distance, copied from the frame comment (±1e-3 Å; §6).
- **`E_hartree`** — converged total electronic energy (Hartree, ORCA `FINAL SINGLE
  POINT ENERGY`). Absolute values fine; only differences are plotted. **Include the
  `d = 14.000 Å` frame** — it is the zero reference.

Energies in **eV** are also accepted (column `E_eV`); a **partial** run (≥3 rows) is fine.

---

## 6. The 42 distances `d` (Å)

```
2.0000 2.1515 2.3030 2.4545 2.6061 2.7576 2.9091 3.0606 3.2121 3.3636 3.5152
3.6667 3.8182 3.9697 4.1212 4.2727 4.4242 4.5758 4.7273 4.8788 5.0303 5.1818
5.3333 5.4848 5.6364 5.7879 5.9394 6.0909 6.2424 6.3939 6.5455 6.6970 6.8485
7.0000 7.6000 8.5143 9.4286 10.3429 11.2571 12.1714 13.0857 14.0000
```

---

## 7. When the CSV is back: plotting

Put the CSV in `interaction_scanning/data/DFT/` (the committed copy is already there)
and redraw the appendix figure from the repo root:

```
python -m paper.paper_figures.appendix.interaction_scanning.plot_interaction_scanning
```

`load_dft()` in that script reads `data/DFT/na_na_dft_energies.csv` next to the
cached model curves in `data/na_na__{model}.npz`. Every curve, DFT included, is
referenced to its largest-`d` value and the force panel is `-dE/dd` of that energy, so
DFT (energies only) is treated on the same footing as the models.
