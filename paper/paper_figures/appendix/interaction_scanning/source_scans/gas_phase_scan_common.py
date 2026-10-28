#!/usr/bin/env python
"""Shared scan loop: run one ASE calculator over all 3 systems, cache per-model npz to data/.

Used by the env-specific MLIP drivers (fairchem / sevennet / mace_polar); the analytic
OPLS-AA curves come from scan_driver_opls.py. Each driver builds its
calculator, then calls run_and_save(model_key, label, make_calc). Geometry/grid/charge come
from gas_phase_geoms so every backend is byte-identical.

Output: data/{system}__{model_key}.npz  with keys
    D        scan grid (A)
    E_raw    absolute model energy (eV)          -> reference at plot time (E - E[-1])
    F_axial  model axial force on the moving fragment (eV/A)
    model, charge, spin, system   (metadata)
"""
import time
from pathlib import Path

import numpy as np

from . import gas_phase_geoms as G

# the caches the figure reads: interaction_scanning/data/
DATA = Path(__file__).resolve().parent.parent / "data"


def _axial_force(system, forces):
    """Project the model force onto the approach axis: force on the *moving* fragment."""
    ax = G.AXIS[system]
    if system == "na_otf" or system == "na_na":
        return float(np.dot(forces[-1], ax))          # force on the last atom (Na that moves)
    if system == "otf_otf":
        return float(np.dot(forces[8:16].sum(0), ax))  # net force on rigid anion B
    raise ValueError(system)


def run_and_save(model_key, label, make_calc, systems=None, device_note=""):
    """make_calc() -> an ASE calculator (built once, reused across all systems)."""
    from ase import Atoms
    DATA.mkdir(exist_ok=True)
    systems = systems or G.SYSTEMS
    t0 = time.time()
    calc = make_calc()
    print(f"[{label}] calculator ready in {time.time()-t0:.0f}s {device_note}", flush=True)
    for sys in systems:
        D = G.GRID[sys]
        E = np.empty(len(D)); Fax = np.empty(len(D))
        t0 = time.time()
        for k, d in enumerate(D):
            symbols, pos = G.build(sys, d)
            atoms = Atoms(symbols=symbols, positions=pos)
            atoms.info["charge"] = G.CHARGE[sys]
            atoms.info["spin"] = G.SPIN[sys]
            atoms.calc = calc
            E[k] = atoms.get_potential_energy()
            Fax[k] = _axial_force(sys, atoms.get_forces())
        out = DATA / f"{sys}__{model_key}.npz"
        np.savez(out, D=D, E_raw=E, F_axial=Fax, model=label,
                 charge=G.CHARGE[sys], spin=G.SPIN[sys], system=sys)
        Er = E - E[-1]; i = int(np.argmin(Er))
        print(f"[{label}] {sys:8s} {len(D)}pts {time.time()-t0:.0f}s -> {out.name} "
              f"(E-extremum {Er[i]:+.3f} eV at d={D[i]:.2f})", flush=True)
