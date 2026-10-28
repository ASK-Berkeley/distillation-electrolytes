#!/usr/bin/env python
"""OPLS-AA driver: analytic rigid-scan energies with the paper's classical baseline.

    python -m paper.paper_figures.appendix.interaction_scanning.source_scans.scan_driver_opls

Writes data/{system}__opls.npz for all 3 systems. No parameter is typed in here:
charges and atom types are read from the topologies the OPLS-AA production runs
use (src/baselines/OPLS/data/molecules/otf.itp and na.itp), and the Lennard-Jones
sigma/epsilon of those types from MANUAL_ATOMTYPES in
src/baselines/OPLS/lib/build_system_lpg.py. That is the CL&P triflate and the
Aqvist Na+ (sources in src/baselines/OPLS/PARAMETERS.md), so the curve is the
force field behind every OPLS-AA number in the paper.

The energy is the intermolecular Coulomb + Lennard-Jones sum between the two
rigid fragments, with geometric combination of both sigma and epsilon (GROMACS
comb-rule 3), full charges and no cutoff. Intramolecular terms do not change
along a rigid scan and are left out. Runs in any env with numpy, ase and tqdm
(e.g. fennol); ase is only needed by the build_system_lpg import.
"""
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.analysis.observables import paths
from src.baselines.OPLS.lib import build_system_lpg as bsl
from src.baselines.OPLS.lib.provenance import sha256

from . import gas_phase_geoms as G
from .gas_phase_scan_common import DATA

KE = 14.399645          # eV*Angstrom / e^2  (Coulomb constant)
KJ2EV = 1.0 / 96.48534  # kJ/mol -> eV
LABEL = "OPLS-AA"
# scan fragment -> topology file in the baseline's molecules/ folder
ITP = {"OTF": "otf.itp", "Na": "na.itp"}


def _section(text, name):
    """Data lines of one `[ name ]` block of a GROMACS topology, comments stripped."""
    m = re.search(rf"^\[\s*{name}\s*\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if m is None:
        raise ValueError(f"no [ {name} ] section")
    lines = (ln.split(";")[0].strip() for ln in m.group(1).splitlines())
    return [ln.split() for ln in lines if ln]


def baseline_parameters():
    """{element: (charge e, sigma A, epsilon eV)} for OTf- and Na+, plus the file hashes."""
    lj = {f[0]: (float(f[5]) * 10.0, float(f[6]) * KJ2EV)      # nm -> A, kJ/mol -> eV
          for f in _section(bsl.MANUAL_ATOMTYPES, "atomtypes")}
    params, hashes = {}, {}
    for itp in ITP.values():
        path = Path(bsl.MANUAL_DIR) / itp
        hashes[itp] = sha256(path)
        for f in _section(path.read_text(), "atoms"):
            atype, name, q = f[1], f[4], float(f[6])
            element = re.sub(r"\d+$", "", name)
            entry = (q, *lj[atype])
            # equivalent atoms (3 O, 3 F) must carry the same type and charge
            if params.setdefault(element, entry) != entry:
                raise ValueError(f"{itp}: atoms of element {element} are not equivalent")
    return params, hashes


def interaction_energy(system, d, params):
    """Coulomb + Lennard-Jones energy (eV) between the fixed and the moving fragment."""
    symbols, pos = G.build(system, d)
    q, sig, eps = (np.array([params[s][k] for s in symbols]) for k in range(3))
    n = G.N_FIXED[system]
    r = np.linalg.norm(pos[:n, None, :] - pos[None, n:, :], axis=-1)
    sr6 = (np.sqrt(np.outer(sig[:n], sig[n:])) / r) ** 6
    lj = 4.0 * np.sqrt(np.outer(eps[:n], eps[n:])) * (sr6 * sr6 - sr6)
    return float(np.sum(KE * np.outer(q[:n], q[n:]) / r + lj))


def main():
    params, hashes = baseline_parameters()
    forcefield = json.dumps({
        "charge_e__sigma_A__epsilon_eV": params,
        "topology_sha256": hashes,
        "code_version": paths.git_sha(),
    })
    for el, (q, s, e) in params.items():
        print(f"[{LABEL}] {el:2s} q={q:+.3f} e  sigma={s:.4f} A  eps={e / KJ2EV:.5f} kJ/mol")
    DATA.mkdir(exist_ok=True)
    for system in tqdm(G.SYSTEMS, desc="OPLS-AA scans"):
        D = G.GRID[system]
        E = np.array([interaction_energy(system, d, params) for d in D])
        out = DATA / f"{system}__opls.npz"
        np.savez(out, D=D, E_raw=E, F_axial=np.full(len(D), np.nan), model=LABEL,
                 charge=G.CHARGE[system], spin=G.SPIN[system], system=system,
                 forcefield=forcefield)
        Er = E - E[-1]; i = int(np.argmin(Er))
        tqdm.write(f"[{LABEL}] {system:8s} {len(D)}pts -> {out.name} "
                   f"(E-extremum {Er[i]:+.3f} eV at d={D[i]:.2f})")


if __name__ == "__main__":
    main()
