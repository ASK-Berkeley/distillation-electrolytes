#!/usr/bin/env python3
"""
build_system_lpg.py
Build GROMACS systems (.gro/.top/.itp) from packed boxes.

Parameter provenance (see ../PARAMETERS.md for the full table and citations):
- Solvents (DME, DEGDME, TEGDME, PC): OPLS-AA from LigParGen with 1.14*CM1A
  charges, vendored under data/molecules_lpg/. Each LigParGen ITP uses
  opls_800-8xx atomtype names that collide across molecules, so every
  molecule's types are renamed with a unique offset.
- Anions (PF6-, OTf-): CL&P (Canongia Lopes & Padua), hand-transcribed from the
  published il.ff into data/molecules/{pf6,otf}.itp; Lennard-Jones types clp_*
  are defined in MANUAL_ATOMTYPES below.
- Cations (Na+, Li+): Aqvist, as distributed in GROMACS oplsaa.ff
  (opls_407, opls_406), copied verbatim into MANUAL_ATOMTYPES.
All ions carry full integer charges (no charge scaling).

Renaming offsets for the LigParGen types:
  DME:    opls_8xx -> opls_8000+xx  (8000-8015)
  DEGDME: opls_8xx -> opls_8100+xx  (8100-8122)
  TEGDME: opls_8xx -> opls_8200+xx  (8200-8236)
  PC:     opls_8xx -> opls_8300+xx  (8300-8312)

Run ../check_topologies.py after touching anything here or under data/.
"""

import sys
import os
import re
import numpy as np
from ase.io import read
from collections import Counter, OrderedDict

# ============================================================
# Paths
# ============================================================
# Reference force-field inputs live in the sibling data/ dir (OPLS/data). Point
# OPLS_DATA_ROOT elsewhere to override.
LIB_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get(
    "OPLS_DATA_ROOT", os.path.join(os.path.dirname(LIB_DIR), "data"))
LPG_DIR = os.path.join(DATA_ROOT, 'molecules_lpg')
MANUAL_DIR = os.path.join(DATA_ROOT, 'molecules')

# ============================================================
# Atomtype renaming: each LigParGen molecule gets a unique offset
# so opls_8xx -> opls_(OFFSET+xx)
# ============================================================
MOL_OFFSETS = {
    'DME':    8000,
    'DEGDME': 8100,
    'TEGDME': 8200,
    'PC':     8300,
}

# OPLS-AA [ defaults ]: LJ (nbfunc 1), geometric combination (comb-rule 3),
# 1-4 pair parameters generated from the atom types, 1-4 LJ and Coulomb both
# scaled by 0.5. LigParGen writes the same block; CL&P assumes it.
DEFAULTS_BLOCK = """\
[ defaults ]
; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ
    1               3              yes            0.5     0.5
"""

# Atom types for the hand-maintained ion topologies in data/molecules/.
# Columns: name  bonded-type  mass  charge(unused, set per atom)  ptype  sigma(nm)  epsilon(kJ/mol)
MANUAL_ATOMTYPES = """\
; === Ion atomtypes (hand-maintained; sources in ../PARAMETERS.md) ===
[ atomtypes ]
; Na+, Li+: Aqvist, J. Phys. Chem. 94, 8021 (1990); values verbatim from
; GROMACS oplsaa.ff/ffnonbonded.itp (opls_407, opls_406)
  opls_407   Na    22.98977    0.000    A    3.33045E-01   1.15980E-02
  opls_406   Li     6.94100    0.000    A    2.12645E-01   7.64793E-02
; PF6-: CL&P il.ff "hexafluorophosphate", Canongia Lopes, Deschamps, Padua,
; J. Phys. Chem. B 108, 2038 (2004); sigma 3.74 / 3.12 A, eps 0.8368 / 0.2552 kJ/mol
  clp_P       P    30.97400    0.000    A    3.74000E-01   8.36800E-01
  clp_FP      F    18.99800    0.000    A    3.12000E-01   2.55200E-01
; OTf-: CL&P il.ff "triflate", Canongia Lopes, Padua, J. Phys. Chem. B 108,
; 16893 (2004); sigma 3.50 / 3.55 / 3.15 / 3.118 A, eps 0.27614 / 1.046 / 0.83736 / 0.2554 kJ/mol
  clp_CBT    CF    12.01100    0.000    A    3.50000E-01   2.76140E-01
  clp_SBT    SB    32.06600    0.000    A    3.55000E-01   1.04600E+00
  clp_OTF    OB    15.99900    0.000    A    3.15000E-01   8.37360E-01
  clp_F1     FB    18.99800    0.000    A    3.11800E-01   2.55400E-01
"""

# ============================================================
# Molecule info for box parsing
# ============================================================
MOLECULES = {
    'DME': {
        'formula': {'C': 4, 'H': 10, 'O': 2},
        'n_atoms': 16,
        'name': 'DME',
        'heavy_seq': ['C', 'O', 'C', 'C', 'O', 'C'],
        'mol_itp': 'DME_mol.itp',
        'resname': 'DME',
        'source': 'lpg',
    },
    'DEGDME': {
        'formula': {'C': 6, 'H': 14, 'O': 3},
        'n_atoms': 23,
        'name': 'DGM',
        'heavy_seq': ['C', 'O', 'C', 'C', 'O', 'C', 'C', 'O', 'C'],
        'mol_itp': 'DEGDME_mol.itp',
        'resname': 'DGM',
        'source': 'lpg',
    },
    'TEGDME': {
        'formula': {'C': 10, 'H': 22, 'O': 5},
        'n_atoms': 37,
        'name': 'TGM',
        'heavy_seq': ['C', 'O', 'C', 'C', 'O', 'C', 'C', 'O', 'C', 'C', 'O', 'C', 'C', 'O', 'C'],
        'mol_itp': 'TEGDME_mol.itp',
        'resname': 'TGM',
        'source': 'lpg',
    },
    'PC': {
        'formula': {'C': 4, 'H': 6, 'O': 3},
        'n_atoms': 13,
        'name': 'PC',
        'heavy_seq': ['C', 'O', 'O', 'C', 'C', 'O', 'C'],
        'mol_itp': 'PC_mol.itp',
        'resname': 'PC',
        'source': 'lpg',
    },
    'OTF': {
        'formula': {'S': 1, 'O': 3, 'C': 1, 'F': 3},
        'n_atoms': 8,
        'name': 'OTF',
        'heavy_seq': ['S', 'O', 'O', 'O', 'C', 'F', 'F', 'F'],
        'mol_itp': 'otf.itp',
        'resname': 'OTF',
        'source': 'manual',
    },
    'NA': {
        'formula': {'Na': 1},
        'n_atoms': 1,
        'name': 'NA',
        'heavy_seq': ['Na'],
        'mol_itp': 'na.itp',
        'resname': 'NA',
        'source': 'manual',
    },
    'LI': {
        'formula': {'Li': 1},
        'n_atoms': 1,
        'name': 'LI',
        'heavy_seq': ['Li'],
        'mol_itp': 'li.itp',
        'resname': 'LI',
        'source': 'manual',
    },
    'PF6': {
        'formula': {'P': 1, 'F': 6},
        'n_atoms': 7,
        'name': 'PF6',
        'heavy_seq': ['P', 'F', 'F', 'F', 'F', 'F', 'F'],
        'mol_itp': 'pf6.itp',
        'resname': 'PF6',
        'source': 'manual',
    },
}


def rename_atomtypes_in_itp(itp_text, mol_key):
    """Rename opls_8xx atomtypes in ITP text to opls_(OFFSET+xx)."""
    offset = MOL_OFFSETS[mol_key]

    def replacer(m):
        num = int(m.group(1))
        if 800 <= num <= 899:
            new_num = offset + (num - 800)
            return f'opls_{new_num}'
        return m.group(0)

    return re.sub(r'opls_(\d+)', replacer, itp_text)


def parse_lpg_itp(itp_file, mol_key):
    """
    Parse LigParGen ITP file.
    Returns:
      - atomtypes_block: str with the [ atomtypes ] section (renamed)
      - mol_block: str from [ moleculetype ] onward (renamed)
      - defaults_block: str with [ defaults ] section
    """
    with open(itp_file) as f:
        text = f.read()

    # Rename all atomtypes
    text = rename_atomtypes_in_itp(text, mol_key)

    # Extract sections
    lines = text.split('\n')

    defaults_lines = []
    atomtypes_lines = []
    mol_lines = []

    section = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('[ defaults'):
            section = 'defaults'
            defaults_lines.append(line)
        elif stripped.startswith('[ atomtypes'):
            section = 'atomtypes'
            atomtypes_lines.append(line)
        elif stripped.startswith('[ moleculetype'):
            section = 'mol'
            mol_lines.append(line)
        elif stripped.startswith('[') and section not in ('atomtypes', 'defaults', 'mol'):
            section = None
        else:
            if section == 'defaults':
                defaults_lines.append(line)
            elif section == 'atomtypes':
                atomtypes_lines.append(line)
            elif section == 'mol':
                mol_lines.append(line)

    return (
        '\n'.join(defaults_lines),
        '\n'.join(atomtypes_lines),
        '\n'.join(mol_lines)
    )


def build_combined_forcefield(mol_keys_lpg, out_dir):
    """
    Build the combined forcefield ITP: one [ defaults ] block, the renamed
    LigParGen [ atomtypes ] of every solvent in the system, and the ion
    atom types from MANUAL_ATOMTYPES (always written, so a system's topology
    is complete whichever ions it contains).
    """
    ff_itp = os.path.join(out_dir, 'forcefield_lpg.itp')
    all_atomtypes = []

    for mol_key in mol_keys_lpg:
        lpg_itp_file = os.path.join(LPG_DIR, f'{mol_key}_out', f'{mol_key}.gmx.itp')
        defaults, atomtypes, _ = parse_lpg_itp(lpg_itp_file, mol_key)
        # LigParGen must agree with the OPLS defaults we write ourselves.
        got = [ln.split() for ln in defaults.splitlines()
               if ln.strip() and not ln.strip().startswith((';', '['))]
        if got != [['1', '3', 'yes', '0.5', '0.5']]:
            raise ValueError(f'{lpg_itp_file}: unexpected [ defaults ] {got}')
        all_atomtypes.append(f'; === {mol_key} atomtypes (LigParGen OPLS-AA/1.14*CM1A) ===')
        all_atomtypes.append(atomtypes)

    with open(ff_itp, 'w') as f:
        f.write('; Combined force field for the OPLS-AA baseline\n')
        f.write('; Generated by build_system_lpg.py; parameter sources in PARAMETERS.md\n\n')
        f.write(DEFAULTS_BLOCK)
        f.write('\n')
        for at in all_atomtypes:
            f.write(at)
            f.write('\n')
        f.write(MANUAL_ATOMTYPES)

    print(f"Wrote combined forcefield: {ff_itp}")
    return ff_itp


def build_mol_itp(mol_key, out_dir):
    """
    Extract [ moleculetype ] onward from LigParGen ITP (renamed atomtypes).
    Write to out_dir/XXX_mol.itp
    """
    mol_info = MOLECULES[mol_key]
    lpg_itp_file = os.path.join(LPG_DIR, f'{mol_key}_out', f'{mol_key}.gmx.itp')
    _, _, mol_block = parse_lpg_itp(lpg_itp_file, mol_key)

    # Fix residue name in mol block to match our convention
    # LigParGen uses the resname we specified, check it's correct
    out_itp = os.path.join(out_dir, mol_info['mol_itp'])

    with open(out_itp, 'w') as f:
        f.write(f'; LigParGen OPLS-AA/CM1A parameters for {mol_key}\n')
        f.write(f'; Atomtypes renamed with offset {MOL_OFFSETS[mol_key]}\n\n')
        f.write(mol_block)
        f.write('\n')

    print(f"Wrote molecule ITP: {out_itp}")
    return out_itp


def identify_molecules(atoms):
    """Identify molecules in ASE Atoms by scanning atom sequences."""
    symbols = atoms.get_chemical_symbols()
    n = len(symbols)
    i = 0
    mol_list = []

    while i < n:
        matched = False
        for mol_type, mol_info in MOLECULES.items():
            mol_n = mol_info['n_atoms']
            if i + mol_n > n:
                continue
            chunk = symbols[i:i + mol_n]
            chunk_formula = dict(Counter(chunk))
            if chunk_formula == mol_info['formula']:
                mol_list.append((mol_type, i, i + mol_n))
                i += mol_n
                matched = True
                break
        if not matched:
            sym = symbols[i]
            if sym == 'Na':
                mol_list.append(('NA', i, i + 1))
                i += 1
                matched = True
            elif sym == 'Li':
                mol_list.append(('LI', i, i + 1))
                i += 1
                matched = True
            if not matched:
                print(f"WARNING: Could not identify molecule at atom {i} ({symbols[i:i+10]})")
                i += 1

    return mol_list


def unwrap_molecule(pos, cell):
    """
    Make one molecule whole across PBC.

    The raw ByteFF-Pol trajectory is per-atom wrapped, so a molecule straddling a
    box face comes back with its atoms in different periodic images (a bond can
    read as ~L instead of ~1.5 Ang, which blows up GROMACS). We reconnect the
    atoms with a minimum-image spanning tree (Prim) grown from atom 0: each atom
    is placed in the image nearest to an already-placed neighbour. This is robust
    even for molecules longer than L/2 (e.g. extended tetraglyme), where a simple
    "min image relative to atom 0" would misplace the far end. Finally the whole
    molecule is shifted so its centroid sits in the primary box.
    """
    pos = pos.copy()
    n = len(pos)
    if n == 1:
        return pos - np.floor(pos / cell) * cell  # lone ion: just wrap into box
    placed = [0]
    remaining = set(range(1, n))
    while remaining:
        best = None
        for i in placed:
            for j in remaining:
                d = pos[j] - pos[i]
                d -= np.round(d / cell) * cell
                dist2 = float(d @ d)
                if best is None or dist2 < best[0]:
                    best = (dist2, i, j, d)
        _, i, j, d = best
        pos[j] = pos[i] + d
        placed.append(j)
        remaining.discard(j)
    pos -= np.floor(pos.mean(axis=0) / cell) * cell
    return pos


# Covalent radii (Angstrom) for geometry-based bond perception.
COVALENT_RADII = {'H': 0.31, 'Li': 1.28, 'C': 0.76, 'N': 0.71, 'O': 0.66,
                  'F': 0.57, 'Na': 1.66, 'P': 1.07, 'S': 1.05}
_MASS_ELEMENT = [(1.008, 'H'), (6.94, 'Li'), (12.011, 'C'), (14.007, 'N'),
                 (15.999, 'O'), (18.998, 'F'), (22.990, 'Na'), (30.974, 'P'),
                 (32.06, 'S')]
# A template bond longer than this (Angstrom) in the trajectory snapshot means the
# molecule is dissociated in that frame (an ML-potential artifact) -> repair it.
# Set well above any real bond (longest here is C-S ~1.82 A, which stretches past
# 1.9 A thermally) but far below a true break (observed dissociations are >10 A).
BROKEN_BOND_ANG = 2.5


def _element_from_mass(m):
    return min(_MASS_ELEMENT, key=lambda me: abs(me[0] - m))[1]


def _template_itp_path(mol_key):
    info = MOLECULES[mol_key]
    if info['source'] == 'lpg':
        return os.path.join(LPG_DIR, f'{mol_key}_out', f'{mol_key}.gmx.itp')
    return os.path.join(MANUAL_DIR, info['mol_itp'])


def parse_template_itp(mol_key):
    """Reference topology of a molecule: (names, elements, edges) in ITP atom order."""
    names, elements, edges = [], [], []
    section = None
    with open(_template_itp_path(mol_key)) as f:
        for line in f:
            s = line.split(';')[0].strip()
            if not s:
                continue
            if s.startswith('['):
                section = s.strip('[] ').split()[0].lower()
                continue
            p = s.split()
            if section == 'atoms':
                names.append(p[4])
                elements.append(_element_from_mass(float(p[7])))
            elif section == 'bonds' and p[0].isdigit() and p[1].isdigit():
                edges.append((int(p[0]) - 1, int(p[1]) - 1))
    return names, elements, edges


def load_template_coords(mol_key):
    """Reference conformer coordinates (Angstrom) in ITP atom order, or None."""
    if MOLECULES[mol_key]['source'] != 'lpg':
        return None
    gro = os.path.join(LPG_DIR, f'{mol_key}_out', f'{mol_key}.gmx.gro')
    with open(gro) as f:
        lines = f.readlines()
    n = int(lines[1])
    coords = [[float(ln[20:28]), float(ln[28:36]), float(ln[36:44])] for ln in lines[2:2 + n]]
    return np.array(coords) * 10.0  # nm -> Ang


def _perceive_graph(positions, symbols, cell, factor):
    import networkx as nx
    n = len(positions)
    g = nx.Graph()
    for i, s in enumerate(symbols):
        g.add_node(i, element=s)
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[j] - positions[i]
            d -= np.round(d / cell) * cell
            ri = COVALENT_RADII.get(symbols[i], 0.77)
            rj = COVALENT_RADII.get(symbols[j], 0.77)
            if np.linalg.norm(d) < factor * (ri + rj):
                g.add_edge(i, j)
    return g


def learn_canonical(mol_key, positions, symbols, instances, cell, templates):
    """
    Learn the trajectory->ITP atom mapping for a molecule type, once.

    All molecules of a type share the same internal atom order in the trajectory,
    so we derive the permutation from the first instance whose geometry yields a
    clean graph isomorphic to the ITP template, then reuse it for every instance
    (including any dissociated one, which would not match on its own).

    Returns (perm, bonds_local): perm[r] = trajectory-local index of ITP atom r;
    bonds_local = ITP bonds expressed in trajectory-local indices.
    """
    import networkx as nx
    _, elems_t, edges_t = templates[mol_key]
    n = len(elems_t)
    if n == 1:
        return [0], []
    g_ref = nx.Graph()
    for i, e in enumerate(elems_t):
        g_ref.add_node(i, element=e)
    g_ref.add_edges_from(edges_t)
    for (s, e) in instances:
        P = unwrap_molecule(positions[s:e], cell)
        S = symbols[s:e]
        for factor in (1.30, 1.35, 1.40, 1.45):
            g = _perceive_graph(P, S, cell, factor)
            if g.number_of_edges() != len(edges_t):
                continue
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                g_ref, g, node_match=lambda a, b: a['element'] == b['element'])
            if matcher.is_isomorphic():
                perm = [matcher.mapping[r] for r in range(n)]
                bonds_local = [(perm[a], perm[b]) for a, b in edges_t]
                return perm, bonds_local
    raise ValueError(f"{mol_key}: could not learn atom mapping from any instance")


def heal_dissociated(coords, edges, cell):
    """
    Repair a molecule dissociated in the trajectory frame by pulling the broken-off
    fragment back to bonding distance along the broken bond. This is minimal: only
    the smaller fragment moves, every other atom keeps its trajectory position, so
    no new steric clashes are introduced (unlike dropping in a fresh conformer).

    Returns the healed coords, or None if the break does not cleanly split the
    molecule in two (e.g. a ring), in which case the caller falls back.
    """
    import networkx as nx
    coords = coords.copy()
    broken = max(edges, key=lambda ab: np.linalg.norm(coords[ab[0]] - coords[ab[1]]))
    g = nx.Graph()
    g.add_nodes_from(range(len(coords)))
    for e in edges:
        if e is not broken:
            g.add_edge(*e)
    comps = list(nx.connected_components(g))
    if len(comps) != 2:
        return None
    a, b = broken
    comp_a = next(c for c in comps if a in c)
    comp_b = next(c for c in comps if b in c)
    if len(comp_b) <= len(comp_a):
        anchor, mover, move_set = a, b, comp_b
    else:
        anchor, mover, move_set = b, a, comp_a
    v = coords[mover] - coords[anchor]
    v -= np.round(v / cell) * cell
    target = coords[anchor] + v / np.linalg.norm(v) * 1.45  # ~bond length, same direction
    shift = target - coords[mover]
    for idx in move_set:
        coords[idx] += shift
    return coords


def bond_guided_unwrap(positions, bonds, cell):
    """Make a molecule whole by walking its real bonds (BFS), placing each atom in
    the image nearest its bonded parent. Robust to molecules split across PBC."""
    from collections import deque
    P = positions.copy()
    n = len(P)
    adj = [[] for _ in range(n)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    seen = {0}
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                d = P[v] - P[u]
                d -= np.round(d / cell) * cell
                P[v] = P[u] + d
                seen.add(v)
                q.append(v)
    return P


def write_gro(atoms, mol_list, outfile):
    """Write GROMACS .gro, reordering each molecule to its ITP atom order, making it
    whole across PBC, and repairing any molecule dissociated in this trajectory frame."""
    positions = atoms.get_positions().copy()  # Angstrom
    cell = atoms.cell.lengths()  # Angstrom
    symbols = atoms.get_chemical_symbols()
    n_total = len(atoms)

    keys = set(mt for mt, _, _ in mol_list)
    templates = {k: parse_template_itp(k) for k in keys}
    tcoords = {k: load_template_coords(k) for k in keys}
    canonical = {}
    for k in keys:
        insts = [(s, e) for mt, s, e in mol_list if mt == k]
        canonical[k] = learn_canonical(k, positions, symbols, insts, cell, templates)

    n_repaired = 0
    with open(outfile, 'w') as f:
        f.write("Generated by build_system_lpg.py (LigParGen OPLS-AA)\n")
        f.write(f"{n_total}\n")

        atom_idx = 1
        resnum = 1
        for mol_type, start, end in mol_list:
            resname = MOLECULES[mol_type]['resname']
            names_t, _, edges_t = templates[mol_type]
            perm, bonds_local = canonical[mol_type]
            n = end - start

            if n == 1:
                ordered = positions[start:end].copy()
                ordered -= np.floor(ordered / cell) * cell  # wrap lone ion into box
            else:
                whole = bond_guided_unwrap(positions[start:end], bonds_local, cell)
                ordered = np.array([whole[perm[r]] for r in range(n)])
                max_bond = max(np.linalg.norm(ordered[a] - ordered[b]) for a, b in edges_t)
                if max_bond > BROKEN_BOND_ANG:
                    healed = heal_dissociated(ordered, edges_t, cell)
                    if healed is not None:
                        ordered = healed
                        n_repaired += 1
                        print(f"  [repair] {mol_type} at atom {start}: dissociated in frame "
                              f"(max bond {max_bond:.1f} A) -> healed broken bond")
                    elif tcoords[mol_type] is not None:
                        ref = tcoords[mol_type]
                        ordered = ref - ref.mean(axis=0) + ordered.mean(axis=0)
                        n_repaired += 1
                        print(f"  [repair] {mol_type} at atom {start}: dissociated in frame "
                              f"(max bond {max_bond:.1f} A) -> replaced with template conformer")
                    else:
                        print(f"  [WARN] {mol_type} at atom {start} dissociated "
                              f"(max bond {max_bond:.1f} A) and could not be repaired")

            for j in range(n):
                aname = names_t[j] if j < len(names_t) else f'X{j}'
                pos = ordered[j] / 10.0  # Ang -> nm
                f.write(f"{resnum:5d}{resname:<5s}{aname:>5s}{atom_idx:5d}{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}\n")
                atom_idx += 1
            resnum += 1

        box = cell / 10.0
        f.write(f"   {box[0]:.5f}   {box[1]:.5f}   {box[2]:.5f}\n")

    extra = f", {n_repaired} repaired" if n_repaired else ""
    print(f"Wrote GRO: {outfile} ({n_total} atoms, {len(mol_list)} molecules{extra})")


def write_top(mol_list, outfile, system_name, work_dir, mol_keys_lpg):
    """Write GROMACS .top file using LigParGen combined forcefield."""
    mol_counts = OrderedDict()
    for mol_type, _, _ in mol_list:
        mol_counts[mol_type] = mol_counts.get(mol_type, 0) + 1

    ff_itp = os.path.join(work_dir, 'forcefield_lpg.itp')

    with open(outfile, 'w') as f:
        f.write('; GROMACS topology, OPLS-AA baseline (parameter sources: PARAMETERS.md)\n')
        f.write(f'; System: {system_name}\n\n')

        # Combined forcefield (defaults + all atomtypes)
        f.write(f'#include "{ff_itp}"\n\n')

        # Molecule ITPs (only moleculetype sections)
        for mol_type in mol_counts:
            mol_info = MOLECULES[mol_type]
            if mol_info['source'] == 'lpg':
                itp_path = os.path.join(work_dir, mol_info['mol_itp'])
            else:
                itp_path = os.path.join(MANUAL_DIR, mol_info['mol_itp'])
            f.write(f'#include "{itp_path}"\n')

        f.write('\n[ system ]\n')
        f.write(f'{system_name}\n\n')

        f.write('[ molecules ]\n')
        f.write('; Compound        nmols\n')
        for mol_type, count in mol_counts.items():
            name = MOLECULES[mol_type]['name']
            f.write(f'{name:<16s}  {count}\n')

    print(f"Wrote TOP: {outfile}")


def build_system(system_name, traj_path, output_dir, temperature=298):
    """Build a complete GROMACS system for a given trajectory."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Building system: {system_name}")
    print(f"Traj: {traj_path}")

    atoms = read(traj_path, index=-1)
    print(f"Read {len(atoms)} atoms, cell={atoms.cell.lengths().round(3)}")

    mol_list = identify_molecules(atoms)
    mol_counts = Counter(mt for mt, _, _ in mol_list)
    print(f"Molecules: {dict(mol_counts)}")

    # Determine which LigParGen molecules are needed
    mol_keys_lpg = [mk for mk in mol_counts if MOLECULES[mk]['source'] == 'lpg']

    # Build combined forcefield ITP
    build_combined_forcefield(mol_keys_lpg, output_dir)

    # Build molecule ITPs (mol-only, no defaults/atomtypes)
    for mol_key in mol_keys_lpg:
        build_mol_itp(mol_key, output_dir)

    # Write GRO
    gro_file = os.path.join(output_dir, f'{system_name}.gro')
    write_gro(atoms, mol_list, gro_file)

    # Write TOP
    top_file = os.path.join(output_dir, f'{system_name}.top')
    write_top(mol_list, top_file, system_name, output_dir, mol_keys_lpg)

    return gro_file, top_file, mol_counts


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: build_system_lpg.py <system_name> <traj_path> <output_dir> [temperature]')
        sys.exit(1)

    system_name = sys.argv[1]
    traj_path = sys.argv[2]
    output_dir = sys.argv[3]
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 298

    build_system(system_name, traj_path, output_dir, temperature)
