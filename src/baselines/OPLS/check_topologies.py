#!/usr/bin/env python
"""Self-check of every molecule topology the OPLS baseline uses.

For each species in build_system_lpg.MOLECULES this script builds a one-molecule
system in vacuum with the same code path as build_opls_boxes.py, runs grompp
with NO warnings allowed, and verifies against the molecule's own bond graph:

  nrexcl       every [ moleculetype ] has nrexcl 3 (OPLS)
  exclusions   the exclusion list in the .tpr (gmx dump) is exactly the set of
               atom pairs separated by <= 3 bonds
  pairs        the [ pairs ] block is exactly the set of 1-4 pairs (separated by
               exactly 3 bonds); for PF6- that set is empty
  charge       the summed partial charges are an integer (0 or +-1) to 1e-4 e
  masses       each atom's mass matches the element implied by its name
  geometry     after energy minimisation in vacuum, every bond is at its b0 and
               every angle at its theta0, within a tolerance (strict for the
               hand-written ion files, looser for the LigParGen solvents whose
               torsions and 1-4 terms legitimately strain a few angles)

The exclusion/pairs checks are what would have caught the PF6- bug described in
PF6_TOPOLOGY_BUG.md (nrexcl 1 plus a [ pairs ] list of all F-F pairs: 19
exclusion entries instead of 49, P-F stretched from 1.58 to 1.74 A). The
geometry check catches any topology whose minimised structure disagrees with
its own parameters.

Writes topology_check.txt next to this file and exits non-zero on any failure.
Run it after touching lib/build_system_lpg.py or anything under data/.

Usage:
    source config.sh            # or export GMXBIN=/path/to/gmx
    python check_topologies.py [--keep] [--only PF6 OTF]
"""
import os
import re
import sys
import shutil
import argparse
import tempfile
import subprocess
from itertools import combinations

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from lib import build_system_lpg as bsl

GMX = os.environ.get("GMXBIN", "gmx")
HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "topology_check.txt")

# Geometry tolerances after minimisation: (bond A, angle deg).
# A molecule whose only intramolecular terms are bonds and angles (PF6-, the
# lone ions) must reproduce its b0 / theta0 essentially exactly. A molecule
# with 1-4 pairs or dihedrals (the solvents, OTf-) is legitimately strained by
# those terms: the 0.5-scaled O...F Coulomb repulsion in triflate stretches S-C
# by ~0.04 A, and the PC ring cannot satisfy all its angle minima at once. The
# loose bound still catches a 0.1 A class error such as the old PF6- topology.
TOL_STRICT = (0.005, 0.5)
TOL_STRAINED = (0.05, 8.0)

EM_MDP = """\
integrator    = {integrator}
nsteps        = {nsteps}
emtol         = {emtol}
emstep        = 0.002
nstlist       = 10
cutoff-scheme = Verlet
pbc           = xyz
rlist         = 1.5
coulombtype   = Cut-off
rcoulomb      = 1.5
vdwtype       = Cut-off
rvdw          = 1.5
nstenergy     = 0
nstlog        = 0
nstxout       = 0
"""


# ---------------------------------------------------------------------------
# topology parsing
# ---------------------------------------------------------------------------
def parse_itp(path):
    """Full parse of a molecule .itp: name, nrexcl, atoms, bonds, pairs, angles."""
    d = {"nrexcl": None, "atoms": [], "bonds": [], "pairs": [], "angles": [],
         "has_dihedrals": False}
    section = None
    with open(path) as f:
        for raw in f:
            line = raw.split(";")[0].strip()
            if not line:
                continue
            if line.startswith("["):
                section = line.strip("[] ").split()[0].lower()
                continue
            p = line.split()
            if section == "moleculetype":
                d["name"], d["nrexcl"] = p[0], int(p[1])
            elif section == "atoms":
                d["atoms"].append({"nr": int(p[0]), "type": p[1], "name": p[4],
                                   "charge": float(p[6]), "mass": float(p[7])})
            elif section == "bonds":
                d["bonds"].append((int(p[0]), int(p[1]), float(p[3]) * 10.0))      # b0 in A
            elif section == "pairs":
                d["pairs"].append((int(p[0]), int(p[1])))
            elif section == "angles":
                d["angles"].append((int(p[0]), int(p[1]), int(p[2]), float(p[4])))
            elif section == "dihedrals":
                d["has_dihedrals"] = True
    return d


def graph_distances(n, bonds):
    """All-pairs bond-count distance matrix (BFS), 1-based atoms -> 0-based array."""
    adj = [[] for _ in range(n)]
    for a, b, _ in bonds:
        adj[a - 1].append(b - 1)
        adj[b - 1].append(a - 1)
    dist = np.full((n, n), 10**6, dtype=int)
    for s in range(n):
        dist[s, s] = 0
        q = [s]
        while q:
            u = q.pop(0)
            for v in adj[u]:
                if dist[s, v] > dist[s, u] + 1:
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)
    return dist


def element_from_name(name):
    m = re.match(r"([A-Z][a-z]?)", name)
    sym = m.group(1)
    if sym not in atomic_numbers:          # e.g. 'C1' -> 'C', 'Cl' stays
        sym = sym[0]
    return sym


# ---------------------------------------------------------------------------
# coordinates for a single molecule
# ---------------------------------------------------------------------------
def ideal_coords(mol_key, itp):
    """Starting coordinates (A) in ITP atom order. LigParGen molecules use their
    vendored conformer; the hand-written ions are built from internal coordinates
    (anything topologically right is fine, minimisation does the rest)."""
    if bsl.MOLECULES[mol_key]["source"] == "lpg":
        return bsl.load_template_coords(mol_key)
    if mol_key in ("NA", "LI"):
        return np.zeros((1, 3))
    if mol_key == "PF6":
        r = 1.606
        return np.array([[0, 0, 0], [r, 0, 0], [0, r, 0], [0, 0, r],
                         [0, -r, 0], [0, 0, -r], [-r, 0, 0]], float)
    if mol_key == "OTF":
        # S at origin, C on -z; O's at 102.6 deg from C, F's at 111.7 deg from S
        S = np.zeros(3)
        C = np.array([0, 0, -1.818])
        th_o = np.radians(180 - 102.6)
        Os = [1.437 * np.array([np.sin(th_o) * np.cos(p), np.sin(th_o) * np.sin(p), np.cos(th_o)])
              for p in np.radians([0, 120, 240])]
        th_f = np.radians(111.7)
        Fs = [C + 1.323 * np.array([np.sin(th_f) * np.cos(p), np.sin(th_f) * np.sin(p), np.cos(th_f)])
              for p in np.radians([60, 180, 300])]
        return np.array([S] + Os + [C] + Fs)
    raise KeyError(mol_key)


def write_gro(path, resname, names, coords_A, box_nm=6.0):
    with open(path, "w") as f:
        f.write(f"{resname} single molecule\n{len(names)}\n")
        for i, (nm, x) in enumerate(zip(names, coords_A), 1):
            p = x / 10.0 + box_nm / 2
            f.write(f"{1:5d}{resname:<5s}{nm:>5s}{i:5d}{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n")
        f.write(f"   {box_nm:.5f}   {box_nm:.5f}   {box_nm:.5f}\n")


def read_gro(path):
    L = open(path).read().splitlines()
    n = int(L[1])
    return np.array([[float(a[20:28]), float(a[28:36]), float(a[36:44])] for a in L[2:2 + n]]) * 10.0


# ---------------------------------------------------------------------------
# GROMACS calls
# ---------------------------------------------------------------------------
def run(cmd, cwd, log):
    with open(log, "a") as lf:
        lf.write("# " + " ".join(cmd) + "\n")
        r = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    return r.returncode == 0


def grompp(work, mdp, gro, top, tpr, log):
    # -maxwarn 0: any grompp WARNING fails the check. NOTEs are allowed.
    ok = run([GMX, "grompp", "-f", mdp, "-c", gro, "-p", top, "-o", tpr, "-maxwarn", "0"],
             work, log)
    text = open(log).read()
    warns = re.findall(r"^WARNING \d+ \[.*?\]:\n((?:  .*\n)+)", text, flags=re.M)
    return ok, [w.strip().replace("\n", " ") for w in warns]


def tpr_exclusions(tpr, work):
    """Set of 0-based excluded pairs (i<j) of the single molecule type in tpr."""
    r = subprocess.run([GMX, "dump", "-s", tpr], cwd=work, capture_output=True, text=True)
    excl = set()
    for m in re.finditer(r"excls\[(\d+)\]\[num=\d+\]=\{([^}]*)\}", r.stdout):
        i = int(m.group(1))
        for j in (int(x) for x in m.group(2).split(",") if x.strip()):
            if j != i:
                excl.add((min(i, j), max(i, j)))
    if not excl and "excls" not in r.stdout:
        raise RuntimeError(f"gmx dump produced no exclusion data for {tpr}")
    return excl


def minimise(work, gro, top, log):
    """steep then cg; returns path of the minimised .gro."""
    cur = gro
    for tag, integ, nsteps, emtol in (("steep", "steep", 5000, 10.0), ("cg", "cg", 20000, 0.1)):
        mdp = os.path.join(work, f"{tag}.mdp")
        open(mdp, "w").write(EM_MDP.format(integrator=integ, nsteps=nsteps, emtol=emtol))
        tpr = os.path.join(work, f"{tag}.tpr")
        ok, warns = grompp(work, mdp, cur, top, tpr, log)
        if not ok or warns:
            raise RuntimeError(f"grompp for {tag} failed/warned: {warns}")
        if not run([GMX, "mdrun", "-deffnm", tag, "-s", tpr, "-ntmpi", "1", "-ntomp", "1", "-nb", "cpu"],
                   work, log):
            raise RuntimeError(f"mdrun {tag} failed, see {log}")
        cur = os.path.join(work, f"{tag}.gro")
    return cur


# ---------------------------------------------------------------------------
def angle_deg(a, b, c):
    u, v = a - b, c - b
    cosang = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def check_molecule(mol_key, root, keep):
    info = bsl.MOLECULES[mol_key]
    work = os.path.join(root, mol_key)
    os.makedirs(work, exist_ok=True)
    log = os.path.join(work, "gmx.log")
    res = {"molecule": mol_key, "source": info["source"], "ok": True, "fail": [], "notes": []}

    # --- topology exactly as the box builder writes it
    lpg_keys = [mol_key] if info["source"] == "lpg" else []
    bsl.build_combined_forcefield(lpg_keys, work)
    if info["source"] == "lpg":
        itp_path = bsl.build_mol_itp(mol_key, work)
    else:
        itp_path = os.path.join(bsl.MANUAL_DIR, info["mol_itp"])
        shutil.copy(itp_path, work)
        itp_path = os.path.join(work, info["mol_itp"])
    top = os.path.join(work, f"{mol_key}.top")
    with open(top, "w") as f:
        f.write('#include "forcefield_lpg.itp"\n')
        f.write(f'#include "{os.path.basename(itp_path)}"\n\n')
        f.write(f"[ system ]\n{mol_key} single molecule\n\n[ molecules ]\n{info['name']}  1\n")

    itp = parse_itp(itp_path)
    n = len(itp["atoms"])
    res["n_atoms"] = n

    # --- nrexcl
    res["nrexcl"] = itp["nrexcl"]
    if itp["nrexcl"] != 3:
        res["fail"].append(f"nrexcl = {itp['nrexcl']} (OPLS requires 3)")

    # --- charge
    q = sum(a["charge"] for a in itp["atoms"])
    res["charge"] = round(q, 5)
    if abs(q - round(q)) > 1e-4:
        res["fail"].append(f"net charge {q:.5f} is not an integer")

    # --- masses vs element from the atom name
    for a in itp["atoms"]:
        el = element_from_name(a["name"])
        ref = atomic_masses[atomic_numbers[el]]
        if abs(a["mass"] - ref) > 0.05:
            res["fail"].append(f"atom {a['name']}: mass {a['mass']} but element {el} has {ref:.3f}")

    # --- bond-graph based expectations
    dist = graph_distances(n, itp["bonds"])
    if n > 1 and dist.max() >= 10**6:
        res["fail"].append("bond graph is disconnected")
    expect_excl = {(i, j) for i, j in combinations(range(n), 2) if dist[i, j] <= 3}
    expect_pairs = {(i, j) for i, j in combinations(range(n), 2) if dist[i, j] == 3}
    got_pairs = {(min(a, b) - 1, max(a, b) - 1) for a, b in itp["pairs"]}
    res["n_pairs"] = len(got_pairs)
    if got_pairs != expect_pairs:
        extra = sorted(got_pairs - expect_pairs)
        missing = sorted(expect_pairs - got_pairs)
        res["fail"].append(f"[ pairs ] != 1-4 set: {len(extra)} not 1-4 (e.g. {extra[:3]}), "
                           f"{len(missing)} missing (e.g. {missing[:3]})")
    # all angle triples must be real 1-3 paths
    for a, b, c, _ in itp["angles"]:
        if not (dist[a - 1, b - 1] == 1 and dist[b - 1, c - 1] == 1):
            res["fail"].append(f"angle {a}-{b}-{c} is not bonded a-b-c")

    # --- grompp with -maxwarn 0, then the exclusion list that mdrun would use
    names = [a["name"] for a in itp["atoms"]]
    gro = os.path.join(work, "start.gro")
    write_gro(gro, info["resname"], names, ideal_coords(mol_key, itp))
    mdp = os.path.join(work, "check.mdp")
    open(mdp, "w").write(EM_MDP.format(integrator="steep", nsteps=0, emtol=10.0))
    tpr = os.path.join(work, "check.tpr")
    ok, warns = grompp(work, mdp, gro, top, tpr, log)
    if not ok:
        res["fail"].append(f"grompp failed (see {log})")
        return res
    if warns:
        res["fail"].append("grompp WARNING(s): " + " | ".join(warns))
    got_excl = tpr_exclusions(tpr, work)
    res["n_excl_pairs"] = len(got_excl)
    res["n_excl_entries"] = n + 2 * len(got_excl)          # gmx dump counts i->i and both directions
    if got_excl != expect_excl:
        res["fail"].append(f"tpr exclusions ({len(got_excl)} pairs) != pairs within 3 bonds "
                           f"({len(expect_excl)} pairs)")

    # --- minimise and compare geometry with the topology's own b0 / theta0
    try:
        mgro = minimise(work, gro, top, log)
    except RuntimeError as e:
        res["fail"].append(str(e))
        return res
    x = read_gro(mgro)
    bdev = [abs(np.linalg.norm(x[a - 1] - x[b - 1]) - b0) for a, b, b0 in itp["bonds"]]
    adev = [abs(angle_deg(x[a - 1], x[b - 1], x[c - 1]) - t0) for a, b, c, t0 in itp["angles"]]
    strained = bool(itp["pairs"]) or itp["has_dihedrals"]
    tol_b, tol_a = TOL_STRAINED if strained else TOL_STRICT
    res["geometry_tolerance"] = "strained" if strained else "strict"
    res["max_bond_dev_A"] = round(max(bdev), 4) if bdev else 0.0
    res["max_angle_dev_deg"] = round(max(adev), 2) if adev else 0.0
    if bdev and max(bdev) > tol_b:
        k = int(np.argmax(bdev)); a, b, b0 = itp["bonds"][k]
        res["fail"].append(f"bond {names[a-1]}-{names[b-1]} minimises to "
                           f"{np.linalg.norm(x[a-1]-x[b-1]):.3f} A, b0 = {b0:.3f} A")
    if adev and max(adev) > tol_a:
        k = int(np.argmax(adev)); a, b, c, t0 = itp["angles"][k]
        res["fail"].append(f"angle {names[a-1]}-{names[b-1]}-{names[c-1]} minimises to "
                           f"{angle_deg(x[a-1], x[b-1], x[c-1]):.1f} deg, theta0 = {t0:.1f}")

    res["ok"] = not res["fail"]
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", help="molecule keys to check (default all)")
    ap.add_argument("--keep", action="store_true", help="keep the per-molecule work dirs")
    ap.add_argument("--workdir", default=None, help="where to build (default: temp dir)")
    args = ap.parse_args()

    if shutil.which(GMX) is None and not os.path.exists(GMX):
        sys.exit(f"gmx not found: GMXBIN={GMX}")
    gmx_version = subprocess.run([GMX, "--version"], capture_output=True, text=True).stdout
    gmx_version = next((l.split(":", 1)[1].strip() for l in gmx_version.splitlines()
                        if l.startswith("GROMACS version")), "unknown")

    keys = args.only or list(bsl.MOLECULES)
    root = args.workdir or tempfile.mkdtemp(prefix="opls_topocheck_")
    os.makedirs(root, exist_ok=True)

    results = [check_molecule(k, root, args.keep) for k in keys]
    for r in results:
        r["ok"] = not r["fail"]

    hdr = (f"{'molecule':<8}{'src':<8}{'atoms':>6}{'nrexcl':>7}{'excl':>6}{'pairs':>6}"
           f"{'charge':>8}{'dBond(A)':>10}{'dAng(deg)':>10}{'tol':>10}  result")
    lines = [f"# OPLS baseline topology check  (GROMACS {gmx_version})",
             "# excl  = exclusion entries as gmx dump counts them (7-atom PF6: 49 correct, 19 buggy)",
             "# pairs = [ pairs ] entries, must equal the number of 1-4 pairs in the bond graph",
             "# dBond/dAng = largest |minimised - b0| / |minimised - theta0|; tol: strict = bonds+angles only,",
             "#   strained = molecule also has 1-4 pairs / dihedrals that legitimately shift the minimum",
             hdr, "-" * len(hdr)]
    for r in results:
        lines.append(f"{r['molecule']:<8}{r['source']:<8}{r.get('n_atoms', 0):>6}{str(r.get('nrexcl')):>7}"
                     f"{r.get('n_excl_entries', '-'):>6}{r.get('n_pairs', '-'):>6}"
                     f"{r.get('charge', float('nan')):>8.3f}{r.get('max_bond_dev_A', float('nan')):>10.4f}"
                     f"{r.get('max_angle_dev_deg', float('nan')):>10.2f}{r.get('geometry_tolerance', '-'):>10}"
                     f"  {'PASS' if r['ok'] else 'FAIL'}")
        for f_ in r["fail"]:
            lines.append(f"    ! {f_}")
    n_fail = sum(not r["ok"] for r in results)
    lines.append("")
    lines.append(f"{len(results) - n_fail}/{len(results)} topologies pass")
    text = "\n".join(lines)
    print(text)
    with open(REPORT, "w") as f:
        f.write(text + "\n")
    if not args.keep and not args.workdir:
        shutil.rmtree(root, ignore_errors=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
