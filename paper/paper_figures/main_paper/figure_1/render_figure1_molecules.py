"""Render the molecule images for Figure 1, headless, with no external renderer.

Produces two kinds of CPK-colored, transparent-background PNGs from an electrolyte
simulation box, ready to drop into the Figure 1 overview (see
figure1_overview_prompt.md):

  Image A -- the full periodic simulation box (a few view angles).
  Image B -- a single Na+ solvation cluster extracted from that box (with bonds
             and dashed Na-O coordination lines).

Self-contained orthographic renderer built on matplotlib (depth-sorted spheres
with depth shading); no POV-Ray / VMD / OVITO needed. Run with the fairchemV3
env, which has ase + matplotlib:

    mamba activate fairchemV3
    python -m paper.paper_figures.main_paper.figure_1.render_figure1_molecules

Defaults render the 1 M NaPF6/DME box; pass --box / --out to change.
"""
import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from ase import Atoms
from ase.io import read
from ase.data import covalent_radii
from ase.data.colors import jmol_colors

from src.analysis.observables import paths

DEFAULT_BOX = (str(paths.CFS / "m5024/distillation_project/initial_boxes") + "/"
               "initial_boxes_for_UMA_data_generation/1M/napf6_dme/napf6_dme.traj")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders")


def rot_matrix(ax_deg, ay_deg):
    ax, ay = np.radians(ax_deg), np.radians(ay_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    return Ry @ Rx


def shade(rgb, depth01):
    """Darken a color toward the back (depth01: 0 = front, 1 = back)."""
    f = 1.0 - 0.45 * depth01
    return tuple(np.clip(np.array(rgb) * f, 0, 1))


def render(positions, numbers, ax_deg, ay_deg, path, radius_scale=0.75,
           bonds=None, cell=None, pad=2.0, dpi=300, lw_edge=0.5,
           coord_lines=None, figsize=6.0):
    """Orthographic painter's-algorithm render of one structure to a PNG."""
    R = rot_matrix(ax_deg, ay_deg)
    p = positions @ R.T
    z = p[:, 2]
    order = np.argsort(z)                         # back -> front
    zmin, zmax = z.min(), z.max()
    span = max(zmax - zmin, 1e-6)

    xs, ys = p[:, 0], p[:, 1]
    xmin, xmax = xs.min() - pad, xs.max() + pad
    ymin, ymax = ys.min() - pad, ys.max() + pad

    fig, ax = plt.subplots(figsize=(figsize, figsize * (ymax - ymin) / (xmax - xmin)))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.axis("off")

    if cell is not None:                           # cell wireframe, behind atoms
        corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        cc = (corners @ cell) @ R.T
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                 if np.sum(np.abs(corners[a] - corners[b])) == 1]
        segs = [[cc[a, :2], cc[b, :2]] for a, b in edges]
        ax.add_collection(LineCollection(segs, colors="#9AA1AD", linewidths=0.8,
                                         alpha=0.5, zorder=0))

    if coord_lines:                                # Na--O coordination (dashed)
        segs = [[p[i, :2], p[j, :2]] for i, j in coord_lines]
        ax.add_collection(LineCollection(segs, colors="#7A4Fd6", linewidths=1.4,
                                         linestyles=(0, (2, 2)), alpha=0.9, zorder=1))

    if bonds:                                      # covalent bonds (solid)
        segs = [[p[i, :2], p[j, :2]] for i, j in bonds]
        ax.add_collection(LineCollection(segs, colors="#3A3F4B", linewidths=2.2,
                                         alpha=0.95, zorder=2))

    for idx in order:
        Z = numbers[idx]
        d01 = (zmax - z[idx]) / span
        rgb = shade(jmol_colors[Z], d01)
        rad = covalent_radii[Z] * radius_scale
        ax.add_patch(Circle((p[idx, 0], p[idx, 1]), rad, facecolor=rgb,
                            edgecolor=shade(jmol_colors[Z] * 0.45, d01),
                            linewidth=lw_edge, zorder=3 + d01))
    fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", default=DEFAULT_BOX, help="ASE-readable structure (last frame used).")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output directory for PNGs.")
    ap.add_argument("--name", default="napf6_dme_1M", help="Tag used in output filenames.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    atoms = read(args.box, index=-1)               # do NOT wrap: keep molecules whole
    sym = np.array(atoms.get_chemical_symbols())
    pos = atoms.get_positions()
    num = atoms.get_atomic_numbers()
    cell = atoms.get_cell().array
    print(f"box: {len(atoms)} atoms, cell diag {np.diag(cell).round(1)}, "
          f"species {sorted(set(sym))}")

    # ---- Image A: full box, three views ----
    center = pos.mean(0)
    for tag, (axd, ayd) in {"v1": (18, -28), "v2": (12, 32), "v3": (90, 0)}.items():
        render(pos - center, num, axd, ayd, f"{args.out}/boxA_{args.name}_{tag}.png",
               radius_scale=0.62, cell=cell, pad=1.5, figsize=7.0, lw_edge=0.4)

    # ---- Image B: Na+ (or Li+) solvation cluster ----
    cation = "Na" if "Na" in sym else "Li"
    cat_idx = np.where(sym == cation)[0]
    oxy = np.where(sym == "O")[0]
    best, best_no = cat_idx[0], -1
    for i in cat_idx:                              # pick the best-coordinated cation
        d = atoms.get_distances(i, oxy, mic=True)
        if (d < 2.8).sum() > best_no:
            best_no, best = int((d < 2.8).sum()), i
    print(f"chosen {cation} index {best} with {best_no} O within 2.8 A")

    allidx = np.arange(len(atoms))
    vecs = atoms.get_distances(best, allidx, mic=True, vector=True)   # cation -> j
    dist = np.linalg.norm(vecs, axis=1)
    sel = list(allidx[dist <= 4.6])

    # include the nearest whole anion centre (P for PF6 / S for OTf) + its halogens
    for centre, hal, rbond in (("P", "F", 1.9), ("S", "F", 1.9)):
        cidx = np.where(sym == centre)[0]
        if not len(cidx):
            continue
        cd = atoms.get_distances(best, cidx, mic=True)
        near = cidx[int(np.argmin(cd))]
        if cd.min() < 7.5:
            hidx = np.where(sym == hal)[0]
            hd = atoms.get_distances(near, hidx, mic=True)
            sel += [near] + list(hidx[hd < rbond])
        break
    sel = np.unique(sel)

    cpos = pos[best] + atoms.get_distances(best, sel, mic=True, vector=True)  # unwrapped
    clu = Atoms(numbers=num[sel], positions=cpos)
    csym = clu.get_chemical_symbols()
    print(f"cluster: {len(clu)} atoms, species {sorted(set(csym))}")

    # bonds: covalent within cluster (exclude cation); cation-O drawn as coordination
    bonds, coord = [], []
    cp = clu.get_positions()
    cnum = clu.get_atomic_numbers()
    for a in range(len(clu)):
        for b in range(a + 1, len(clu)):
            d = np.linalg.norm(cp[a] - cp[b])
            if csym[a] == cation or csym[b] == cation:
                if {csym[a], csym[b]} == {cation, "O"} and d < 2.9:
                    coord.append((a, b))
                continue
            if d < (covalent_radii[cnum[a]] + covalent_radii[cnum[b]]) * 1.2:
                bonds.append((a, b))

    cc = cp - cp.mean(0)
    for tag, (axd, ayd) in {"v1": (15, -20), "v2": (20, 35)}.items():
        render(cc, cnum, axd, ayd, f"{args.out}/clusterB_{cation.lower()}_solvation_{tag}.png",
               radius_scale=0.9, bonds=bonds, coord_lines=coord, pad=1.2,
               figsize=5.0, lw_edge=0.6, dpi=300)


if __name__ == "__main__":
    main()
