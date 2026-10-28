"""
Pick a production seed frame out of each equilibrated NPT trajectory.

Selection rule (one frame per box):

  1. mean density  <rho>  over the second half of the NPT run, 500 -> 1000 ps,
     taken from npt.edr (0.5 ps energy grid, so 1001 samples).
  2. among the trajectory frames actually written to npt.xtc in that same
     window (nstxout-compressed = 1000 steps = 1 ps, so 501 candidates),
     take the one whose instantaneous density is closest to <rho>.
  3. dump that frame with `gmx trjconv -dump` -> <box>/seed.gro.

The first 500 ps is discarded because the packmol boxes start ~0.8x (0.6x for
TEGDME) too dilute and the c-rescale barostat spends the early part of the run
collapsing the cell -- including it would drag <rho> below the equilibrated
plateau and bias every seed low.

Density of a trajectory frame is computed as M / V from the cell vectors in the
frame itself, with the system mass M taken from the .edr as <Density * Volume>
(constant in NPT). That keeps step 2 exactly consistent with the .gro that is
finally written: the same box vectors produce both the selection metric and the
reported molarity, so the seed's molarity is not an estimate.

Molarity of the seed uses n_salt_pairs from meta.json and the seed cell volume,
and is compared against the label (target_molarity_M) in a parity plot.

Usage:
    source config.sh
    python prepare_production.py             # -> $CM_PROD_ROOT
    python prepare_production.py --dry-run   # select + report, write nothing
"""

import os
import csv
import glob
import json
import shutil
import struct
import argparse
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
EQ_ROOT = os.environ.get("CM_EQ_ROOT", os.path.join(_HERE, "runs", "equilibrated"))
PROD_ROOT = os.environ.get(
    "CM_PROD_ROOT", os.path.join(_HERE, "runs", "production_ready"))
GMX = os.environ.get("GMXBIN", "gmx")

NA = 6.02214076e23
T_BEGIN_PS = 500.0     # start of the averaging / selection window
T_END_PS = 1000.0


# --- .edr ------------------------------------------------------------------

def edr_series(run_dir, terms=("Volume", "Density")):
    """{term: [(t_ps, value), ...]} via `gmx energy` on npt.edr."""
    edr = os.path.join(run_dir, "npt.edr")
    out = {}
    for term in terms:
        xvg = os.path.join(run_dir, f"_prod_{term.lower()}.xvg")
        subprocess.run(
            [GMX, "energy", "-f", edr, "-o", xvg],
            input=f"{term}\n", text=True, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        pts = []
        with open(xvg) as f:
            for line in f:
                if line[0] in "#@":
                    continue
                t, v = line.split()[:2]
                pts.append((float(t), float(v)))
        os.remove(xvg)
        out[term] = pts
    return out


# --- .xtc ------------------------------------------------------------------

def xtc_frames(xtc):
    """[(time_ps, [box_xx, box_yy, box_zz]), ...] by walking the xtc headers.

    An xtc frame header is magic(1995), natoms, step, time, then the 3x3 box as
    floats; after that comes the compressed coordinate block whose byte length
    is stored in its own header, so frames can be skipped without decoding any
    coordinates. Everything is big-endian.
    """
    frames = []
    with open(xtc, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            magic, = struct.unpack(">i", head)
            if magic != 1995:
                raise ValueError(f"{xtc}: bad frame magic {magic}")
            natoms, _step = struct.unpack(">ii", f.read(8))
            time, = struct.unpack(">f", f.read(4))
            box = struct.unpack(">9f", f.read(36))
            frames.append((time, [box[0], box[4], box[8]]))
            # coordinate block
            n, = struct.unpack(">i", f.read(4))
            if n <= 9:                       # stored uncompressed
                f.seek(n * 3 * 4, os.SEEK_CUR)
            else:
                # precision(float) + minint[3] + maxint[3] + smallidx(int)
                f.seek(4 + 12 + 12 + 4, os.SEEK_CUR)
                nbytes, = struct.unpack(">i", f.read(4))
                f.seek((nbytes + 3) // 4 * 4, os.SEEK_CUR)          # padded to 4 bytes
    return frames


# --- .gro ------------------------------------------------------------------

def gro_box(gro):
    """Cell vector lengths (nm) from the last line of a .gro file."""
    with open(gro) as f:
        last = f.read().rstrip("\n").splitlines()[-1]
    v = [float(x) for x in last.split()]
    return v[:3]


def molarity(n_pairs, volume_nm3):
    """mol/L for n_pairs formula units in volume_nm3."""
    # 1 nm = 1e-7 cm  ->  1 nm^3 = 1e-21 cm^3 = 1e-24 L
    return n_pairs / (NA * volume_nm3 * 1e-24)


# --- per-box ---------------------------------------------------------------

def select_seed(run_dir, dry_run=False):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    name = meta["name"]
    log = os.path.join(run_dir, "npt.log")
    if "Finished mdrun" not in open(log).read():
        return None

    ser = edr_series(run_dir)
    win = [(t, v) for t, v in ser["Density"] if T_BEGIN_PS <= t <= T_END_PS]
    rho_mean = sum(v for _, v in win) / len(win)
    rho_std = (sum((v - rho_mean) ** 2 for _, v in win) / len(win)) ** 0.5

    # system mass from <rho*V>, constant in NPT -- lets a frame's density come
    # straight from its own cell vectors
    vol = dict(ser["Volume"])
    mass_kg = sum(v * vol[t] for t, v in ser["Density"] if t in vol) / len(
        [t for t, _ in ser["Density"] if t in vol]
    ) * 1e-27   # kg/m^3 * nm^3 -> kg

    frames = [(t, b) for t, b in xtc_frames(os.path.join(run_dir, "npt.xtc"))
              if T_BEGIN_PS <= t <= T_END_PS]
    cand = []
    for t, b in frames:
        v_nm3 = b[0] * b[1] * b[2]
        cand.append((t, b, v_nm3, mass_kg / (v_nm3 * 1e-27)))
    t_sel, box_sel, v_sel, rho_sel = min(cand, key=lambda c: abs(c[3] - rho_mean))

    row = {
        "box": name,
        "solvent": meta["system"].split("_")[1],
        "target_M": meta["target_molarity_M"],
        "n_salt_pairs": meta["n_salt_pairs"],
        "rho_mean_kgm3": round(rho_mean, 3),
        "rho_std_kgm3": round(rho_std, 3),
        "n_edr_samples": len(win),
        "n_xtc_candidates": len(cand),
        "seed_time_ps": t_sel,
        "seed_rho_kgm3": round(rho_sel, 3),
        "rho_dev_kgm3": round(rho_sel - rho_mean, 4),
        "rho_dev_sigma": round((rho_sel - rho_mean) / rho_std, 4) if rho_std else 0.0,
        "L_seed_nm": round(box_sel[0], 5),
        "V_seed_nm3": round(v_sel, 4),
        "seed_M": round(molarity(meta["n_salt_pairs"], v_sel), 5),
    }
    row["seed_over_target"] = round(row["seed_M"] / row["target_M"], 4)

    if dry_run:
        return row

    out_dir = os.path.join(PROD_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    seed_gro = os.path.join(out_dir, "seed.gro")
    subprocess.run(
        [GMX, "trjconv", "-s", os.path.join(run_dir, "npt.tpr"),
         "-f", os.path.join(run_dir, "npt.xtc"), "-o", seed_gro,
         "-dump", str(t_sel), "-pbc", "mol"],
        input="System\n", text=True, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # the dumped cell must be the cell we selected on, else the reported
    # molarity describes a different frame than the one on disk
    got = gro_box(seed_gro)
    if max(abs(g - s) for g, s in zip(got, box_sel)) > 1e-3:
        raise RuntimeError(f"{name}: dumped box {got} != selected {box_sel}")

    # topology travels with the seed so the production run is self-contained
    for src in glob.glob(os.path.join(run_dir, "*.itp")) + [
        os.path.join(run_dir, f"{name}.top")
    ]:
        shutil.copy2(src, out_dir)

    json.dump({**meta, "seed": row, "source_run": run_dir,
               "window_ps": [T_BEGIN_PS, T_END_PS]},
              open(os.path.join(out_dir, "seed_meta.json"), "w"), indent=2)
    return row


def parity_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solvents = sorted({r["solvent"] for r in rows})
    colors = plt.get_cmap("tab10")
    cof = {s: colors(i) for i, s in enumerate(solvents)}
    # only three distinct target molarities, so the parity panel alone hides
    # the per-box spread -- the residual panel carries the actual information
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.6, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})

    lim = [0, max(max(r["target_M"] for r in rows),
                  max(r["seed_M"] for r in rows)) * 1.12]
    ax.plot(lim, lim, "-", color="0.4", lw=1, zorder=1, label="parity")
    for frac, ls in ((0.95, ":"), (1.05, ":")):
        ax.plot(lim, [frac * x for x in lim], ls, color="0.7", lw=1, zorder=1)

    for s in solvents:
        sub = [r for r in rows if r["solvent"] == s]
        ax.scatter([r["target_M"] for r in sub], [r["seed_M"] for r in sub],
                   s=46, color=cof[s], edgecolor="white", linewidth=0.6,
                   zorder=3, label=s)

    dev = [abs(r["seed_over_target"] - 1) for r in rows]
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("assigned molarity  (mol/L)")
    ax.set_ylabel("production seed molarity  (mol/L)")
    ax.set_title(f"Seed vs assigned molarity, n={len(rows)}\n"
                 f"max |dev| {max(dev) * 100:.1f}%,  mean |dev| "
                 f"{sum(dev) / len(dev) * 100:.1f}%   (dotted = ±5%)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_aspect("equal")

    order = sorted(rows, key=lambda r: r["seed_over_target"])
    y = range(len(order))
    ax2.axvline(0, color="0.4", lw=1)
    for frac in (-5, 5):
        ax2.axvline(frac, ls=":", color="0.7", lw=1)
    ax2.barh(list(y), [(r["seed_over_target"] - 1) * 100 for r in order],
             color=[cof[r["solvent"]] for r in order], height=0.68)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels([r["box"] for r in order], fontsize=7)
    ax2.set_xlabel("seed molarity deviation from label  (%)")
    ax2.set_title("per-box residual", fontsize=10)
    ax2.set_ylim(-0.7, len(order) - 0.3)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    print(f"parity plot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="select and report, write nothing")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(EQ_ROOT, "*"))):
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        r = select_seed(d, dry_run=args.dry_run)
        if r is None:
            print(f"  skip {os.path.basename(d)}: npt not finished")
            continue
        rows.append(r)
        print(f"  {r['box']:<24} t={r['seed_time_ps']:>7.1f} ps  "
              f"rho {r['seed_rho_kgm3']:>8.2f} (<rho> {r['rho_mean_kgm3']:>8.2f}, "
              f"{r['rho_dev_sigma']:+.3f} sigma)  "
              f"M {r['seed_M']:.3f} / {r['target_M']:.2f} = {r['seed_over_target']:.3f}")

    if not rows:
        print(f"No finished boxes under {EQ_ROOT}")
        return
    if args.dry_run:
        return

    os.makedirs(PROD_ROOT, exist_ok=True)
    csv_p = os.path.join(PROD_ROOT, "production_seeds.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} seed(s) -> {PROD_ROOT}\n{csv_p}")

    parity_plot(rows, os.path.join(PROD_ROOT, "molarity_parity.png"))


if __name__ == "__main__":
    main()
