"""
Packed vs equilibrated molarity and density for every equilibrated box.

The packmol boxes were sized from a target mass density rather than from
molarity, so each one starts dilute relative to its label. The NPT stage relaxes
the volume at fixed particle count, so it moves the molarity by exactly the
volume ratio -- this script measures how much of the labelled concentration that
recovers.

Reads <box>.gro (packed) and npt.gro (equilibrated) for the cell vector, and
meta.json for the salt count and label. Averages come from the NPT log, so a
box is only reported once its npt has finished.

Usage:
    source config.sh
    python analyze_equilibrated.py            # -> stdout table + equilibrated_molarity.csv
"""

import os
import re
import csv
import glob
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)   # the OPLS baseline root (tools/ lives under it)
EQ_ROOT = os.environ.get("CM_EQ_ROOT", os.path.join(_BASE, "runs", "equilibrated"))
NA = 6.02214076e23


def box_length_nm(gro):
    """Cubic cell side (nm) from the last line of a .gro file."""
    with open(gro) as f:
        last = f.read().rstrip("\n").splitlines()[-1]
    return float(last.split()[0])


def molarity(n_pairs, L_nm):
    """mol/L for n_pairs salt formula units in a cubic cell of side L_nm."""
    volume_L = (L_nm * 1e-7) ** 3 / 1000.0   # 1 nm = 1e-7 cm; 1 L = 1000 cm^3
    return n_pairs / (NA * volume_L)


def log_average(log, key):
    """Pull one quantity out of the A V E R A G E S block of an mdrun log."""
    if not os.path.exists(log):
        return None
    with open(log) as f:
        text = f.read()
    i = text.find("A V E R A G E S")
    if i < 0:
        return None
    block = text[i:i + 4000].splitlines()
    for j, line in enumerate(block[:-1]):
        if key in line:
            # values sit on the line below the header, in the same column
            col = line.index(key)
            nxt = block[j + 1]
            vals = list(re.finditer(r"-?\d+\.\d+e[+-]\d+", nxt))
            if not vals:
                continue
            # the value whose field ends nearest the header's end
            best = min(vals, key=lambda m: abs(m.end() - (col + len(key))))
            return float(best.group())
    return None


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(EQ_ROOT, "*"))):
        meta_p = os.path.join(d, "meta.json")
        npt_gro = os.path.join(d, "npt.gro")
        if not (os.path.exists(meta_p) and os.path.exists(npt_gro)):
            continue
        log = os.path.join(d, "npt.log")
        if "Finished mdrun" not in open(log).read():
            continue   # still running; npt.gro is a partial write
        meta = json.load(open(meta_p))
        name = meta["name"]
        packed_gro = os.path.join(d, f"{name}.gro")
        L0, L1 = box_length_nm(packed_gro), box_length_nm(npt_gro)
        n = meta["n_salt_pairs"]
        m0, m1 = molarity(n, L0), molarity(n, L1)
        target = meta["target_molarity_M"]
        rows.append({
            "box": name,
            "solvent": meta["system"].split("_")[1],
            "target_M": target,
            "packed_M": round(m0, 4),
            "equil_M": round(m1, 4),
            "packed_over_target": round(m0 / target, 4),
            "equil_over_target": round(m1 / target, 4),
            "L_packed_nm": round(L0, 4),
            "L_equil_nm": round(L1, 4),
            "volume_ratio": round((L1 / L0) ** 3, 4),
            "T_avg_K": log_average(log, "Temperature"),
            "P_avg_bar": log_average(log, "Pressure (bar)"),
        })

    if not rows:
        print(f"No finished boxes under {EQ_ROOT}")
        return

    hdr = f"{'box':<24}{'target':>7}{'packed':>8}{'equil':>8}{'pk/tg':>7}{'eq/tg':>7}{'V1/V0':>7}{'T_avg':>8}{'P_avg':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        t = f"{r['T_avg_K']:.1f}" if r["T_avg_K"] is not None else "--"
        p = f"{r['P_avg_bar']:.1f}" if r["P_avg_bar"] is not None else "--"
        print(f"{r['box']:<24}{r['target_M']:>7.2f}{r['packed_M']:>8.3f}{r['equil_M']:>8.3f}"
              f"{r['packed_over_target']:>7.2f}{r['equil_over_target']:>7.2f}"
              f"{r['volume_ratio']:>7.3f}{t:>8}{p:>9}")

    out = os.path.join(_BASE, "equilibrated_molarity.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} box(es) -> {out}")

    # Per-solvent summary: the packed deficit fell into a 0.80x and a 0.60x
    # group, so report whether equilibration closes both the same way.
    print("\nby solvent (mean of equil/target):")
    for solv in sorted({r["solvent"] for r in rows}):
        sub = [r for r in rows if r["solvent"] == solv]
        pk = sum(r["packed_over_target"] for r in sub) / len(sub)
        eq = sum(r["equil_over_target"] for r in sub) / len(sub)
        print(f"  {solv:<10} n={len(sub):<3} packed {pk:.2f}x -> equilibrated {eq:.2f}x")


if __name__ == "__main__":
    main()
