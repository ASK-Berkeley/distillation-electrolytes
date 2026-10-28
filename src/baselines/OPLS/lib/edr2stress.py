"""
GROMACS .edr  ->  Green-Kubo stress log in the fixed column format below.

    # Green-Kubo shear-viscosity stress log. pressure_unit=atm stress_interval[steps]=2 dt[fs]=1.0000
    # time_ps  Pxx Pyy Pzz Pxy Pxz Pyz  volume[Ang^3]  T[Kelvin]
    0.00200 7.361351e+02 ... -1.541734e+03 1.163783e+03 31149.8906 293.697

GROMACS does not write this file itself, but it has everything in it: with
`nstcalcenergy = nstenergy = 2` and `dt = 1 fs` the .edr carries the full
pressure tensor, the volume and the temperature every 2 fs, which is exactly
the sampling of the reference log. This module pulls those terms out with
`gmx energy` and reformats them.

Two conversions and one symmetrisation happen here:

  * pressure  bar -> atm   (GROMACS reports bar; the reference log is atm)
  * volume    nm^3 -> Ang^3
  * off-diagonals are averaged, Pxy = (Pres-XY + Pres-YX)/2. The GROMACS
    virial is not exactly symmetric frame to frame (PME mesh + the order in
    which the pair force is accumulated), and the Green-Kubo shear estimator
    wants the symmetric traceless part, so the average is the right input and
    it also halves the noise on each off-diagonal.

Streaming: `gmx energy` writes an .xvg that is as large as the output, so both
are processed line by line and the .xvg is removed afterwards. Nothing is held
in memory, which matters at 10^7 frames.

Usage:
    python edr2stress.py prod.part0001.edr -o run.stress            # write
    python edr2stress.py prod.part0002.edr -o run.stress --append   # continue
"""

import os
import re
import argparse
import subprocess

GMXBIN = os.environ.get("GMXBIN", "gmx")

BAR_TO_ATM = 1.0 / 1.01325          # 1 atm = 1.01325 bar exactly
NM3_TO_A3 = 1000.0

# `gmx energy` emits the selected terms in *.edr index* order, not in the order
# they were asked for, so the columns are located by their xvg legend rather
# than by position -- see legend_columns().
TERMS = ["Pres-XX", "Pres-YY", "Pres-ZZ",
         "Pres-XY", "Pres-YX", "Pres-XZ", "Pres-ZX", "Pres-YZ", "Pres-ZY",
         "Volume", "Temperature"]

# In a constant-volume run GROMACS writes no Box-*/Volume/Density terms at all
# -- there is nothing to record -- so Volume is absent from an NVT .edr and has
# to be supplied from the box itself (`volume_nm3`). It is genuinely constant
# there, so the column is just filled in.
OPTIONAL = {"Volume"}

HEADER = ("# Green-Kubo shear-viscosity stress log. pressure_unit=atm "
          "stress_interval[steps]={interval} dt[fs]={dt_fs:.4f}\n"
          "# time_ps  Pxx Pyy Pzz Pxy Pxz Pyz  volume[Ang^3]  T[Kelvin]\n")


def edr_to_xvg(edr, xvg):
    """Dump TERMS from `edr` into `xvg`, keeping the xmgrace legends."""
    sel = "\n".join(TERMS) + "\n"
    r = subprocess.run(
        [GMXBIN, "energy", "-f", edr, "-o", xvg],
        input=sel, text=True, capture_output=True,
    )
    if r.returncode != 0 or not os.path.exists(xvg):
        raise RuntimeError(f"gmx energy failed on {edr}:\n{r.stderr[-2000:]}")
    return xvg


def legend_columns(xvg):
    """{term: column index} from the `@ sN legend "term"` lines.

    Column 0 is time, so legend sN lives in column N+1. Every term in TERMS
    except the OPTIONAL ones must be found -- a silently missing one would
    shift the whole mapping and produce a plausible-looking but wrong log.
    """
    col = {}
    with open(xvg) as f:
        for line in f:
            if line[0] not in "#@":
                break
            m = re.match(r'@\s+s(\d+)\s+legend\s+"(.*)"', line.strip())
            if m:
                col[m.group(2)] = int(m.group(1)) + 1
    missing = [t for t in TERMS if t not in col and t not in OPTIONAL]
    if missing:
        raise RuntimeError(f"{xvg}: terms not found in legends: {missing}")
    return col


def convert(edr, out, interval=2, dt_fs=1.0, append=False, keep_xvg=False,
            t_offset=None, volume_nm3=None, t_min=None):
    """edr -> stress log. Returns (n_frames, last_time_ps).

    volume_nm3 is required for an NVT .edr, which carries no Volume term.

    t_min drops frames at or before that time. A checkpoint restart re-emits
    the frame it restarted from, so without this the segment boundary appears
    twice in the log and shows up as a spurious zero-lag repeat in the ACF.
    """
    xvg = os.path.splitext(edr)[0] + "_stress.xvg"
    edr_to_xvg(edr, xvg)
    c = legend_columns(xvg)
    ncol = max(c.values()) + 1
    if "Volume" not in c and volume_nm3 is None:
        raise RuntimeError(
            f"{edr} has no Volume term (constant-volume run) and no "
            "volume_nm3 was given -- pass the box volume explicitly")
    const_vol = None if "Volume" in c else volume_nm3 * NM3_TO_A3

    n, last_t = 0, None
    mode = "a" if append and os.path.exists(out) else "w"
    with open(xvg) as fin, open(out, mode) as fout:
        if mode == "w":
            fout.write(HEADER.format(interval=interval, dt_fs=dt_fs))
        for line in fin:
            if not line or line[0] in "#@":
                continue
            f = line.split()
            if len(f) < ncol:
                continue
            t = float(f[0])
            if t_offset is not None:
                t += t_offset
            if t_min is not None and t <= t_min + 1e-9:
                continue
            pxx = float(f[c["Pres-XX"]]) * BAR_TO_ATM
            pyy = float(f[c["Pres-YY"]]) * BAR_TO_ATM
            pzz = float(f[c["Pres-ZZ"]]) * BAR_TO_ATM
            pxy = (float(f[c["Pres-XY"]]) + float(f[c["Pres-YX"]])) * 0.5 * BAR_TO_ATM
            pxz = (float(f[c["Pres-XZ"]]) + float(f[c["Pres-ZX"]])) * 0.5 * BAR_TO_ATM
            pyz = (float(f[c["Pres-YZ"]]) + float(f[c["Pres-ZY"]])) * 0.5 * BAR_TO_ATM
            vol = const_vol if const_vol is not None else float(f[c["Volume"]]) * NM3_TO_A3
            temp = float(f[c["Temperature"]])
            fout.write(f"{t:.5f} {pxx:.6e} {pyy:.6e} {pzz:.6e} "
                       f"{pxy:.6e} {pxz:.6e} {pyz:.6e} {vol:.4f} {temp:.3f}\n")
            n += 1
            last_t = t

    if not keep_xvg:
        os.remove(xvg)
    return n, last_t


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("edr")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--interval", type=int, default=2,
                   help="stress dump interval in steps, for the header (default 2)")
    p.add_argument("--dt-fs", type=float, default=1.0,
                   help="timestep in fs, for the header (default 1.0)")
    p.add_argument("--append", action="store_true",
                   help="append to an existing log instead of rewriting its header")
    p.add_argument("--keep-xvg", action="store_true")
    p.add_argument("--volume-nm3", type=float, default=None,
                   help="cell volume; required for an NVT .edr, which has no "
                        "Volume term")
    a = p.parse_args()
    n, t = convert(a.edr, a.out, a.interval, a.dt_fs, a.append, a.keep_xvg,
                   volume_nm3=a.volume_nm3)
    print(f"{a.edr} -> {a.out}: {n} frame(s), last t = {t} ps")


if __name__ == "__main__":
    main()
