"""Compute pairwise RDFs for the 8 UMA 1 M (293 K) production trajectories that
also have training datasets (NaOTf and NaPF6 in DME, DEGDME, TEGDME and PC), and write one ``rdf.csv`` per system into a flat
output tree.

    python -m src.analysis.observables.structural.compute_uma_1M_rdfs            # all 8, parallel
    python -m src.analysis.observables.structural.compute_uma_1M_rdfs --systems naotf_dme,napf6_pc

These trajectories are the raw flat ``<system>.traj`` files under one shared
folder (not per-run folders). This thin batch layer reuses the package's
tested ``RDFAccumulator`` + lazy frame ``io`` + the per-system RDF-pair registry,
and just handles (a) the flat trajectory layout, (b) a fraction-based
equilibration drop (these are flexible-cell NPT runs with no fixed dt, so a
frame *fraction* is more robust than ``--eq-time-ps``), and (c) the requested
``{out}/{system}_1M/rdf.csv`` output layout.

Three pairs per system (the registry defaults — the most important interactions
for electrolyte solvation / ion pairing):
  * ``Na-O``      cation--solvent (and anion O for OTf): the solvation shell
  * ``Na-F``      cation--anion via fluorine: contact ion pairing
  * ``Na-{P,S}``cation--anion centre (PF6 / OTf): ion-pair association

``r_max`` is chosen **per system**: minimum-image is only valid out to half the
smallest cell perpendicular width, so each system's g(r) is binned wide (the
``--r-max-target``, default 12 Å) and then truncated to ``min(target,
½·min_width)`` over the frames actually used. Truncation is exact — minimum-image
at any r < ½·width is computed correctly regardless of the binning ceiling. This
gives the ideal 12 Å for the well-behaved boxes and only clips the systems whose
flexible-cell NPT box deformed (e.g. naotf_dme elongates to a ~17.7 Å short axis
-> ½ = 8.8 Å, so its g(r) honestly cannot reach 10 Å with isotropic min-image).

These are ~1 ns runs saved every 10 fs (~106k frames); the first ``--eq-frac``
(default 0.30, ~0.3 ns) is dropped as equilibration and ~``--n-frames`` are
sampled evenly from the rest.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .. import io, paths
from ..rdf import RDFAccumulator
from ..systems import SYSTEMS

# The 8 systems with a 1 M / 293 K UMA trajectory *and* a training dataset at 1 M
# (lipf6_dme was trained at 0.5 M only, so it has no entry here)
# (see docs/uma_trajectory_coverage_293K.md). For each, the system_id equals the
# trajectory file's basename in the source folder.
TRAJ_ROOT = paths.UMA_293K_1ns
OUT_ROOT = paths.OUT_UMA_293K_RDF
SYSTEMS_1M = [
    "naotf_diglyme", "naotf_dme", "naotf_pc", "naotf_tgdme",
    "napf6_diglyme", "napf6_dme", "napf6_pc", "napf6_tgdme",
]

# Trajectories are saved every 10 fs (~106k frames over ~1 ns).
DT_PS = 0.01



def _perp_min_width(cell) -> float:
    """Smallest perpendicular width of a (triclinic-capable) cell — the hard
    ceiling for minimum-image distances is half this value."""
    a, b, c = np.asarray(cell)
    vol = abs(np.dot(a, np.cross(b, c)))
    return min(vol / np.linalg.norm(np.cross(b, c)),
               vol / np.linalg.norm(np.cross(a, c)),
               vol / np.linalg.norm(np.cross(a, b)))


def _valid_r_max(min_width: float, target: float) -> float:
    """Largest min-image-valid r_max <= target, floored to a clean 0.05 grid
    edge (and < ½·min_width, so every used frame is safe)."""
    half = 0.5 * min_width
    return float(min(target, np.floor(half / 0.05) * 0.05))


def _first_peak(r, g):
    """First g(r) maximum beyond the 1.5 Å self/bonded core."""
    mask = r > 1.5
    if not np.any(g[mask] > 0):
        return None
    i = int(np.argmax(g[mask]) + np.searchsorted(r, 1.5))
    return {"r": float(r[i]), "g": float(g[i])}



def compute_one(system_id: str, n_frames: int, eq_frac: float,
                r_max_target: float, dr: float, out_root: Path, git_sha: str) -> dict:
    """RDF for one system over the post-equilibration frame window; writes
    ``{out_root}/{system_id}_1M/rdf.csv`` (+ ``rdf_summary.json``).

    Bins out to ``r_max_target`` while tracking the smallest cell width over the
    used frames, then truncates g(r) to the min-image-valid range for *this*
    system (exact: min-image at r < ½·width is unaffected by the bin ceiling)."""
    spec = SYSTEMS[system_id]
    pairs = list(spec.rdf_pairs)
    traj = TRAJ_ROOT / f"{system_id}.traj"

    total = io.estimate_n_frames(traj)
    start = int(total * eq_frac)
    avail = max(total - start, 1)
    stride = max(1, avail // n_frames)

    acc = RDFAccumulator(pairs, r_max=r_max_target, dr=dr)
    used = 0
    min_width = np.inf
    for atoms in io.iter_frames(traj, start=start, stride=stride, max_count=n_frames):
        acc.update(atoms)
        min_width = min(min_width, _perp_min_width(atoms.get_cell()))
        used += 1
    res = acc.finalize()

    # Truncate each g(r) to this system's min-image-valid range.
    r_max = _valid_r_max(min_width, r_max_target)
    keep = res.r <= r_max + 1e-9
    r_out = res.r[keep]
    g_out = {p: res.g[p][keep] for p in pairs}
    peaks = {p: pk for p in pairs if (pk := _first_peak(r_out, g_out[p])) is not None}

    out_dir = out_root / f"{system_id}_1M"
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [r_out] + [g_out[p] for p in pairs]
    header = ",".join(["r"] + [f"g_{p}" for p in pairs])
    np.savetxt(out_dir / "rdf.csv", np.column_stack(cols),
               delimiter=",", header=header, comments="", fmt="%.6g")

    summary = {
        "system_id": system_id,
        "cation": spec.cation, "anion": spec.anion, "solvent": spec.solvent,
        "concentration_M": "1M", "temperature_K": "293K",
        "rdf": {
            "pairs": pairs, "r_max": r_max, "r_max_target": r_max_target,
            "dr": dr, "min_perp_width_window_A": round(float(min_width), 3),
            "peaks": peaks, "n_frames": used,
        },
        "sampling": {
            "dt_ps": DT_PS, "n_frames_requested": n_frames, "n_frames_used": used,
            "eq_frac_dropped": eq_frac, "eq_time_ps": round(start * DT_PS, 1),
            "window_ps": round((total - start) * DT_PS, 1),
            "start_frame": start, "stride": stride, "n_frames_total": int(total),
        },
        "provenance": {"traj_path": str(traj), "code_version": git_sha},
    }
    (out_dir / "rdf_summary.json").write_text(json.dumps(summary, indent=2))

    return {"system_id": system_id, "out": str(out_dir / "rdf.csv"),
            "frames_used": used, "pairs": pairs, "r_max": r_max,
            "min_width": round(float(min_width), 1),
            "peak_r": {k: round(v["r"], 2) for k, v in peaks.items()}}


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=None,
                    help="Comma list of system_ids (default: all 8).")
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="Output root; each system writes to <root>/<system>_1M/rdf.csv.")
    ap.add_argument("--n-frames", type=int, default=3000,
                    help="Frames sampled evenly across the post-eq window (default 3000).")
    ap.add_argument("--eq-frac", type=float, default=0.30,
                    help="Fraction of frames dropped as equilibration (default 0.30, ~0.3 ns).")
    ap.add_argument("--r-max-target", type=float, default=12.0,
                    help="Ideal r_max (Å); each system is truncated to min(target, ½·min_width).")
    ap.add_argument("--dr", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=9, help="Parallel systems (default 9).")
    ap.add_argument("--out-suffix", default="",
                    help="Appended to the output directory name (e.g. _test), so a rerun "
                         "does not overwrite the results the figures read.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the trajectories that would be processed and exit.")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    systems = ([s.strip() for s in args.systems.split(",")] if args.systems
               else list(SYSTEMS_1M))
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        raise SystemExit(f"Unknown system_ids: {unknown}. Known: {sorted(SYSTEMS)}")
    out_root = Path(args.out_root + args.out_suffix)
    git_sha = paths.git_sha()

    if args.dry_run:
        print(f"{len(systems)} systems -> {out_root}")
        for s in systems:
            traj = TRAJ_ROOT / f"{s}.traj"
            print(f"  {s:16s} {'ok     ' if traj.is_file() else 'MISSING'} {traj}")
        return

    print(f"Computing RDFs for {len(systems)} systems -> {out_root}")
    print(f"  pairs per system: registry defaults (Na-O, Na-F, Na-<centre>)")
    print(f"  n_frames={args.n_frames}  eq_frac={args.eq_frac} (~{args.eq_frac*1.06:.2f} ns dropped)  "
          f"r_max_target={args.r_max_target} Å  dr={args.dr} Å")

    results = []
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(systems))) as ex:
        futs = {ex.submit(compute_one, s, args.n_frames, args.eq_frac,
                          args.r_max_target, args.dr, out_root, git_sha): s
                for s in systems}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="systems"):
            s = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                tqdm.write(f"  [done] {s:16s} {res['pairs']}  r_max={res['r_max']:.2f} Å "
                           f"(minW={res['min_width']:.1f})  peak_r={res['peak_r']}")
            except Exception as e:
                tqdm.write(f"  [FAIL] {s}: {e}")

    print(f"\nWrote {len(results)} rdf.csv files under {out_root}")


if __name__ == "__main__":
    main()
