"""The PaiNN 293 K RDFs: one RDF per replica, plus the replica mean every figure
reads.

This is the only PaiNN RDF producer -- it supersedes the former
``compute_painn_1M_rdfs.py``, which computed a single RDF per system from
the r0 trajectory alone. It covers 9 systems: the 8 Na-salt systems at 1 M
(replicas r0-r4) and lipf6_dme at 0.5 M (r0 only). Each system's replicas (r0
plus the 1 ns reruns r1-r4, which differ from r0 only in FeNNol's
``random_seed``) are computed independently and then collapsed into an
aggregate.

``lipf6_dme`` is not a special case: it is a system with a different run base
(0.5 M) and a single replica, and it flows through the same path to an n=1
aggregate whose ``sd_`` columns are NaN.

Sampling params (``--n-frames 3000``, ``--eq-frac 0.30``, ``--r-max-target
12.0``, ``--dr 0.05``) match the UMA RDFs from ``compute_uma_1M_rdfs.py``, so
the two sides are directly comparable.

LAYOUT -- ``{out_root}/{system_id}_{conc}/``::

    rdf.csv              canonical: r, g_<pair> (replica mean), sd_<pair> (std)
    rdf_summary.json     n, replica list, peaks of the mean, per-replica provenance
    replicas/
        rdf_r<N>.csv          the per-replica curves the mean is built from
        rdf_r<N>_summary.json

The mean keeps the plain ``g_<pair>`` name the single-trajectory files used, so
existing readers pick up the replica mean with no code change. The spread is
``sd_<pair>`` and NOT ``g_<pair>_std`` on purpose: the figures detect pairs with
``[c[2:] for c in cols if c.startswith("g_")]``, so a ``g_``-prefixed std column
would be silently plotted as a phantom pair. This mirrors the UMA tree exactly,
which has no replicas and therefore no ``sd_`` columns.

Blown-up runs are skipped automatically. Five of the 32 replicas evaporated --
the box expanded to ~450 Å and the density collapsed to ~4e-4 g/cm^3 while the
thermostat held 293 K, so they exited cleanly with full-length trajectories.
Exit codes can't distinguish them, so every run is gated on its final-frame
density lying in ``[--rho-min, --rho-max]`` instead. That gate also covers any
future rerun dropped into the replica tree.

Those same 5 (naotf_pc r2, napf6_pc r1-r4) were rerun from a different seed; the
healthy reruns replaced the evaporated originals in the trajectory tree, so the
set is complete at n=5 for all 8 systems and the gate now rejects nothing.

CAVEAT: those 5 do NOT share ``start.xyz`` with the rest of the set. Their seed
is the 100 ps frame of the UMA 293 K / 1 M / 1 ns trajectory itself
(``paths.UMA_293K_1ns/<system>.traj``, frame 10000), i.e. a relaxed
non-cubic cell, rather than the cubic packing box r0 and r1-r4 start from. That
packing box is UMA's own frame 0 (rho=1.205, unrelaxed), and collapsing it to
~1.38 over the first ~100 ps is what made the PC systems evaporate; the 100 ps
frame is past the collapse but still at the initial Na-F pairing (CN 4.118, same
as t=0), so it skips the transient without changing structural basin. Same
composition and atom count, same state point, same MD settings. Defensible for
an equilibrium observable like g(r) after the 30 % equilibration drop; note it
for anything sensitive to the initial condition, especially napf6_pc where 4 of
its 5 replicas come from the new seed.
``shares_start_with_r0`` detects them by comparing ``start.xyz`` md5s rather
than by a hardcoded list, and records the answer in each
``rdf_r<N>_summary.json`` as ``provenance.shares_start_xyz_with_r0``. See the
README beside the trajectories.

    python -m src.analysis.observables.structural.compute_painn_replica_rdfs
    python -m src.analysis.observables.structural.compute_painn_replica_rdfs --systems naotf_pc,napf6_pc
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase.io.trajectory import Trajectory
from tqdm import tqdm

from .. import io, paths
from ..rdf import RDFAccumulator
from ..systems import SYSTEMS

# The whole replica set lives in one tree: r0 plus r1-r4 per system.
ROOT = paths.PAINN_293K_1ns
OUT_ROOT = paths.OUT_PAINN_293K_RDF

# system_id -> (run subfolder base, replicas that exist).
# The 8 Na systems each ran 5 replicas at 1 M. lipf6_dme is not a special case:
# it is simply a system with a different run base and a single replica, and it
# flows through the same code path to an n=1 aggregate (sd_ columns are NaN).
RUN_SPEC = {
    "napf6_dme":     ("npt_1M_293K_1ns_100fs",   [0, 1, 2, 3, 4]),
    "napf6_pc":      ("npt_1M_293K_1ns_100fs",   [0, 1, 2, 3, 4]),
    "napf6_diglyme": ("npt_1M_293K_1ns_100fs",   [0, 1, 2, 3, 4]),
    "napf6_tgdme":   ("npt_1M_293K_1ns_100fs",   [0, 1, 2, 3, 4]),
    "naotf_dme":     ("npt_1M_293K_1ns_100fs",   [0, 1, 2, 3, 4]),
    "naotf_pc":      ("npt_1M_293K_1ns_100fs",   [0, 1, 2, 3, 4]),
    "naotf_diglyme": ("npt_1M_293K_1ns_100fs",   [0, 1, 2, 3, 4]),
    "naotf_tgdme":   ("npt_1M_293K_1ns_100fs",   [0, 1, 2, 3, 4]),
    "lipf6_dme":     ("npt_0_5M_293K_1ns_100fs", [0]),
}
ALL_SYSTEMS = list(RUN_SPEC)

DT_PS = 0.1          # dt 1 fs, nprint 100 -> a frame every 100 fs
AMU_A3_TO_G_CM3 = 1.66053906660



def run_base(system_id: str) -> str:
    return RUN_SPEC[system_id][0]


def replicas_for(system_id: str) -> list[int]:
    return list(RUN_SPEC[system_id][1])


def conc_label(system_id: str) -> str:
    """``npt_1M_293K_…`` -> ``1M``; ``npt_0_5M_293K_…`` -> ``0.5M``."""
    token = run_base(system_id).split("_293K")[0].removeprefix("npt_")
    return token.replace("_", ".")


def out_dir_for(out_root: Path, system_id: str) -> Path:
    return out_root / f"{system_id}_{conc_label(system_id)}"


def run_dir(system_id: str, rep: int) -> Path:
    """r0 sits under the bare run name, r1-r4 under ``<run>_r<N>``."""
    base = run_base(system_id)
    if rep == 0:
        return ROOT / system_id / base
    return ROOT / system_id / f"{base}_r{rep}"


def traj_path(system_id: str, rep: int) -> Path:
    d = run_dir(system_id, rep)
    return d / f"{d.name}.traj"


def shares_start_with_r0(system_id: str, rep: int) -> bool | None:
    """Whether this replica started from the same frame as r0.

    True for the whole set *except* the 5 that were rerun on Delta (naotf_pc r2,
    napf6_pc r1-r4), which seed from a 100 ps UMA transport frame rather than the
    cubic packing box -- see the README beside the trajectories. Decided by
    comparing ``start.xyz`` against r0's rather than by a hardcoded list, so it
    stays correct if runs are moved or re-run. ``None`` if either file is absent.
    """
    if rep == 0:
        return True
    mine, theirs = run_dir(system_id, rep) / "start.xyz", run_dir(system_id, 0) / "start.xyz"
    if not (mine.exists() and theirs.exists()):
        return None
    return _md5(mine) == _md5(theirs)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _density(atoms) -> float:
    return float(atoms.get_masses().sum() / atoms.get_volume() * AMU_A3_TO_G_CM3)


def final_density(path: Path) -> float | None:
    """Density of the last frame, or None if the trajectory is unreadable/empty.
    One random access -- cheap enough to gate on before the full RDF pass."""
    try:
        tr = Trajectory(str(path))
        n = len(tr)
        return _density(tr[n - 1]) if n else None
    except Exception:
        return None


def _perp_min_width(cell) -> float:
    """Smallest perpendicular width of a (triclinic-capable) cell -- the hard
    ceiling for minimum-image distances is half this value."""
    a, b, c = np.asarray(cell)
    vol = abs(np.dot(a, np.cross(b, c)))
    return min(vol / np.linalg.norm(np.cross(b, c)),
               vol / np.linalg.norm(np.cross(a, c)),
               vol / np.linalg.norm(np.cross(a, b)))


def _valid_r_max(min_width: float, target: float) -> float:
    """Largest min-image-valid r_max <= target, floored to a clean 0.05 grid edge."""
    return float(min(target, np.floor(0.5 * min_width / 0.05) * 0.05))


def _first_peak(r, g):
    """First g(r) maximum beyond the 1.5 Å self/bonded core."""
    mask = r > 1.5
    if not np.any(g[mask] > 0):
        return None
    i = int(np.argmax(g[mask]) + np.searchsorted(r, 1.5))
    return {"r": float(r[i]), "g": float(g[i])}



def compute_one(system_id: str, rep: int, n_frames: int, eq_frac: float,
                r_max_target: float, dr: float, rho_min: float, rho_max: float,
                out_root: Path, git_sha: str) -> dict:
    """RDF for one replica. Returns a status dict; ``status`` is one of
    ``ok`` / ``missing`` / ``blown_up``."""
    traj = traj_path(system_id, rep)
    shares = shares_start_with_r0(system_id, rep)
    tag = f"{system_id} r{rep}"

    if not traj.exists():
        return {"system_id": system_id, "rep": rep, "status": "missing", "detail": str(traj)}

    rho_end = final_density(traj)
    if rho_end is None:
        return {"system_id": system_id, "rep": rep, "status": "missing",
                "detail": f"unreadable/empty: {traj}"}
    if not (rho_min < rho_end < rho_max):
        # Evaporated (or otherwise unphysical) -- excluded from the average.
        return {"system_id": system_id, "rep": rep, "status": "blown_up",
                "detail": f"final density {rho_end:.4g} g/cm^3"}

    spec = SYSTEMS[system_id]
    pairs = list(spec.rdf_pairs)

    total = io.estimate_n_frames(traj)
    start = int(total * eq_frac)
    avail = max(total - start, 1)
    stride = max(1, avail // n_frames)

    acc = RDFAccumulator(pairs, r_max=r_max_target, dr=dr)
    used = 0
    min_width = np.inf
    rho_sum = 0.0
    for atoms in io.iter_frames(traj, start=start, stride=stride, max_count=n_frames):
        acc.update(atoms)
        min_width = min(min_width, _perp_min_width(atoms.get_cell()))
        rho_sum += _density(atoms)
        used += 1
    res = acc.finalize()

    r_max = _valid_r_max(min_width, r_max_target)
    keep = res.r <= r_max + 1e-9
    r_out = res.r[keep]
    g_out = {p: res.g[p][keep] for p in pairs}
    peaks = {p: pk for p in pairs if (pk := _first_peak(r_out, g_out[p])) is not None}
    rho_mean = rho_sum / max(used, 1)

    # Per-replica curves are the *inputs* to the aggregate, so they sit one level
    # down; <system>_<conc>/rdf.csv at the top is what the figures read.
    out_dir = out_dir_for(out_root, system_id) / "replicas"
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [r_out] + [g_out[p] for p in pairs]
    header = ",".join(["r"] + [f"g_{p}" for p in pairs])
    np.savetxt(out_dir / f"rdf_r{rep}.csv", np.column_stack(cols),
               delimiter=",", header=header, comments="", fmt="%.6g")

    summary = {
        "system_id": system_id, "replica": rep,
        "cation": spec.cation, "anion": spec.anion, "solvent": spec.solvent,
        "concentration_M": conc_label(system_id), "temperature_K": "293K",
        "rdf": {
            "pairs": pairs, "r_max": r_max, "r_max_target": r_max_target, "dr": dr,
            "min_perp_width_window_A": round(float(min_width), 3),
            "peaks": peaks, "n_frames": used,
        },
        "sampling": {
            "dt_ps": DT_PS, "n_frames_requested": n_frames, "n_frames_used": used,
            "eq_frac_dropped": eq_frac, "eq_time_ps": round(start * DT_PS, 1),
            "window_ps": round((total - start) * DT_PS, 1),
            "start_frame": start, "stride": stride, "n_frames_total": int(total),
        },
        "health": {"mean_density_window_g_cm3": round(rho_mean, 4),
                   "final_density_g_cm3": round(rho_end, 4)},
        "provenance": {
            "traj_path": str(traj), "code_version": git_sha,
            # False for the 5 Delta reruns: they seed from a 100 ps UMA transport
            # frame rather than the cubic packing box the rest of the set shares.
            "shares_start_xyz_with_r0": shares,
        },
    }
    (out_dir / f"rdf_r{rep}_summary.json").write_text(json.dumps(summary, indent=2))

    return {"system_id": system_id, "rep": rep, "status": "ok", "tag": tag,
            "shares_start": shares, "frames_used": used, "r_max": r_max,
            "rho": round(rho_mean, 3),
            "peak_r": {k: round(v["r"], 2) for k, v in peaks.items()}}


def write_aggregate(out_root: Path, system_id: str, git_sha: str) -> dict | None:
    """Collapse ``replicas/rdf_r*.csv`` into the canonical ``rdf.csv``.

    Columns are ``r``, ``g_<pair>`` (mean across replicas) and ``sd_<pair>``
    (sample std, ddof=1). The mean deliberately keeps the plain ``g_<pair>``
    name the single-trajectory files used, so every existing reader picks up the
    replica mean with no code change; and the spread is ``sd_`` rather than
    ``g_<pair>_std`` because the figures detect pairs with
    ``[c[2:] for c in cols if c.startswith("g_")]`` -- a ``g_``-prefixed std
    column would be silently plotted as a phantom pair.

    Only sigma is stored: ``n_replicas`` is in the summary, so SEM = sd/sqrt(n)
    is recoverable, whereas sigma cannot be recovered from SEM alone.
    """
    d = out_dir_for(out_root, system_id)
    files = sorted((d / "replicas").glob("rdf_r*.csv"),
                   key=lambda f: int(f.stem.split("_r")[-1]))
    if not files:
        return None
    reps = [int(f.stem.split("_r")[-1]) for f in files]
    arrs = [np.genfromtxt(f, delimiter=",", names=True) for f in files]

    # Replicas can stop at different r_max (the min-image ceiling follows each
    # run's smallest box), so truncate to the shortest common grid.
    n = min(a.shape[0] for a in arrs)
    r = arrs[0]["r"][:n]
    # genfromtxt mangles "g_Na-O" -> "g_NaO"; recover the real pair names.
    with open(files[0]) as fh:
        raw_cols = fh.readline().strip().split(",")
    pairs = [c[2:] for c in raw_cols if c.startswith("g_")]

    cols, header = [r], ["r"]
    peaks, spreads = {}, {}
    for pair, key in zip(pairs, [c for c in arrs[0].dtype.names if c != "r"]):
        stack = np.vstack([a[key][:n] for a in arrs])
        mean = stack.mean(axis=0)
        sd = (stack.std(axis=0, ddof=1) if len(arrs) > 1
              else np.full_like(mean, np.nan))    # ddof=1 is undefined for n=1
        cols += [mean, sd]
        header += [f"g_{pair}", f"sd_{pair}"]
        if (pk := _first_peak(r, mean)) is not None:
            peaks[pair] = pk
        spreads[pair] = round(float(np.nanmean(sd)), 4) if len(arrs) > 1 else None

    np.savetxt(d / "rdf.csv", np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="", fmt="%.6g")

    # Roll the per-replica summaries up, so the aggregate is self-describing.
    per_rep = {}
    for rep in reps:
        p = d / "replicas" / f"rdf_r{rep}_summary.json"
        if p.exists():
            s = json.loads(p.read_text())
            per_rep[f"r{rep}"] = {
                "shares_start_xyz_with_r0": s["provenance"]["shares_start_xyz_with_r0"],
                "mean_density_window_g_cm3": s["health"]["mean_density_window_g_cm3"],
                "n_frames": s["rdf"]["n_frames"],
            }
    spec = SYSTEMS[system_id]
    mixed = sorted(k for k, v in per_rep.items()
                   if v["shares_start_xyz_with_r0"] is False)
    summary = {
        "system_id": system_id,
        "cation": spec.cation, "anion": spec.anion, "solvent": spec.solvent,
        "concentration_M": conc_label(system_id), "temperature_K": "293K",
        "rdf": {
            "pairs": pairs, "r_max": float(r[-1]), "dr": float(r[1] - r[0]),
            "peaks": peaks,               # of the replica mean
            "mean_sd": spreads,           # mean of sd_<pair> over r, a one-number scale
        },
        "replicas": {
            "n": len(reps), "list": [f"r{x}" for x in reps],
            "spread_convention": "sd_<pair> = sample std across replicas (ddof=1); "
                                 "SEM = sd/sqrt(n). NaN when n=1.",
            "different_start_xyz": mixed,
            "per_replica": per_rep,
        },
        "provenance": {"code_version": git_sha,
                       "built_from": str((d / "replicas").relative_to(out_root))},
    }
    (d / "rdf_summary.json").write_text(json.dumps(summary, indent=2))
    return {"system_id": system_id, "n": len(reps), "reps": reps,
            "mixed": mixed, "peaks": peaks}


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=None, help="Comma list of system_ids (default: all 9).")
    ap.add_argument("--replicas", default=None,
                    help="Comma list of replica indices (default: whichever exist "
                         "per system -- 0-4 for the Na systems, 0 only for lipf6_dme).")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--n-frames", type=int, default=3000,
                    help="Frames sampled evenly across the post-eq window (default 3000).")
    ap.add_argument("--eq-frac", type=float, default=0.30,
                    help="Fraction of frames dropped as equilibration (default 0.30).")
    ap.add_argument("--r-max-target", type=float, default=12.0)
    ap.add_argument("--dr", type=float, default=0.05)
    ap.add_argument("--rho-min", type=float, default=0.7,
                    help="Reject a replica whose final density is below this (g/cm^3).")
    ap.add_argument("--rho-max", type=float, default=2.0,
                    help="Reject a replica whose final density is above this (g/cm^3).")
    ap.add_argument("--jobs", type=int, default=16, help="Parallel replicas (default 16).")
    ap.add_argument("--out-suffix", default="",
                    help="Appended to the output directory name (e.g. _test), so a rerun "
                         "does not overwrite the results the figures read.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the trajectories that would be processed and exit.")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    systems = ([s.strip() for s in args.systems.split(",")] if args.systems else list(ALL_SYSTEMS))
    unknown = [s for s in systems if s not in RUN_SPEC]
    if unknown:
        raise SystemExit(f"Unknown system_ids: {unknown} (known: {ALL_SYSTEMS})")
    out_root = Path(args.out_root + args.out_suffix)
    git_sha = paths.git_sha()

    # Replica sets are per-system: lipf6_dme has only r0. An explicit --replicas
    # intersects with what exists rather than requesting missing runs.
    override = [int(r) for r in args.replicas.split(",")] if args.replicas else None
    tasks = [(s, r) for s in systems for r in replicas_for(s)
             if override is None or r in override]
    print(f"Computing per-replica RDFs: {len(systems)} systems, {len(tasks)} runs "
          f"-> {out_root}")
    print(f"  n_frames={args.n_frames}  eq_frac={args.eq_frac}  "
          f"r_max_target={args.r_max_target} Å  dr={args.dr} Å")
    print(f"  health gate: {args.rho_min} < final density < {args.rho_max} g/cm^3")
    if args.dry_run:
        for s, r in tasks:
            t = traj_path(s, r)
            print(f"  {s:16s} r{r}  {'ok     ' if t.is_file() else 'MISSING'} {t}")
        return

    ok, blown, missing = [], [], []
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as ex:
        futs = {ex.submit(compute_one, s, r, args.n_frames, args.eq_frac,
                          args.r_max_target, args.dr, args.rho_min, args.rho_max,
                          out_root, git_sha): (s, r)
                for s, r in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="replicas"):
            s, r = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                tqdm.write(f"  [FAIL] {s} r{r}: {e}")
                continue
            if res["status"] == "ok":
                ok.append(res)
                src = "" if res["shares_start"] is not False else "  [different start.xyz]"
                tqdm.write(f"  [ok]      {s:16s} r{r}  rho={res['rho']:.3f}  "
                           f"r_max={res['r_max']:.2f} Å  peak_r={res['peak_r']}{src}")
            elif res["status"] == "blown_up":
                blown.append(res)
                tqdm.write(f"  [SKIP]    {s:16s} r{r}  BLOWN UP ({res['detail']})")
            else:
                missing.append(res)
                tqdm.write(f"  [missing] {s:16s} r{r}  {res['detail']}")

    print(f"\n{len(ok)} replica RDFs written under {out_root}")
    if blown:
        print(f"{len(blown)} skipped as blown up: "
              + ", ".join(f"{b['system_id']}/r{b['rep']}" for b in sorted(
                  blown, key=lambda d: (d['system_id'], d['rep']))))
    if missing:
        print(f"{len(missing)} missing: "
              + ", ".join(f"{m['system_id']}/r{m['rep']}" for m in sorted(
                  missing, key=lambda d: (d['system_id'], d['rep']))))

    counts, reseeded = {}, {}
    for res in ok:
        counts.setdefault(res["system_id"], []).append(res["rep"])
        if res["shares_start"] is False:
            reseeded.setdefault(res["system_id"], []).append(res["rep"])
    print("\nusable replicas per system:")
    for s in systems:
        got = sorted(counts.get(s, []))
        # Flag the reruns: they do not share start.xyz with r0.
        fd = sorted(reseeded.get(s, []))
        note = f"   ({', '.join('r%d' % r for r in fd)}: different start.xyz)" if fd else ""
        print(f"  {s:16s} n={len(got)}  {['r%d' % r for r in got]}{note}")

    # Collapse each system's replicas into the canonical rdf.csv the figures read.
    print("\naggregates (rdf.csv = replica mean, sd_<pair> = std across replicas):")
    for s in systems:
        agg = write_aggregate(out_root, s, git_sha)
        if agg is None:
            print(f"  {s:16s} -- no replica CSVs, skipped")
            continue
        pk = "  ".join(f"{k}: r={v['r']:.2f} g={v['g']:.2f}" for k, v in agg["peaks"].items())
        note = "  [n=1, sd=NaN]" if agg["n"] == 1 else (
            f"  [†{len(agg['mixed'])}/{agg['n']} different start.xyz]" if agg["mixed"] else "")
        print(f"  {s:16s} n={agg['n']}  {pk}{note}")
    print(f"\n-> {out_root}/<system>_<conc>/rdf.csv  (+ replicas/rdf_r<N>.csv)")


if __name__ == "__main__":
    main()
