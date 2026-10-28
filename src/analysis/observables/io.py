"""Trajectory resolution, dt detection, and lazy frame streaming.

Handles both formats the project produces:
  * ``{run}.traj``         — ASE binary (random access), the going-forward default
  * ``{run}.traj.extxyz``  — text extxyz (older pilots), and ``{run}_every<N>ps.extxyz`` subsamples

Per-frame time is not reliably stored in binary ``.traj``; the extxyz comment
carries ``Time_ps``. dt resolution order: explicit override -> Time_ps delta of
the first two frames -> ``tdump[ps]`` from the sibling ``.fennol.yaml`` -> fallback.

The extxyz reader parses only species + positions + lattice (skipping the
forces/velocities columns) and never loads the whole file into RAM.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io.trajectory import Trajectory

_LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
_TIME_RE = re.compile(r"Time_ps=([-\d.eE+]+)")


# ── Trajectory resolution ───────────────────────────────────────────────────
def resolve_traj(run_dir) -> Path:
    """Return the run's own trajectory file.

    Keyed strictly off the run-folder basename so we never pick up the
    ``<system>.traj`` symlink-to-UMA-seed that lives in pilot folders.
    """
    run_dir = Path(run_dir)
    base = run_dir.name
    # Preference: random-access binary .traj (new runs) > analysis-ready subsample
    # (legacy pilots; small + carries Time_ps) > full text trajectory.
    if (run_dir / f"{base}.traj").is_file():
        return run_dir / f"{base}.traj"
    subsamples = sorted(run_dir.glob(f"{base}_every*ps.extxyz"))
    if subsamples:
        return subsamples[0]
    if (run_dir / f"{base}.traj.extxyz").is_file():
        return run_dir / f"{base}.traj.extxyz"
    raise FileNotFoundError(
        f"No trajectory found for run {base!r} in {run_dir} "
        f"(looked for {base}.traj, {base}_every*ps.extxyz, {base}.traj.extxyz)"
    )


def load_config(run_dir) -> dict | None:
    """Parse the sibling ``{run}.fennol.yaml`` if present."""
    run_dir = Path(run_dir)
    cfg = run_dir / f"{run_dir.name}.fennol.yaml"
    if not cfg.is_file():
        hits = sorted(run_dir.glob("*.fennol.yaml"))
        if not hits:
            return None
        cfg = hits[0]
    with open(cfg) as f:
        return yaml.safe_load(f)


def is_extxyz(path) -> bool:
    s = str(path)
    return s.endswith(".extxyz") or s.endswith(".xyz")


# ── Frame streaming ─────────────────────────────────────────────────────────
def _parse_extxyz_comment(comment: str):
    m = _LATTICE_RE.search(comment)
    cell = np.array(list(map(float, m.group(1).split()))).reshape(3, 3) if m else None
    tm = _TIME_RE.search(comment)
    time_ps = float(tm.group(1)) if tm else None
    return cell, time_ps


def _iter_extxyz(path, start, stride, max_count):
    """Lazily yield Atoms (species+positions+cell only) from an extxyz file."""
    n_yielded = 0
    frame_idx = 0
    with open(path) as f:
        while True:
            header = f.readline()
            if not header:
                break
            n_at = int(header.strip())
            comment = f.readline()
            selected = (frame_idx >= start and (frame_idx - start) % stride == 0)
            if not selected:
                for _ in range(n_at):
                    f.readline()
                frame_idx += 1
                continue
            cell, time_ps = _parse_extxyz_comment(comment)
            symbols = []
            positions = np.empty((n_at, 3))
            for j in range(n_at):
                parts = f.readline().split()
                symbols.append(parts[0])
                positions[j] = (float(parts[1]), float(parts[2]), float(parts[3]))
            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
            if time_ps is not None:
                atoms.info["Time_ps"] = time_ps
            yield atoms
            n_yielded += 1
            frame_idx += 1
            if max_count is not None and n_yielded >= max_count:
                return


def _iter_traj(path, start, stride, max_count):
    """Lazily yield Atoms from an ASE binary trajectory (random access)."""
    traj = Trajectory(str(path), mode="r")
    try:
        n = len(traj)
        n_yielded = 0
        for i in range(start, n, stride):
            yield traj[i]
            n_yielded += 1
            if max_count is not None and n_yielded >= max_count:
                return
    finally:
        traj.close()


def iter_frames(path, start: int = 0, stride: int = 1, max_count: int | None = None):
    """Unified lazy frame iterator for either trajectory format."""
    if is_extxyz(path):
        yield from _iter_extxyz(path, start, stride, max_count)
    else:
        yield from _iter_traj(path, start, stride, max_count)


def estimate_n_frames(path) -> int:
    """Number of frames (exact for .traj; estimated from file size for extxyz)."""
    path = Path(path)
    if not is_extxyz(path):
        traj = Trajectory(str(path), mode="r")
        try:
            return len(traj)
        finally:
            traj.close()
    # extxyz: bytes of the first frame -> estimate
    with open(path) as f:
        n_at = int(f.readline().strip())
        f.readline()  # comment
        for _ in range(n_at):
            f.readline()
        bytes_per_frame = f.tell()
    if bytes_per_frame == 0:
        return 0
    return max(1, os.path.getsize(path) // bytes_per_frame)


def resolve_dt_ps(run_dir, traj_path, override: float | None = None,
                  fallback: float = 0.1):
    """Resolve picoseconds-per-frame. Returns (dt_ps, source_str)."""
    if override is not None:
        return float(override), "override"
    # Try Time_ps delta from the first two frames
    times = [a.info.get("Time_ps") for a in iter_frames(traj_path, max_count=2)]
    if len(times) == 2 and times[0] is not None and times[1] is not None:
        dt = float(times[1]) - float(times[0])
        if dt > 0:
            return dt, "Time_ps delta"
    cfg = load_config(run_dir)
    if cfg and cfg.get("tdump[ps]") is not None:
        return float(cfg["tdump[ps]"]), "config tdump[ps]"
    return float(fallback), "fallback"
