#!/usr/bin/env python
"""Shared discovery + GROMACS I/O for the OPLS ``production_replicas`` transport set.

The OPLS reference runs are 16 boxes x 4 replicas of 20 ns GROMACS NVT
(v-rescale/Bussi, tau_t = 10 ps), living at::

    <root>/<salt>_<solvent>_<conc>M_<temp>/r<k>/
        prod.xtc                       # concatenated, 20001 frames @ 1 ps
        prod.tpr, seed.gro             # topology
        <box>_r<k>.stress              # GK stress log, 2 fs, identical format
                                       # to the FeNNol ``.stress`` logs
        rep_meta.json, prod_state.json

This module exists so ``diffusivity_opls_replicas.py`` and
``viscosity_gk_opls_replicas.py`` can run the *same* estimators the PaiNN/FeNNol
transport scripts use, over GROMACS inputs. It provides:

  * :func:`discover_runs`      -- parse the tree into per-replica records
  * :func:`xtc_frames`         -- an ``io.iter_frames``-compatible frame source
                                  backed by mdtraj (ASE ``Atoms`` per frame)
  * :func:`groups_from_topology` -- cation/anion/solvent molecule grouping taken
                                  from the GROMACS topology instead of this
                                  project's element-sequence patterns
  * :func:`load_exp_table`     -- the 293/298 K experimental reference tables

Why grouping comes from the topology (the one unavoidable deviation)
--------------------------------------------------------------------
``observables.grouping.build_groups`` finds molecules by matching an exact
*element sequence* (``solvent_dict["DME"] = [H, C, O, H, H, C, C, ...]``), which
encodes the atom ordering the FeNNol boxes were built with. LigParGen/GROMACS
writes DME as ``C1 O1 C2 C3 O2 C4 H1..H10`` -- the same molecule, a different
atom order -- so the pattern matcher cannot be used as-is. Here each GROMACS
residue is taken as one molecule and its element *multiset* is validated against
the same ``cation_dict``/``anion_dict``/``solvent_dict`` entries, so the species
are provably identical; only the within-molecule order differs, and every
downstream quantity (mass-weighted molecular COM, MSD, D) is invariant to it.

Requires ``mdtraj`` (present in the ``fennol`` env); MDAnalysis is not needed.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms


from .. import paths
from ..grouping import Groups
from ..systems import (
    SYSTEMS, anion_dict, cation_dict, solvent_dict,
)

# ── paths / naming ───────────────────────────────────────────────────────────
DEFAULT_ROOT = paths.OPLS_20ns_298K
DEFAULT_OUT_ROOT = paths.OUT_OPLS_TRANSPORT
GROUND_TRUTH = paths.GROUND_TRUTH

# <salt>_<solvent>_<conc>M_<temp>, e.g. lipf6_dme_0_5M_273 / napf6_tgdme_1M_298
BOX_RE = re.compile(r"^(?P<salt>[a-z0-9]+)_(?P<solvent>[a-z]+)_"
                    r"(?P<conc>\d+(?:_\d+)?)M_(?P<temp>\d+)$")
REPLICA_RE = re.compile(r"^r(?P<rep>\d+)$")

# tokens this project registers (system_id = "<salt>_<solvent>")
_SALT = {"napf6": "NaPF6", "naotf": "NaOTf", "lipf6": "LiPF6"}
_SOLVENT = {"dme": "DME", "pc": "PC", "diglyme": "DEGDME", "tgdme": "TEGDME"}
_SAMPLE_RE = re.compile(r"^([\d.]+)\s*M\s+(\S+)\s+in\s+(\S+)$")

# experimental table per box temperature (only these two are curated)
EXP_CSV_BY_T = {293: GROUND_TRUTH / "293p2K_exp_values.csv",
                298: GROUND_TRUTH / "298p2K_exp_values.csv"}


def conc_to_molar(tok: str) -> float:
    """'0_1'->0.1, '0_5'->0.5, '1'->1.0 (the token *without* the trailing M)."""
    return float(tok.replace("_", "."))


def conc_label(tok: str) -> str:
    """Normalise a box concentration token to the PaiNN label form ('0_1M')."""
    return f"{tok}M"


def _f(x):
    if x is None:
        return None
    x = str(x).strip()
    return None if x in ("", "---") else float(x)


# ── experimental reference ───────────────────────────────────────────────────
def load_exp_table(temp_k: int) -> dict:
    """(salt_token, solvent_token, conc_M) -> dict(eta [mPa*s], D_exp_* [1e-10 m^2/s])
    for the curated table at ``temp_k``. Empty dict if no table exists for that
    temperature (the 273 / 323 K LiPF6 boxes have no measurements)."""
    csv_path = EXP_CSV_BY_T.get(int(round(temp_k)))
    if csv_path is None or not csv_path.is_file():
        return {}
    inv_salt = {v: k for k, v in _SALT.items()}
    inv_solv = {v: k for k, v in _SOLVENT.items()}
    out = {}
    for _, r in pd.read_csv(csv_path).iterrows():
        m = _SAMPLE_RE.match(str(r["Sample"]).strip())
        if not m:
            continue
        conc, salt, solv = float(m.group(1)), inv_salt.get(m.group(2)), inv_solv.get(m.group(3))
        if salt and solv:
            out[(salt, solv, conc)] = dict(
                eta=_f(r.get("Dynamic Viscosity (mPa*s)")),
                D_exp_cation=_f(r.get("D*_+ (x 10^-10 m^2/s)")),
                D_exp_anion=_f(r.get("D*_- (x 10^-10 m^2/s)")),
                D_exp_solvent=_f(r.get("D*_0 (x 10^-10 m^2/s)")),
            )
    return out


_EXP_CACHE: dict[int, dict] = {}


def exp_ref(salt: str, solvent: str, conc_m: float, temp_k: float) -> dict:
    """Experimental row for one box, or an all-None dict when unavailable."""
    t = int(round(temp_k))
    if t not in _EXP_CACHE:
        _EXP_CACHE[t] = load_exp_table(t)
    return _EXP_CACHE[t].get((salt, solvent, conc_m), {})


# ── discovery ────────────────────────────────────────────────────────────────
def discover_runs(root: Path, need: str = "traj", replicas: list[int] | None = None,
                  only: str | None = None) -> tuple[list, list]:
    """Walk ``root`` and return ``(runs, skipped)`` records, one per replica.

    ``need`` selects the required artefact: ``"traj"`` (``prod.xtc`` + a topology,
    for diffusivity) or ``"stress"`` (the ``*.stress`` GK log, for viscosity).
    ``replicas`` restricts to those replica indices; ``only`` is a substring
    filter on the box name.
    """
    runs, skipped = [], []
    paths.require(root)   # readable error if the tree is on another machine
    for box_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        m = BOX_RE.match(box_dir.name)
        if not m:
            continue
        salt, solv = m.group("salt"), m.group("solvent")
        system_id = f"{salt}_{solv}"
        if system_id not in SYSTEMS:
            continue
        if only is not None and only not in box_dir.name:
            continue
        conc_tok, temp_k = m.group("conc"), float(m.group("temp"))
        conc_m = conc_to_molar(conc_tok)
        ref = exp_ref(salt, solv, conc_m, temp_k)

        for rep_dir in sorted(box_dir.glob("r*")):
            rm = REPLICA_RE.match(rep_dir.name)
            if not rm:
                continue
            rep = int(rm.group("rep"))
            if replicas is not None and rep not in replicas:
                continue
            rec = dict(
                system_id=system_id, salt=salt, solvent=solv,
                box=box_dir.name, conc_label=conc_label(conc_tok), conc_M=conc_m,
                temperature_K=temp_k, replica=rep,
                run_name=f"{box_dir.name}_r{rep}", run_dir=str(rep_dir),
                eta_exp=ref.get("eta"),
                D_exp_cation=ref.get("D_exp_cation"),
                D_exp_anion=ref.get("D_exp_anion"),
                D_exp_solvent=ref.get("D_exp_solvent"),
            )
            if need == "stress":
                if not list(rep_dir.glob("*.stress")):
                    rec["skip_reason"] = "no .stress log"
                    skipped.append(rec)
                    continue
            else:
                parts = resolve_traj_parts(rep_dir)
                if not parts:
                    rec["skip_reason"] = "no prod.xtc / prod.part*.xtc"
                    skipped.append(rec)
                    continue
                try:
                    rec["topology"] = str(resolve_topology(rep_dir))
                except FileNotFoundError as e:
                    rec["skip_reason"] = str(e)
                    skipped.append(rec)
                    continue
                # a single concatenated ``prod.xtc`` collapses to a 1-element list, so
                # every consumer (xtc_dt_ps / xtc_n_frames / make_xtc_frame_source) sees
                # the same list interface for both the old and segmented layouts.
                rec["traj"] = [str(p) for p in parts]
                rec["n_parts"] = len(parts)
            runs.append(rec)
    return runs, skipped


def resolve_traj_parts(run_dir) -> list[Path]:
    """The production trajectory as an ordered list of ``.xtc`` files.

    Two on-disk layouts exist and both are supported transparently:
      * a single concatenated ``prod.xtc`` (the original ``production_replicas`` set)
        -> ``[prod.xtc]``;
      * ten 2 ns segments ``prod.part0001.xtc`` .. ``prod.part0010.xtc`` written by
        GROMACS ``-cpi`` restarts (the current OPLS set) -> the sorted list.
        Each segment after the first repeats, as its frame 0, the last frame of the
        previous segment; that duplicate is dropped downstream (see
        :func:`make_xtc_frame_source` / :func:`xtc_n_frames`).
    Returns ``[]`` if neither is present."""
    run_dir = Path(run_dir)
    single = run_dir / "prod.xtc"
    if single.is_file():
        return [single]
    return sorted(run_dir.glob("prod.part*.xtc"))


def _as_parts(path) -> list[str]:
    """Normalise a trajectory reference (a single path or a list of segment paths)
    to a list of ``str`` paths."""
    if isinstance(path, (list, tuple)):
        return [str(p) for p in path]
    return [str(path)]


def resolve_topology(run_dir) -> Path:
    """Topology for mdtraj: the seed ``.gro`` (or the first segment's ``.gro``).

    ``prod.tpr`` is *not* used -- mdtraj cannot parse a ``.tpr``, and the ``.gro``
    carries everything the analysis needs (residue blocks + elements)."""
    run_dir = Path(run_dir)
    for cand in [run_dir / "seed.gro", *sorted(run_dir.glob("prod.part*.gro"))]:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no .gro topology in {run_dir}")


def _parse_itp(itp_path):
    """(moleculetype_name, charges) from a GROMACS .itp [atoms] block."""
    name, charges, section = None, [], None
    for raw in Path(itp_path).read_text().splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.lower().replace(" ", "")
            continue
        if section == "[moleculetype]" and name is None:
            name = line.split()[0]
        elif section == "[atoms]":
            parts = line.split()
            if len(parts) > 6:
                charges.append(float(parts[6]))
    return name, charges


def charges_from_itp(run_dir, spec) -> tuple[float, float]:
    """Formal (z_cation, z_anion) from the summed partial charges in the run's
    ``.itp`` files, rounded to integers — exactly what ``code_utils``'
    ``build_species_dicts`` did. Falls back to (+1, -1) if the files are absent
    or ambiguous. Raises if a parsed charge is not within 0.05 e of an integer."""
    run_dir = Path(run_dir)
    by_name = {}
    for itp in sorted(run_dir.glob("*.itp")):
        if "forcefield" in itp.name.lower():
            continue
        name, charges = _parse_itp(itp)
        if name and charges:
            by_name[name.upper()] = sum(charges)
    # residue names as spelled in the .gro / .top (NA, PF6, OTF, DME, ...)
    cat_names = {spec.cation.upper(), "NA", "LI"} & set(by_name)
    an_names = {spec.anion.upper()} & set(by_name)
    z_cat, z_an = 1.0, -1.0
    for names, default, label in ((cat_names, 1.0, "cation"), (an_names, -1.0, "anion")):
        if not names:
            continue
        q = by_name[sorted(names)[0]]
        if abs(q - round(q)) > 0.05:
            raise RuntimeError(f"{label} charge {q:.3f} in {run_dir} is not near-integral")
        if label == "cation":
            z_cat = float(round(q))
        else:
            z_an = float(round(q))
    return z_cat, z_an


def replica_meta(run_dir) -> dict:
    p = Path(run_dir) / "rep_meta.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ── GROMACS trajectory access ────────────────────────────────────────────────
def _mdtraj():
    try:
        import mdtraj  # noqa: F401
    except ImportError as e:                                     # pragma: no cover
        raise RuntimeError(
            "mdtraj is required to read the OPLS .xtc trajectories. It ships in the "
            "`fennol` conda env: /pscratch/sd/i/ishan_a/conda/envs/fennol/bin/python"
        ) from e
    import mdtraj
    return mdtraj


def load_topology(topology_path):
    """mdtraj Topology for a run (cheap -- a .gro parse)."""
    md = _mdtraj()
    return md.load(str(topology_path)).topology


def xtc_dt_ps(traj_path) -> float:
    """Frame spacing in ps, read from the first segment's own time stamps.

    Accepts a single path or the segment list from :func:`resolve_traj_parts`."""
    md = _mdtraj()
    first = _as_parts(traj_path)[0]
    with md.formats.XTCTrajectoryFile(str(first)) as f:
        _, time, _, _ = f.read(2)
    if len(time) < 2:
        raise RuntimeError(f"{first}: fewer than 2 frames, cannot infer dt")
    return float(time[1] - time[0])


def _part_time_info(part) -> tuple[int, float, float, float | None]:
    """``(n_frames, t_first, t_last, dt)`` for one segment, read cheaply from the
    xtc offset index plus the first two and last time stamps (no full decode)."""
    md = _mdtraj()
    with md.formats.XTCTrajectoryFile(str(part)) as f:
        n = int(len(f))
        if n == 0:
            return 0, float("nan"), float("nan"), None
        f.seek(0)
        _, t_head, _, _ = f.read(min(2, n))
        f.seek(n - 1)
        _, t_tail, _, _ = f.read(1)
    dt = float(t_head[1] - t_head[0]) if len(t_head) > 1 else None
    return n, float(t_head[0]), float(t_tail[-1]), dt


def xtc_n_frames(traj_path) -> int:
    """De-duplicated logical frame count across one or more ``.xtc`` segments.

    Each restart segment repeats the previous segment's last frame as its own
    frame 0; that overlap is subtracted so the count matches exactly what
    :func:`make_xtc_frame_source` yields (a single concatenated ``prod.xtc`` gives
    its plain length)."""
    parts = _as_parts(traj_path)
    total, prev_last, dt = 0, None, None
    for part in parts:
        n, t0, t_last, pdt = _part_time_info(part)
        if pdt is not None:
            dt = pdt
        dup = (prev_last is not None and dt is not None and abs(t0 - prev_last) < 0.5 * dt)
        total += n - (1 if dup else 0)
        prev_last = t_last
    return total


def make_xtc_frame_source(topology_path, chunk: int = 500):
    """Return an ``io.iter_frames``-compatible callable for GROMACS ``.xtc``.

    The returned ``frames(path, start=, stride=, max_count=)`` yields ASE
    ``Atoms`` (numbers + positions in A + cell in A + pbc), which is exactly the
    interface ``observables.diffusivity._unwrap_com_frame`` consumes, so the OPLS
    trajectories flow through the identical unwrap / COM-frame / MSD code as the
    PaiNN ``.traj`` runs.

    ``path`` may be a single ``.xtc`` or the ordered segment list from
    :func:`resolve_traj_parts`. Segments are streamed back-to-back as one logical
    trajectory: the frame each restart repeats at a segment boundary is detected by
    its time stamp and dropped, and ``start`` / ``stride`` / ``max_count`` index into
    that de-duplicated stream (identically to a single concatenated ``prod.xtc``).
    """
    md = _mdtraj()
    top = load_topology(topology_path)
    numbers = np.array([a.element.number for a in top.atoms], dtype=int)

    def frames(path, start: int = 0, stride: int = 1, max_count: int | None = None):
        parts = _as_parts(path)
        n_yielded = 0          # frames emitted (honours stride + max_count)
        g = 0                  # index into the de-duplicated logical stream
        last_t = None          # time stamp of the last logical frame kept
        dt = None              # inferred frame spacing (for the overlap test)
        for part in parts:
            for block in md.iterload(str(part), top=top, chunk=chunk):
                xyz = np.asarray(block.xyz) * 10.0                  # nm -> A
                cells = np.asarray(block.unitcell_vectors) * 10.0   # nm -> A
                times = np.asarray(block.time)
                for i in range(block.n_frames):
                    t = float(times[i])
                    if last_t is not None and dt is None and t > last_t:
                        dt = t - last_t
                    # drop a segment's repeated boundary frame (t <= previous frame)
                    if last_t is not None and dt is not None and t <= last_t + 0.5 * dt:
                        continue
                    last_t = t
                    if g >= start and (g - start) % stride == 0:
                        yield Atoms(numbers=numbers, positions=xyz[i],
                                    cell=cells[i], pbc=True)
                        n_yielded += 1
                        if max_count is not None and n_yielded >= max_count:
                            return
                    g += 1

    return frames


# ── molecule grouping from the GROMACS topology ──────────────────────────────
def groups_from_topology(topology_path, spec) -> Groups:
    """Cation / anion / solvent molecule groups taken from the GROMACS residues.

    Every residue is validated against this project's species dictionaries by
    element multiset, so the molecules are provably the same species the PaiNN
    grouping matches; only the intra-molecule atom order differs (see module
    docstring). Raises if any residue is unclassifiable or if the species'
    molecules are not all identical, which the vectorised COM kernel assumes.
    """
    top = load_topology(topology_path)
    want = {
        "cation": Counter(cation_dict[spec.cation]),
        "anion": Counter(anion_dict[spec.anion]),
        "solvent": Counter(solvent_dict[spec.solvent]),
    }
    found = {"cation": [], "anion": [], "solvent": []}
    for res in top.residues:
        idx = np.array([a.index for a in res.atoms], dtype=int)
        comp = Counter(a.element.symbol for a in res.atoms)
        for key, target in want.items():
            if comp == target:
                found[key].append(idx)
                break
        else:
            raise RuntimeError(
                f"residue {res.name} ({sorted(comp.items())}) in {topology_path} matches "
                f"no species of system '{spec.system_id}' "
                f"(cation={spec.cation}, anion={spec.anion}, solvent={spec.solvent})"
            )

    n_grouped = sum(len(g) for gs in found.values() for g in gs)
    n_total = top.n_atoms
    if n_grouped != n_total:
        raise RuntimeError(f"grouping mismatch: {n_grouped} of {n_total} atoms grouped")
    for key, gs in found.items():
        if not gs:
            raise RuntimeError(f"no {key} molecules found in {topology_path}")
        sizes = {len(g) for g in gs}
        if len(sizes) != 1:
            raise RuntimeError(f"{key} molecules have differing sizes {sizes}")

    # element order must be identical across molecules of a species: the COM
    # kernel takes the per-atom mass vector from the first molecule only.
    syms = [a.element.symbol for a in top.atoms]
    for key, gs in found.items():
        ref = [syms[i] for i in gs[0]]
        for g in gs[1:]:
            if [syms[i] for i in g] != ref:
                raise RuntimeError(f"{key} molecules do not share one atom ordering")

    return Groups(cat_idx=[int(g[0]) for g in found["cation"]],
                  anion_groups=found["anion"], solvent_groups=found["solvent"])


# ── misc ─────────────────────────────────────────────────────────────────────
def decimate(n: int, target: int = 2000) -> np.ndarray:
    """Indices thinning ``n`` points to ~``target``, keeping the endpoints."""
    if n <= target:
        return np.arange(n)
    return np.unique(np.r_[np.arange(0, n, n // target), n - 1])


def per_run_dir(out_dir: Path, rec: dict) -> Path:
    """``<out_dir>/per_run/<box>/r<k>/`` -- the OPLS tree itself is read-only
    (owned by another user), so per-run artefacts are mirrored here."""
    d = Path(out_dir) / "per_run" / rec["box"] / f"r{rec['replica']}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# plot encoding, matched to the PaiNN transport scripts
SOLV_COLOR = {"dme": "#E68310", "diglyme": "#11A579", "pc": "#E73F74", "tgdme": "#3969AC"}
SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}
SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}
SALT_MARKER = {"napf6": "o", "naotf": "s", "lipf6": "^"}


def box_label(rec: dict) -> str:
    return (f"{SALT_LABEL.get(rec['salt'], rec['salt'])} {SOLV_LABEL[rec['solvent']]} "
            f"{rec['conc_label'].replace('_', '.')} {int(rec['temperature_K'])}K")
