#!/usr/bin/env python
"""Single source of truth for **where every dataset lives**.

Every analysis script imports its roots from here instead of hardcoding an
absolute path, so this file alone answers "where is my data?".

Run it to print the manifest with a reachability check for the current machine::

    python -m src.analysis.observables.paths

Inputs are pinned to the machine that physically holds them — they are *not*
relative to the current host. `/work/nvme` (Delta) is unreachable from
Perlmutter and vice-versa, so a script whose input lives on the other machine
simply cannot run there; use `require()` to say so with a readable message
instead of a bare ``FileNotFoundError``.

| constant                       | machine    | used by                          |
|--------------------------------|------------|----------------------------------|
| OPLS_20ns_298K                 | perlmutter | {viscosity_gk,diffusivity,conductivity}_opls_replicas |
| OPLS_20ns_298K_OLD             | perlmutter | superseded pre-2026-10 OPLS tree, comparisons only    |
| PAINN_NVT_REPLICAS             | delta      | {viscosity_gk,diffusivity,conductivity}_replicas      |
| PAINN_NPT_REPLICAS_FOR_DENSITIES | perlmutter | density_replicas                                  |
| PAINN_293K_1ns                 | perlmutter | compute_painn_replica_rdfs                            |
| UMA_293K_1ns                   | perlmutter | compute_uma_1M_rdfs                                   |
| UMA_298K_1ns                   | delta      | uma_densities                                         |
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

# ── machine anchors ──────────────────────────────────────────────────────────
PERLMUTTER_SCRATCH = Path("/pscratch/sd/i/ishan_a/electrolytes_data")
DELTA_SCRATCH = Path("/work/nvme/beye/iamin/electrolytes_data")
CFS = Path("/global/cfs/cdirs")

# The MLP transport baseline was produced on Perlmutter; this is the group-readable
# copy in a shared project directory. Override with ELECTROLYTES_ANALYSIS_ROOT to run it elsewhere.
PERLMUTTER_ANALYSIS = Path(os.environ.get(
    "ELECTROLYTES_ANALYSIS_ROOT",
    "/global/cfs/cdirs/m5250/distillation_proj/general_analysis",
))

REPO_ROOT = Path(__file__).resolve().parents[3]      # …/electrolytes
DATA_DIR = REPO_ROOT / "data"

# Committed stand-in for the observable outputs. ``data/`` is a symlink to
# scratch and holds the full trees (hundreds of MB of per_run detail); only the
# summary CSVs and the RDF curves are small enough to track, and they live here.
# Everything downstream reads the roots below, which prefer live scratch and
# fall back to this copy when ``data/`` is absent — as on a fresh clone, where
# the figures then still draw. Refresh it with
# ``python -m src.analysis.observables.export_results``.
RESULTS = REPO_ROOT / "results"
OBS_ROOT = DATA_DIR if DATA_DIR.exists() else RESULTS

# The released inputs (trained students, simulation boxes), committed; written by
# ``python -m src.export_paper_data`` from DATA_DIR.
PAPER_DATA = REPO_ROOT / "paper_data"
GROUND_TRUTH = REPO_ROOT / "src" / "analysis" / "observables" / "ground_truth"


def current_machine() -> str:
    """``'perlmutter'`` | ``'delta'`` | ``'unknown'``."""
    host = os.environ.get("NERSC_HOST")
    if host:
        return host
    if DELTA_SCRATCH.parent.is_dir():
        return "delta"
    return "unknown"


def git_sha() -> str:
    """Short SHA of the repo HEAD, or ``'unknown'`` outside a checkout. Every
    driver records this in its ``*_config.yaml`` for provenance."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


# ── inputs: simulation trajectories ──────────────────────────────────────────

# OPLS/GROMACS reference set, 16 boxes x 4 replicas, 20 ns NVT @ 1 ps, 298 K.
# The 2026-10 rerun with the corrected and fully cited force field (CL&P PF6-/OTf-,
# Aqvist Na+/Li+, LigParGen solvents; see src/baselines/OPLS/PARAMETERS.md), with the
# boxes repacked at this force field's own equilibrium density. OPLS_CLP_ROOT also
# holds what produced the runs: ``pass1``/``pass2`` (density run and repacked-box
# equilibration), ``production_ready`` (seed frames + molarity table), ``provenance``
# and ``code`` (the exact source the jobs ran), and ``analysis``.
# The trajectory is stored as ten 2 ns segments ``prod.part0001.xtc``..``prod.part0010.xtc``
# rather than a single concatenated ``prod.xtc``; ``opls_runs`` reads either layout
# (de-duplicating the frame each restart repeats at the segment boundary). All
# artefacts are mirrored into OUT_OPLS_TRANSPORT.
OPLS_CLP_ROOT = CFS / "m5024/electrolytes_data/OPLS/updated_clp_param_runs"
OPLS_20ns_298K = OPLS_CLP_ROOT / "production_replicas"

# Superseded: the earlier ``correct_parameter`` tree (18 boxes, LigParGen triflate, and
# a PF6- topology with the exclusion bug described in
# src/baselines/OPLS/PF6_TOPOLOGY_BUG.md). Kept only for old-vs-new comparisons; no
# driver reads it by default.
OPLS_20ns_298K_OLD = (
    CFS / "m5250/distillation_proj/simulation_results/OPLS"
        / "corrected_molarity/correct_parameter"
)

# PaiNN NVT transport, 4-replica tree (the ``boxes_correct_molar`` set). Layout
# ``nvt_replica_<N>/<system>/nvt_transport_<conc>_298K_20ns_1ps/`` (N = 1..4). The
# transport drivers auto-detect this layout (see ``iter_run_dirs``). The head-to-head
# partner of OPLS_20ns_298K: viscosity, diffusivity and conductivity all read it.
#   * ``nvt_replica_4`` is bit-identical to ``exp04_correct_molarity_samebox`` r0
#     (last frames verified equal to 0 A for every box); replicas 1-3 are new
#     independent runs.  15 boxes x 4 replicas (lipf6_dme absent; napf6_tgdme 0.1 M
#     absent).
# Read-only (another user's tree) so all artefacts go to OUT_PAINN_TRANSPORT_REPLICAS.
PAINN_NVT_REPLICAS = Path(
    "/work/nvme/bfoy/yjian1/electrolytes_data/simulation/boxes_correct_molar/painn_nvt"
)

# PaiNN NPT replicas (mixed-noise, all systems) — the density observable set.
PAINN_NPT_REPLICAS_FOR_DENSITIES = (
    PERLMUTTER_SCRATCH / "experiments" / "exp01_mixed_noise_all_systems"
)

# PaiNN 293 K / 1 ns NPT runs used for the RDF comparison against UMA.
# The whole 5-replica set now lives here: r0 under ``<system>/<run>/`` and r1-r4
# under ``<system>/<run>_r<N>/``. (r1-r4 were staged on Perlmutter scratch and 5
# of them rerun on Delta; both were consolidated into this one tree — see its
# README, which also documents that those 5 carry a different ``start.xyz``.)
PAINN_293K_1ns = (
    CFS / "m5250/distillation_proj/simulation_results/PAINN/electrolytes_data"
        / "simulation/FP32_simulation/In_distribution_exp/NPT_langevin"
        / "npt_1ns_293K_for_compare_rdf_with_UMA"
)

# UMA 1 M / 293 K / 1 ns trajectories (10 fs saves), the RDF baseline.
UMA_293K_1ns = (
    CFS / "m5024/distillation_project/data/raw_data_from_UMA_simulation"
        / "293K/UMA_trajs_293K_1M_1ns"
)

# UMA 298 K / 1 ns NPT runs (1 ps saves), one <system>_<conc>/ folder per system:
# the teacher densities of Fig. 2b. Written by
# src/simulate/fairchem_configs/launch_uma_298K_density.sh.
UMA_298K_1ns = DELTA_SCRATCH / "UMA_298K_1ns"

# ── outputs: computed observables ────────────────────────────────────────────
OUT_OPLS_TRANSPORT = OBS_ROOT / "opls_transport_observables"

# Multi-replica transport observables, computed from PAINN_NVT_REPLICAS. Mirrors the
# OPLS tree's layout exactly (per_run/<box>/r<k>/, *_master.csv, *_by_box.csv,
# *_config.yaml, figure_*.png) so the two sets are trivially comparable; the
# 4-replica set means *_by_box.csv carries real mean +/- s.d. across replicas.
# Default output root of the {viscosity_gk,diffusivity,conductivity}_replicas
# drivers; ``--time 1ns|5ns`` appends a suffix (see transport/time_presets.py).
OUT_PAINN_TRANSPORT_REPLICAS = OBS_ROOT / "painn_transport_observables_replicas"

# PaiNN static observables: densities at the top, the 293 K RDFs in a subdir.
OUT_PAINN_STATIC = OBS_ROOT / "painn_static_observables_NPT"
# One directory per system: ``rdf.csv`` is the replica *mean* (with ``sd_<pair>``
# spread columns) and is what every figure reads; the per-replica curves it was
# built from sit in ``<system>_<conc>/replicas/rdf_r<N>.csv``. Same shape as
# OUT_UMA_293K_RDF below, which has no replicas and so no ``sd_`` columns.
OUT_PAINN_293K_RDF = OUT_PAINN_STATIC / "rdfs_293K"

OUT_UMA_STATIC = OBS_ROOT / "uma_static_observables_1ns_NPT"
OUT_UMA_293K_RDF = OUT_UMA_STATIC / "rdfs_293K"
# UMA 298 K densities (uma_densities.py). On Delta this is the trajectory folder
# itself, since data/ points at DELTA_SCRATCH.
OUT_UMA_298K_DENSITY = OBS_ROOT / "UMA_298K_1ns"

# ── models ───────────────────────────────────────────────────────────────────
# The UMA-S teacher. uma-s-1p1 is the only teacher used in the paper; numbers
# from another checkpoint (1p2) are not comparable. Same file on both machines,
# different project dirs; override with the UMA_S_1P1 env var.
_UMA_S_1P1 = {
    "delta": Path("/projects/beye/iamin/distillation_project/models/uma-s-1p1.pt"),
    "perlmutter": CFS / "m4558/distillation_project/models/uma-s-1p1.pt",
}
UMA_S_1P1 = Path(os.environ.get(
    "UMA_S_1P1", _UMA_S_1P1.get(current_machine(), _UMA_S_1P1["delta"])))

# ── inputs: external analysis trees ──────────────────────────────────────────

# Transport numbers for the MLP baseline of the SI parity figure, merged per
# system. Read by paper_figures/appendix/MLP_results.
MLP_BASELINE = PERLMUTTER_ANALYSIS / "meta_data" / "correct_molarity" / "MLP"

# ── outputs: figures ─────────────────────────────────────────────────────────
# The manuscript is a separate repository; its sources are not part of this one.
# Figure scripts write their PNGs into this tree, which the manuscript build
# picks up, and create it on first run. Point FIGURES elsewhere (or set
# ELECTROLYTES_FIGURES) to write somewhere else.
FIGURES = Path(os.environ.get(
    "ELECTROLYTES_FIGURES", REPO_ROOT / "paper" / "paper_electrolytes" / "Figures"))
FIGURES_MAIN = FIGURES / "main_paper"
FIGURES_APPENDIX = FIGURES / "appendix"

# ── the manifest, for `require()` and the CLI ────────────────────────────────
# name -> (path, machine)
INPUTS: dict[str, tuple[Path, str]] = {
    "OPLS_20ns_298K": (OPLS_20ns_298K, "perlmutter"),
    "OPLS_20ns_298K_OLD": (OPLS_20ns_298K_OLD, "perlmutter"),
    "PAINN_NVT_REPLICAS": (PAINN_NVT_REPLICAS, "delta"),
    "PAINN_NPT_REPLICAS_FOR_DENSITIES": (PAINN_NPT_REPLICAS_FOR_DENSITIES, "perlmutter"),
    "PAINN_293K_1ns": (PAINN_293K_1ns, "perlmutter"),
    "UMA_293K_1ns": (UMA_293K_1ns, "perlmutter"),
    "UMA_298K_1ns": (UMA_298K_1ns, "delta"),
    "MLP_BASELINE": (MLP_BASELINE, "perlmutter"),
}
OUTPUTS: dict[str, Path] = {
    "OUT_OPLS_TRANSPORT": OUT_OPLS_TRANSPORT,
    "OUT_PAINN_TRANSPORT_REPLICAS": OUT_PAINN_TRANSPORT_REPLICAS,
    "OUT_PAINN_STATIC": OUT_PAINN_STATIC,
    "OUT_PAINN_293K_RDF": OUT_PAINN_293K_RDF,
    "OUT_UMA_STATIC": OUT_UMA_STATIC,
    "OUT_UMA_293K_RDF": OUT_UMA_293K_RDF,
    "OUT_UMA_298K_DENSITY": OUT_UMA_298K_DENSITY,
    "GROUND_TRUTH": GROUND_TRUTH,
    "FIGURES": FIGURES,
}

_BY_PATH = {p: (name, machine) for name, (p, machine) in INPUTS.items()}


def require(path: Path, name: str | None = None) -> Path:
    """Return ``path``, or raise ``SystemExit`` with a message that says *why*
    it is missing — a wrong-machine dataset is the common case."""
    path = Path(path)
    if path.exists():
        return path

    name, machine = _BY_PATH.get(path, (name, None))
    here = current_machine()
    label = name or str(path)
    if machine and machine != here:
        raise SystemExit(
            f"{label} lives on {machine}, but you are on {here}.\n"
            f"  {path}\n"
            f"Run this on {machine}, or pass an explicit root on the command line."
        )
    raise SystemExit(
        f"{label} not found:\n  {path}\n"
        f"(machine: {here}). Pass an explicit root on the command line if it moved."
    )


def _main() -> None:
    here = current_machine()
    print(f"machine: {here}\nrepo:    {REPO_ROOT}\n")
    width = max(len(n) for n in (*INPUTS, *OUTPUTS))

    print("INPUTS")
    for name, (path, machine) in INPUTS.items():
        if machine != here:
            mark = "–"                       # not this machine; can't say
        else:
            mark = "ok" if path.exists() else "MISSING"
        print(f"  {name:<{width}}  {mark:<8} [{machine}] {path}")

    print("\nOUTPUTS")
    for name, path in OUTPUTS.items():
        mark = "ok" if path.exists() else "MISSING"
        print(f"  {name:<{width}}  {mark:<8} {path}")


if __name__ == "__main__":
    _main()
