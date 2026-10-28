"""Electrolyte system registry: species patterns and per-system specs.

A ``system_id`` is ``<salt>_<solvent>`` (e.g. ``napf6_dme``) and keys into
``SYSTEMS``. Runs live under ``<system_id>/<run_name>/``, so
``system_id_from_parent_dir`` resolves a run's system from its parent folder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ── Species patterns ────────────────────────────────────────────────────────
# Exact element-sequence of one molecule, matched against the atomic-symbol list
# in order (see grouping.direct_groups_from_species). Relocated unchanged from
# the former diffusivity/utils/component_dictionary.py.
cation_dict = {
    "Na": ["Na"],
    "Li": ["Li"],
}

anion_dict = {
    "PF6": ["P", "F", "F", "F", "F", "F", "F"],
    "OTf": ["C", "S", "F", "F", "F", "O", "O", "O"],
}

solvent_dict = {
    "Diglyme": ["H", "C", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "H", "H", "H"],
    "DME": ["H", "C", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "H", "H", "H"],
    "PC": ["C", "H", "O", "C", "C", "H", "H", "O", "O", "C", "H", "H", "H"],
    "TGDME": ["H", "C", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "C",
              "H", "H", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "H", "H", "H"],
    "DEG": ["H", "O", "C", "C", "H", "H", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "H"],
}


@dataclass(frozen=True)
class SystemSpec:
    """Static description of one salt/solvent system."""
    system_id: str          # "napf6_dme"
    cation: str             # key into cation_dict, also the element ("Na"/"Li")
    anion: str              # key into anion_dict
    solvent: str            # key into solvent_dict
    rdf_pairs: tuple        # default element pairs for the RDF, e.g. ("Na-O", ...)


def _rdf_pairs(cation: str, anion: str) -> tuple:
    """Default RDF pairs: cation against solvent O and the anion's two heavy
    species (always a halogen + the anion centre)."""
    third = {"PF6": "P", "OTf": "S"}[anion]
    return (f"{cation}-O", f"{cation}-F", f"{cation}-{third}")


# (system_id, cation, anion, solvent) for every system the project runs.
_TABLE = [
    ("napf6_dme",     "Na", "PF6",  "DME"),
    ("napf6_pc",      "Na", "PF6",  "PC"),
    ("napf6_diglyme", "Na", "PF6",  "Diglyme"),
    ("napf6_tgdme",   "Na", "PF6",  "TGDME"),
    ("naotf_dme",     "Na", "OTf",  "DME"),
    ("naotf_pc",      "Na", "OTf",  "PC"),
    ("naotf_diglyme", "Na", "OTf",  "Diglyme"),
    ("naotf_tgdme",   "Na", "OTf",  "TGDME"),
    ("lipf6_dme",     "Li", "PF6",  "DME"),
]

SYSTEMS = {
    sid: SystemSpec(sid, cat, an, solv, _rdf_pairs(cat, an))
    for sid, cat, an, solv in _TABLE
}


def lookup_system(system_id: str) -> SystemSpec:
    if system_id not in SYSTEMS:
        raise KeyError(
            f"Unknown system '{system_id}'. Known systems: {sorted(SYSTEMS)}"
        )
    return SYSTEMS[system_id]


_CONC_RE = re.compile(r"^\d+(\.\d+)?M$")


def system_id_from_parent_dir(run_dir) -> str:
    """Resolve a run's system from its parent folder (the project layout is
    ``experiments/<exp>/<system>/<run>/``, e.g. ``.../napf6_dme_1M/<run>``).

    Used by ``src/simulate/fennol_npt.py`` to resolve the system of a run folder.
    Matches the parent folder against the registry *exactly* (after dropping a
    trailing concentration token like ``_1M``); raises if it isn't a known
    system, rather than silently guessing.
    """
    parent = Path(run_dir).parent.name
    candidates = [parent]
    toks = parent.split("_")
    if toks and _CONC_RE.match(toks[-1]):
        candidates.append("_".join(toks[:-1]))   # napf6_dme_1M -> napf6_dme
    for cand in candidates:
        if cand in SYSTEMS:
            return cand
    raise KeyError(
        f"Cannot resolve a registered system from parent folder {parent!r} of "
        f"{run_dir}. Pass --system explicitly. Known systems: {sorted(SYSTEMS)}"
    )
