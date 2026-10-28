"""Molecule grouping and periodic-image-correct centre-of-mass.

Atoms in a trajectory are grouped into cation / anion / solvent molecules by
matching each species' exact element-sequence against the (ordered) atomic-symbol
list. Molecular COMs are then computed with minimum-image unwrapping.

Why the unwrapping matters (load-bearing — do not "simplify" away):
trajectory positions are *wrapped* into the box. For a single atom (Na+/Li+)
that's fine, but a multi-atom molecule straddling a periodic boundary has its
atoms split across opposite faces, so a naive mass-weighted COM lands in the
middle of the box instead of at the molecule. Which molecules are split changes
every frame, so the COMs jump wildly and the MSD inflates by ~5-13x (PF6/DME D
went from ~38 down to ~3 x10^-10 m^2/s once fixed). ``diffusivity._species_coms``
places all atoms on the same side (MIC relative to the first atom) before averaging.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .systems import SystemSpec, anion_dict, cation_dict, solvent_dict


def direct_groups_from_species(symbols, species_list):
    """Find molecules by matching an exact element-sequence pattern.

    Slides a window of ``len(species_list)`` over ``symbols`` and records every
    exact match (element identity and order). Overlapping matches are dropped,
    keeping the first occurrence, so each atom belongs to at most one group.

    Returns a list of int index-arrays, one per matched molecule.
    """
    groups = []
    window_len = len(species_list)
    for i in range(len(symbols) - window_len + 1):
        if list(symbols[i:i + window_len]) == list(species_list):
            groups.append(np.arange(i, i + window_len, dtype=int))

    if not groups:
        return groups

    unique_groups = []
    seen = set()
    for group in groups:
        gset = set(group.tolist())
        if not gset & seen:
            unique_groups.append(group)
            seen |= gset
    return unique_groups


@dataclass
class Groups:
    """Per-frame atom groupings for one system."""
    cat_idx: list                 # indices of cation atoms (single-atom species)
    anion_groups: list            # list of index-arrays, one per anion molecule
    solvent_groups: list          # list of index-arrays, one per solvent molecule


def build_groups(symbols, spec: SystemSpec) -> Groups:
    """Group atoms into cation/anion/solvent for ``spec`` and sanity-check that
    every atom is accounted for."""
    cat_groups = direct_groups_from_species(symbols, cation_dict[spec.cation])
    anion_groups = direct_groups_from_species(symbols, anion_dict[spec.anion])
    solvent_groups = direct_groups_from_species(symbols, solvent_dict[spec.solvent])

    n_total = len(symbols)
    n_grouped = sum(len(g) for g in cat_groups + anion_groups + solvent_groups)
    if n_grouped != n_total:
        raise RuntimeError(
            f"Grouping mismatch for system '{spec.system_id}': matched {n_grouped} "
            f"atoms but the frame has {n_total}. Check the species patterns."
        )

    cat_idx = [i for i, s in enumerate(symbols) if s == spec.cation]
    if not cat_idx:
        raise RuntimeError(f"No {spec.cation} cation atoms found.")

    return Groups(cat_idx=cat_idx, anion_groups=anion_groups, solvent_groups=solvent_groups)
