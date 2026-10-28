#!/usr/bin/env python
"""fairchemV2_new driver: UMA (uma-s-1p1) + OrbMol-v2 for all 3 systems.

    mamba activate fairchemV2_new
    python -m paper.paper_figures.appendix.interaction_scanning.source_scans.scan_driver_fairchem

Writes data/{system}__uma.npz and data/{system}__orb.npz.
UMA + Orb read atoms.info['charge'/'spin'] (0 / -2 / +2 for na_otf / otf_otf / na_na).
"""
from . import gas_phase_scan_common as C

from src.analysis.observables import paths

UMA_PATH = str(paths.UMA_S_1P1)


def make_uma():
    import torch
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return FAIRChemCalculator(load_predict_unit(UMA_PATH, device=dev), task_name="omol")


def make_orb():
    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ff, adapter = pretrained.orbmol_v2(device=dev)
    return ORBCalculator(ff, atoms_adapter=adapter, device=dev)


if __name__ == "__main__":
    C.run_and_save("uma", "UMA (uma-s-1p1 MLIP)", make_uma, device_note="(GPU if avail)")
    C.run_and_save("orb", "OrbMol-v2 (Orb MLIP)", make_orb)
