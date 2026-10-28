#!/usr/bin/env python
"""mace_polar driver: MACE-Polar (PolarMACE polar-1-m) for all 3 systems.

    micromamba run -n mace_polar python -m paper.paper_figures.appendix.interaction_scanning.source_scans.scan_driver_mace_polar

Writes data/{system}__mace.npz. MACE pins e3nn==0.4.4 (incompatible with fairchem), so it
runs in its own env. Reads atoms.info['charge'/'spin'] (0 / -2 / +2).
"""
from . import gas_phase_scan_common as C

MODEL = "polar-1-m"


def make_calc():
    import torch
    from mace.calculators import mace_polar
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return mace_polar(model=MODEL, device=dev, default_dtype="float64")


if __name__ == "__main__":
    C.run_and_save("mace", "MACE-Polar (polar-1-m)", make_calc)
