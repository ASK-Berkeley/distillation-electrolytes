#!/usr/bin/env python
"""sevennet driver: SevenNet-Omni (7net-omni, modal omol25_high) for all 3 systems.

    micromamba run -n sevennet python -m paper.paper_figures.appendix.interaction_scanning.source_scans.scan_driver_sevennet

Writes data/{system}__sevennet.npz. NB: SevenNet-Omni does NOT read a per-system charge/
spin, so for the charged like-charge systems (otf_otf = -2, na_na = +2) it sees only the
bare geometry -- a genuine limitation the plots expose (it cannot know the net charge).
"""
from . import gas_phase_scan_common as C

MODEL = "7net-omni"
MODAL = "omol25_high"     # OMol25 wB97M-V/def2-TZVPD fidelity (matches the DFT reference)


def make_calc():
    import torch
    from sevenn.calculator import SevenNetCalculator
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return SevenNetCalculator(MODEL, modal=MODAL, device=dev)


if __name__ == "__main__":
    C.run_and_save("sevennet", "SevenNet-Omni (7net-omni, omol25_high)", make_calc)
