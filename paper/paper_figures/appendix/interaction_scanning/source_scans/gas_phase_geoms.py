#!/usr/bin/env python
"""Shared geometry / grid / charge definitions for the three gas-phase ion scans.

Single source of truth imported by every env-specific scan driver (fairchem, sevennet,
mace_polar) and the plot script, so all models see the IDENTICAL rigid geometry, approach
axis, scan grid and total charge/spin for each system:

  na_otf  : Na+ approaches the SO3 face of a rigid OTf-        (net charge  0, attraction)
  otf_otf : two rigid OTf-, SO3-face to SO3-face               (net charge -2, like-charge)
  na_na   : two bare Na+ point ions                            (net charge +2, like-charge)

All geometries/grids are copied verbatim from the original per-system scripts
(gas_phase_Na_OTf_scan.ipynb cell 2, scan_otf_otf.py, scan_na_na.py) so results are
byte-compatible with the previously published curves.

For each system this module exposes:
  GRID[sys]                     -> np.ndarray of scan distances d (A)
  CHARGE[sys], SPIN[sys]        -> ints for atoms.info
  build(sys, d)                 -> (symbols:list[str], positions:np.ndarray)  for any model
  N_FIXED[sys]                  -> atoms in the fixed fragment (the rest is the moving one)
  AXIS[sys]                     -> approach axis (for optional axial-force projection)
  XLABEL[sys], REF_D[sys]       -> plot x-axis label and the reference (dissociated) distance
"""
import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────
HARTREE_EV = 27.211386245988

# ── OTf- rigid geometry (one equilibrated m5024 MD frame), shared by na_otf & otf_otf ──
OTF_ELEMENTS = ["C", "S", "F", "F", "F", "O", "O", "O"]
OTF_POS = np.array([
    [23.010, 12.330,  8.480],   # C  (CF3 carbon)
    [23.290, 11.460, 10.080],   # S  (sulfonate)
    [23.130, 11.480,  7.450],   # F
    [21.780, 12.870,  8.460],   # F
    [23.930, 13.280,  8.380],   # F
    [23.170, 12.570, 11.040],   # O  (sulfonate O)
    [24.590, 10.930,  9.900],   # O
    [22.140, 10.560, 10.050],   # O
], dtype=float)

# ── na_otf: Na+ into the OTf- SO3 face ──────────────────────────────────────────
_iC = OTF_ELEMENTS.index("C")
_O_idx = [i for i, e in enumerate(OTF_ELEMENTS) if e == "O"]
_Oc_A = OTF_POS[_O_idx].mean(0)
_AXIS_OTF = (_Oc_A - OTF_POS[_iC]); _AXIS_OTF /= np.linalg.norm(_AXIS_OTF)


def _na_otf_na_pos(d):
    return _Oc_A + d * _AXIS_OTF


# ── otf_otf: reflect anion A through the SO3-face plane, translate to O3c sep = d ──
def _reflect(p):
    return p - 2.0 * np.outer(p @ _AXIS_OTF, _AXIS_OTF)


_OTF_POS_B0 = _reflect(OTF_POS)
_Oc_B0 = _OTF_POS_B0[_O_idx].mean(0)


def _otf_B_pos(d):
    return _OTF_POS_B0 + ((_Oc_A + d * _AXIS_OTF) - _Oc_B0)


# ── na_na: two bare Na+ on a line ───────────────────────────────────────────────
_NA_A = np.array([0.0, 0.0, 0.0])
_AXIS_NANA = np.array([1.0, 0.0, 0.0])

# ── per-system registries ───────────────────────────────────────────────────────
GRID = {
    "na_otf":  np.unique(np.concatenate([np.linspace(1.6, 6.0, 34), np.linspace(6.4, 12.0, 8)])),
    "otf_otf": np.unique(np.concatenate([np.linspace(2.5, 7.0, 34), np.linspace(7.6, 14.0, 8)])),
    "na_na":   np.unique(np.concatenate([np.linspace(2.0, 7.0, 34), np.linspace(7.6, 14.0, 8)])),
}
CHARGE = {"na_otf": 0, "otf_otf": -2, "na_na": 2}
SPIN   = {"na_otf": 1, "otf_otf": 1, "na_na": 1}
AXIS   = {"na_otf": _AXIS_OTF, "otf_otf": _AXIS_OTF, "na_na": _AXIS_NANA}
REF_D  = {"na_otf": 12.0, "otf_otf": 14.0, "na_na": 14.0}
XLABEL = {
    "na_otf":  r"Na–O$_3$(centroid) distance  $d$  (Å)",
    "otf_otf": r"O$_3$(A)–O$_3$(B) centroid distance  $d$  (Å)",
    "na_na":   r"Na$^+$–Na$^+$ distance  $d$  (Å)",
}
N_FIXED = {"na_otf": len(OTF_ELEMENTS), "otf_otf": len(OTF_ELEMENTS), "na_na": 1}
SYSTEMS = ["na_otf", "otf_otf", "na_na"]


def build(system, d):
    """Return (symbols, positions) of `system` at separation d; the fixed fragment comes first."""
    if system == "na_otf":
        return OTF_ELEMENTS + ["Na"], np.vstack([OTF_POS, _na_otf_na_pos(d)])
    if system == "otf_otf":
        return OTF_ELEMENTS + OTF_ELEMENTS, np.vstack([OTF_POS, _otf_B_pos(d)])
    if system == "na_na":
        return ["Na", "Na"], np.vstack([_NA_A, _NA_A + d * _AXIS_NANA])
    raise ValueError(f"unknown system {system!r}")


def min_contact(system, d):
    """Smallest inter-fragment atom distance at separation d (diagnostic)."""
    if system == "na_na":
        return d
    if system == "na_otf":
        na = _na_otf_na_pos(d)
        return min(np.linalg.norm(na - OTF_POS[i]) for i in range(len(OTF_ELEMENTS)))
    pB = _otf_B_pos(d)
    return min(np.linalg.norm(pB[j] - OTF_POS[i])
               for i in range(len(OTF_ELEMENTS)) for j in range(len(OTF_ELEMENTS)))
