"""Mass density observable — a streaming per-frame accumulator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import density_g_cm3


@dataclass
class DensityResult:
    mean_g_cm3: float
    std_g_cm3: float
    n_frames: int


class DensityAccumulator:
    """Accumulates per-frame mass density so it can ride a shared streaming pass."""

    def __init__(self):
        self._rho = []

    def update(self, atoms) -> None:
        mass_amu = float(np.sum(atoms.get_masses()))
        vol_a3 = float(atoms.get_volume())
        self._rho.append(density_g_cm3(mass_amu, vol_a3))

    def finalize(self) -> DensityResult:
        arr = np.asarray(self._rho, dtype=float)
        if arr.size == 0:
            return DensityResult(float("nan"), float("nan"), 0)
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return DensityResult(float(arr.mean()), std, int(arr.size))
