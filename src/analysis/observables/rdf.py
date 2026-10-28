"""Radial distribution functions — a streaming per-frame accumulator.

g(r) for each element pair "A-B" via minimum-image distances. Lifted from the
former example_rdf.py, split into update/finalize so density and RDF share one
trajectory pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np


@dataclass
class RDFResult:
    r: np.ndarray                 # bin midpoints (A)
    g: dict                       # pair -> g(r) array
    pairs: list
    peaks: dict                   # pair -> {"r": float, "g": float} (first peak beyond 1.5 A)
    r_max: float
    dr: float
    n_frames: int


class RDFAccumulator:
    def __init__(self, pairs, r_max: float = 12.0, dr: float = 0.05):
        self.pairs = list(pairs)
        self.parsed = [tuple(p.split("-")) for p in self.pairs]
        self.bins = np.arange(0.0, r_max + dr, dr)
        self.r_mid = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.shell_vol = (4.0 / 3.0) * pi * (self.bins[1:] ** 3 - self.bins[:-1] ** 3)
        self.r_max = r_max
        self.dr = dr
        self.hists = {p: np.zeros(len(self.r_mid), dtype=np.float64) for p in self.pairs}
        self.n_central = {p: 0.0 for p in self.pairs}
        self.n_partner = {p: 0.0 for p in self.pairs}
        self.vol_sum = 0.0
        self.n_frames = 0
        self._box_warned = False

    def update(self, atoms) -> None:
        syms = np.array(atoms.get_chemical_symbols())
        pos = atoms.get_positions()
        cell = np.array(atoms.get_cell())
        if not self._box_warned:   # minimum-image is only valid out to half the box
            self._box_warned = True
            half_min_box = 0.5 * float(np.linalg.norm(cell, axis=1).min())
            if self.r_max > half_min_box:
                print(f"[warn] RDF r_max={self.r_max} Å > half the min box length "
                      f"({half_min_box:.1f} Å); g(r) past {half_min_box:.1f} Å is unreliable.")
        inv_cell = np.linalg.inv(cell)
        self.vol_sum += atoms.get_volume()
        self.n_frames += 1

        for pstr, (a, b) in zip(self.pairs, self.parsed):
            idx_a = np.where(syms == a)[0]
            idx_b = np.where(syms == b)[0]
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue
            disp = (pos[idx_b][None, :, :] - pos[idx_a][:, None, :]).reshape(-1, 3)
            frac = disp @ inv_cell
            frac -= np.round(frac)                       # minimum image
            d = np.linalg.norm(frac @ cell, axis=1)
            self.hists[pstr] += np.histogram(d, bins=self.bins)[0]
            self.n_central[pstr] += len(idx_a)
            self.n_partner[pstr] += len(idx_b)

    def finalize(self) -> RDFResult:
        g, peaks = {}, {}
        vol_avg = self.vol_sum / self.n_frames if self.n_frames else float("nan")
        for pstr in self.pairs:
            n_cen = self.n_central[pstr]
            if n_cen == 0 or self.n_frames == 0:
                g[pstr] = np.zeros_like(self.r_mid)
                continue
            rho_partner = (self.n_partner[pstr] / self.n_frames) / vol_avg
            gr = (self.hists[pstr] / n_cen) / (rho_partner * self.shell_vol)
            g[pstr] = gr
            mask = self.r_mid > 1.5                       # ignore the self/bonded core
            if np.any(gr[mask] > 0):
                idx = int(np.argmax(gr[mask]) + np.searchsorted(self.r_mid, 1.5))
                peaks[pstr] = {"r": float(self.r_mid[idx]), "g": float(gr[idx])}
        return RDFResult(self.r_mid, g, self.pairs, peaks, self.r_max, self.dr, self.n_frames)
