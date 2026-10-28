"""Diffusion coefficients from MSD.

Streams a trajectory, removes system COM drift, MIC-unwraps cation atoms and
anion/solvent molecule COMs in time, computes a time-origin-averaged MSD per
species, and fits the Einstein slope. Physics lifted verbatim from the former
diffusivity/msd_with_com.py; decoupled from its hardcoded NERSC config and
plotting, and made to stream (no whole-trajectory load).
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from ase.geometry import find_mic
from tqdm import tqdm

from . import io
from .grouping import build_groups
from .systems import SystemSpec
from .units import yeh_hummer_correction_a2ps


@dataclass
class DiffResult:
    tau_ps: np.ndarray
    msd_cat: np.ndarray
    msd_anion: np.ndarray | None
    msd_solvent: np.ndarray | None
    d_cation_a2ps: float
    d_anion_a2ps: float | None
    d_solvent_a2ps: float | None
    subsample_stride: int
    frames_used: int
    effective_dt_ps: float
    analysis_window_ps: float
    eq_time_ps: float
    tau_min_fit_ps: float
    tau_max_fit_ps: float
    method: str = "fft"
    box_length_a: float = 0.0
    # Yeh-Hummer finite-size correction (all None unless it was requested)
    viscosity_cp: float | None = None
    yh_correction_a2ps: float | None = None


# ── MSD ─────────────────────────────────────────────────────────────────────
def msd_time_origin(unwrapped: np.ndarray) -> np.ndarray:
    """Time-origin averaged MSD over particles and origins. unwrapped: (T, M, 3)."""
    T = unwrapped.shape[0]
    msd = np.zeros(T, dtype=float)
    for lag in range(T):
        d = unwrapped[lag:] - unwrapped[:T - lag]
        msd[lag] = np.sum(d * d, axis=2).mean()
    return msd


def _msd_chunk(args):
    unwrapped, lag_start, lag_end = args
    T = unwrapped.shape[0]
    out = np.zeros(lag_end - lag_start)
    for k, lag in enumerate(range(lag_start, lag_end)):
        d = unwrapped[lag:] - unwrapped[:T - lag]
        out[k] = np.sum(d * d, axis=2).mean()
    return lag_start, out


def msd_time_origin_parallel(unwrapped: np.ndarray, n_workers: int = 4,
                             chunk_size: int = 200) -> np.ndarray:
    """Parallel MSD over lag chunks. Numerically identical to msd_time_origin
    (verify with the equivalence test before relying on it)."""
    T = unwrapped.shape[0]
    chunks = [(unwrapped, s, min(s + chunk_size, T))
              for s in range(0, T, chunk_size)]
    msd = np.zeros(T)
    if n_workers <= 1:
        for lag_start, arr in map(_msd_chunk, chunks):
            msd[lag_start:lag_start + len(arr)] = arr
    else:
        with Pool(processes=n_workers) as pool:
            for lag_start, arr in pool.imap(_msd_chunk, chunks):
                msd[lag_start:lag_start + len(arr)] = arr
    return msd


def msd_fft(unwrapped: np.ndarray) -> np.ndarray:
    """Time-origin averaged MSD via the FFT (Fast Correlation) algorithm.

    Identical result to :func:`msd_time_origin` (averaged over particles and all
    time origins) but **O(T log T)** instead of O(T^2), so the whole trajectory
    can be used cheaply. ``unwrapped``: (T, M, 3); returns (T,).

    Ported from byteff2's ``onsager_conductivity.correlate_xy`` (the Calandrini /
    ``tidynamics`` decomposition):
        MSD(tau) = [S1(tau) - 2 S2(tau)] / (M * (T - tau)),
    where S1 is the per-frame sum-of-squares (via a cumulative-sum identity) and
    S2 = sum_t r(t).r(t+tau) is the position autocorrelation (via FFT). Sums are
    accumulated over particles, so the divide by M*(T-tau) is the joint average.
    """
    T, M = unwrapped.shape[0], unwrapped.shape[1]
    if T == 0:
        return np.zeros(0)

    # S1(tau) = sum_{t=0}^{T-1-tau} (D[t] + D[t+tau]), with D[t] = sum over
    # particles & dims of r(t)^2.  Built from cumulative sums of D (the identity
    # S1[tau] = Q - cumsum(D)[tau-1] - cumsum(D[::-1])[tau-1], Q = 2*sum(D)).
    D = np.sum(unwrapped * unwrapped, axis=(1, 2))        # (T,)
    Q = 2.0 * D.sum()
    cm1 = np.cumsum(D)
    cm2 = np.cumsum(D[::-1])
    s1 = np.empty(T)
    s1[0] = Q
    s1[1:] = Q - cm1[:-1] - cm2[:-1]

    # S2(tau) = sum_t r(t).r(t+tau), summed over particles & dims, via the
    # FFT autocorrelation (zero-padded to >= 2T-1 to make it linear, not circular).
    n = 1 << (2 * T - 1).bit_length()                     # next power of two >= 2T-1
    F = np.fft.rfft(unwrapped, n=n, axis=0)               # (n//2+1, M, 3)
    acf = np.fft.irfft(F * np.conjugate(F), n=n, axis=0)[:T]   # (T, M, 3)
    s2 = np.sum(acf, axis=(1, 2))                         # (T,)

    norm = M * np.arange(T, 0, -1)                        # M * (T - tau)
    return (s1 - 2.0 * s2) / norm


def fit_diffusion(tau_ps, msd, tau_min_ps, tau_max_ps=None):
    """Linear (Einstein) fit; D = slope/6. Returns (D_a2ps, slope, intercept, mask)."""
    mask = tau_ps >= tau_min_ps
    if tau_max_ps is not None:
        mask &= tau_ps <= tau_max_ps
    x, y = tau_ps[mask], msd[mask]
    if x.size < 10:
        raise RuntimeError(
            f"Not enough points ({x.size}) in fit window [{tau_min_ps}, {tau_max_ps}] ps."
        )
    slope, intercept = np.polyfit(x, y, 1)
    return slope / 6.0, slope, intercept, mask


# ── Streaming unwrap (true system-COM frame) ─────────────────────────────────
# NOTE (load-bearing): self-diffusion must be measured in the frame of the *true*
# system centre of mass.  These runs are Langevin-thermostatted (thermostat LGV),
# so total momentum is NOT conserved and the unwrapped-system COM random-walks
# many Angstrom over a run; leaving it in inflates every MSD (up to ~2.5x, worst
# for the slowest species).  ``ase.Atoms.get_center_of_mass()`` is computed from
# the *wrapped* coordinates and cannot see this drift, so we build the COM from
# the time-unwrapped molecule COMs ourselves and subtract it.  (An earlier version
# subtracted the wrapped ase COM and therefore over-estimated D.)

def _species_coms(raw, group_arr, mass_per_mol, cell, pbc):
    """Whole-molecule COMs for one species, vectorised over molecules.

    ``group_arr``: (n_mol, n_atoms) atom-index array (all molecules same size);
    ``mass_per_mol``: (n_atoms,) atomic masses of one molecule.  Atoms are made
    whole via MIC relative to each molecule's first atom before the mass average,
    so a molecule straddling a box face gives a COM at the molecule.
    """
    P = raw[group_arr]                                   # (n_mol, n_atoms, 3)
    if P.shape[1] == 1:                                  # monoatomic (cation)
        return P[:, 0, :]
    ref = P[:, :1, :]
    mic, _ = find_mic((P[:, 1:, :] - ref).reshape(-1, 3), cell, pbc=pbc)
    whole = np.concatenate([ref, ref + mic.reshape(P.shape[0], -1, 3)], axis=1)
    return (whole * mass_per_mol[None, :, None]).sum(axis=1) / mass_per_mol.sum()


def _unwrap_com_frame(traj_path, spec, dt_ps, eq_time_ps, n_total, target_frames,
                      show_progress=False, frames_source=None, groups_override=None):
    """Stream frames after ``eq_time_ps`` and return per-species molecule-COM time
    series in the *true system-COM frame*, MIC-unwrapped in time.

    Returns ``(tau_ps, pos_cat, pos_anion, pos_solvent, stride, T, box_a,
    molar_masses, com_drift_a)``: ``pos_*`` are (T, N_species, 3); ``box_a`` is the
    window-mean cubic-equivalent edge V^(1/3) (A); ``molar_masses`` the per-species
    molecular masses [cat, anion, solvent]; ``com_drift_a`` the net system-COM
    displacement over the window (diagnostic for the Langevin drift removed).

    ``frames_source`` / ``groups_override`` are optional injection points for
    trajectory formats this package does not read natively (GROMACS ``.xtc`` via
    the OPLS transport scripts). ``frames_source(path, start, stride, max_count)``
    must yield ASE-like frames exactly as :func:`io.iter_frames` does, and
    ``groups_override`` a ready-made :class:`grouping.Groups` (needed when the
    per-molecule *atom order* differs from this project's element-sequence
    patterns, as it does for GROMACS topologies). Defaults reproduce the previous
    behaviour byte-for-byte; everything downstream is the same code either way.
    """
    if n_total < 3:
        raise RuntimeError("Trajectory has too few frames to analyze.")
    i_start = min(max(int(np.floor(eq_time_ps / dt_ps)), 0), n_total - 2)
    avail = n_total - i_start
    stride = max(1, int(np.ceil(avail / target_frames)))
    T = len(range(i_start, n_total, stride))

    iter_frames = frames_source if frames_source is not None else io.iter_frames
    frames = iter_frames(traj_path, start=i_start, stride=stride, max_count=T)
    f0 = next(frames)
    masses = np.asarray(f0.get_masses())
    groups = (groups_override if groups_override is not None
              else build_groups(f0.get_chemical_symbols(), spec))

    # molecule groups in species blocks: [cations | anions | solvents]
    cat_arr = np.array(groups.cat_idx, dtype=int)[:, None]        # (n_cat, 1)
    an_arr = np.array(groups.anion_groups, dtype=int)             # (n_an, k_an)
    sol_arr = np.array(groups.solvent_groups, dtype=int)          # (n_sol, k_sol)
    n_cat, n_an, n_sol = len(cat_arr), len(an_arr), len(sol_arr)
    n_mol = n_cat + n_an + n_sol

    m_cat_atoms, m_an_atoms, m_sol_atoms = masses[cat_arr[0]], masses[an_arr[0]], masses[sol_arr[0]]
    mol_mass = np.concatenate([
        np.full(n_cat, m_cat_atoms.sum()),
        np.full(n_an, m_an_atoms.sum()),
        np.full(n_sol, m_sol_atoms.sum()),
    ])
    molar_masses = np.array([m_cat_atoms.sum(), m_an_atoms.sum(), m_sol_atoms.sum()])

    def frame_coms(f):
        raw, cell, pbc = f.get_positions(), f.get_cell(), f.get_pbc()
        return np.concatenate([
            _species_coms(raw, cat_arr, m_cat_atoms, cell, pbc),
            _species_coms(raw, an_arr, m_an_atoms, cell, pbc),
            _species_coms(raw, sol_arr, m_sol_atoms, cell, pbc),
        ], axis=0), cell, pbc, f.get_volume()

    molU = np.zeros((T, n_mol, 3))
    prev, cell, pbc, vol = frame_coms(f0)
    molU[0] = prev
    vol_sum = vol
    k = 1
    for f in tqdm(frames, total=T - 1, desc="unwrap", disable=not show_progress):
        cur, cell, pbc, vol = frame_coms(f)
        disp, _ = find_mic(cur - prev, cell, pbc=pbc)         # COM moves << box/2
        molU[k] = molU[k - 1] + disp
        prev = cur
        vol_sum += vol
        k += 1

    if k < T:                                                 # trajectory ended early
        molU = molU[:k]
        T = k
    comU = np.einsum("tmj,m->tj", molU, mol_mass) / mol_mass.sum()   # true system COM
    r = molU - comU[:, None, :]

    pos_cat = r[:, :n_cat, :]
    pos_anion = r[:, n_cat:n_cat + n_an, :]
    pos_solvent = r[:, n_cat + n_an:, :]
    box_a = (vol_sum / k) ** (1.0 / 3.0)
    tau_ps = np.arange(T, dtype=float) * stride * dt_ps
    com_drift_a = float(np.linalg.norm(comU[-1] - comU[0]))
    return (tau_ps, pos_cat, pos_anion, pos_solvent, stride, T, box_a,
            molar_masses, com_drift_a)


def compute_diffusivity(traj_path, spec: SystemSpec, dt_ps: float,
                        eq_time_ps: float = 2000.0, tau_min_fit_ps: float = 1000.0,
                        tau_max_fit_ps: float = None, tau_max_fit_frac: float = 0.5,
                        target_frames: int = None, n_workers: int = 1,
                        method: str = "fft", yeh_hummer: bool = False,
                        viscosity_cp: float = None, temperature_k: float = None,
                        max_frames: int = None, frames_source=None,
                        groups_override=None, n_frames: int = None) -> DiffResult:
    """MSD over the whole post-equilibration trajectory; Einstein-fit over
    [tau_min_fit_ps, cutoff], where cutoff = tau_max_fit_ps if given else
    tau_max_fit_frac * (full lag window). The full curve is computed/returned
    regardless; only the fit is bounded, so the noisy few-origin tail is shown
    but not fit.

    ``method``: "fft" (default, O(T log T) Fast-Correlation MSD) or "direct"
    (the O(T^2) time-origin loop; ``n_workers`` parallelizes it). Both give the
    same numbers (see tests/test_msd_fft.py).

    ``yeh_hummer``: if True, add the finite-size correction
    dD = k_B T xi / (6 pi eta L) to every species' D (needs ``viscosity_cp`` and
    ``temperature_k``); both the raw D_PBC and the corrected D_inf are reported.
    Off by default.

    ``frames_source`` / ``groups_override`` / ``n_frames`` are the injection
    points documented on :func:`_unwrap_com_frame` — used by the OPLS transport
    scripts to feed GROMACS ``.xtc`` frames through this same estimator."""
    n_total = n_frames if n_frames is not None else io.estimate_n_frames(traj_path)
    if max_frames is not None:                 # cap analysis to first max_frames frames
        n_total = min(n_total, max_frames)
    if target_frames is None:
        # The FFT MSD is O(T log T), so this no longer bounds an O(T^2) cost; it
        # caps the linear unwrap/IO pass + memory. Default high so a ~20 ns @ 1 ps
        # run (~20k frames) uses *every* frame (stride 1) for best statistics.
        target_frames = 50000

    (tau, pos_cat, pos_anion, pos_solvent, stride, T, box_a,
     _molar_masses, _com_drift_a) = _unwrap_com_frame(
        traj_path, spec, dt_ps, eq_time_ps, n_total, target_frames,
        frames_source=frames_source, groups_override=groups_override,
    )
    window_ps = float(tau[-1]) if len(tau) else 0.0
    cutoff = tau_max_fit_ps if tau_max_fit_ps is not None else tau_max_fit_frac * window_ps

    def _msd(x):
        if x is None:
            return None
        if method == "fft":
            return msd_fft(x)
        return msd_time_origin_parallel(x, n_workers) if n_workers > 1 else msd_time_origin(x)

    msd_cat = _msd(pos_cat)
    msd_anion = _msd(pos_anion)
    msd_solvent = _msd(pos_solvent)

    d_cat, _, _, _ = fit_diffusion(tau, msd_cat, tau_min_fit_ps, cutoff)
    d_an = fit_diffusion(tau, msd_anion, tau_min_fit_ps, cutoff)[0] if msd_anion is not None else None
    d_sol = fit_diffusion(tau, msd_solvent, tau_min_fit_ps, cutoff)[0] if msd_solvent is not None else None

    yh = None
    if yeh_hummer:
        if viscosity_cp is None or temperature_k is None:
            raise RuntimeError(
                "Yeh-Hummer correction requested but viscosity_cp/temperature_k missing."
            )
        yh = yeh_hummer_correction_a2ps(temperature_k, viscosity_cp, box_a)

    return DiffResult(
        tau_ps=tau, msd_cat=msd_cat, msd_anion=msd_anion, msd_solvent=msd_solvent,
        d_cation_a2ps=d_cat, d_anion_a2ps=d_an, d_solvent_a2ps=d_sol,
        subsample_stride=stride, frames_used=T, effective_dt_ps=stride * dt_ps,
        analysis_window_ps=window_ps,
        eq_time_ps=eq_time_ps, tau_min_fit_ps=tau_min_fit_ps, tau_max_fit_ps=cutoff,
        method=method, box_length_a=box_a,
        viscosity_cp=(viscosity_cp if yeh_hummer else None), yh_correction_a2ps=yh,
    )
