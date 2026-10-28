"""Onsager + Nernst-Einstein ionic conductivity from MD trajectories.

Native re-implementation of the byteff2 ``onsager_conductivity`` physics,
decoupled from that external submodule and made to conform to this package's
streaming style (mirrors :mod:`diffusivity`). Same unwrap, same FFT MSD, so the
per-species self-diffusivities reported here are *identical* to the ones the
diffusivity pipeline produces for the same trajectory + fit window.

Pipeline (one streaming pass, reused from diffusivity):
  1. Stream frames after ``eq_time_ps``; build per-species molecule-COM time
     series (system-COM removed, intra-molecular bonds made whole, then
     MIC-unwrapped in time).  See :func:`diffusivity._unwrap_com_frame`.
  2. Self term: FFT MSD per species -> Einstein slope -> D_self_i (A^2/ps).
  3. Collective term: R_i(t) = sum over molecules of species i of its COM.
     The charge-current cross-MSD  C_ij(tau) = <dR_i(tau).dR_j(tau)>  gives the
     Onsager matrix L_ij (slope / 6 / N_total), corrected for the fixed-COM
     constraint (``_remove_com_error``).

From those two we report, in one row per run:
  - ``sigma_NE``        Nernst-Einstein conductivity (ignores ion-ion corr.)
  - ``sigma_onsager``   full Onsager conductivity (includes cross-correlations)
  - the Onsager matrix ``L_ij`` (raw + COM-corrected, 10^-10 m^2/s)
  - self-diffusivities D_i (D_PBC and Yeh-Hummer D_inf)
  - transference numbers t_+ (NE + Onsager) and ionicity sigma_O / sigma_NE

Conventions match byteff2 so numbers line up with ``ground_truth/conductivity_all.csv``:
sigma_NE is built from the *PBC* self-diffusivities (the finite-size correction
is applied only to the reported D_inf, not to sigma).

The self-diffusivity fit reuses the diffusivity module verbatim; only the
collective cross term is new here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import io
from .diffusivity import _unwrap_com_frame, fit_diffusion, msd_fft
from .systems import SystemSpec
from .units import KB_J_PER_K, a2ps_to_m2s, yeh_hummer_correction_a2ps

E_CHARGE_C = 1.602176634e-19          # elementary charge (C)
S_PER_M_TO_MS_PER_CM = 10.0           # 1 S/m = 10 mS/cm


# ── cross-correlation (collective charge-current) ────────────────────────────
def cross_msd_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Time-origin-averaged cross displacement correlation of two collective
    coordinates, via the FFT (Fast-Correlation) algorithm.

        C(tau) = < (a(t+tau) - a(t)) . (b(t+tau) - b(t)) >_t ,

    averaged over all time origins t.  ``a``, ``b``: (T, 3) collective vectors;
    returns (T,).  For ``a is b`` this reduces to the (single-particle) MSD.

    Same Calandrini / ``tidynamics`` decomposition as :func:`diffusivity.msd_fft`
    (ported from byteff2's ``onsager_conductivity.correlate_xy``):
        C = [S1 - S2_ab - S2_ba] / (T - tau),
    with S1 the running sum of the per-frame dot a(t).b(t) and S2 the two
    position cross-correlations (a<->b) via FFT.
    """
    T = a.shape[0]
    if T == 0:
        return np.zeros(0)

    d = np.einsum("ij,ij->i", a, b)                       # (T,) per-frame a.b
    cm1 = np.cumsum(d)
    cm2 = np.cumsum(d[::-1])
    Q = 2.0 * cm1[-1]
    s1 = np.empty(T)
    s1[0] = Q
    s1[1:] = Q - cm1[:-1] - cm2[:-1]

    n = 1 << (2 * T - 1).bit_length()                     # >= 2T-1, power of two
    A = np.fft.rfft(a, n=n, axis=0)
    B = np.fft.rfft(b, n=n, axis=0)
    c_ab = np.fft.irfft(A * np.conjugate(B), n=n, axis=0)[:T].sum(axis=1)
    c_ba = np.fft.irfft(np.conjugate(A) * B, n=n, axis=0)[:T].sum(axis=1)

    div = np.arange(T, 0, -1)                             # T - tau
    return (s1 - c_ab - c_ba) / div


def _remove_com_error(L: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Project out the fixed-centre-of-mass constraint from the Onsager matrix.

    In a closed simulation cell the total (mass-weighted) momentum is conserved,
    so the species collective displacements are linearly dependent; this removes
    that spurious contribution.  ``masses`` are the per-species *molar* masses.
    (byteff2 ``remove_center_of_mass_error``, eq. A.6.)
    """
    total_m = masses.sum()
    Lm = L @ masses
    mLm = float(masses @ Lm) / total_m**2
    Lm = Lm / total_m
    return L - Lm[:, None] - Lm[None, :] + mLm


# ── result container ─────────────────────────────────────────────────────────
@dataclass
class ConductivityResult:
    # curves (for reproducibility / plotting)
    tau_ps: np.ndarray
    msd_cat: np.ndarray
    msd_anion: np.ndarray
    msd_solvent: np.ndarray
    cross_pp: np.ndarray               # ++  (cation-cation collective)
    cross_pm: np.ndarray               # +-  (cation-anion)
    cross_mm: np.ndarray               # --  (anion-anion)

    # self-diffusivities (A^2/ps, PBC / finite box)
    d_cation_a2ps: float
    d_anion_a2ps: float
    d_solvent_a2ps: float

    # Onsager matrix (10^-10 m^2/s), species order [cat, anion, solvent]
    lambda_raw: np.ndarray            # (3,3) before COM-error removal
    lambda_corr: np.ndarray          # (3,3) after COM-error removal

    # conductivities (mS/cm)
    sigma_ne_mS_cm: float
    sigma_onsager_mS_cm: float

    # transport summaries
    t_cation_ne: float               # NE cation transference number
    t_cation_onsager: float          # Onsager cation transference number
    ionicity: float                  # sigma_onsager / sigma_ne (inverse Haven)

    # counts / thermodynamic state
    n_cation: int
    n_anion: int
    n_solvent: int
    z_cation: float
    z_anion: float
    volume_a3: float
    box_length_a: float
    temperature_k: float
    com_drift_a: float               # net system-COM displacement over window (diagnostic)

    # provenance / fit window
    subsample_stride: int
    frames_used: int
    effective_dt_ps: float
    analysis_window_ps: float
    eq_time_ps: float
    tau_min_fit_ps: float
    tau_max_fit_ps: float

    # Yeh-Hummer finite-size correction (None unless requested)
    viscosity_cp: float | None = None
    yh_correction_a2ps: float | None = None

    # ── scaled views (10^-10 m^2/s) ─────────────────────────────────────────
    def _scaled(self, d):
        return None if d is None else a2ps_to_m2s(d) * 1e10

    @property
    def d_cation_x1e10(self):
        return self._scaled(self.d_cation_a2ps)

    @property
    def d_anion_x1e10(self):
        return self._scaled(self.d_anion_a2ps)

    @property
    def d_solvent_x1e10(self):
        return self._scaled(self.d_solvent_a2ps)


# ── conductivity from L_ij / D_self (SI, returns mS/cm) ──────────────────────
def _sigma_ne_mS_cm(counts, charges, d_m2s, volume_a3, T_K) -> float:
    """Nernst-Einstein sigma = e^2/(V kB T) * sum_i N_i z_i^2 D_i  ->  mS/cm."""
    counts = np.asarray(counts, float)
    charges = np.asarray(charges, float)
    d_m2s = np.asarray(d_m2s, float)
    V_m3 = volume_a3 * 1e-30
    sigma_si = (E_CHARGE_C**2 * np.sum(counts * charges**2 * d_m2s)
                / (V_m3 * KB_J_PER_K * T_K))            # S/m
    return sigma_si * S_PER_M_TO_MS_PER_CM


def _sigma_onsager_mS_cm(lambda_m2s, charges, n_total, volume_a3, T_K) -> float:
    """Onsager sigma = e^2 N/(V kB T) * z^T L z  ->  mS/cm.

    ``lambda_m2s`` is the COM-corrected matrix in m^2/s (already divided by
    6*N_total on construction, so the leading N_total restores the extensive
    charge-current variance).
    """
    charges = np.asarray(charges, float)
    V_m3 = volume_a3 * 1e-30
    zLz = float(charges @ lambda_m2s @ charges)
    sigma_si = (E_CHARGE_C**2 * zLz * n_total
                / (V_m3 * KB_J_PER_K * T_K))            # S/m
    return sigma_si * S_PER_M_TO_MS_PER_CM


# ── main entry ───────────────────────────────────────────────────────────────
def compute_conductivity(
    traj_path,
    spec: SystemSpec,
    dt_ps: float,
    z_cation: float = 1.0,
    z_anion: float = -1.0,
    eq_time_ps: float = 2000.0,
    tau_min_fit_ps: float = 1000.0,
    tau_max_fit_ps: float | None = None,
    tau_max_fit_frac: float = 0.5,
    target_frames: int | None = None,
    yeh_hummer: bool = False,
    viscosity_cp: float | None = None,
    temperature_k: float | None = None,
    show_progress: bool = True,
    frames_source=None,
    groups_override=None,
    n_frames: int | None = None,
) -> ConductivityResult:
    """Onsager + Nernst-Einstein conductivity for one ASE trajectory.

    The self-diffusivity is measured in the *true system-COM frame* (see
    :func:`_unwrap_com_frame`); this is the physically correct choice and matches
    both byteff2 and the diffusivity pipeline, which shares the same
    ``_unwrap_com_frame`` streaming unwrap.
    ``sigma`` itself needs no viscosity; the Yeh-Hummer correction only enters the
    reported D_inf columns, so pass ``yeh_hummer=True`` (+ ``viscosity_cp``) only
    when you want those.  ``temperature_k`` sets the T used in the sigma prefactor
    (required).

    All 1:1 salts here have z = +/-1; override ``z_cation`` / ``z_anion`` otherwise.

    ``frames_source`` / ``groups_override`` / ``n_frames`` are the injection points
    documented on :func:`diffusivity._unwrap_com_frame` — used by the OPLS
    conductivity script to feed GROMACS ``.xtc`` frames through this same estimator.
    """
    if temperature_k is None:
        raise RuntimeError("compute_conductivity requires temperature_k for the sigma prefactor.")

    n_total = n_frames if n_frames is not None else io.estimate_n_frames(traj_path)
    if target_frames is None:
        target_frames = 50000            # ~all frames of a 20 ns @ 1 ps run

    # COM-frame streaming unwrap: per-species molecule-COM series in the TRUE
    # system-COM frame, MIC-unwrapped in time; shapes (T, N_species, 3).
    (tau, pos_cat, pos_anion, pos_solvent, stride, T, box_a,
     molar_masses, com_drift_a) = _unwrap_com_frame(
        traj_path, spec, dt_ps, eq_time_ps, n_total, target_frames,
        show_progress=show_progress, frames_source=frames_source,
        groups_override=groups_override,
    )

    window_ps = float(tau[-1]) if len(tau) else 0.0
    cutoff = tau_max_fit_ps if tau_max_fit_ps is not None else tau_max_fit_frac * window_ps

    n_cat = pos_cat.shape[1]
    n_an = pos_anion.shape[1]
    n_sol = pos_solvent.shape[1]
    n_total_mol = n_cat + n_an + n_sol
    volume_a3 = box_a**3

    # ── self term (COM-frame MSD) ────────────────────────────────────────────
    msd_cat = msd_fft(pos_cat)
    msd_anion = msd_fft(pos_anion)
    msd_solvent = msd_fft(pos_solvent)
    d_cat = fit_diffusion(tau, msd_cat, tau_min_fit_ps, cutoff)[0]
    d_an = fit_diffusion(tau, msd_anion, tau_min_fit_ps, cutoff)[0]
    d_sol = fit_diffusion(tau, msd_solvent, tau_min_fit_ps, cutoff)[0]

    # ── collective term -> Onsager matrix L_ij ───────────────────────────────
    # R_i(t) = sum over molecules of species i of its (system-COM-removed) COM.
    R = [pos_cat.sum(axis=1), pos_anion.sum(axis=1), pos_solvent.sum(axis=1)]
    cross = {}                          # (i,j) -> C_ij(tau) curve
    lambda_a2ps = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            c_ij = cross_msd_fft(R[i], R[j])
            cross[(i, j)] = c_ij
            # slope/6 from fit_diffusion; collective term divides by N_total
            slope_over_6 = fit_diffusion(tau, c_ij, tau_min_fit_ps, cutoff)[0]
            lambda_a2ps[i, j] = lambda_a2ps[j, i] = slope_over_6 / n_total_mol

    lambda_raw = a2ps_to_m2s(lambda_a2ps) * 1e10                 # 10^-10 m^2/s
    lambda_corr = _remove_com_error(lambda_raw, molar_masses)

    # ── conductivities ───────────────────────────────────────────────────────
    charges = np.array([z_cation, z_anion, 0.0])
    d_pbc_m2s = np.array([a2ps_to_m2s(d_cat), a2ps_to_m2s(d_an), a2ps_to_m2s(d_sol)])
    counts = np.array([n_cat, n_an, n_sol])
    sigma_ne = _sigma_ne_mS_cm(counts, charges, d_pbc_m2s, volume_a3, temperature_k)
    sigma_ons = _sigma_onsager_mS_cm(lambda_corr * 1e-10, charges, n_total_mol,
                                     volume_a3, temperature_k)

    # ── transference numbers + ionicity ──────────────────────────────────────
    # NE: t_+ = N_+ z_+^2 D_+ / (N_+ z_+^2 D_+ + N_- z_-^2 D_-)
    ne_cat = n_cat * z_cation**2 * a2ps_to_m2s(d_cat)
    ne_an = n_an * z_anion**2 * a2ps_to_m2s(d_an)
    t_ne = ne_cat / (ne_cat + ne_an) if (ne_cat + ne_an) else float("nan")
    # Onsager: t_+ = (z_+^2 L_++ + z_+ z_- L_+-) / (z^T L z)  over the ion block
    Lpp, Lpm, Lmm = lambda_corr[0, 0], lambda_corr[0, 1], lambda_corr[1, 1]
    denom = z_cation**2 * Lpp + 2 * z_cation * z_anion * Lpm + z_anion**2 * Lmm
    t_ons = ((z_cation**2 * Lpp + z_cation * z_anion * Lpm) / denom
             if denom else float("nan"))
    ionicity = sigma_ons / sigma_ne if sigma_ne else float("nan")

    # ── Yeh-Hummer self-diffusivity correction (optional, sigma-independent) ──
    yh = None
    if yeh_hummer:
        if viscosity_cp is None:
            raise RuntimeError("yeh_hummer=True needs viscosity_cp.")
        yh = yeh_hummer_correction_a2ps(temperature_k, viscosity_cp, box_a)

    return ConductivityResult(
        tau_ps=tau, msd_cat=msd_cat, msd_anion=msd_anion, msd_solvent=msd_solvent,
        cross_pp=cross[(0, 0)], cross_pm=cross[(0, 1)], cross_mm=cross[(1, 1)],
        d_cation_a2ps=d_cat, d_anion_a2ps=d_an, d_solvent_a2ps=d_sol,
        lambda_raw=lambda_raw, lambda_corr=lambda_corr,
        sigma_ne_mS_cm=sigma_ne, sigma_onsager_mS_cm=sigma_ons,
        t_cation_ne=t_ne, t_cation_onsager=t_ons, ionicity=ionicity,
        n_cation=n_cat, n_anion=n_an, n_solvent=n_sol,
        z_cation=z_cation, z_anion=z_anion,
        volume_a3=volume_a3, box_length_a=box_a, temperature_k=temperature_k,
        com_drift_a=com_drift_a,
        subsample_stride=stride, frames_used=T, effective_dt_ps=stride * dt_ps,
        analysis_window_ps=window_ps, eq_time_ps=eq_time_ps,
        tau_min_fit_ps=tau_min_fit_ps, tau_max_fit_ps=cutoff,
        viscosity_cp=(viscosity_cp if yeh_hummer else None),
        yh_correction_a2ps=yh,
    )
