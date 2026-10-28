"""Shear viscosity from equilibrium Green-Kubo stress-autocorrelation.

Each GK run (``fennol_npt.py --stress-log``, launched by
``src/simulate/launch_viscosity_gk.sh``) writes a ``<name>.stress`` log:

    # Green-Kubo shear-viscosity stress log. pressure_unit=atm stress_interval[steps]=N dt[fs]=..
    # time_ps  Pxx Pyy Pzz Pxy Pxz Pyz  volume[Ang^3]  T[Kelvin]
    <t> <Pxx> <Pyy> <Pzz> <Pxy> <Pxz> <Pyz> <V> <T>

Viscosity is the stress-autocorrelation integral

    eta = (V / kB T) * integral_0^inf < sigma_ab(0) sigma_ab(t) > dt ,

averaged over the 5 independent traceless stress components
(sigma_xy, sigma_xz, sigma_yz, (sigma_xx-sigma_yy)/2, (sigma_yy-sigma_zz)/2)
to reduce noise. The ACF is computed by FFT; the running integral eta(t_cut)
is returned so the plateau can be inspected (the tail is noise-dominated).

Used by ``transport/viscosity_gk_replicas.py`` (PaiNN) and
``transport/viscosity_gk_opls_replicas.py`` (OPLS) via ``eta_running_curve`` and
``fit_eta_tdm``; see ``transport/METHODS.md`` for the method.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .units import ANGSTROM3_TO_M3, KB_J_PER_K

_P_TO_PA = {"atm": 101325.0, "bar": 1e5, "Pa": 1.0}
_PUNIT_RE = re.compile(r"pressure_unit=(\w+)")


def _read_stress(path: Path):
    """Load a .stress log -> (dt_ps, DataFrame, p_to_pa). Robust to a
    partially-written final line (job still running). ``p_to_pa`` converts the
    log's pressure unit (parsed from the header line; every log the simulator
    has produced so far says atm) to Pa; an unrecognized unit raises rather
    than silently assuming atm."""
    path = Path(path)
    with open(path) as f:
        first_line = f.readline()
    m = _PUNIT_RE.search(first_line)
    unit = m.group(1) if m else "atm"          # headerless legacy log
    if unit not in _P_TO_PA:
        raise RuntimeError(f"{path.name}: unknown pressure_unit={unit!r} in stress header")
    p_to_pa = _P_TO_PA[unit]
    cols = ["time_ps", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz", "volume", "T"]
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=cols,
                     dtype=np.float64, on_bad_lines="skip", engine="c")
    df = df[np.isfinite(df).all(axis=1)]
    t = df["time_ps"].to_numpy()
    # the log is append-mode and restarts checkpoint only every ndump steps, so a
    # requeue can replay up to one dump window; keep strictly-increasing rows only.
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        keep = np.concatenate([[True], t[1:] > np.maximum.accumulate(t)[:-1]])
        df = df[keep]
        t = df["time_ps"].to_numpy()
    dt_ps = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    return dt_ps, df, p_to_pa


def _acf_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Unbiased autocorrelation <x(0)x(t)> for lags 0..max_lag-1 via FFT."""
    x = x - x.mean()
    n = len(x)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
    acf /= (n - np.arange(n))          # unbiased: divide by # overlapping pairs
    return acf[:max_lag]


# ── running η(τ) curve + best-practice (TDM) window estimator ────────────────
# For the transport_* NVT+BUSSI runs: the running GK integral η(τ) out to ~1 ns
# WITH a block-resampling error band, and a Zhang-Maginn time-decomposition fit
# that reads off η∞ with a data-driven window (no hand-picked plateau).


def _running_integral(acf: np.ndarray, dt_s: float, scale) -> np.ndarray:
    """Trapezoid cumulative integral of an ACF, then apply `scale` (-> cP)."""
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (acf[1:] + acf[:-1]) * dt_s)])
    return scale(integ)


def eta_running_curve(run_dir, max_lag_ps: float = 1000.0, nblock: int = 5,
                      t_K: float = 298.0, eq_time_ps: float = 0.0,
                      max_time_ps: float | None = None) -> dict:
    """Running Green-Kubo viscosity integral η(τ) [cP] out to `max_lag_ps`.

    Returns the full running integral averaged over the 5 traceless stress
    components, plus a block-resampling error band (std over `nblock` equal
    time-blocks, each re-integrated and interpolated onto the full-run lag grid).
    The prefactor uses the nominal setpoint `t_K` (the .stress T column is a
    known-incorrect on-step KE quantity; see ``transport/METHODS.md`` Sec. 4).
    ``eq_time_ps`` drops stress rows with t < eq_time_ps before the ACF, so the
    post-restart thermal transient can be excluded (matches the 2 ns cut the
    diffusivity/conductivity estimators apply); ``t_total_ns`` still reports the
    full simulated length. ``max_time_ps`` (if set) keeps only stress rows with
    t <= max_time_ps *before* the eq cut, i.e. analyses the run as if only its
    first ``max_time_ps`` had been simulated (the short-trajectory convergence
    study); ``t_total_ns`` then reports the truncated length.
    """
    run_dir = Path(run_dir)
    hits = sorted(run_dir.glob("*.stress"))
    if not hits:
        raise FileNotFoundError(f"No .stress log in {run_dir}")
    dt_ps, df, p_to_pa = _read_stress(hits[0])
    if not np.isfinite(dt_ps) or dt_ps <= 0:
        raise RuntimeError(f"{run_dir.name}: could not infer stress sampling dt")
    if max_time_ps is not None:
        df = df[df["time_ps"] <= max_time_ps]
        if len(df) < 1000:
            raise RuntimeError(
                f"{run_dir.name}: only {len(df)} stress rows within the first "
                f"{max_time_ps} ps")
    t_total_ns = float(df["time_ps"].iloc[-1] / 1000.0)
    if eq_time_ps > 0:
        df = df[df["time_ps"] >= eq_time_ps]
        if len(df) < 1000:
            raise RuntimeError(
                f"{run_dir.name}: only {len(df)} stress rows after the "
                f"{eq_time_ps} ps equilibration cut")

    P = {k: df[k].to_numpy() for k in ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")}
    comps = [c * p_to_pa for c in
             (P["Pxy"], P["Pxz"], P["Pyz"],
              0.5 * (P["Pxx"] - P["Pyy"]), 0.5 * (P["Pyy"] - P["Pzz"]))]
    V_m3 = float(df["volume"].mean()) * ANGSTROM3_TO_M3
    pref, dt_s = V_m3 / (KB_J_PER_K * t_K), dt_ps * 1e-12
    n = len(df)
    ml = min(n - 1, int(round(max_lag_ps / dt_ps)))
    lags = np.arange(ml) * dt_ps
    scale = lambda I: pref * I * 1000.0                       # Pa.s -> cP

    def _run(idx):
        acf = np.mean([_acf_fft(c[idx], ml) for c in comps], axis=0)
        return acf, _running_integral(acf, dt_s, scale)

    acf_full, full = _run(slice(None))
    bl = n // max(nblock, 1)
    blocks = []
    for b in range(nblock):
        if bl < 2:
            break
        mlb = min(bl - 1, ml)
        yb = _run(slice(b * bl, (b + 1) * bl))[1][:mlb]
        blocks.append(np.interp(lags, np.arange(mlb) * dt_ps, yb))
    band = np.std(blocks, axis=0) if blocks else np.zeros_like(full)

    return {
        "run": run_dir.name,
        "lags_ps": lags,
        "running_cp": full,
        "band": band,
        "acf": acf_full,
        "dt_ps": dt_ps,
        "n_samples": n,
        "eq_time_ps": eq_time_ps,
        "t_total_ns": t_total_ns,
        "T_mean_K": float(df["T"].mean()),
        "V_m3": V_m3,
    }


def _eta_model(t, b1, b2, t1, t2):
    """Running GK integral of a two-exponential stress ACF (Zhang-Maginn TDM):
    ACF = a1 e^{-t/tau1} + a2 e^{-t/tau2}  =>  eta(t) = sum_i b_i (1 - e^{-t/tau_i}),
    b_i = a_i tau_i * V/(kB T);  eta_inf = b1 + b2."""
    return b1 * (1.0 - np.exp(-t / t1)) + b2 * (1.0 - np.exp(-t / t2))


def _smooth(x, w):
    return x if w <= 1 else np.convolve(x, np.ones(w) / w, mode="same")


def fit_eta_tdm(lags, full, band, t_floor_ps: float = 1.0, rel_noise_cut: float = 0.4,
                smooth_ps: float = 5.0, se_rel_max: float = 0.5) -> dict:
    """Time-decomposition estimate of eta (Zhang, Otani & Maginn, JCTC 2015).

    Fit the running integral over [t_floor, t_cut] -- t_cut = where the smoothed
    block band first exceeds `rel_noise_cut` * value (their 40% rule) -- with the
    two-exponential saturation model, weighting each point by 1/band so the
    random-walk tail cannot drag the fit. Reports eta_inf = b1 + b2 with the fit
    SE combined with the block band at the plateau-onset lag, and converged=True
    only if the data actually reach >= 90% of eta_inf inside the trusted window
    (i.e. eta_inf is a read-off, not an extrapolation) with rel. err < se_rel_max.
    """
    from scipy.optimize import curve_fit
    dt = lags[1] - lags[0]
    n = len(full)
    i0 = max(1, int(t_floor_ps / dt))
    sm = max(1, int(smooth_ps / dt))
    band_s, val_s = _smooth(band, sm), np.abs(_smooth(full, sm))
    bad = np.where((np.arange(n) > i0) & (band_s > rel_noise_cut * np.maximum(val_s, 1e-12)))[0]
    i_cut = int(bad[0]) if len(bad) else n - 1
    t_cut = float(lags[i_cut])

    idx = np.unique(np.round(np.geomspace(i0, i_cut, 400)).astype(int))
    t, y = lags[idx], full[idx]
    sig = np.maximum(band_s[idx], 1e-3 * max(float(np.abs(y).max()), 1e-12))
    y_end = max(float(full[i_cut]), 1e-6)
    p0 = [0.5 * y_end, 0.6 * y_end, 2.0, max(0.2 * t_cut, 20.0)]
    bounds = ([0.0, 0.0, 0.05, 1.0], [np.inf, np.inf, t_cut, 10.0 * t_cut])
    try:
        popt, pcov = curve_fit(_eta_model, t, y, p0=p0, sigma=sig,
                               bounds=bounds, maxfev=50000)
    except Exception:
        return dict(est=float(full[i_cut]), err=float(band[i_cut]), conv=False,
                    t_lo=float(lags[i0]), t_hi=t_cut, fit_curve=None,
                    tau_slow_ps=np.nan, frac_recovered=np.nan)
    eta_inf = float(popt[0] + popt[1])
    J = np.array([1.0, 1.0, 0.0, 0.0])
    fit_se = float(np.sqrt(J @ pcov @ J))
    tau_slow = float(max(popt[2], popt[3]))
    frac = float(_eta_model(t_cut, *popt) / max(eta_inf, 1e-12))
    i_pl = min(i_cut, int(3.0 * tau_slow / dt))
    err = float(np.hypot(fit_se, band_s[i_pl]))
    conv = bool(frac >= 0.90 and err / max(eta_inf, 1e-12) < se_rel_max)
    return dict(est=eta_inf, err=err, conv=conv, t_lo=float(lags[i0]), t_hi=t_cut,
                fit_curve=(t, _eta_model(t, *popt)), tau_slow_ps=tau_slow,
                frac_recovered=frac)

