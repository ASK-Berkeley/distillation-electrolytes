#!/usr/bin/env python
"""Einstein self-diffusivity (Yeh-Hummer corrected) for the transport_* NVT runs.

Sibling of ``viscosity_gk_replicas.py`` and structured the same way. For every
transport run this streams the trajectory once, builds the time-origin MSD per
species (cation / anion / solvent) in the true system-COM frame, and Einstein-fits
D = slope/6 with BOTH fit methods on that same curve: the fixed
``[tau_min_fit_ps, tau_max_fit_frac * window]`` window (primary ``D_*`` columns)
and the per-species Fickian beta~=1 window (``D_*_fick_*`` columns, incl. beta +
converged flags). Pick either family downstream; no re-analysis needed.

Run tree (auto-detected, ``iter_run_dirs``): the default is the 4-replica set
``PAINN_NVT_REPLICAS`` (``nvt_replica_<N>/<system>/nvt_transport_<conc>_298K_20ns_1ps/``,
298 K only -- off-temperature runs are skipped); the legacy exp04 ``r0``-only layout
still works if passed via ``--experiment-root``.

OUTPUT (default): results go ONLY to the external ``--out-root``
(``data/painn_transport_observables_replicas`` by default), leaving the run tree
untouched -- ``diffusivity<suffix>/{per_run/<box>/r<k>/, *_master.csv, *_by_box.csv,
*_config.yaml, figure_*.png}``. Pass ``--write-internal`` to ALSO write the per-run
``<run_dir>/observables/`` and aggregate ``<exp>/transport/`` outputs described below.

The finite-size (Yeh-Hummer) correction ``dD = kB T xi / (6 pi eta L)`` is applied
TWO ways so its sensitivity to the viscosity input is captured:
  * ``D_inf_exp`` -- eta = experimental shear viscosity (298 K exp table).
  * ``D_inf_sim`` -- eta = this run's own simulation GK viscosity (read from the
                     ``transport/gk_viscosity/viscosity_gk_master.csv`` produced by
                     ``viscosity_gk_replicas.py`` -- run that first).

Per run it writes into ``<run_dir>/observables/diffusivity_<N>ns/`` (N = simulated ns):
  * ``diffusivity.csv`` -- one row: D_PBC, D_inf_sim, D_inf_exp, both etas + box,
                           the fit window, and the experimental D for reference.
  * ``msd.csv``         -- tau_ps + the three species MSDs (full curve).
  * ``diffusivity_msd.png`` -- per-run log-log MSD panel with Einstein fits.
  * merges a ``diffusivity`` block into the existing ``summary.json``.

Then it rolls everything up into ``<exp>/transport/diffusivity/``:
  * ``diffusivity_master.csv``     -- one row per run.
  * ``diffusivity_config.yaml``    -- settings + provenance sidecar.
  * ``figure_transport_diff_msd16.png``  -- the 16-panel MSD + fit grid.
  * ``figure_transport_diff_parity.png`` -- the 2x3 parity (rows: YH exp-eta /
    YH sim-eta; cols: cation / anion / solvent).

CPU only; conda env ``fennol``. Run from the repo root as a module (the
package uses relative imports):

    python -m src.analysis.observables.transport.diffusivity_replicas --dry-run
    python -m src.analysis.observables.transport.diffusivity_replicas --jobs 8
"""
from __future__ import annotations

import os

# Keep per-process BLAS/FFT single-threaded so the ProcessPool doesn't oversubscribe.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml


from .. import io, paths
from ..paths import git_sha
from ..diffusivity import compute_diffusivity
from ..systems import SYSTEMS
from ..units import a2ps_to_m2s, yeh_hummer_correction_a2ps

from . import time_presets as TP

# ── paths / constants ────────────────────────────────────────────────────────
DEFAULT_ROOT = paths.PAINN_NVT_REPLICAS
DEFAULT_EXP_CSV = paths.GROUND_TRUTH / "298p2K_exp_values.csv"
T_K = 298.0

# Two run-tree layouts are supported transparently (see iter_run_dirs):
#   * legacy exp04 (r0-only): <system>/transport_<conc>_298K_r0/
#   * 4-replica tree:         nvt_replica_<N>/<system>/nvt_transport_<conc>_298K_20ns_1ps/
RUN_GLOB = "transport_*_298K_r0"
RUN_RE = re.compile(r"^transport_(?P<conc>.+?)_(?P<temp>\d+)K_r(?P<rep>\d+)$")
REPLICA_DIR_RE = re.compile(r"^nvt_replica_(?P<rep>\d+)$")
REPLICA_RUN_GLOB = "nvt_transport_*"
REPLICA_RUN_RE = re.compile(r"^nvt_transport_(?P<conc>.+?)_(?P<temp>\d+)K_")

# Two fit-window methods. This driver computes BOTH on the same MSD curve for
# every run (the MSD pass is the expensive part; a second Einstein window is
# free) and writes both column families to the master, so plotting/reporting
# can pick either without re-analysis:
#
#  * "fixed"   -- one hard-coded window [tau_min_fit_ps, tau_max_fit_frac*window]
#                 applied to every species (the original driver's behaviour).
#                 Fills the primary (unsuffixed) D_* columns.
#  * "fickian" -- a per-species, data-driven window: locate the contiguous
#                 log-log region where the local slope beta = d ln MSD / d ln tau
#                 is within beta_tol of 1 (the genuinely-Fickian / diffusive
#                 regime) and Einstein-fit there. Reports per-species beta,
#                 window, and a converged flag. This is the more robust estimator
#                 for the sparse 0.1 M boxes, where a fixed window fit sits in
#                 the noisy few-ion long-lag tail (replica CV ~45% vs ~14%).
#                 Fills the D_*_fick_* columns. (Reconstructs the trusted exp01
#                 pipeline.)
#
# The single MSD is computed with SETTINGS_FIXED's eq/stride (2 ns cut, every
# frame) and both windows are fitted on it. SETTINGS_FICKIAN therefore holds only
# the window-search parameters; it has no equilibration cut or stride of its own.
SETTINGS_FIXED = dict(
    method="fixed",
    eq_time_ps=2000.0,
    tau_min_fit_ps=1000.0,
    tau_max_fit_ps=None,
    tau_max_fit_frac=0.5,
    target_frames=50000,   # >= available => stride 1 => use every frame
    msd_method="fft",
    temperature_k=T_K,
)

SETTINGS_FICKIAN = dict(
    method="fickian",
    t_floor_ps=5.0,        # ignore the ballistic/caging rise below this lag
    t_max_frac=0.4,        # search the Fickian window only within 0-40% of the lag axis
    beta_tol=0.15,         # |beta - 1| <= this counts as diffusive
    min_pts=12,            # min log-spaced samples in a valid beta~=1 run
    n_log=200,             # log-spaced samples used to estimate beta(tau)
    smooth_win=5,          # smoothing (samples) of ln MSD before differencing
)

# system_id token -> experimental-CSV name
_SALT = {"napf6": "NaPF6", "naotf": "NaOTf", "lipf6": "LiPF6"}
_SOLVENT = {"dme": "DME", "pc": "PC", "diglyme": "DEGDME", "tgdme": "TEGDME"}
_SAMPLE_RE = re.compile(r"^([\d.]+)\s*M\s+(\S+)\s+in\s+(\S+)$")

# plot encoding (self-contained, matches viscosity_gk_replicas.py)
SOLV_COLOR = {"dme": "#E68310", "diglyme": "#11A579", "pc": "#E73F74", "tgdme": "#3969AC"}
SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}
SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}
SALT_MARKER = {"napf6": "o", "naotf": "s", "lipf6": "^"}

# species encoding for the MSD panels (label / colour / exp-D CSV column key)
SPECIES = [
    ("cation",  r"$D_+$", "#1B7837", "D_exp_cation"),
    ("anion",   r"$D_-$", "#762A83", "D_exp_anion"),
    ("solvent", r"$D_0$", "#2166AC", "D_exp_solvent"),
]
SPEC_COLOR = {sp: c for sp, _, c, _ in SPECIES}


def conc_to_molar(tok: str) -> float:
    """'0_1M'->0.1, '0_5M'->0.5, '1M'->1.0."""
    m = re.match(r"^(\d+)_(\d+)M$", tok)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.match(r"^(\d+)M$", tok)
    if m:
        return float(m.group(1))
    raise ValueError(f"Cannot parse concentration token {tok!r}")


def is_replica_tree(root) -> bool:
    """True if ``root`` is the 4-replica tree (has ``nvt_replica_<N>`` subdirs)."""
    return any(Path(root).glob("nvt_replica_*"))


def iter_run_dirs(root):
    """Yield ``(system_id, salt, solvent, conc_tok, conc_M, temp_k, replica, run_dir)``
    for every transport run under ``root``, transparently handling both the legacy
    exp04 (r0-only) layout and the 4-replica ``nvt_replica_<N>`` tree. Shared by all
    three transport drivers (conductivity imports this; viscosity mirrors it)."""
    root = Path(paths.require(root))   # readable error if the tree is on another machine
    if is_replica_tree(root):
        for rep_dir in sorted((p for p in root.glob("nvt_replica_*") if p.is_dir()),
                              key=lambda p: p.name):
            rm = REPLICA_DIR_RE.match(rep_dir.name)
            if not rm:
                continue
            replica = int(rm.group("rep"))
            for sys_dir in sorted(p for p in rep_dir.iterdir() if p.is_dir()):
                toks = sys_dir.name.split("_", 1)
                if len(toks) != 2 or toks[0] not in _SALT or toks[1] not in _SOLVENT:
                    continue
                for rd in sorted(sys_dir.glob(REPLICA_RUN_GLOB)):
                    m = REPLICA_RUN_RE.match(rd.name)
                    if not m:
                        continue
                    conc_tok = m.group("conc")
                    yield (sys_dir.name, toks[0], toks[1], conc_tok,
                           conc_to_molar(conc_tok), float(m.group("temp")), replica, rd)
    else:
        for sys_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            toks = sys_dir.name.split("_", 1)
            if len(toks) != 2 or toks[0] not in _SALT or toks[1] not in _SOLVENT:
                continue
            for rd in sorted(sys_dir.glob(RUN_GLOB)):
                m = RUN_RE.match(rd.name)
                if not m:
                    continue
                conc_tok = m.group("conc")
                yield (sys_dir.name, toks[0], toks[1], conc_tok,
                       conc_to_molar(conc_tok), float(m.group("temp")),
                       int(m.group("rep")), rd)


def _f(x):
    """Permissive float: None for missing values (None, empty, '---', NaN --
    a blank CSV cell arrives from pandas as float nan, which must not leak
    through as a 'present' value)."""
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "---", "nan", "NaN"):
        return None
    v = float(s)
    return v if np.isfinite(v) else None


def load_exp_table(csv_path: Path) -> dict:
    """(salt_token, solvent_token, conc_M) -> dict(eta [mPa*s], D_exp_* [1e-10 m^2/s])."""
    inv_salt = {v: k for k, v in _SALT.items()}
    inv_solv = {v: k for k, v in _SOLVENT.items()}
    out = {}
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        m = _SAMPLE_RE.match(str(r["Sample"]).strip())
        if not m:
            continue
        conc, salt, solv = float(m.group(1)), inv_salt.get(m.group(2)), inv_solv.get(m.group(3))
        if salt and solv:
            out[(salt, solv, conc)] = dict(
                eta=_f(r.get("Dynamic Viscosity (mPa*s)")),
                D_exp_cation=_f(r.get("D*_+ (x 10^-10 m^2/s)")),
                D_exp_anion=_f(r.get("D*_- (x 10^-10 m^2/s)")),
                D_exp_solvent=_f(r.get("D*_0 (x 10^-10 m^2/s)")),
            )
    return out


def load_sim_viscosity(out_root: Path, suffix: str = "") -> dict:
    """(system_id, conc_label) -> simulation GK eta [cP], read from the external
    viscosity output produced by viscosity_gk_replicas.py under
    ``OUT_ROOT/viscosity<suffix>/``. Prefers the per-box mean
    (``viscosity_gk_by_box.csv``) so a box's YH correction uses one eta across its
    replicas; falls back to the per-run master. Empty (with a warning) if absent --
    the sim-eta YH column is then NaN. ``suffix`` selects the matching viscosity
    folder."""
    vdir = Path(out_root) / f"viscosity{suffix}"
    by_box = vdir / "viscosity_gk_by_box.csv"
    if by_box.is_file():
        df = pd.read_csv(by_box)
        return {(str(r["system_id"]), str(r["conc_label"])): _f(r["eta_gk_mean_cP"])
                for _, r in df.iterrows()}
    master = vdir / "viscosity_gk_master.csv"
    if master.is_file():
        df = pd.read_csv(master)
        return {(str(r["system_id"]), str(r["conc_label"])): _f(r["eta_gk_cP"])
                for _, r in df.iterrows()}
    print(f"[warn] no sim-viscosity output under {vdir}; sim-eta YH column will be NaN.",
          flush=True)
    return {}


def discover_runs(root: Path, exp: dict, sim_eta: dict) -> tuple[list, list]:
    """Return (runs, skipped) for every transport run under `root` (both layouts)."""
    runs, skipped = [], []
    for system_id, salt, solv, conc_tok, conc_m, temp_k, replica, rd in iter_run_dirs(root):
        if system_id not in SYSTEMS:
            continue
        # This is a 298 K pipeline (exp tables, YH prefactor and box labelling all key
        # off 298 K); skip any off-temperature runs (e.g. the lipf6_dme 0.5 M 273/323 K
        # series) rather than silently mislabelling them as 298 K.
        if int(round(temp_k)) != int(round(T_K)):
            skipped.append(dict(system_id=system_id, conc_label=conc_tok, run_name=rd.name,
                                skip_reason=f"non-{int(round(T_K))}K (T={int(round(temp_k))}K)"))
            continue
        ref = exp.get((salt, solv, conc_m), {})
        rec = dict(
            system_id=system_id, salt=salt, solvent=solv,
            conc_label=conc_tok, conc_M=conc_m, replica=replica,
            run_name=rd.name, run_dir=str(rd),
            box=f"{system_id}_{conc_tok}_{int(round(temp_k))}",
            eta_exp=ref.get("eta"), eta_sim=sim_eta.get((system_id, conc_tok)),
            D_exp_cation=ref.get("D_exp_cation"),
            D_exp_anion=ref.get("D_exp_anion"),
            D_exp_solvent=ref.get("D_exp_solvent"),
        )
        try:
            io.resolve_traj(rd)
        except FileNotFoundError:
            rec["skip_reason"] = "no trajectory"
            skipped.append(rec)
        else:
            runs.append(rec)
    return runs, skipped


# ── per-run compute + output ─────────────────────────────────────────────────
def _decimate(n: int, target: int = 2000) -> np.ndarray:
    """Indices that thin `n` points down to ~`target`, keeping endpoints."""
    if n <= target:
        return np.arange(n)
    return np.unique(np.r_[np.arange(0, n, n // target), n - 1])


def _yh_dD_1e10(eta_cp, box_a):
    """Yeh-Hummer dD in 1e-10 m^2/s for a viscosity (cP) and box edge (A)."""
    if eta_cp is None or box_a is None:
        return np.nan
    if not (np.isfinite(eta_cp) and eta_cp > 0 and np.isfinite(box_a) and box_a > 0):
        return np.nan
    return a2ps_to_m2s(yeh_hummer_correction_a2ps(T_K, eta_cp, box_a)) * 1e10


def _merge_summary(obs_dir: Path, block: dict) -> None:
    p = obs_dir / "summary.json"
    summary = {}
    if p.is_file():
        try:
            summary = json.loads(p.read_text())
        except Exception:
            summary = {}
    summary["diffusivity"] = block
    p.write_text(json.dumps(summary, indent=2))


def _per_run_external_dir(settings: dict, rec: dict) -> Path:
    """External per-run target: OUT_ROOT/diffusivity<suffix>/per_run/<box>/r<k>/
    (the OPLS-mirrored layout)."""
    return (settings["out_root"] / f"diffusivity{settings['out_suffix']}"
            / "per_run" / rec["box"] / f"r{rec['replica']}")


def _mirror_per_run(obs_dir: Path, rec: dict, settings: dict) -> None:
    """Copy this run's per-run outputs (written internally into <run_dir>/observables/)
    out into the external per_run/<box>/r<k>/ mirror. Only used with --write-internal;
    without it the files are written straight into the external dir (no copy)."""
    mirror_dir = _per_run_external_dir(settings, rec)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for f in obs_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, mirror_dir / f.name)


_SPECIES_MSD = (("cation", "msd_cat"), ("anion", "msd_anion"), ("solvent", "msd_solvent"))


def _einstein(tau, msd, lo, hi):
    """Einstein slope D = slope/6 (A^2/ps) of MSD vs tau over [lo, hi].
    Forgiving (no minimum-point exception); returns (D, slope, intercept)."""
    sel = (tau >= lo) & (tau <= hi) & (np.asarray(msd) > 0)
    x, y = tau[sel], np.asarray(msd)[sel]
    if x.size < 3:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return slope / 6.0, float(slope), float(intercept)


def _smooth(y, w):
    # Zero-padded ("same") moving average, kept as used for the paper's numbers.
    # It pulls the last few log-spaced samples of ln MSD toward zero, so beta is
    # unusable on the last 3 of the n_log=200 samples. With t_max_frac=0.4 over the
    # 18 ns analysed window, Fickian windows therefore end by ~6.45 ns, i.e. ~36% of
    # the maximum lag; Methods states the search range as 5 ps to 36%.
    if w <= 1 or y.size < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def _longest_true_run(mask):
    """(start, end) inclusive indices of the longest contiguous True run, or
    (None, None) if there are none."""
    best_len = 0
    best = (None, None)
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 > best_len:
                best_len = j - i + 1
                best = (i, j)
            i = j + 1
        else:
            i += 1
    return best


def _fickian_window(tau, msd, s):
    """Locate the diffusive (beta~=1) window per one species' MSD and Einstein-fit
    it. Returns (tlo, thi, slope, intercept, D_a2ps, beta_win, converged)."""
    tau = np.asarray(tau)
    msd = np.asarray(msd)
    t_max = s["t_max_frac"] * float(tau[-1]) if len(tau) else s["t_floor_ps"]
    sel = (tau > 0) & (msd > 0) & (tau >= s["t_floor_ps"]) & (tau <= t_max)
    t, y = tau[sel], msd[sel]
    if t.size < s["min_pts"]:
        lo, hi = (float(t[0]), float(t[-1])) if t.size >= 2 else (s["t_floor_ps"], t_max)
        D, slope, ic = _einstein(tau, msd, lo, hi)
        return lo, hi, slope, ic, D, np.nan, False

    # log-spaced samples -> local log-log slope beta(tau)
    ts = np.geomspace(t[0], t[-1], min(s["n_log"], t.size))
    idx = np.unique(np.clip(np.searchsorted(t, ts), 0, t.size - 1))
    tt, yy = t[idx], y[idx]
    lt, ly = np.log(tt), _smooth(np.log(yy), s["smooth_win"])
    beta = np.gradient(ly, lt)
    good = np.abs(beta - 1.0) <= s["beta_tol"]

    lo_i, hi_i = _longest_true_run(good)
    converged = lo_i is not None and (hi_i - lo_i + 1) >= s["min_pts"]
    if not converged:
        # No sufficiently-long beta~=1 run: fall back to the min_pts-wide block
        # whose mean |beta-1| is smallest, so we still report a best-effort D.
        w = min(s["min_pts"], len(beta))
        c = np.convolve(np.abs(beta - 1.0), np.ones(w) / w, mode="valid")
        start = int(np.argmin(c)) if len(c) else 0
        lo_i, hi_i = start, min(start + w - 1, len(beta) - 1)

    tlo, thi = float(tt[lo_i]), float(tt[hi_i])
    D, slope, ic = _einstein(tau, msd, tlo, thi)
    beta_win = float(np.mean(beta[lo_i:hi_i + 1]))
    return tlo, thi, slope, ic, D, beta_win, bool(converged)


def _compute_fits(dres, settings):
    """Dispatch to the fixed or Fickian window per species. Returns
    (fits, fmeta) where fits[sp]=(tlo,thi,slope,intercept,D_a2ps) for the plot/D
    and fmeta[sp]=(beta, converged)."""
    fits, fmeta = {}, {}
    for sp, attr in _SPECIES_MSD:
        msd = getattr(dres, attr)
        if msd is None:
            fits[sp] = None
            fmeta[sp] = (None, None)
            continue
        if settings["method"] == "fickian":
            tlo, thi, slope, ic, D, beta, conv = _fickian_window(dres.tau_ps, msd, settings)
        else:
            tlo, thi = dres.tau_min_fit_ps, dres.tau_max_fit_ps
            D, slope, ic = _einstein(dres.tau_ps, msd, tlo, thi)
            beta, conv = None, True
        fits[sp] = (tlo, thi, float(slope), float(ic), float(D))
        fmeta[sp] = (beta, conv)
    return fits, fmeta


def _plot_msd_panel(ax, tau, msds, fits, title, fits2=None):
    """Log-log MSD panel: the 3 species curves + their Einstein fit lines, with
    the fit window shaded. ``msds``/``fits`` keyed by species name. ``fits2``
    (optional) draws a second fit set (the Fickian windows) in orange dash-dot;
    only the ``fits`` windows are shaded."""
    # tau=0 is MSD≡0 (comes out as ±1e-12 float noise) and can't be shown on a
    # log axis, so require tau>0 everywhere — otherwise a species whose zero-lag
    # value rounds to a tiny *positive* number sneaks a ~1e-12 point onto the
    # plot and blows the log y-axis down 12 decades.
    tpos = np.asarray(tau) > 0
    # Fit windows are per-species (identical for the "fixed" method, distinct for
    # "fickian"); bound the y-range on data up to the widest window end, excluding
    # the noisy far tail (few time origins => msd_fft can spike/go negative there).
    wins = [f for f in fits.values() if f is not None]
    tlo_all = min((f[0] for f in wins), default=None)
    thi_all = max((f[1] for f in wins), default=None)
    t_ylim = thi_all if thi_all is not None else (tau.max() if len(tau) else None)
    ymin = ymax = None
    for sp, splab, color, _ in SPECIES:
        msd = msds.get(sp)
        if msd is None:
            continue
        m = np.asarray(msd)
        pos = (m > 0) & tpos
        core = pos & (tau <= t_ylim) if t_ylim is not None else pos
        if core.any():
            lo, hi = float(m[core].min()), float(m[core].max())
            ymin = lo if ymin is None else min(ymin, lo)
            ymax = hi if ymax is None else max(ymax, hi)
        ax.plot(tau[pos], m[pos], color=color, lw=1.4, label=splab)
        for fset, fcolor, fls in ((fits, "0.15", "--"), (fits2, "#D95F02", "-.")):
            fit = fset.get(sp) if fset is not None else None
            if fit is not None:
                tmin, tmax, slope, intercept, _ = fit
                xf = np.linspace(tmin, tmax, 50)
                yf = slope * xf + intercept
                ax.plot(xf[yf > 0], yf[yf > 0], color=fcolor, lw=1.1, ls=fls)
    if tlo_all is not None:
        ax.axvspan(tlo_all, thi_all, color="0.6", alpha=0.15)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=8)
    ax.set_ylabel(r"MSD [$\mathrm{\AA}^2$]", fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7); ax.grid(True, which="both", ls=":", alpha=0.4)
    # Pin y to the MSD data range so a fit line dipping toward zero near the
    # window's low end (small/negative intercept) can't blow out the log axis.
    # Set last: set_yscale("log") re-autoscales, so this must come after it.
    if ymin is not None and ymax > ymin > 0:
        ax.set_ylim(ymin * 0.5, ymax * 2.0)


def analyze_one(rec: dict, settings: dict, plots: bool) -> dict:
    """Compute MSD + Einstein D for one run, write per-run outputs, return a
    light-weight result (scalar row + decimated MSD/fits for the aggregate figure)."""
    rd = Path(rec["run_dir"])
    traj = io.resolve_traj(rd)
    dt, dt_source = io.resolve_dt_ps(rd, traj)
    n_total = io.estimate_n_frames(traj)
    # --time 1ns/5ns: analyze only the first T of the run (convergence study).
    max_frames = TP.cap_frames(settings.get("time", TP.DEFAULT_TIME), dt)
    if max_frames is not None:
        n_total = min(n_total, max_frames)
    t_total_ns = n_total * dt / 1000.0

    spec = SYSTEMS[rec["system_id"]]
    dres = compute_diffusivity(
        traj, spec, dt,
        eq_time_ps=settings["eq_time_ps"], tau_min_fit_ps=settings["tau_min_fit_ps"],
        tau_max_fit_ps=settings["tau_max_fit_ps"], tau_max_fit_frac=settings["tau_max_fit_frac"],
        target_frames=settings["target_frames"], method=settings["msd_method"],
        yeh_hummer=False, max_frames=max_frames,
    )
    fits, fmeta = _compute_fits(dres, settings)                 # fixed window
    fick_fits, fick_meta = _compute_fits(dres, settings["fick"])  # beta~=1 window
    box_a = dres.box_length_a
    dD_sim = _yh_dD_1e10(rec["eta_sim"], box_a)
    dD_exp = _yh_dD_1e10(rec["eta_exp"], box_a)

    # Per-run output target. With --write-internal: the run's own
    # observables/diffusivity_<N>ns/ subfolder (kept separate from the viscosity /
    # conductivity observables in the run). Default: straight into the external
    # per_run/<box>/r<k>/ mirror, leaving the (read-only) run tree untouched.
    if settings["write_internal"]:
        obs_dir = rd / "observables" / f"diffusivity_{round(t_total_ns)}ns{settings['out_suffix']}"
    else:
        obs_dir = _per_run_external_dir(settings, rec)
    obs_dir.mkdir(parents=True, exist_ok=True)

    # D_PBC per species comes from the (method-dependent) per-species Einstein fit.
    D_pbc = {sp: (None if fits[sp] is None else a2ps_to_m2s(fits[sp][4]) * 1e10)
             for sp, _, _, _ in SPECIES}
    row = {
        "system_id": rec["system_id"], "salt": rec["salt"], "solvent": rec["solvent"],
        "conc_label": rec["conc_label"], "conc_M": rec["conc_M"], "replica": rec["replica"],
        "run_name": rec["run_name"], "fit_method": "fixed+fickian",
        "t_total_ns": round(t_total_ns, 3), "dt_ps": round(dt, 5), "dt_source": dt_source,
        "frames_used": dres.frames_used, "effective_dt_ps": round(dres.effective_dt_ps, 5),
        "analysis_window_ps": round(dres.analysis_window_ps, 2),
        "eq_time_ps": settings["eq_time_ps"],
        "msd_method": dres.method,
        "box_L_A": round(box_a, 3),
        "eta_sim_cP": rec["eta_sim"], "eta_exp_cP": rec["eta_exp"],
        "yh_dD_sim_1e10": None if np.isnan(dD_sim) else round(dD_sim, 4),
        "yh_dD_exp_1e10": None if np.isnan(dD_exp) else round(dD_exp, 4),
    }
    all_conv = True
    for sp, _, _, expkey in SPECIES:
        d = D_pbc[sp]
        beta, conv = fmeta[sp]
        fit = fits[sp]
        row[f"D_{sp}_pbc_1e10"] = None if d is None else round(d, 4)
        row[f"D_{sp}_inf_sim_1e10"] = (None if d is None or np.isnan(dD_sim)
                                       else round(d + dD_sim, 4))
        row[f"D_{sp}_inf_exp_1e10"] = (None if d is None or np.isnan(dD_exp)
                                       else round(d + dD_exp, 4))
        row[f"D_{sp}_exp_1e10"] = rec[expkey]
        row[f"D_{sp}_t_lo_ps"] = None if fit is None else round(fit[0], 1)
        row[f"D_{sp}_t_hi_ps"] = None if fit is None else round(fit[1], 1)
        row[f"D_{sp}_beta"] = None if beta is None or (isinstance(beta, float) and np.isnan(beta)) else round(beta, 3)
        row[f"D_{sp}_converged"] = None if conv is None else bool(conv)
        if conv is False:
            all_conv = False
    row["all_converged"] = all_conv

    # Fickian (beta~=1 window) column family, from the SAME MSD curve.
    all_conv_fick = True
    for sp, _, _, _ in SPECIES:
        fit = fick_fits[sp]
        beta, conv = fick_meta[sp]
        d = None if fit is None else a2ps_to_m2s(fit[4]) * 1e10
        row[f"D_{sp}_fick_pbc_1e10"] = None if d is None else round(d, 4)
        row[f"D_{sp}_fick_inf_sim_1e10"] = (None if d is None or np.isnan(dD_sim)
                                            else round(d + dD_sim, 4))
        row[f"D_{sp}_fick_inf_exp_1e10"] = (None if d is None or np.isnan(dD_exp)
                                            else round(d + dD_exp, 4))
        row[f"D_{sp}_fick_t_lo_ps"] = None if fit is None else round(fit[0], 1)
        row[f"D_{sp}_fick_t_hi_ps"] = None if fit is None else round(fit[1], 1)
        row[f"D_{sp}_fick_beta"] = (None if beta is None or (isinstance(beta, float) and np.isnan(beta))
                                    else round(beta, 3))
        row[f"D_{sp}_fick_converged"] = None if conv is None else bool(conv)
        if conv is False:
            all_conv_fick = False
    row["all_converged_fick"] = all_conv_fick

    # per-run flat outputs
    pd.DataFrame([row]).to_csv(obs_dir / "diffusivity.csv", index=False)
    msd_out = {"tau_ps": dres.tau_ps, "msd_cation": dres.msd_cat}
    msd_out["msd_anion"] = dres.msd_anion if dres.msd_anion is not None else np.nan
    msd_out["msd_solvent"] = dres.msd_solvent if dres.msd_solvent is not None else np.nan
    pd.DataFrame(msd_out).to_csv(obs_dir / "msd.csv", index=False)
    _merge_summary(obs_dir, {**{k: row[k] for k in (
        "fit_method", "t_total_ns", "frames_used", "effective_dt_ps", "analysis_window_ps",
        "eq_time_ps", "msd_method", "box_L_A", "all_converged", "all_converged_fick",
        "eta_sim_cP", "eta_exp_cP", "yh_dD_sim_1e10", "yh_dD_exp_1e10")},
        **{k: row[k] for k in row if k.startswith("D_")},
        "code_version": git_sha(), "generated": settings["generated"]})

    msds = {"cation": dres.msd_cat, "anion": dres.msd_anion, "solvent": dres.msd_solvent}
    if plots:
        fig, ax = plt.subplots(figsize=(5, 4))
        _plot_msd_panel(ax, dres.tau_ps, msds, fits,
                        f"{rec['run_name']} ({round(t_total_ns)} ns)", fits2=fick_fits)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout(); fig.savefig(obs_dir / "diffusivity_msd.png", dpi=130,
                                        facecolor="white")
        plt.close(fig)

    if settings["write_internal"]:
        _mirror_per_run(obs_dir, rec, settings)
    # else obs_dir already *is* the external per_run target; nothing to copy.

    # decimated MSD for the aggregate 16-panel figure (keep IPC light)
    d2 = _decimate(len(dres.tau_ps), 2000)
    plot_msds = {sp: (None if m is None else np.asarray(m)[d2]) for sp, m in msds.items()}
    return {"rec": rec, "row": row,
            "plot": {"tau": dres.tau_ps[d2], "msds": plot_msds, "fits": fits,
                     "fits2": fick_fits, "t_total_ns": t_total_ns}}


# ── aggregate figures ────────────────────────────────────────────────────────
def msd_grid(results: list, out: Path, exp_label: str):
    results = sorted(results, key=lambda r: (r["rec"]["salt"], r["rec"]["solvent"],
                                             r["rec"]["conc_M"]))
    n = len(results)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, res in zip(axes, results):
        rec, pl = res["rec"], res["plot"]
        title = (f"{SALT_LABEL.get(rec['salt'], rec['salt'])} {SOLV_LABEL[rec['solvent']]} "
                 f"{rec['conc_label'].replace('_', '.')} ({round(pl['t_total_ns'])} ns)")
        _plot_msd_panel(ax, pl["tau"], pl["msds"], pl["fits"], title,
                        fits2=pl.get("fits2"))
    for ax in axes[n:]:
        ax.axis("off")
    handles = [Line2D([], [], color=c, lw=1.6, label=lab)
               for _, lab, c, _ in SPECIES]
    handles.append(Line2D([], [], color="0.15", lw=1.2, ls="--", label="fixed-window fit"))
    handles.append(Line2D([], [], color="#D95F02", lw=1.2, ls="-.", label=r"Fickian ($\beta$≈1) fit"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Self-diffusivity — time-origin MSD (log-log) with fixed-window Einstein fit "
                 "(black dashed, window grey) and Fickian $\\beta$≈1 fit (orange dash-dot)"
                 f"  —  {exp_label}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print("saved ->", out, flush=True)


def _fin(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _parity_panel(ax, df, xcol, ycol, title, conv_col=None, err_col=None):
    """Simulation D_inf vs experiment (PFG-NMR), log-log. Solvent = color,
    salt = marker, 0.1 M = hollow; factor-2 band. Points whose per-species fit did
    not converge (``conv_col`` False) are marked 'x' and excluded from the MAPD.
    ``err_col`` (optional) draws a +/- error bar per point (replica s.d.)."""
    vals = [v for _, r in df.iterrows()
            for v in (_fin(r[xcol]), _fin(r[ycol])) if np.isfinite(v) and v > 0]
    if not vals:
        ax.set_title(title + " (no data)"); ax.set_xscale("log"); ax.set_yscale("log")
        return
    lo, hi = min(vals) / 1.8, max(vals) * 1.8
    ax.fill_between([lo, hi], [lo/2, hi/2], [lo*2, hi*2], color="0.8", alpha=0.3, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.45", ls="--", lw=1.3, zorder=1)
    devs = []
    for _, r in df.iterrows():
        xe, ys = _fin(r[xcol]), _fin(r[ycol])
        if not (np.isfinite(xe) and np.isfinite(ys) and xe > 0 and ys > 0):
            continue
        c = SOLV_COLOR[r["solvent"]]
        filled = float(r["conc_M"]) != 0.1
        if err_col is not None:
            es = _fin(r.get(err_col))
            if np.isfinite(es) and es > 0:
                ax.errorbar(xe, ys, yerr=es, fmt="none", ecolor=c, elinewidth=1.2,
                            capsize=3, alpha=0.65, zorder=2)
        ax.scatter(xe, ys, s=130, marker=SALT_MARKER.get(r["salt"], "o"),
                   facecolor=c if filled else "white", edgecolor=c, linewidths=1.9, zorder=3)
        conv = True if conv_col is None else bool(r.get(conv_col, True))
        if conv:
            devs.append(abs(ys - xe) / xe)
        else:
            ax.scatter(xe, ys, s=70, marker="x", color="0.15", linewidths=1.6, zorder=4)
    txt = (f"MAPD = {100*np.mean(devs):.0f}%\n(conv n = {len(devs)}/{len(df)})"
           if devs else f"none converged (n=0/{len(df)})")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7", alpha=0.85))
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi)); ax.set_aspect("equal")
    ax.set_xlabel(r"Experiment  ($10^{-10}$ m$^2$/s)")
    ax.set_ylabel(r"Simulation  ($10^{-10}$ m$^2$/s)")
    ax.set_title(title)
    ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.55)
    ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.2)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def parity_grid(bybox: pd.DataFrame, out: Path, exp_label: str, fick: bool = False):
    """2x3 parity on the per-box replica means (one point per box, error bar =
    s.d. over its replicas): rows = YH with experimental eta / YH with simulation
    GK eta; columns = cation / anion / solvent. ``fick=True`` plots the Fickian
    (``D_*_fick_*``) column family instead of the fixed-window one. Takes the
    ``aggregate_by_box`` frame (``*_mean_1e10`` / ``*_std_1e10`` columns)."""
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                         "axes.facecolor": "white"})
    pre = "fick_" if fick else ""
    method_lab = r"Fickian $\beta$≈1 window" if fick else "fixed 1-9 ns window"
    ROWS = [("inf_exp", r"YH: experimental $\eta$"),
            ("inf_sim", r"YH: simulation GK $\eta$")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    for i, (suffix, rowlab) in enumerate(ROWS):
        for j, (sp, splab, _, _) in enumerate(SPECIES):
            ax = axes[i, j]
            _parity_panel(ax, bybox, f"D_{sp}_exp_1e10",
                          f"D_{sp}_{pre}{suffix}_mean_1e10",
                          f"{splab}  ({rowlab})", conv_col=f"D_{sp}_{pre}converged",
                          err_col=f"D_{sp}_{pre}{suffix}_std_1e10")

    salts = [s for s in ("napf6", "naotf", "lipf6") if (bybox["salt"] == s).any()]
    solvs = [s for s in ("dme", "diglyme", "pc", "tgdme") if (bybox["solvent"] == s).any()]
    h = [Line2D([], [], marker=SALT_MARKER[s], color="0.3", ls="", ms=9, mfc="0.3",
                label=SALT_LABEL[s]) for s in salts]
    h += [Line2D([], [], marker="o", color=SOLV_COLOR[s], ls="", ms=9, label=SOLV_LABEL[s])
          for s in solvs]
    h += [Line2D([], [], marker="o", color="0.3", ls="", ms=9, mfc="white", label="0.1 M (hollow)"),
          Line2D([], [], marker="x", color="0.15", ls="", ms=9, label="not converged")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Self-diffusivity parity ({method_lab}) — Einstein MSD slope, Yeh-Hummer "
                 "finite-size correction with experimental η (top) vs simulation GK η (bottom)\n"
                 "dD = kBT·ξ/(6πηL); per-box mean ± s.d. over replicas  —  "
                 f"{exp_label}",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print("saved ->", out, flush=True)


# ── per-box aggregate (shared with diffusivity_opls_replicas.py) ──────────────
def aggregate_by_box(master: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- s.d. of every D variant over each box's replicas.

    The mean and s.d. always use every finite replica. A replica whose
    Fickian window search failed contributes its fallback fit (the twelve-point
    stretch closest to beta = 1), so every box is averaged over the same
    replicas; ``*_n_converged`` records how many met the beta criterion, and a
    box counts as converged when at least one did. Requires a "box" and
    "temperature_K" column on ``master`` (the PaiNN driver's own master doesn't
    carry these -- they're added only for the OPLS-mirrored output)."""
    rows = []
    for box, g in master.groupby("box", sort=False):
        first = g.iloc[0]
        row = {
            "box": box, "system_id": first["system_id"], "salt": first["salt"],
            "solvent": first["solvent"], "conc_label": first["conc_label"],
            "conc_M": first["conc_M"], "temperature_K": first["temperature_K"],
            "n_replicas": int(len(g)), "fit_method": first["fit_method"],
            "box_L_A": first["box_L_A"],
            "eta_sim_cP": first["eta_sim_cP"], "eta_exp_cP": first["eta_exp_cP"],
            "yh_dD_sim_1e10": first["yh_dD_sim_1e10"], "yh_dD_exp_1e10": first["yh_dD_exp_1e10"],
        }
        for sp, _, _, _ in SPECIES:
            # one aggregation block per column family: unsuffixed (fixed window)
            # always; "fick_" when the master carries the Fickian columns.
            prefixes = [""] + (["fick_"] if f"D_{sp}_fick_pbc_1e10" in g.columns else [])
            for pre in prefixes:
                conv_mask = g[f"D_{sp}_{pre}converged"].fillna(False).astype(bool)
                row[f"D_{sp}_{pre}converged"] = bool(conv_mask.any())
                row[f"D_{sp}_{pre}n_converged"] = int(conv_mask.sum())
                for variant in ("pbc", "inf_sim", "inf_exp"):
                    col = f"D_{sp}_{pre}{variant}_1e10"
                    vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
                    row[f"D_{sp}_{pre}{variant}_mean_1e10"] = (round(float(vals.mean()), 4)
                                                               if len(vals) else np.nan)
                    row[f"D_{sp}_{pre}{variant}_std_1e10"] = (round(float(vals.std(ddof=1)), 4)
                                                              if len(vals) > 1 else 0.0)
            row[f"D_{sp}_exp_1e10"] = first[f"D_{sp}_exp_1e10"]
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values(["salt", "solvent", "conc_M", "temperature_K"])
            .reset_index(drop=True))


# ── driver ───────────────────────────────────────────────────────────────────
MASTER_ORDER = [
    "system_id", "salt", "solvent", "conc_label", "conc_M", "replica", "run_name",
    "fit_method", "t_total_ns", "dt_ps", "dt_source", "frames_used", "effective_dt_ps",
    "analysis_window_ps", "eq_time_ps", "msd_method", "box_L_A",
    "eta_sim_cP", "eta_exp_cP", "yh_dD_sim_1e10", "yh_dD_exp_1e10",
    "all_converged", "all_converged_fick",
    "D_cation_pbc_1e10", "D_cation_inf_sim_1e10", "D_cation_inf_exp_1e10", "D_cation_exp_1e10",
    "D_cation_t_lo_ps", "D_cation_t_hi_ps", "D_cation_beta", "D_cation_converged",
    "D_anion_pbc_1e10", "D_anion_inf_sim_1e10", "D_anion_inf_exp_1e10", "D_anion_exp_1e10",
    "D_anion_t_lo_ps", "D_anion_t_hi_ps", "D_anion_beta", "D_anion_converged",
    "D_solvent_pbc_1e10", "D_solvent_inf_sim_1e10", "D_solvent_inf_exp_1e10", "D_solvent_exp_1e10",
    "D_solvent_t_lo_ps", "D_solvent_t_hi_ps", "D_solvent_beta", "D_solvent_converged",
    "D_cation_fick_pbc_1e10", "D_cation_fick_inf_sim_1e10", "D_cation_fick_inf_exp_1e10",
    "D_cation_fick_t_lo_ps", "D_cation_fick_t_hi_ps", "D_cation_fick_beta", "D_cation_fick_converged",
    "D_anion_fick_pbc_1e10", "D_anion_fick_inf_sim_1e10", "D_anion_fick_inf_exp_1e10",
    "D_anion_fick_t_lo_ps", "D_anion_fick_t_hi_ps", "D_anion_fick_beta", "D_anion_fick_converged",
    "D_solvent_fick_pbc_1e10", "D_solvent_fick_inf_sim_1e10", "D_solvent_fick_inf_exp_1e10",
    "D_solvent_fick_t_lo_ps", "D_solvent_fick_t_hi_ps", "D_solvent_fick_beta", "D_solvent_fick_converged",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--exp-csv", type=Path, default=DEFAULT_EXP_CSV)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--time", choices=TP.TIME_CHOICES, default=TP.DEFAULT_TIME,
                    help="Analyze only the first 1ns/5ns of each run with windows scaled "
                         "by T/20ns (short-trajectory convergence study; see time_presets.py). "
                         "Default 20ns = full run, current behaviour.")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="External output root (default data/painn_transport_observables_replicas"
                         ", suffixed _1ns/_5ns when --time is set). All results go here.")
    ap.add_argument("--write-internal", action="store_true",
                    help="ALSO write per-run outputs into each <run_dir>/observables/ and "
                         "the aggregate into <experiment-root>/transport/diffusivity. Default: "
                         "write ONLY to --out-root, leaving the run tree untouched.")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix appended to the diffusivity output folder (per-run + "
                         "aggregate), e.g. '_NEW'. The sim-GK viscosity is read from the "
                         "matching viscosity<visc-suffix> folder under --out-root "
                         "(see --visc-suffix).")
    ap.add_argument("--visc-suffix", default=None,
                    help="Suffix of the viscosity<...> folder (under --out-root) to read the "
                         "sim-GK viscosity from. Defaults to --out-suffix. Viscosity is "
                         "fit-method-independent, so point a '_fickian' diffusivity run at "
                         "the existing viscosity with --visc-suffix ''.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    if args.out_root is None:
        args.out_root = TP.out_root_for(args.time)

    visc_suffix = args.out_suffix if args.visc_suffix is None else args.visc_suffix
    exp = load_exp_table(args.exp_csv)
    sim_eta = load_sim_viscosity(args.out_root, visc_suffix)
    runs, skipped = discover_runs(args.experiment_root, exp, sim_eta)
    print(f"Discovered {len(runs)} runs to analyze, {len(skipped)} skipped.")

    if args.dry_run:
        for r in sorted(runs, key=lambda x: (x["salt"], x["solvent"], x["conc_M"])):
            e, es = r["eta_exp"], r["eta_sim"]
            print(f"  {r['system_id']:16s} {r['conc_label']:5s}  "
                  f"eta_exp={'--' if e is None else f'{e:.3f}'} "
                  f"eta_sim={'--' if es is None else f'{es:.3f}'} cP  "
                  f"D_exp(+/-/0)={r['D_exp_cation']}/{r['D_exp_anion']}/{r['D_exp_solvent']}")
        for r in skipped:
            print(f"  SKIP {r['system_id']} {r['conc_label']}: {r['skip_reason']}")
        print(f"\nDRY RUN: {len(runs)} runs, {len(skipped)} skipped. Nothing computed.")
        return

    # One MSD pass per run (SETTINGS_FIXED's eq/stride); both fit methods are
    # applied to that curve — fixed fills the primary columns, fickian the
    # D_*_fick_* family.
    # --time scales eq/fit windows proportionally (see time_presets.py); at the
    # default 20ns the overlays are identity copies of the base dicts.
    base = TP.scale_diffusivity(SETTINGS_FIXED, args.time)
    settings = dict(base, fick=TP.scale_diffusivity(SETTINGS_FICKIAN, args.time),
                    time=args.time,
                    out_suffix=args.out_suffix, out_root=args.out_root,
                    write_internal=args.write_internal,
                    generated=datetime.now().isoformat(timespec="seconds"))

    results, n = [], len(runs)
    print(f"[progress] launching {n} runs on {args.jobs} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(analyze_one, r, settings, not args.no_plots): i
                for i, r in enumerate(runs)}
        for done, fut in enumerate(as_completed(futs), 1):
            r = runs[futs[fut]]
            tag = f"{r['system_id']} {r['conc_label']}"
            try:
                res = fut.result()
                results.append(res)
                row = res["row"]
                print(f"[progress] {done}/{n}  {tag:22s} "
                      f"D_inf_sim(+/-/0)={row['D_cation_inf_sim_1e10']}/"
                      f"{row['D_anion_inf_sim_1e10']}/{row['D_solvent_inf_sim_1e10']}  "
                      f"({row['t_total_ns']:.1f} ns)", flush=True)
            except Exception as e:
                print(f"[progress] {done}/{n}  {tag:22s} ERROR: {e}", flush=True)

    if not results:
        print("No runs analyzed; nothing to aggregate.")
        return

    master = pd.DataFrame([res["row"] for res in results])
    master = master.reindex(columns=[c for c in MASTER_ORDER if c in master.columns]
                            + [c for c in master.columns if c not in MASTER_ORDER])
    master = master.sort_values(["salt", "solvent", "conc_M", "replica"]).reset_index(drop=True)
    exp_label = args.experiment_root.name.replace("_", " ")

    cfg = dict(
        observable="diffusivity",
        estimator="einstein_fixed_window (primary D_* columns) + "
                  "einstein_fickian_beta1 (D_*_fick_* columns), same MSD curve",
        fit_method="fixed+fickian",
        yeh_hummer=True, yh_variants=["experimental_eta", "simulation_gk_eta"],
        experiment_root=str(args.experiment_root),
        exp_source=str(args.exp_csv),
        sim_viscosity_source=str(Path(args.out_root) / f"viscosity{visc_suffix}"
                                 / "viscosity_gk_by_box.csv"),
        settings={k: v for k, v in base.items()},
        fickian_settings={k: v for k, v in SETTINGS_FICKIAN.items()},
        write_internal=args.write_internal,
        n_runs_analyzed=len(results), n_skipped=len(skipped),
        n_all_converged_fick=(int(master["all_converged_fick"].sum())
                              if "all_converged_fick" in master else None),
        skipped_runs=[f"{r['system_id']} {r['conc_label']} ({r['skip_reason']})"
                      for r in skipped],
        code_version=git_sha(), generated=settings["generated"],
    )

    # Per-box aggregate (replica mean ± s.d.) — built up front because the parity
    # figures plot one point per box with the replica s.d. as the error bar.
    # "box" computed straight from columns (not a run_name-keyed lookup): run_name
    # alone isn't unique across systems ("nvt_transport_1M_298K_20ns_1ps" repeats
    # per system).
    mirror_master = master.copy()
    mirror_master["box"] = (mirror_master["system_id"] + "_" + mirror_master["conc_label"]
                            + "_" + str(int(round(T_K))))
    mirror_master["temperature_K"] = T_K
    mirror_bybox = aggregate_by_box(mirror_master)

    # ── internal aggregate (opt-in): <experiment-root>/transport/diffusivity<suffix>/ ─
    if args.write_internal:
        out_dir = args.experiment_root / "transport" / f"diffusivity{args.out_suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        master.to_csv(out_dir / "diffusivity_master.csv", index=False)
        (out_dir / "diffusivity_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        if not args.no_plots:
            msd_grid(results, out_dir / "figure_transport_diff_msd16.png", exp_label)
            parity_grid(mirror_bybox, out_dir / "figure_transport_diff_parity.png", exp_label)
            parity_grid(mirror_bybox, out_dir / "figure_transport_diff_parity_fickian.png",
                        exp_label, fick=True)

    # ── external output (always): OUT_ROOT/diffusivity<suffix>/, OPLS-mirrored layout ─
    mirror_dir = Path(args.out_root) / f"diffusivity{args.out_suffix}"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    mirror_master.to_csv(mirror_dir / "diffusivity_master.csv", index=False)
    mirror_bybox.to_csv(mirror_dir / "diffusivity_by_box.csv", index=False)
    mirror_cfg = dict(cfg, forcefield="PaiNN/FeNNol",
                      opls_counterpart="transport/diffusivity_opls_replicas.py",
                      out_dir=str(mirror_dir), n_boxes=len(mirror_bybox))
    (mirror_dir / "diffusivity_config.yaml").write_text(yaml.safe_dump(mirror_cfg, sort_keys=False))
    if not args.no_plots:
        if args.write_internal:
            shutil.copy2(out_dir / "figure_transport_diff_msd16.png",
                         mirror_dir / "figure_transport_diff_msd.png")
            shutil.copy2(out_dir / "figure_transport_diff_parity.png",
                         mirror_dir / "figure_transport_diff_parity.png")
            shutil.copy2(out_dir / "figure_transport_diff_parity_fickian.png",
                         mirror_dir / "figure_transport_diff_parity_fickian.png")
        else:
            msd_grid(results, mirror_dir / "figure_transport_diff_msd.png", exp_label)
            parity_grid(mirror_bybox, mirror_dir / "figure_transport_diff_parity.png", exp_label)
            parity_grid(mirror_bybox, mirror_dir / "figure_transport_diff_parity_fickian.png",
                        exp_label, fick=True)

    print("\n=== Self-diffusivity (Einstein MSD, 1e-10 m^2/s) ===")
    cols = ["system_id", "conc_label", "replica", "t_total_ns", "box_L_A", "eta_sim_cP",
            "eta_exp_cP", "D_cation_pbc_1e10", "D_cation_inf_sim_1e10",
            "D_cation_inf_exp_1e10", "D_cation_exp_1e10"]
    print(master[[c for c in cols if c in master.columns]].to_string(index=False))
    print(f"\nDone: {len(results)} runs -> {mirror_dir}/diffusivity_master.csv")
    print(f"  external output -> {mirror_dir}/ (per_run/<box>/r<k>, *_master.csv, *_by_box.csv)")
    if args.write_internal:
        print(f"  internal too    -> {out_dir}/ + <run_dir>/observables/diffusivity_<N>ns/")


if __name__ == "__main__":
    main()
