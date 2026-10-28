#!/usr/bin/env python
"""Shared settings, experimental reference, row schema and plots for the two
Onsager/Nernst-Einstein conductivity drivers.

``conductivity_opls_replicas.py`` (OPLS/GROMACS) and ``conductivity_replicas.py``
(PaiNN/FeNNol) both compute sigma with ``observables.conductivity.compute_conductivity``
and both build their outputs here, so the two sets are directly comparable.

Verified equivalence
--------------------
``observables/conductivity.py`` is a native re-implementation of byteff2's
``onsager_calc`` — the kernel ``code_utils`` used for the OPLS set. Fed the same
trajectory and the same fit window they agree to **2.8e-7 relative** on
sigma_onsager, sigma_NE and every D_self (checked on
``napf6_dme_1M_298/r0``; float32 xtc round-off is the residual). So "same
calculation" here is a measured statement, not an assumption.

Fit windows
-----------
``code_utils`` did *not* use one window for both backends, so a head-to-head has
to pick. The presets below make the choice explicit:

  * ``byteff2_ase`` (default) -- what ``code_utils``' ASE path defaulted to
    (``run_onsager_conductivity``: ``eq_cut_ns=2``, ``load_dt_ps=10``,
    ``nt_start/nt_end = 50/200``) -> 2 ns cut, 10 ps stride, **500-2000 ps** fit.
    Fitting the collective MSD in this later, cleanly-diffusive window (rather
    than the early 50-199 ps of ``byteff2_gromacs``) removes a systematic
    high-side bias in sigma_onsager and improves the vs-experiment correlation
    (log-log Pearson r 0.56 -> 0.65 on the 0.5/1 M PaiNN set).
  * ``byteff2_gromacs`` -- what the OPLS set was *originally* produced with:
    ``onsager_calc`` drops the first 200 frames and fits MSD lags 50..199. At the
    GROMACS 1 ps spacing that is a 200 ps equilibration cut and a **50-199 ps**
    fit, every frame. Kept for reproducing the legacy byteff2 GROMACS numbers,
    but the 50-199 ps lags sit in the collective MSD's sub-diffusive crossover,
    so it is no longer the default.
  * ``transport`` -- this project's transport convention, matching
    ``diffusivity_replicas.py`` / the OPLS diffusivity driver: 2 ns cut, every
    frame, fit ``[1000 ps, 0.5 x window]``. Correct for the self term, but the
    1-9 ns lags are too noisy for the (one-coordinate) collective term.

Both drivers default to ``byteff2_ase`` so PaiNN and OPLS are computed the
same way; ``--window`` switches either one. Use the *same* preset on both sides
of any PaiNN-vs-OPLS comparison.

Note the collective (Onsager) term is far noisier than the self term, so the
window genuinely moves sigma_onsager.
Report which preset a number came from.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


from .. import paths

GROUND_TRUTH = paths.GROUND_TRUTH

# ── fit-window presets ───────────────────────────────────────────────────────
WINDOWS = {
    "byteff2_gromacs": dict(
        eq_time_ps=200.0, tau_min_fit_ps=50.0, tau_max_fit_ps=199.0,
        tau_max_fit_frac=None, load_dt_ps=None,
    ),
    "byteff2_ase": dict(
        eq_time_ps=2000.0, tau_min_fit_ps=500.0, tau_max_fit_ps=2000.0,
        tau_max_fit_frac=None, load_dt_ps=10.0,
    ),
    "transport": dict(
        eq_time_ps=2000.0, tau_min_fit_ps=1000.0, tau_max_fit_ps=None,
        tau_max_fit_frac=0.5, load_dt_ps=None,
    ),
    # ``byteff2_ase`` scaled by trajectory length (f = T / 20 ns) for the
    # short-trajectory convergence study; the 10 ps stride would drop below the
    # 1 ps dump interval, so these use every frame. Selected automatically by
    # ``--time 1ns/2ns/5ns`` in the PaiNN driver (see time_presets.py).
    "byteff2_ase_1ns": dict(
        eq_time_ps=100.0, tau_min_fit_ps=25.0, tau_max_fit_ps=100.0,
        tau_max_fit_frac=None, load_dt_ps=None,
    ),
    "byteff2_ase_2ns": dict(
        eq_time_ps=200.0, tau_min_fit_ps=50.0, tau_max_fit_ps=200.0,
        tau_max_fit_frac=None, load_dt_ps=None,
    ),
    "byteff2_ase_5ns": dict(
        eq_time_ps=500.0, tau_min_fit_ps=125.0, tau_max_fit_ps=500.0,
        tau_max_fit_frac=None, load_dt_ps=None,
    ),
}
DEFAULT_WINDOW = "byteff2_ase"


def target_frames_for(window: dict, n_total: int, dt_ps: float) -> int:
    """``target_frames`` for :func:`compute_conductivity` realising this preset's
    ``load_dt_ps`` (None -> every frame, i.e. stride 1)."""
    if window["load_dt_ps"] is None:
        return 10 ** 9                                    # >> n_total => stride 1
    i_start = min(max(int(math.floor(window["eq_time_ps"] / dt_ps)), 0), max(n_total - 2, 0))
    avail = max(n_total - i_start, 1)
    stride = max(1, round(window["load_dt_ps"] / dt_ps))
    return max(1, math.ceil(avail / stride))


# ── experimental reference ───────────────────────────────────────────────────
# `conductivity_all.csv` is a results file from the PaiNN NPT-replica runs, but
# it carries the concentration- and temperature-matched experimental conductivity
# per system, which is the only such table in the repo. Harvest just that column.
_SYS_RE = re.compile(r"^(?P<salt>[a-z0-9]+)_(?P<solv>[a-z]+)__npt_"
                     r"(?P<conc>\d+(?:_\d+)?)M_(?P<T>\d+)K")


def load_exp_conductivity(csv_path: Path | None = None) -> dict:
    """(salt, solvent, conc_M, T_K) -> experimental conductivity [mS/cm]."""
    csv_path = csv_path or (GROUND_TRUTH / "conductivity_all.csv")
    if not Path(csv_path).is_file():
        print(f"[warn] no experimental conductivity table at {csv_path}; "
              f"exp column will be NaN.", flush=True)
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for system, g in df.groupby("system"):
        m = _SYS_RE.match(str(system))
        if not m:
            continue
        vals = g["exp_conductivity_mS_cm"].dropna().unique()
        if not len(vals):
            continue
        key = (m.group("salt"), m.group("solv"),
               float(m.group("conc").replace("_", ".")), float(m.group("T")))
        out[key] = float(vals[0])
    return out


def exp_sigma(table: dict, salt: str, solvent: str, conc_m: float, temp_k: float):
    return table.get((salt, solvent, float(conc_m), float(temp_k)))


# ── row schema ───────────────────────────────────────────────────────────────
ROW_ORDER = [
    "system_id", "salt", "solvent", "conc_label", "conc_M", "temperature_K",
    "replica", "run_name", "window", "eq_time_ps", "tau_min_fit_ps", "tau_max_fit_ps",
    "dt_ps", "effective_dt_ps", "frames_used", "analysis_window_ps",
    "box_L_A", "volume_A3", "n_cation", "n_anion", "n_solvent",
    "z_cation", "z_anion", "com_drift_A",
    "sigma_onsager_mS_cm", "sigma_NE_mS_cm", "sigma_exp_mS_cm",
    "ionicity", "t_cation_NE", "t_cation_onsager",
    "D_cation_pbc_1e10", "D_anion_pbc_1e10", "D_solvent_pbc_1e10",
    "L_pp_1e10", "L_pm_1e10", "L_mm_1e10",
]


def build_row(res, meta: dict, window_name: str, dt_ps: float) -> dict:
    """Flat row from a ``ConductivityResult`` + run metadata."""
    L = res.lambda_corr
    row = {
        "system_id": meta["system_id"], "salt": meta["salt"], "solvent": meta["solvent"],
        "conc_label": meta["conc_label"], "conc_M": meta["conc_M"],
        "temperature_K": meta["temperature_K"], "replica": meta.get("replica"),
        "run_name": meta["run_name"], "window": window_name,
        "eq_time_ps": res.eq_time_ps,
        "tau_min_fit_ps": round(res.tau_min_fit_ps, 3),
        "tau_max_fit_ps": round(res.tau_max_fit_ps, 3),
        "dt_ps": round(dt_ps, 5),
        "effective_dt_ps": round(res.effective_dt_ps, 5),
        "frames_used": res.frames_used,
        "analysis_window_ps": round(res.analysis_window_ps, 2),
        "box_L_A": round(res.box_length_a, 3),
        "volume_A3": round(res.volume_a3, 2),
        "n_cation": res.n_cation, "n_anion": res.n_anion, "n_solvent": res.n_solvent,
        "z_cation": res.z_cation, "z_anion": res.z_anion,
        "com_drift_A": round(res.com_drift_a, 3),
        "sigma_onsager_mS_cm": round(res.sigma_onsager_mS_cm, 5),
        "sigma_NE_mS_cm": round(res.sigma_ne_mS_cm, 5),
        "sigma_exp_mS_cm": meta.get("sigma_exp"),
        "ionicity": round(res.ionicity, 5),
        "t_cation_NE": round(res.t_cation_ne, 5),
        "t_cation_onsager": round(res.t_cation_onsager, 5),
        "D_cation_pbc_1e10": round(res.d_cation_x1e10, 5),
        "D_anion_pbc_1e10": round(res.d_anion_x1e10, 5),
        "D_solvent_pbc_1e10": round(res.d_solvent_x1e10, 5),
        "L_pp_1e10": round(float(L[0, 0]), 6),
        "L_pm_1e10": round(float(L[0, 1]), 6),
        "L_mm_1e10": round(float(L[1, 1]), 6),
    }
    return row


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=[c for c in ROW_ORDER if c in df.columns]
                      + [c for c in df.columns if c not in ROW_ORDER])


# ── plotting ─────────────────────────────────────────────────────────────────
SPEC_COLOR = {"cation": "#1B7837", "anion": "#762A83", "solvent": "#2166AC"}
CROSS_COLOR = {"pp": "#1B7837", "pm": "#D95F02", "mm": "#762A83"}
SOLV_COLOR = {"dme": "#E68310", "diglyme": "#11A579", "pc": "#E73F74", "tgdme": "#3969AC"}
SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}
SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}
SALT_MARKER = {"napf6": "o", "naotf": "s", "lipf6": "^"}


def decimate(n: int, target: int = 2000) -> np.ndarray:
    if n <= target:
        return np.arange(n)
    return np.unique(np.r_[np.arange(0, n, n // target), n - 1])


def plot_cross_panel(ax, tau, cross, t_lo, t_hi, title):
    """Log-log |C_ij(tau)| for the three ion-ion collective correlations, fit
    window shaded. Dashed = the curve is predominantly negative over the window
    (anticorrelated ion motion, the usual sign for +-)."""
    pos = np.asarray(tau) > 0
    ymin = ymax = None
    for key, lab in (("pp", r"$C_{++}$"), ("pm", r"$C_{+-}$"), ("mm", r"$C_{--}$")):
        c = np.asarray(cross[key])
        m = pos & (np.abs(c) > 0)
        if not m.any():
            continue
        win = m & (tau >= t_lo) & (tau <= t_hi)
        ls = "--" if win.any() and (c[win] < 0).mean() > 0.5 else "-"
        y = np.abs(c[m])
        ax.plot(tau[m], y, color=CROSS_COLOR[key], lw=1.3, ls=ls, label=lab)
        core = y[(tau[m] >= t_lo) & (tau[m] <= t_hi)] if win.any() else y
        if len(core):
            lo, hi = float(core.min()), float(core.max())
            ymin = lo if ymin is None else min(ymin, lo)
            ymax = hi if ymax is None else max(ymax, hi)
    ax.axvspan(t_lo, t_hi, color="0.6", alpha=0.15)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=8)
    ax.set_ylabel(r"$|C_{ij}(\tau)|$ [$\mathrm{\AA}^2$]", fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7); ax.grid(True, which="both", ls=":", alpha=0.4)
    if ymin is not None and ymax > ymin > 0:
        ax.set_ylim(ymin * 0.2, ymax * 5.0)


def plot_self_panel(ax, tau, msds, t_lo, t_hi, title):
    """Log-log per-species self MSD with the fit window shaded."""
    pos = np.asarray(tau) > 0
    for key, lab in (("cation", r"MSD$_+$"), ("anion", r"MSD$_-$"), ("solvent", r"MSD$_0$")):
        m = np.asarray(msds[key])
        sel = pos & (m > 0)
        if sel.any():
            ax.plot(tau[sel], m[sel], color=SPEC_COLOR[key], lw=1.3, label=lab)
    ax.axvspan(t_lo, t_hi, color="0.6", alpha=0.15)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=8)
    ax.set_ylabel(r"MSD [$\mathrm{\AA}^2$]", fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7); ax.grid(True, which="both", ls=":", alpha=0.4)


def run_figure(res, title, out_path):
    """Per-run 2-panel diagnostic: self MSD | ion-ion collective correlations."""
    d = decimate(len(res.tau_ps), 3000)
    tau = res.tau_ps[d]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    plot_self_panel(a1, tau, {"cation": res.msd_cat[d], "anion": res.msd_anion[d],
                              "solvent": res.msd_solvent[d]},
                    res.tau_min_fit_ps, res.tau_max_fit_ps, f"{title} — self MSD")
    a1.legend(fontsize=8, frameon=False)
    plot_cross_panel(a2, tau, {"pp": res.cross_pp[d], "pm": res.cross_pm[d],
                               "mm": res.cross_mm[d]},
                     res.tau_min_fit_ps, res.tau_max_fit_ps,
                     f"{title} — collective\n"
                     f"$\\sigma_O$={res.sigma_onsager_mS_cm:.3f} "
                     f"$\\sigma_{{NE}}$={res.sigma_ne_mS_cm:.3f} mS/cm")
    a2.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)


def cross_grid(results: list, out: Path, suptitle: str, ncol: int = 4):
    """Grid of the collective-correlation panels, one per run."""
    n = len(results)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.6 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax, r in zip(axes, results):
        plot_cross_panel(ax, r["plot"]["tau"], r["plot"]["cross"],
                         r["row"]["tau_min_fit_ps"], r["row"]["tau_max_fit_ps"],
                         f"{r['label']}\n$\\sigma_O$={r['row']['sigma_onsager_mS_cm']:.3f} "
                         f"$\\sigma_{{NE}}$={r['row']['sigma_NE_mS_cm']:.3f} "
                         f"exp={r['row']['sigma_exp_mS_cm'] if r['row']['sigma_exp_mS_cm'] is not None else float('nan'):.3f}")
    for ax in axes[n:]:
        ax.axis("off")
    h = [Line2D([], [], color=CROSS_COLOR[k], lw=1.6, label=lab)
         for k, lab in (("pp", r"$C_{++}$"), ("pm", r"$C_{+-}$"), ("mm", r"$C_{--}$"))]
    h.append(Line2D([], [], color="0.4", lw=1.6, ls="--", label="negative over fit window"))
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.015, 1, 0.985])
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    print("saved ->", out, flush=True)


def _fin(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _parity_panel(ax, df, xcol, ycol, ecol, title):
    """sigma_sim vs sigma_exp, log-log; solvent = colour, salt = marker,
    0.1 M = hollow, factor-2 band, MAPD over the plotted points."""
    vals = [v for _, r in df.iterrows() for v in (_fin(r[xcol]), _fin(r[ycol]))
            if np.isfinite(v) and v > 0]
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
        if ecol is not None:
            es = _fin(r[ecol])
            ax.errorbar(xe, ys, yerr=es if np.isfinite(es) else None, fmt="none",
                        ecolor=c, elinewidth=1.1, capsize=2, alpha=0.6, zorder=2)
        ax.scatter(xe, ys, s=130, marker=SALT_MARKER.get(r["salt"], "o"),
                   facecolor=c if filled else "white", edgecolor=c, linewidths=1.9, zorder=3)
        devs.append(abs(ys - xe) / xe)
    txt = (f"MAPD = {100*np.mean(devs):.0f}%\n(n = {len(devs)}/{len(df)})"
           if devs else f"no pairs (n=0/{len(df)})")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7", alpha=0.85))
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi)); ax.set_aspect("equal")
    ax.set_xlabel("Experiment  (mS/cm)"); ax.set_ylabel("Simulation  (mS/cm)")
    ax.set_title(title)
    ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.55)
    ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.2)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def parity_figure(df: pd.DataFrame, out: Path, suptitle: str,
                  ons_col="sigma_onsager_mS_cm", ne_col="sigma_NE_mS_cm",
                  ons_err=None, ne_err=None):
    """Two-panel parity: Onsager (left) and Nernst-Einstein (right) vs experiment."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    _parity_panel(axes[0], df, "sigma_exp_mS_cm", ons_col, ons_err,
                  r"Onsager $\sigma_O$ (with ion-ion cross terms)")
    _parity_panel(axes[1], df, "sigma_exp_mS_cm", ne_col, ne_err,
                  r"Nernst-Einstein $\sigma_{NE}$ (self term only)")
    salts = [s for s in ("napf6", "naotf", "lipf6") if (df["salt"] == s).any()]
    solvs = [s for s in ("dme", "diglyme", "pc", "tgdme") if (df["solvent"] == s).any()]
    h = [Line2D([], [], marker=SALT_MARKER[s], color="0.3", ls="", ms=9, mfc="0.3",
                label=SALT_LABEL[s]) for s in salts]
    h += [Line2D([], [], marker="o", color=SOLV_COLOR[s], ls="", ms=9, label=SOLV_LABEL[s])
          for s in solvs]
    h += [Line2D([], [], marker="o", color="0.3", ls="", ms=9, mfc="white", label="0.1 M (hollow)")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print("saved ->", out, flush=True)


# ── per-group aggregate (over replicas) ──────────────────────────────────────
AGG_COLS = ["sigma_onsager_mS_cm", "sigma_NE_mS_cm", "ionicity",
            "t_cation_NE", "t_cation_onsager",
            "D_cation_pbc_1e10", "D_anion_pbc_1e10", "D_solvent_pbc_1e10"]


def aggregate(master: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Mean +/- s.d. over the replicas of each ``group_col`` value."""
    rows = []
    for key, g in master.groupby(group_col, sort=False):
        first = g.iloc[0]
        row = {group_col: key, "system_id": first["system_id"], "salt": first["salt"],
               "solvent": first["solvent"], "conc_label": first["conc_label"],
               "conc_M": first["conc_M"], "temperature_K": first["temperature_K"],
               "n_replicas": int(len(g)), "window": first["window"],
               "tau_min_fit_ps": first["tau_min_fit_ps"],
               "tau_max_fit_ps": first["tau_max_fit_ps"],
               "box_L_A": first["box_L_A"], "volume_A3": first["volume_A3"],
               "sigma_exp_mS_cm": first["sigma_exp_mS_cm"]}
        for c in AGG_COLS:
            v = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy()
            row[f"{c}_mean"] = round(float(v.mean()), 5) if len(v) else np.nan
            row[f"{c}_std"] = round(float(v.std(ddof=1)), 5) if len(v) > 1 else 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    df["onsager_over_exp"] = df["sigma_onsager_mS_cm_mean"] / df["sigma_exp_mS_cm"]
    df["NE_over_exp"] = df["sigma_NE_mS_cm_mean"] / df["sigma_exp_mS_cm"]
    return (df.sort_values(["salt", "solvent", "conc_M", "temperature_K"])
            .reset_index(drop=True))
