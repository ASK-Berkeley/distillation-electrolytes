#!/usr/bin/env python
"""Green-Kubo shear viscosity for the transport_* NVT+BUSSI runs.

For every transport run this computes the running GK stress-ACF integral eta(tau)
out to ``--max-lag-ps`` (default 1 ns) with a block-resampling error band, then
reads off eta_inf with the Zhang-Maginn time-decomposition fit (JCTC 2015) -- a
data-driven window, no hand-picked plateau (method and validation: ``METHODS.md`` Secs. 4 and 8).
The first ``--eq-time-ps`` (default 2000 ps) of the stress log is discarded
before the ACF, matching the 2 ns equilibration cut the diffusivity and
conductivity estimators apply to the trajectory.

Run tree (auto-detected, ``iter_run_dirs``): the default is the 4-replica set
``PAINN_NVT_REPLICAS`` (``nvt_replica_<N>/<system>/nvt_transport_<conc>_298K_20ns_1ps/``,
298 K only -- off-temperature runs are skipped); the legacy exp04 ``r0``-only layout
still works if passed via ``--experiment-root``.

OUTPUT (default): results go ONLY to the external ``--out-root``
(``data/painn_transport_observables_replicas`` by default), leaving the run tree
untouched -- ``viscosity<suffix>/{per_run/<box>/r<k>/, *_master.csv, *_by_box.csv,
*_config.yaml, figure_*.png}``. Pass ``--write-internal`` to ALSO write the per-run
``<run_dir>/observables/`` and aggregate ``<exp>/transport/`` outputs described below.

Per run it writes into ``<run_dir>/observables/viscosity_<N>ns/`` (N = simulated ns):
  * ``viscosity_gk.csv``        -- one row: eta_inf, error, converged, fit window,
                                   tau_slow, frac_recovered, eta at fixed cutoffs,
                                   + the experimental eta for reference.
  * ``viscosity_gk_curve.csv``  -- the running average of the integral,
                                   tau_ps / eta_running_cp / band (decimated).
  * ``viscosity_gk.png``        -- per-run convergence panel (unless --no-plots).
  * merges a ``viscosity_gk`` block into the existing ``summary.json``.

Then it rolls everything up into ``<exp>/transport/gk_viscosity/``:
  * ``viscosity_gk_master.csv`` -- one row per run.
  * ``viscosity_gk_config.yaml``-- settings + provenance sidecar.
  * ``figure_transport_visc_convergence16.png`` -- the 16-panel eta(tau) grid.

CPU only; conda env ``fennol``. Run from the repo root as a module (the
package uses relative imports):

    python -m src.analysis.observables.transport.viscosity_gk_replicas --dry-run
    python -m src.analysis.observables.transport.viscosity_gk_replicas --jobs 8
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
import numpy as np
import pandas as pd
import yaml


from .. import paths
from ..paths import git_sha
from ..viscosity_gk import eta_running_curve, fit_eta_tdm

from . import time_presets as TP

# ── paths / constants ────────────────────────────────────────────────────────
DEFAULT_ROOT = paths.PAINN_NVT_REPLICAS
DEFAULT_EXP_CSV = paths.GROUND_TRUTH / "298p2K_exp_values.csv"
T_K = 298.0
NBLOCK = 5
CUTOFFS_PS = (100.0, 250.0, 500.0, 1000.0)   # fixed read-offs, for reference only

# Two run-tree layouts are supported transparently (see iter_run_dirs):
#   * legacy exp04 (r0-only): <system>/transport_<conc>_298K_r0/
#   * 4-replica tree:         nvt_replica_<N>/<system>/nvt_transport_<conc>_298K_20ns_1ps/
RUN_GLOB = "transport_*_298K_r0"
RUN_RE = re.compile(r"^transport_(?P<conc>.+?)_(?P<temp>\d+)K_r(?P<rep>\d+)$")
REPLICA_DIR_RE = re.compile(r"^nvt_replica_(?P<rep>\d+)$")
REPLICA_RUN_GLOB = "nvt_transport_*"
REPLICA_RUN_RE = re.compile(r"^nvt_transport_(?P<conc>.+?)_(?P<temp>\d+)K_")

# system_id token -> experimental-CSV name
_SALT = {"napf6": "NaPF6", "naotf": "NaOTf", "lipf6": "LiPF6"}
_SOLVENT = {"dme": "DME", "pc": "PC", "diglyme": "DEGDME", "tgdme": "TEGDME"}
_SAMPLE_RE = re.compile(r"^([\d.]+)\s*M\s+(\S+)\s+in\s+(\S+)$")

# plot encoding (self-contained, matches paper/plot_transport_gk.py)
SOLV_COLOR = {"dme": "#E68310", "diglyme": "#11A579", "pc": "#E73F74", "tgdme": "#3969AC"}
SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}
SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}
SALT_MARKER = {"napf6": "o", "naotf": "s", "lipf6": "^"}


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
    exp04 (r0-only) layout and the 4-replica ``nvt_replica_<N>`` tree. Kept local so
    this driver stays standalone (matches diffusivity_replicas.iter_run_dirs)."""
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


def load_exp_viscosity(csv_path: Path) -> dict:
    """(salt_token, solvent_token, conc_M) -> experimental eta [mPa*s]."""
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
            out[(salt, solv, conc)] = _f(r.get("Dynamic Viscosity (mPa*s)"))
    return out


def discover_runs(root: Path, exp: dict) -> tuple[list, list]:
    """Return (runs, skipped) for every transport run under `root` (both layouts)."""
    runs, skipped = [], []
    for system_id, salt, solv, conc_tok, conc_m, temp_k, replica, rd in iter_run_dirs(root):
        # 298 K pipeline: skip off-temperature runs (the lipf6_dme 0.5 M 273/323 K
        # series in replicas 1-3) rather than mislabelling them as 298 K.
        if int(round(temp_k)) != int(round(T_K)):
            skipped.append(dict(system_id=system_id, conc_label=conc_tok, run_name=rd.name,
                                skip_reason=f"non-{int(round(T_K))}K (T={int(round(temp_k))}K)"))
            continue
        rec = dict(system_id=system_id, salt=salt, solvent=solv,
                   conc_label=conc_tok, conc_M=conc_m, replica=replica,
                   run_name=rd.name, run_dir=str(rd),
                   box=f"{system_id}_{conc_tok}_{int(round(temp_k))}",
                   eta_exp=exp.get((salt, solv, conc_m)))
        if not list(rd.glob("*.stress")):
            rec["skip_reason"] = "no .stress log"
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


def _merge_summary(obs_dir: Path, block: dict) -> None:
    p = obs_dir / "summary.json"
    summary = {}
    if p.is_file():
        try:
            summary = json.loads(p.read_text())
        except Exception:
            summary = {}
    summary["viscosity_gk"] = block
    p.write_text(json.dumps(summary, indent=2))


def _per_run_external_dir(settings: dict, rec: dict) -> Path:
    """External per-run target: OUT_ROOT/viscosity<suffix>/per_run/<box>/r<k>/
    (the OPLS-mirrored layout)."""
    return (settings["out_root"] / f"viscosity{settings['out_suffix']}"
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


def _plot_panel(ax, lags, full, band, p, eta_exp, title):
    idx = _decimate(len(lags), 4000)
    lp, fp, bp = lags[idx], full[idx], band[idx]
    c = "#3969AC"
    ax.plot(lp, fp, color=c, lw=1.5)
    ax.fill_between(lp, fp - bp, fp + bp, color=c, alpha=0.22, lw=0)
    ax.axvspan(p["t_lo"], p["t_hi"], color="0.6", alpha=0.18)
    if p.get("fit_curve") is not None:
        tf, yf = p["fit_curve"]
        ax.plot(tf, yf, color="0.15", lw=1.1, ls="--")
        ax.axhline(p["est"], color="0.15", lw=0.8, ls=":")
    have_exp = eta_exp is not None and np.isfinite(eta_exp)
    if have_exp:
        ax.axhline(eta_exp, ls="--", color="r", lw=1.2)
    top = max(2.5 * abs(p["est"]), 1.5 * (eta_exp if have_exp else 0.0), 1e-3)
    ax.set_xlim(0, lags[-1]); ax.set_ylim(0, top)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=8); ax.set_ylabel(r"$\eta(\tau)$ [cP]", fontsize=8)
    ax.set_title(title, fontsize=8.5, color=("k" if p["conv"] else "#b00"))
    ax.tick_params(labelsize=7); ax.grid(True, ls=":", alpha=0.4)


def analyze_one(rec: dict, settings: dict, plots: bool) -> dict:
    """Compute eta(tau) + TDM fit for one run, write per-run outputs, return a
    light-weight result (scalar row + decimated curve for the aggregate figure)."""
    rd = Path(rec["run_dir"])
    c = eta_running_curve(rd, max_lag_ps=settings["max_lag_ps"],
                          nblock=settings["nblock"], t_K=settings["temperature_k"],
                          eq_time_ps=settings["eq_time_ps"],
                          max_time_ps=settings.get("max_time_ps"))
    lags, full, band = c["lags_ps"], c["running_cp"], c["band"]
    p = fit_eta_tdm(lags, full, band)

    # Per-run output target. With --write-internal: the run's own
    # observables/viscosity_<N>ns/ subfolder (kept separate from the diffusivity /
    # conductivity observables in the run). Default: straight into the external
    # per_run/<box>/r<k>/ mirror, leaving the (read-only) run tree untouched.
    if settings["write_internal"]:
        obs_dir = rd / "observables" / f"viscosity_{round(c['t_total_ns'])}ns{settings['out_suffix']}"
    else:
        obs_dir = _per_run_external_dir(settings, rec)
    obs_dir.mkdir(parents=True, exist_ok=True)

    def _cut(tc):
        i = min(len(full) - 1, int(round(tc / c["dt_ps"])))
        return float(full[i])
    eta_cut = {tc: _cut(tc) for tc in settings["cutoffs_ps"]}

    row = {
        "system_id": rec["system_id"], "salt": rec["salt"], "solvent": rec["solvent"],
        "conc_label": rec["conc_label"], "conc_M": rec["conc_M"], "replica": rec["replica"],
        "run_name": rec["run_name"],
        "eta_gk_cP": round(p["est"], 4), "eta_err_cP": round(p["err"], 4),
        "converged": bool(p["conv"]),
        "t_lo_ps": round(p["t_lo"], 2), "t_hi_ps": round(p["t_hi"], 2),
        "tau_slow_ps": round(p["tau_slow_ps"], 2), "frac_recovered": round(p["frac_recovered"], 4),
        "t_total_ns": round(c["t_total_ns"], 3), "T_mean_K": round(c["T_mean_K"], 2),
        "dt_fs": round(c["dt_ps"] * 1000.0, 3), "n_stress_samples": c["n_samples"],
        "eq_time_ps": settings["eq_time_ps"],
        "max_lag_ps": settings["max_lag_ps"], "T_prefactor_K": settings["temperature_k"],
        "eta_exp_cP": rec["eta_exp"],
    }
    for tc in settings["cutoffs_ps"]:
        row[f"eta_at_{int(tc)}ps_cP"] = round(eta_cut[tc], 4)

    # per-run flat outputs
    pd.DataFrame([row]).to_csv(obs_dir / "viscosity_gk.csv", index=False)
    d = _decimate(len(lags), settings["curve_points"])
    pd.DataFrame({"tau_ps": lags[d], "eta_running_cP": full[d], "band_cP": band[d]}
                 ).to_csv(obs_dir / "viscosity_gk_curve.csv", index=False)
    _merge_summary(obs_dir, {**{k: row[k] for k in (
        "eta_gk_cP", "eta_err_cP", "converged", "t_lo_ps", "t_hi_ps", "tau_slow_ps",
        "frac_recovered", "t_total_ns", "eq_time_ps", "T_prefactor_K", "max_lag_ps")},
        "code_version": git_sha(),
        "generated": settings["generated"], "eta_exp_cP": rec["eta_exp"]})

    if plots:
        fig, ax = plt.subplots(figsize=(5, 4))
        flag = "OK" if p["conv"] else "NOT conv"
        _plot_panel(ax, lags, full, band, p, rec["eta_exp"],
                    f"{rec['run_name']} ({round(c['t_total_ns'])} ns)\n"
                    f"$\\eta$={p['est']:.2f} exp={rec['eta_exp'] or float('nan'):.2f} [{flag}]")
        fig.tight_layout(); fig.savefig(obs_dir / "viscosity_gk.png", dpi=130, facecolor="white")
        plt.close(fig)

    if settings["write_internal"]:
        _mirror_per_run(obs_dir, rec, settings)
    # else obs_dir already *is* the external per_run target; nothing to copy.

    # decimated curve for the aggregate 16-panel figure (keep IPC light)
    d2 = _decimate(len(lags), 4000)
    return {"rec": rec, "row": row,
            "plot": {"lags": lags[d2], "full": full[d2], "band": band[d2],
                     "fit_curve": p["fit_curve"], "est": p["est"], "conv": p["conv"],
                     "t_lo": p["t_lo"], "t_hi": p["t_hi"], "t_total_ns": c["t_total_ns"]}}


# ── aggregate figure ─────────────────────────────────────────────────────────
def convergence_grid(results: list, out: Path, exp_label: str):
    results = sorted(results, key=lambda r: (r["rec"]["salt"], r["rec"]["solvent"], r["rec"]["conc_M"]))
    n = len(results)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, res in zip(axes, results):
        rec, pl = res["rec"], res["plot"]
        flag = "OK" if pl["conv"] else "NOT conv"
        e = rec["eta_exp"]
        p = {"t_lo": pl["t_lo"], "t_hi": pl["t_hi"], "fit_curve": pl["fit_curve"],
             "est": pl["est"], "conv": pl["conv"]}
        title = (f"{SALT_LABEL.get(rec['salt'], rec['salt'])} {SOLV_LABEL[rec['solvent']]} "
                 f"{rec['conc_label'].replace('_', '.')} ({round(pl['t_total_ns'])} ns)\n"
                 f"est={pl['est']:.2f} exp={e if e is not None else float('nan'):.2f} [{flag}]")
        _plot_panel(ax, pl["lags"], pl["full"], pl["band"], p, e, title)
        ax.get_lines()[0].set_color(SOLV_COLOR[rec["solvent"]])
        for coll in ax.collections:
            coll.set_color(SOLV_COLOR[rec["solvent"]]); coll.set_alpha(0.22)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("GK viscosity — running integral η(τ) to 1 ns, block band, "
                 "Zhang-Maginn TDM fit (black dashed) → η∞ (dotted), exp (red), "
                 f"fit region (grey)  —  {exp_label}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print("saved ->", out, flush=True)


def parity_plot(master: pd.DataFrame, out: Path, exp_label: str):
    """Simulation GK eta vs experiment, log-log. Solvent = color, salt = marker,
    0.1 M = hollow; factor-2 band; non-converged marked x and excluded from MAPD."""
    from matplotlib.lines import Line2D

    def _fin(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    fig, (ax, al) = plt.subplots(1, 2, figsize=(8.4, 6),
                                 gridspec_kw={"width_ratios": [1, 0.36]})
    vals = [v for _, r in master.iterrows()
            for v in (_fin(r["eta_exp_cP"]), _fin(r["eta_gk_cP"])) if np.isfinite(v) and v > 0]
    lo, hi = min(vals) / 1.8, max(vals) * 1.8
    ax.fill_between([lo, hi], [lo/2, hi/2], [lo*2, hi*2], color="0.8", alpha=0.3, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.45", ls="--", lw=1.3, zorder=1)
    devs = []
    for _, r in master.iterrows():
        xe, ys, es = _fin(r["eta_exp_cP"]), _fin(r["eta_gk_cP"]), _fin(r["eta_err_cP"])
        if not (np.isfinite(xe) and np.isfinite(ys) and xe > 0 and ys > 0):
            continue
        c = SOLV_COLOR[r["solvent"]]
        filled = float(r["conc_M"]) != 0.1
        ax.errorbar(xe, ys, yerr=es if np.isfinite(es) else None, fmt="none", ecolor=c,
                    elinewidth=1.1, capsize=2, alpha=0.6, zorder=2)
        ax.scatter(xe, ys, s=130, marker=SALT_MARKER.get(r["salt"], "o"),
                   facecolor=c if filled else "white", edgecolor=c, linewidths=1.9, zorder=3)
        if bool(r["converged"]):
            devs.append(abs(ys - xe) / xe)
        else:
            ax.scatter(xe, ys, s=70, marker="x", color="0.15", linewidths=1.6, zorder=4)
    txt = (f"MAPD = {100*np.mean(devs):.0f}%\n(converged n = {len(devs)}/{len(master)})"
           if devs else f"none converged (n=0/{len(master)})")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7", alpha=0.85))
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi)); ax.set_aspect("equal")
    ax.set_xlabel(r"Experiment  (mPa$\cdot$s)"); ax.set_ylabel(r"Simulation GK  (cP)")
    ax.set_title(rf"Shear viscosity $\eta$ (GK) — {exp_label}")
    ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.55)
    ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.2)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    al.axis("off")
    salts = [s for s in ("napf6", "naotf", "lipf6") if (master["salt"] == s).any()]
    solvs = [s for s in ("dme", "diglyme", "pc", "tgdme") if (master["solvent"] == s).any()]
    h = [Line2D([], [], marker=SALT_MARKER[s], color="0.3", ls="", ms=9, mfc="0.3",
                label=SALT_LABEL[s]) for s in salts]
    h += [Line2D([], [], marker="o", color=SOLV_COLOR[s], ls="", ms=9, label=SOLV_LABEL[s])
          for s in solvs]
    h += [Line2D([], [], marker="o", color="0.3", ls="", ms=9, mfc="white", label="0.1 M (hollow)"),
          Line2D([], [], marker="x", color="0.15", ls="", ms=9, label="not converged")]
    al.legend(handles=h, loc="center", fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    print("saved ->", out, flush=True)


# ── per-box aggregate (shared with viscosity_gk_opls_replicas.py) ─────────────
def aggregate_by_box(master: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- s.d. of eta over each box's replicas.

    Converged replicas only when at least one converged; otherwise fall back to
    all finite replicas so the box still reports a number. Requires a "box" and
    "temperature_K" column on ``master`` (the PaiNN driver's own master doesn't
    carry these -- they're added only for the OPLS-mirrored output)."""
    rows = []
    for box, g in master.groupby("box", sort=False):
        conv = g[g["converged"] & g["eta_gk_cP"].notna()]
        use = conv if not conv.empty else g[g["eta_gk_cP"].notna()]
        vals = use["eta_gk_cP"].to_numpy(dtype=float)
        first = g.iloc[0]
        rows.append({
            "box": box, "system_id": first["system_id"], "salt": first["salt"],
            "solvent": first["solvent"], "conc_label": first["conc_label"],
            "conc_M": first["conc_M"], "temperature_K": first["temperature_K"],
            "n_replicas": int(len(g)), "n_converged": int(g["converged"].sum()),
            "n_used": int(len(vals)),
            "eta_gk_mean_cP": round(float(np.mean(vals)), 4) if len(vals) else np.nan,
            "eta_gk_std_cP": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
            "eta_gk_sem_cP": (round(float(np.std(vals, ddof=1) / np.sqrt(len(vals))), 4)
                              if len(vals) > 1 else 0.0),
            "eta_exp_cP": first["eta_exp_cP"],
        })
    df = pd.DataFrame(rows)
    df["sim_over_exp"] = df["eta_gk_mean_cP"] / df["eta_exp_cP"]
    return df.sort_values(["salt", "solvent", "conc_M", "temperature_K"]).reset_index(drop=True)


# ── driver ───────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--exp-csv", type=Path, default=DEFAULT_EXP_CSV)
    ap.add_argument("--time", choices=TP.TIME_CHOICES, default=TP.DEFAULT_TIME,
                    help="Analyze only the first 1ns/5ns of each run with windows scaled "
                         "by T/20ns (short-trajectory convergence study; see time_presets.py). "
                         "Default 20ns = full run, current behaviour.")
    ap.add_argument("--max-lag-ps", type=float, default=None,
                    help="Running integral / block band computed out to this lag. Default: "
                         "resolved from --time (1000 ps at 20ns; min(1000, 0.2*(T-eq)) else).")
    ap.add_argument("--eq-time-ps", type=float, default=None,
                    help="Discard stress rows before this time, matching the diffusivity/"
                         "conductivity equilibration cut. Default: resolved from --time "
                         "(2000 ps at 20ns, scaled by T/20ns else). 0 = use all.")
    ap.add_argument("--nblock", type=int, default=NBLOCK, help="Blocks for the error band.")
    ap.add_argument("--curve-points", type=int, default=2000,
                    help="Points kept in the saved per-run curve CSV (default 2000).")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out-root", type=Path, default=None,
                    help="External output root (default data/painn_transport_observables_replicas"
                         ", suffixed _1ns/_5ns when --time is set). All results go here.")
    ap.add_argument("--write-internal", action="store_true",
                    help="ALSO write per-run outputs into each <run_dir>/observables/ and the "
                         "aggregate into <experiment-root>/transport/gk_viscosity. Default: "
                         "write ONLY to --out-root, leaving the run tree untouched.")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix appended to the viscosity output folder (per-run + aggregate), "
                         "e.g. '_NEW', so a re-analysis doesn't clobber existing outputs.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    # Resolve --time into windows / cap / output root (explicit flags win).
    tp = TP.visc_overrides(args.time)
    if args.max_lag_ps is None:
        args.max_lag_ps = tp["max_lag_ps"]
    if args.eq_time_ps is None:
        args.eq_time_ps = tp["eq_time_ps"]
    if args.out_root is None:
        args.out_root = TP.out_root_for(args.time)
    max_time_ps = None if args.time == TP.DEFAULT_TIME else TP.TRAJ_PS[args.time]

    exp = load_exp_viscosity(args.exp_csv)
    runs, skipped = discover_runs(args.experiment_root, exp)
    print(f"Discovered {len(runs)} runs to analyze, {len(skipped)} skipped.")

    if args.dry_run:
        for r in sorted(runs, key=lambda x: (x["salt"], x["solvent"], x["conc_M"])):
            e = r["eta_exp"]
            print(f"  {r['system_id']:16s} {r['conc_label']:5s}  eta_exp="
                  f"{'--' if e is None else f'{e:.3f}'} mPa*s   {r['run_dir']}")
        for r in skipped:
            print(f"  SKIP {r['system_id']} {r['conc_label']}: {r['skip_reason']}")
        print(f"\nDRY RUN: {len(runs)} runs, {len(skipped)} skipped. Nothing computed.")
        return

    settings = dict(
        max_lag_ps=args.max_lag_ps, nblock=args.nblock, temperature_k=T_K,
        eq_time_ps=args.eq_time_ps,
        time=args.time, max_time_ps=max_time_ps,
        cutoffs_ps=CUTOFFS_PS, curve_points=args.curve_points,
        out_suffix=args.out_suffix, out_root=args.out_root,
        write_internal=args.write_internal,
        generated=datetime.now().isoformat(timespec="seconds"),
    )

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
                print(f"[progress] {done}/{n}  {tag:22s} eta={row['eta_gk_cP']:.2f} cP "
                      f"±{row['eta_err_cP']:.2f}  conv={row['converged']}  "
                      f"tau_slow={row['tau_slow_ps']:.0f}ps  ({row['t_total_ns']:.1f} ns)",
                      flush=True)
            except Exception as e:
                print(f"[progress] {done}/{n}  {tag:22s} ERROR: {e}", flush=True)

    if not results:
        print("No runs analyzed; nothing to aggregate.")
        return

    master = pd.DataFrame([res["row"] for res in results]).sort_values(
        ["salt", "solvent", "conc_M", "replica"]).reset_index(drop=True)
    exp_label = args.experiment_root.name.replace("_", " ")

    cfg = dict(
        observable="viscosity_gk", estimator="zhang_maginn_tdm",
        experiment_root=str(args.experiment_root), viscosity_exp_source=str(args.exp_csv),
        settings={k: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                  for k, v in settings.items() if k not in ("generated", "write_internal")},
        write_internal=args.write_internal,
        n_runs_analyzed=len(results), n_skipped=len(skipped),
        n_converged=int(master["converged"].sum()),
        skipped_runs=[f"{r['system_id']} {r['conc_label']} ({r['skip_reason']})" for r in skipped],
        code_version=git_sha(), generated=settings["generated"],
    )

    # ── internal aggregate (opt-in): <experiment-root>/transport/gk_viscosity<suffix>/ ─
    if args.write_internal:
        out_dir = args.experiment_root / "transport" / f"gk_viscosity{args.out_suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        master.to_csv(out_dir / "viscosity_gk_master.csv", index=False)
        (out_dir / "viscosity_gk_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        if not args.no_plots:
            convergence_grid(results, out_dir / "figure_transport_visc_convergence16.png", exp_label)
            parity_plot(master, out_dir / "figure_transport_visc_parity.png", exp_label)

    # ── external output (always): OUT_ROOT/viscosity<suffix>/, OPLS-mirrored layout ──
    mirror_dir = Path(args.out_root) / f"viscosity{args.out_suffix}"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    mirror_master = master.copy()
    # computed straight from columns (not a run_name-keyed lookup): run_name alone
    # isn't unique across systems ("nvt_transport_1M_298K_20ns_1ps" repeats per system).
    mirror_master["box"] = (mirror_master["system_id"] + "_" + mirror_master["conc_label"]
                            + "_" + str(int(round(T_K))))
    mirror_master["temperature_K"] = T_K
    mirror_master.to_csv(mirror_dir / "viscosity_gk_master.csv", index=False)
    mirror_bybox = aggregate_by_box(mirror_master)
    mirror_bybox.to_csv(mirror_dir / "viscosity_gk_by_box.csv", index=False)
    mirror_cfg = dict(cfg, forcefield="PaiNN/FeNNol",
                      opls_counterpart="transport/viscosity_gk_opls_replicas.py",
                      out_dir=str(mirror_dir), n_boxes=len(mirror_bybox))
    (mirror_dir / "viscosity_gk_config.yaml").write_text(yaml.safe_dump(mirror_cfg, sort_keys=False))
    if not args.no_plots:
        if args.write_internal:
            shutil.copy2(out_dir / "figure_transport_visc_convergence16.png",
                         mirror_dir / "figure_transport_visc_convergence.png")
            shutil.copy2(out_dir / "figure_transport_visc_parity.png",
                         mirror_dir / "figure_transport_visc_parity.png")
        else:
            convergence_grid(results, mirror_dir / "figure_transport_visc_convergence.png", exp_label)
            parity_plot(mirror_master, mirror_dir / "figure_transport_visc_parity.png", exp_label)

    print("\n=== GK viscosity (TDM fit) ===")
    cols = ["system_id", "conc_label", "replica", "t_total_ns", "eta_gk_cP", "eta_err_cP",
            "converged", "t_hi_ps", "tau_slow_ps", "frac_recovered", "eta_exp_cP"]
    print(master[cols].to_string(index=False))
    print(f"\nDone: {len(results)} runs "
          f"({int(master['converged'].sum())} converged) -> {mirror_dir}/viscosity_gk_master.csv")
    print(f"  external output -> {mirror_dir}/ (per_run/<box>/r<k>, *_master.csv, *_by_box.csv)")
    if args.write_internal:
        print(f"  internal too    -> {out_dir}/ + <run_dir>/observables/viscosity_<N>ns/")


if __name__ == "__main__":
    main()
