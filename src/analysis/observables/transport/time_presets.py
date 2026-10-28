#!/usr/bin/env python
"""Trajectory-length presets for the transport convergence study.

The appendix convergence study recomputes diffusivity, conductivity and GK
viscosity from only the first 1 ns / 2 ns / 5 ns of the 20 ns PaiNN NVT runs, with
every analysis window scaled proportionally to the trajectory length
(``f = T / 20 ns``) plus a few floors where linear scaling breaks physics or
statistics:

* fickian window search: ``t_floor_ps`` / ``tau_min_fit_ps`` (5 ps) stay
  absolute -- the ballistic/caging floor is physics, not statistics; the upper
  bounds are already fractions of the lag axis and scale automatically.
* GK viscosity ``max_lag_ps``: capped at one block of five,
  ``min(1000 ps, 0.2 * (T - eq))``, so the block error band stays defined.
* conductivity stride (``load_dt_ps``): 10 ps * f would drop below the 1 ps
  dump interval, so the short presets just use every frame.

``--time 20ns`` (the default everywhere) resolves to the existing behaviour
bit-identically: unscaled windows and the unsuffixed output root.

This module is imported by the three ``*_replicas.py`` drivers and never
imports them (base settings dicts are passed in as arguments), so there are no
circular imports and the OPLS drivers -- which import those base dicts -- are
unaffected.
"""
from __future__ import annotations

from pathlib import Path

from .. import paths

TIME_CHOICES = ("1ns", "2ns", "5ns", "20ns")
DEFAULT_TIME = "20ns"
FULL_PS = 20000.0
TRAJ_PS = {"1ns": 1000.0, "2ns": 2000.0, "5ns": 5000.0, "20ns": FULL_PS}

# Base (20 ns) values whose scaled versions are computed by rule below. Kept as
# names, not copies, so the single sources of truth stay in the driver scripts
# (SETTINGS_FIXED) and conductivity_common (WINDOWS["byteff2_ase"]).
_VISC_BASE_EQ_PS = 2000.0
_VISC_BASE_MAX_LAG_PS = 1000.0


def factor(label: str) -> float:
    return TRAJ_PS[label] / FULL_PS


def cap_frames(label: str, dt_ps: float) -> int | None:
    """Total-frame cap realising "first T of the run" (None = no cap, 20 ns)."""
    if label == DEFAULT_TIME:
        return None
    return int(round(TRAJ_PS[label] / dt_ps))


def out_root_for(label: str) -> Path:
    """``…/painn_transport_observables_replicas[_1ns|_5ns]``."""
    base = paths.OUT_PAINN_TRANSPORT_REPLICAS
    if label == DEFAULT_TIME:
        return base
    return base.with_name(f"{base.name}_{label}")


def scale_diffusivity(base: dict, label: str) -> dict:
    """Overlay a diffusivity SETTINGS dict with proportionally scaled times.

    Scales ``eq_time_ps`` always, and ``tau_min_fit_ps`` only for the fixed
    method (the fickian 5 ps value is an absolute ballistic/caging floor).
    All upper fit bounds are fractions of the lag axis and need no scaling.
    """
    if label == DEFAULT_TIME:
        return dict(base)
    f = factor(label)
    out = dict(base)
    if "eq_time_ps" in base:          # the Fickian window settings have no cut of their own
        out["eq_time_ps"] = base["eq_time_ps"] * f
    if base.get("method") == "fixed":
        out["tau_min_fit_ps"] = base["tau_min_fit_ps"] * f
    return out


def visc_overrides(label: str) -> dict:
    """Resolved ``eq_time_ps`` / ``max_lag_ps`` for the GK viscosity driver."""
    f = factor(label)
    eq = _VISC_BASE_EQ_PS * f
    return {
        "eq_time_ps": eq,
        "max_lag_ps": min(_VISC_BASE_MAX_LAG_PS, 0.2 * (TRAJ_PS[label] - eq)),
    }


def cond_window_name(label: str) -> str:
    """Name of the conductivity_common.WINDOWS preset for this length."""
    return "byteff2_ase" if label == DEFAULT_TIME else f"byteff2_ase_{label}"
