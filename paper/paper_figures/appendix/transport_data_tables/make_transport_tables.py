"""Generate the two appendix transport-data tables (LaTeX) from the replica CSVs.

Reads the per-replica master CSVs for the distilled PaiNN model and the OPLS
reference and emits, as ``mean(std)`` over replicas:

  Tables 1a/1b (diffusivity, one per model):  raw self-diffusivities D+/D-/D0
     (no Yeh--Hummer), the additive Yeh--Hummer correction dD_YH computed with
     the simulated and with the experimental viscosity (species-independent),
     and the experimental D+/D-/D0 for reference. All in 1e-10 m^2/s.

  Table 2 (conductivity + viscosity):  Onsager conductivity sigma (mS/cm) and
     Green--Kubo viscosity eta (cP) for PaiNN, OPLS, and experiment.

All simulated entries are mean +/- std over replicas.

The Onsager (not Nernst--Einstein) conductivity is used, matching figure 3 of
the main text. Experimental columns are single reference values (no std).

    python make_transport_tables.py            # prints both tables
    python make_transport_tables.py --write     # also writes tables_*.tex here
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.observables import paths

REPO = Path(__file__).resolve().parents[4]
PAINN = paths.OUT_PAINN_TRANSPORT_REPLICAS
OPLS = paths.OUT_OPLS_TRANSPORT

# (salt, solvent, conc_M) in the display order used by the existing table.
ROWS = [
    ("lipf6", "dme", 0.5),
    ("napf6", "dme", 0.1), ("napf6", "dme", 1.0),
    ("napf6", "pc", 0.1), ("napf6", "pc", 1.0),
    ("napf6", "diglyme", 0.1), ("napf6", "diglyme", 1.0),
    ("napf6", "tgdme", 1.0),
    ("naotf", "dme", 0.1), ("naotf", "dme", 1.0),
    ("naotf", "pc", 0.1), ("naotf", "pc", 1.0),
    ("naotf", "diglyme", 0.1), ("naotf", "diglyme", 1.0),
    ("naotf", "tgdme", 0.1), ("naotf", "tgdme", 1.0),
]

SALT_TEX = {"lipf6": r"LiPF$_6$", "napf6": r"NaPF$_6$", "naotf": r"NaOTf"}
SOLV_TEX = {"dme": "DME", "pc": "PC", "diglyme": "DEGDME", "tgdme": "TEGDME"}


def load():
    d = {}
    d["painn_diff"] = pd.read_csv(PAINN / "diffusivity" / "diffusivity_master.csv")
    d["opls_diff"] = pd.read_csv(OPLS / "diffusivity" / "diffusivity_master.csv")
    d["painn_cond"] = pd.read_csv(PAINN / "conductivity" / "conductivity_master.csv")
    d["opls_cond"] = pd.read_csv(OPLS / "conductivity" / "conductivity_master.csv")
    d["painn_visc"] = pd.read_csv(PAINN / "viscosity" / "viscosity_gk_master.csv")
    d["opls_visc"] = pd.read_csv(OPLS / "viscosity" / "viscosity_gk_master.csv")
    return d


def _sel(df, salt, solvent, conc):
    m = (df["salt"] == salt) & (df["solvent"] == solvent) & (np.isclose(df["conc_M"], conc))
    return df[m]


def ms(df, salt, solvent, conc, col):
    """(mean, std) of ``col`` over replicas, NaNs dropped. (nan, nan) if empty."""
    sub = _sel(df, salt, solvent, conc)
    if sub.empty or col not in sub:
        return np.nan, np.nan
    v = pd.to_numeric(sub[col], errors="coerce").dropna().values
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def const(df, salt, solvent, conc, col):
    """A single reference value (experimental column, constant over replicas)."""
    m, _ = ms(df, salt, solvent, conc, col)
    return m


def fmt_ms(mean, std, nd=2):
    """``mean $\\pm$ std`` cell; ``--`` when absent."""
    if mean is None or not np.isfinite(mean):
        return "--"
    if std is None or not np.isfinite(std):
        return f"{mean:.{nd}f}"
    return rf"{mean:.{nd}f} $\pm$ {std:.{nd}f}"


def fmt_one(v, nd=2):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


# ── Table 1a / 1b: diffusivity, one upright table per model ──────────────────────
def diffusivity_table(df, model_name, label):
    """Per-model diffusivity table: raw D+/D-/D0, the two Yeh--Hummer corrections,
    and experimental D+/D-/D0 for reference."""
    lines = []
    for salt, solvent, conc in ROWS:
        cells = [SALT_TEX[salt], SOLV_TEX[solvent], f"{conc:g}"]
        for sp in ("cation", "anion", "solvent"):          # raw D (no YH), Fickian window as in Fig. 4
            cells.append(fmt_ms(*ms(df, salt, solvent, conc, f"D_{sp}_fick_pbc_1e10")))
        # dD_YH is essentially constant across replicas, so report the mean only.
        cells.append(fmt_one(ms(df, salt, solvent, conc, "yh_dD_sim_1e10")[0]))  # dD (sim visc)
        cells.append(fmt_one(ms(df, salt, solvent, conc, "yh_dD_exp_1e10")[0]))  # dD (exp visc)
        for sp in ("cation", "anion", "solvent"):          # experimental D
            cells.append(fmt_one(const(df, salt, solvent, conc, f"D_{sp}_exp_1e10")))
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)

    return rf"""\begin{{table}}[h]
\centering
\caption{{Raw (uncorrected) self-diffusivities and additive Yeh--Hummer finite-size corrections $\Delta D_{{\mathrm{{YH}}}}$ for {model_name}, with experimental self-diffusivities for reference. Simulated entries are mean\,$\pm$\,std over replicas; $\Delta D_{{\mathrm{{YH}}}}$ is species-independent (``sim''/``exp'' denote the viscosity used to evaluate it, and the finite-size corrected diffusivity is calculated by adding it to the raw value). All values in $10^{{-10}}$~m$^2$/s; ``--'' denotes values not available.}}
\label{{{label}}}
\footnotesize
\setlength{{\tabcolsep}}{{3.5pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lll ccc cc ccc}}
\toprule
 & & & \multicolumn{{3}}{{c}}{{$D_{{\mathrm{{raw}}}}$ (sim.)}}
     & \multicolumn{{2}}{{c}}{{$\Delta D_{{\mathrm{{YH}}}}$}}
     & \multicolumn{{3}}{{c}}{{Experimental}} \\
\cmidrule(lr){{4-6}}\cmidrule(lr){{7-8}}\cmidrule(lr){{9-11}}
Salt & Solvent & Conc. & $D^+$ & $D^-$ & $D^0$ & sim & exp & $D^+$ & $D^-$ & $D^0$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}"""


# ── Table 2: conductivity + viscosity ───────────────────────────────────────────
def cond_visc_table(d):
    pnc, opc = d["painn_cond"], d["opls_cond"]
    pnv, opv = d["painn_visc"], d["opls_visc"]
    lines = []
    for salt, solvent, conc in ROWS:
        cells = [SALT_TEX[salt], SOLV_TEX[solvent], f"{conc:g}"]
        # conductivity: PaiNN, OPLS (Onsager, mean(std)), then experiment
        cells.append(fmt_ms(*ms(pnc, salt, solvent, conc, "sigma_onsager_mS_cm")))
        cells.append(fmt_ms(*ms(opc, salt, solvent, conc, "sigma_onsager_mS_cm")))
        cells.append(fmt_one(const(pnc, salt, solvent, conc, "sigma_exp_mS_cm")))
        # viscosity: PaiNN, OPLS (GK, mean(std)), then experiment
        cells.append(fmt_ms(*ms(pnv, salt, solvent, conc, "eta_gk_cP")))
        cells.append(fmt_ms(*ms(opv, salt, solvent, conc, "eta_gk_cP")))
        cells.append(fmt_one(const(pnv, salt, solvent, conc, "eta_exp_cP")))
        lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(lines)

    return rf"""\begin{{table}}[h]
\centering
\caption{{Simulated and experimental ionic conductivity $\sigma$ (Onsager, mS/cm) and shear viscosity $\eta$ (cP). Simulated entries are mean\,$\pm$\,std over replicas for the distilled PaiNN model and OPLS-AA; ``--'' denotes values not available.}}
\label{{tab:transport-cond-visc}}
\small
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{lll ccc ccc}}
\toprule
 & & & \multicolumn{{3}}{{c}}{{$\sigma$ (mS/cm)}}
     & \multicolumn{{3}}{{c}}{{$\eta$ (cP)}} \\
\cmidrule(lr){{4-6}}\cmidrule(lr){{7-9}}
Salt & Solvent & Conc. & PaiNN & OPLS-AA & Exp. & PaiNN & OPLS-AA & Exp. \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="Write tables_diffusivity.tex / tables_cond_visc.tex here.")
    args = ap.parse_args(argv)

    d = load()
    t1a = diffusivity_table(d["painn_diff"], "PaiNN", "tab:transport-diffusivity-painn")
    t1b = diffusivity_table(d["opls_diff"], "OPLS-AA", "tab:transport-diffusivity-opls")
    t2 = cond_visc_table(d)
    for t in (t1a, t1b, t2):
        print(t)
        print()

    if args.write:
        here = Path(__file__).resolve().parent
        (here / "table_diffusivity_painn.tex").write_text(t1a + "\n")
        (here / "table_diffusivity_opls.tex").write_text(t1b + "\n")
        (here / "table_cond_visc.tex").write_text(t2 + "\n")
        print(f"wrote -> {here}/table_diffusivity_painn.tex, "
              "table_diffusivity_opls.tex, table_cond_visc.tex")


if __name__ == "__main__":
    main()
