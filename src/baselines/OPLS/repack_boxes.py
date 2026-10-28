#!/usr/bin/env python
"""Second pass of the molarity correction: repack every box at the density the
CURRENT force field actually equilibrates to.

The packed boxes fix the solute:solvent ratio so that, at the force field's own
equilibrium density, the molarity equals the label. That density depends on the
parameters, so after a force-field change the ratio has to be recomputed:

  pass 1   NPT on the existing boxes (run_equilibration.py --npt-ns 6): a
           density-determination run, longer than the 1 ns seed equilibration
           because viscous boxes (tetraglyme) relax their density slowly
  here     read <rho> over the last half of each pass-1 NPT, check that it is
           no longer drifting (second vs first half of that window), solve the same
           mass balance as src/data_prep/box_creation/build_boxes_correct_M.py
               V_target  = n_salt / (M_target * N_A)
               n_solvent = round((rho * V_target - n_salt * m_pair) / m_solvent)
           and pack the new composition with Packmol
  pass 2   build_opls_boxes.py + run_equilibration.py on the repacked boxes

The salt count is kept; only the solvent count changes. The packed cell is
scaled from the pass-1 packed cell by (N_atoms_new / N_atoms_old)^(1/3), so the
packing looseness is the same as before. Molecule files, masses and the Packmol
deck writer are imported from build_boxes_correct_M.py so the two cannot drift.

Input:  <eq-root>/<box>/{meta.json,npt.edr,npt.xtc,npt.log}   (pass-1 equilibration)
Output: <out>/<system>/initial_<conc>.xyz + packmol_<conc>.inp
        <out>/repack_summary.csv   densities, old/new counts, predicted molarity

Usage:
    source config.sh
    python repack_boxes.py --eq-root runs/pass1/equilibrated --out runs/pass2/boxes_root
    python repack_boxes.py --validate      # planner reproduces the existing counts
"""
import os
import sys
import csv
import json
import glob
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from ase.io import read, write
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from src.data_prep.box_creation import build_boxes_correct_M as bcm   # noqa: E402
from src.analysis.observables.systems import lookup_system            # noqa: E402

PACKMOL = os.environ.get("PACKMOL", "packmol")


def plan(system, conc_label, n_salt, rho_g_cm3):
    """(n_solvent, predicted molarity at rho) for one box. Same mass balance as
    build_boxes_correct_M.plan_box, with the salt count given instead of read
    from an exp01 seed frame."""
    spec = lookup_system(system)
    m_cat, _ = bcm._mol_info(bcm.CATION_FILE[spec.cation])
    m_an, n_at_an = bcm._mol_info(bcm.ANION_FILE[spec.anion])
    m_solv, n_at_solv = bcm._mol_info(bcm.SOLVENT_FILE[spec.solvent])
    m_pair = m_cat + m_an
    M_target = bcm.CONC_TO_M[conc_label]
    v_target = n_salt / (M_target * bcm.MOL_PER_L_TO_PER_A3)
    mass_tot = bcm.mass_amu_from_density(rho_g_cm3, v_target)
    n_solvent = max(1, round((mass_tot - n_salt * m_pair) / m_solv))
    v_eq = bcm.vol_a3_from_mass_density(n_salt * m_pair + n_solvent * m_solv, rho_g_cm3)
    n_atoms = n_solvent * n_at_solv + n_salt * (1 + n_at_an)
    return dict(spec=spec, n_solvent=n_solvent, M_pred=bcm.molarity(n_salt, v_eq),
                n_atoms=n_atoms, M_target=M_target)


def solvent_count(meta):
    return next(v for k, v in meta["molecule_counts"].items()
                if k not in ("NA", "LI", "PF6", "OTF"))


def pass1_density(run_dir, n_salt, window_frac=0.5):
    """Equilibrium density of a finished pass-1 NPT from npt.edr.

    Averages over the last `window_frac` of the run and reports the drift
    between the two halves of that window, so an unconverged density is visible
    instead of silently setting the composition. Returns None if unfinished."""
    import prepare_production as pp
    log = os.path.join(run_dir, "npt.log")
    if not os.path.exists(log) or "Finished mdrun" not in open(log).read():
        return None
    ser = pp.edr_series(run_dir)
    t_end = ser["Density"][-1][0]
    t0 = t_end * (1.0 - window_frac)
    t_mid = 0.5 * (t0 + t_end)
    rho = [(t, v) for t, v in ser["Density"] if t >= t0]
    vol = [v for t, v in ser["Volume"] if t >= t0]
    mean = sum(v for _, v in rho) / len(rho)
    std = (sum((v - mean) ** 2 for _, v in rho) / len(rho)) ** 0.5
    first = [v for t, v in rho if t < t_mid]
    second = [v for t, v in rho if t >= t_mid]
    drift_pct = 100.0 * (sum(second) / len(second) - sum(first) / len(first)) / mean
    v_mean = sum(vol) / len(vol)
    return {"rho_mean_kgm3": mean, "rho_std_kgm3": std, "drift_pct": drift_pct,
            "window_ps": (t0, t_end), "M": pp.molarity(n_salt, v_mean)}


def pack_one(task):
    (system, conc, n_salt, n_solvent, L, out_root, tolerance) = task
    spec = lookup_system(system)
    out_dir = Path(out_root) / system
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"initial_{conc}.xyz"
    if final.exists():
        return f"{system} {conc}: exists, skipped"
    inp = out_dir / f"packmol_{conc}.inp"
    packed = out_dir / f"packed_{conc}.xyz"
    bcm.write_packmol_input(inp, packed, spec, n_solvent, n_salt, L, tolerance=tolerance)
    r = subprocess.run([PACKMOL], stdin=open(inp), capture_output=True, text=True, timeout=3600)
    if r.returncode != 0 or not packed.exists() or "Success!" not in r.stdout:
        (out_dir / f"packmol_{conc}.log").write_text(r.stdout[-5000:] + "\n" + r.stderr[-2000:])
        return f"{system} {conc}: ** FAILED (see packmol_{conc}.log)"
    atoms = read(str(packed))
    atoms.set_cell([L, L, L])
    atoms.set_pbc(True)
    write(str(final), atoms)
    packed.unlink()
    return f"{system} {conc}: OK  L={L:.2f} A  solvent={n_solvent}"


def validate():
    """With the densities the existing boxes were built from, the planner must
    return exactly the solvent counts those boxes contain."""
    old = bcm.load_opls_densities(bcm.OPLS_CSV, 298)
    boxes_root = os.environ.get("OPLS_XYZ_ROOT", str(HERE / "boxes_root"))
    bad = 0
    for system, conc in bcm.SYSTEMS:
        xyz = Path(boxes_root) / system / f"initial_{conc}.xyz"
        if not xyz.exists():
            print(f"  {system} {conc}: no packed box at {xyz}")
            bad += 1
            continue
        atoms = read(str(xyz))
        spec = lookup_system(system)
        n_salt = atoms.get_chemical_symbols().count(spec.cation)
        _, n_at_an = bcm._mol_info(bcm.ANION_FILE[spec.anion])
        _, n_at_solv = bcm._mol_info(bcm.SOLVENT_FILE[spec.solvent])
        n_solv_old = (len(atoms) - n_salt * (1 + n_at_an)) // n_at_solv
        p = plan(system, conc, n_salt, old[bcm._opls_key(system, conc)])
        ok = p["n_solvent"] == n_solv_old
        bad += not ok
        print(f"  {system+' '+conc:22s} salt {n_salt:3d}  solvent old {n_solv_old:4d}  "
              f"planner {p['n_solvent']:4d}  {'ok' if ok else 'MISMATCH'}")
    print("planner reproduces the existing boxes" if not bad else f"{bad} mismatch(es)")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eq-root", help="pass-1 equilibration root")
    ap.add_argument("--out", help="output boxes root for pass 2")
    ap.add_argument("--tolerance", type=float, default=2.0, help="Packmol tolerance, A")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count()))
    ap.add_argument("--window-frac", type=float, default=0.5,
                    help="average the density over this last fraction of the pass-1 NPT (default 0.5)")
    ap.add_argument("--max-drift-pct", type=float, default=0.5,
                    help="flag a box whose density changes by more than this between the two "
                         "halves of the averaging window (default 0.5)")
    ap.add_argument("--dry-run", action="store_true", help="plan and report, pack nothing")
    ap.add_argument("--validate", action="store_true",
                    help="check the planner against the existing packed boxes and exit")
    args = ap.parse_args()

    if args.validate:
        sys.exit(1 if validate() else 0)
    if not args.eq_root or not args.out:
        ap.error("--eq-root and --out are required")

    rows, tasks = [], []
    dirs = [d for d in sorted(glob.glob(os.path.join(args.eq_root, "*")))
            if os.path.exists(os.path.join(d, "meta.json"))]
    for d in tqdm(dirs, desc="pass-1 densities"):
        meta = json.load(open(os.path.join(d, "meta.json")))
        seed = pass1_density(d, meta["n_salt_pairs"], args.window_frac)
        if seed is None:
            print(f"  {meta['name']}: pass-1 NPT not finished", file=sys.stderr)
            continue
        rho = seed["rho_mean_kgm3"] / 1000.0
        n_salt = meta["n_salt_pairs"]
        n_solv_old = solvent_count(meta)
        p = plan(meta["system"], meta["concentration_label"], n_salt, rho)
        L = meta["box_length_A"] * (p["n_atoms"] / meta["n_atoms"]) ** (1.0 / 3.0)
        rows.append({
            "box": meta["name"], "system": meta["system"],
            "conc_label": meta["concentration_label"], "target_M": p["M_target"],
            "n_salt": n_salt,
            "rho_pass1_g_cm3": round(rho, 5),
            "rho_pass1_std_g_cm3": round(seed["rho_std_kgm3"] / 1000.0, 5),
            "rho_drift_pct": round(seed["drift_pct"], 3),
            "density_window_ps": f"{seed['window_ps'][0]:.0f}-{seed['window_ps'][1]:.0f}",
            "M_pass1": round(seed["M"], 5),
            "M_pass1_dev_pct": round(100 * (seed["M"] / p["M_target"] - 1), 2),
            "n_solvent_old": n_solv_old, "n_solvent_new": p["n_solvent"],
            "M_pred_pass2": round(p["M_pred"], 5),
            "M_pred_pass2_dev_pct": round(100 * (p["M_pred"] / p["M_target"] - 1), 2),
            "L_packed_A": round(L, 4),
        })
        tasks.append((meta["system"], meta["concentration_label"], n_salt, p["n_solvent"],
                      L, args.out, args.tolerance))

    print(f"\n{'box':<26}{'rho':>8}{'drift%':>8}{'M pass1':>9}{'dev%':>7}{'solv old':>9}{'new':>6}{'M pred':>8}{'dev%':>7}")
    for r in rows:
        flag = "  <-- density still drifting" if abs(r["rho_drift_pct"]) > args.max_drift_pct else ""
        print(f"{r['box']:<26}{r['rho_pass1_g_cm3']:>8.4f}{r['rho_drift_pct']:>8.3f}{r['M_pass1']:>9.4f}{r['M_pass1_dev_pct']:>7.2f}"
              f"{r['n_solvent_old']:>9d}{r['n_solvent_new']:>6d}{r['M_pred_pass2']:>8.4f}{r['M_pred_pass2_dev_pct']:>7.2f}{flag}")
    n_drift = sum(abs(r["rho_drift_pct"]) > args.max_drift_pct for r in rows)
    if n_drift:
        print(f"\nWARNING: {n_drift} box(es) drift more than {args.max_drift_pct}% across the "
              "averaging window; their pass-1 density may not be converged.")
    if not rows:
        sys.exit("no finished pass-1 boxes")
    if args.dry_run:
        return

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "repack_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(tqdm(ex.map(pack_one, tasks), total=len(tasks), desc="packmol"))
    print()
    for r in results:
        print("  " + r)
    n_fail = sum("FAILED" in r for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} boxes packed -> {args.out}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
