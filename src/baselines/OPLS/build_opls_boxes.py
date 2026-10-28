#!/usr/bin/env python
"""Turn packed .xyz boxes into self-contained OPLS/GROMACS run dirs.

Reads each initial_*.xyz under a boxes root laid out as
<boxes_root>/<system>/initial_<conc>.xyz (see the README), perceives the
molecules with build_system_lpg (LigParGen OPLS-AA/CM1A), and writes one
directory per (system, concentration, temperature) into opls_ready_boxes/
holding the .gro, the .top and every .itp it needs. The equilibration
(NPT) and production (NVT) mdps are written by run_equilibration.py /
run_production.py, which own the actual MD protocol.

LiPF6 gets three temperature replicas (273/298/323 K); everything else 298 K.
Dir names end in the temperature so run_equilibration.py / run_production.py
get_temperature() picks it up.

Usage:
    python build_opls_boxes.py                        # read ./boxes_root
    python build_opls_boxes.py --src /path/to/boxes_root --out ./opls_ready_boxes
"""
import os
import re
import json
import shutil
import argparse
from ase.io import read

from lib import build_system_lpg as bsl

HERE = os.path.dirname(os.path.abspath(__file__))
# Packed-box input root: <boxes_root>/<system>/initial_<conc>.xyz (see README).
# Override with --src or $OPLS_XYZ_ROOT.
SRC_ROOT = os.environ.get('OPLS_XYZ_ROOT', os.path.join(HERE, 'boxes_root'))
OUT_ROOT = os.environ.get('OPLS_BOX_ROOT', os.path.join(HERE, 'opls_ready_boxes'))

N_AVOGADRO = 6.02214076e23
CATIONS = ('NA', 'LI')


def parse_molarity(label):
    """'0_1M' -> 0.1, '1M' -> 1.0, '0_5M' -> 0.5."""
    return float(label.rstrip('M').replace('_', '.'))


def localize_top(top_path, box_dir):
    """Rewrite absolute #include paths to bare filenames, copying files in.

    build_system_lpg emits absolute includes (the box dir for the LigParGen
    itps, the vendored molecules/ dir for the ion itps). Making them local means
    the run dir can be moved or copied without the topology breaking.
    """
    with open(top_path) as f:
        text = f.read()

    for inc in re.findall(r'#include\s+"([^"]+)"', text):
        base = os.path.basename(inc)
        dest = os.path.join(box_dir, base)
        if not os.path.exists(dest):
            shutil.copy(inc, dest)
        text = text.replace(f'#include "{inc}"', f'#include "{base}"')

    with open(top_path, 'w') as f:
        f.write(text)


def build_one(system, conc_label, xyz_path, temp, out_root):
    name = f'{system}_{conc_label}_{int(temp)}'
    box_dir = os.path.join(out_root, name)
    if os.path.isdir(box_dir):
        shutil.rmtree(box_dir)

    _gro, top, counts = bsl.build_system(name, xyz_path, box_dir, temperature=temp)
    localize_top(top, box_dir)
    # No mdp is written here: the EM, NPT and NVT mdps are owned by
    # run_equilibration.py / run_production.py, where the protocol lives.

    # Molarity from the packed box: n_salt / (N_A * V)
    atoms = read(xyz_path, index=-1)
    volume_L = atoms.get_volume() * 1e-27  # A^3 -> L
    n_salt = sum(v for k, v in counts.items() if k in CATIONS)
    actual = n_salt / (N_AVOGADRO * volume_L)
    target = parse_molarity(conc_label)

    meta = {
        'name': name,
        'system': system,
        'concentration_label': conc_label,
        'target_molarity_M': target,
        'actual_molarity_M': actual,
        'rel_error_pct': 100.0 * (actual - target) / target,
        'temperature_K': temp,
        'n_salt_pairs': n_salt,
        'molecule_counts': dict(counts),
        'box_length_A': float(atoms.cell.lengths()[0]),
        'volume_A3': float(atoms.get_volume()),
        'n_atoms': len(atoms),
        'source_xyz': xyz_path,
    }
    with open(os.path.join(box_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', default=SRC_ROOT,
                    help='boxes root: <system>/initial_<conc>.xyz '
                         f'(default: {SRC_ROOT}; or $OPLS_XYZ_ROOT)')
    ap.add_argument('--out', default=OUT_ROOT,
                    help=f'output dir for the OPLS boxes (default: {OUT_ROOT})')
    ap.add_argument('--lipf6-temps', nargs='+', type=int, default=[273, 298, 323],
                    help='temperatures (K) to build for the lipf6 systems '
                         '(default: 273 298 323; pass 298 for the 16-box paper set)')
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(
            f'boxes root not found: {args.src}\n'
            'Point --src (or $OPLS_XYZ_ROOT) at a boxes root laid out as '
            '<boxes_root>/<system>/initial_<conc>.xyz (see the README).')
    os.makedirs(args.out, exist_ok=True)
    records = []

    for system in sorted(os.listdir(args.src)):
        sys_dir = os.path.join(args.src, system)
        if not os.path.isdir(sys_dir):
            continue
        # LiPF6 is the temperature-series system
        temps = args.lipf6_temps if system.startswith('lipf6') else [298]

        for fname in sorted(os.listdir(sys_dir)):
            if not (fname.startswith('initial_') and fname.endswith('.xyz')):
                continue
            conc_label = fname[len('initial_'):-len('.xyz')]
            xyz_path = os.path.join(sys_dir, fname)
            for temp in temps:
                records.append(build_one(system, conc_label, xyz_path, temp, args.out))

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(records, f, indent=2)

    print(f'\nBuilt {len(records)} boxes into {args.out}')
    for r in records:
        print(f"  {r['name']:<28} target={r['target_molarity_M']:.2f} M  "
              f"actual={r['actual_molarity_M']:.4f} M  "
              f"({r['rel_error_pct']:+.2f}%)")


if __name__ == '__main__':
    main()
