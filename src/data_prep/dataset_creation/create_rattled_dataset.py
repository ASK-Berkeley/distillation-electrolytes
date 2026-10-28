"""Create a rattled training dataset from one or more UMA MD trajectories.

Pass one trajectory per group via ``--labels`` (``--concentrations`` is a
backward-compatible alias). Groups may be different concentrations of one
salt/solvent (0.1 M, 1 M, 2 M) OR entirely different systems (naotf_dme,
napf6_pc, ...). All groups are pooled into a single combined dataset so one
model can be trained across them.

End-to-end pipeline:
    1. Read each UMA trajectory on two disjoint, interleaved time grids within
       the first `duration_ps`: MD frames every `sampling_interval_fs` starting
       at t=0, and noised frames on the *same* grid shifted by `noised_offset_fs`
       (so MD at 0, 50, 100 ... fs and noised at 20, 70, 120 ... fs). The two
       grids never collide, so no noised frame shares a geometry with an MD one.
    2. Rattle the noised set (Gaussian noise added to the atomic positions); the
       MD set is kept as-is.
    3. Label with UMA (energy / forces / stress), one trajectory at a time
       with a freshly-loaded calculator per trajectory. The noised frames are
       always (re)labeled, since displacing the atoms changes the labels. MD
       frames that carry stored UMA labels (labeled trajectories) are kept as-is;
       MD frames from position-only trajectories are UMA-labeled too (same model,
       so the dataset stays single-model consistent). The calculator is reloaded
       per trajectory because merge_mole=True specializes UMA to one fixed
       composition and asserts every frame matches it (see load_uma_calculator).
    4. Split each trajectory's MD and noised frames 90/10 into train/val
       independently, THEN pool the splits across trajectories (so val holds a
       proportional ~10% of every trajectory and source). Compute a per-element
       linear energy reference (linref) from the pooled training set and subtract
       it so energies are dereferenced.
    5. Save 3 pickles — training_all.pkl (MD + noised), training_md_data.pkl
       (MD only), training_noised.pkl (noised only) — plus linref.json. Every
       frame carries an `is_noised` bool (False = pure MD, True = rattled) so
       training_all.pkl is self-describing; FeNNol collates it as an unused
       per-frame array and ignores it.
    6. Dump one simulation_start_<label>.xyz per trajectory: a single frame
       pulled from that trajectory (default: the 100 ps mark) to seed downstream
       MD simulations.

DEFAULTS: assumes each .traj saves a frame every 10 fs. Uses the first 100 ps,
sampling MD frames every 50 fs (2000 frames) and noised frames every 50 fs on a
grid offset by 20 fs (2000 frames) — 2000 noised + 2000 unnoised (MD) = 4000
frames PER trajectory. With 2 trajectories, training_all.pkl holds ~8000
frames. Override via the flags below.

ABLATION (``--no-noise``): read the exact same two grids but skip rattling, so
the offset grid stays at its true MD geometry. The frame count and 90/10 split
are identical to a rattled run (4000 frames -> 3600 train / 400 val per
trajectory), but every frame is pure MD. A pre-labeled trajectory then needs no
UMA pass at all, so the run is fast. Only ``training_all.pkl`` is written (the
MD/noised split is meaningless with no noise). See ``--no-noise`` below.

USAGE (submit as GPU SLURM job):
    python -m src.data_prep.dataset_creation.create_rattled_dataset \
        --traj /path/to/napf6_dme_0.1M.traj /path/to/napf6_dme_1M.traj \
        --labels 0.1M 1M \
        --uma-checkpoint /path/to/uma-s-1p1.pt \
        --output-dir /path/to/training_data_napf6_dme_rattled_v2

CONDA ENV: fairchemV3
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import Trajectory
from tqdm import tqdm


Z_TO_SYMBOL = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 55: 'Cs',
}


# ── Linref utilities (same as convert_xyz_to_fennol_pkl.py) ──────────────────


def compute_linear_reference(frames):
    X = np.vstack([
        np.bincount(fr['species'], minlength=100).astype(int)
        for fr in frames
    ])
    y = np.array([fr['total_energy'] for fr in frames])
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeff


def compute_force_rms(frames):
    all_forces = np.concatenate([fr['forces'].ravel() for fr in frames])
    return float(np.sqrt(np.mean(all_forces ** 2)))


def compute_stress_rms(frames):
    stress_vals = [fr['stress'].ravel() for fr in frames if 'stress' in fr]
    if not stress_vals:
        return None
    all_stress = np.concatenate(stress_vals)
    return float(np.sqrt(np.mean(all_stress ** 2)))


def apply_deref(frames, coeff):
    out = []
    for fr in frames:
        fr2 = dict(fr)
        ref = float(np.dot(np.bincount(fr['species'], minlength=100), coeff))
        fr2['total_energy'] = fr['total_energy'] - ref
        out.append(fr2)
    return out


def build_linref_dict(coeff, force_rms, stress_rms, frames):
    unique_z = sorted(int(z) for z in np.unique(
        np.concatenate([fr['species'] for fr in frames])
    ))
    return {
        'force_rms': force_rms,
        'stress_rms': stress_rms,
        'coeff_by_Z': {str(z): float(coeff[z]) for z in unique_z},
        'coeff_by_symbol': {Z_TO_SYMBOL.get(z, str(z)): float(coeff[z]) for z in unique_z},
        'note': (
            'E_absolute = E_model_output + sum(coeff_by_Z[Z] * count(Z)). '
            'Use force_rms as normalizer_rmsd for energy/force loss terms. '
            'Use stress_rms as normalizer_rmsd for the stress loss term.'
        ),
    }


# ── Pipeline steps ───────────────────────────────────────────────────────────


def atoms_to_dict(atoms, with_labels=True):
    """Geometry (and optionally stored UMA labels) of one frame as a plain dict.

    With ``with_labels=False`` only species/coordinates/cell are read — used for
    frames that get (re)labeled by UMA anyway: all noised frames, plus MD frames
    from position-only trajectories that carry no stored energy/forces/stress.
    """
    d = {
        'species': np.array(atoms.get_atomic_numbers(), dtype=np.int32),
        'coordinates': atoms.get_positions().astype(np.float64),
        'cell': atoms.cell.array.astype(np.float64),
        # Provenance flag: False = pure MD frame. rattle_frames flips it to True
        # for noised frames so training_all.pkl is self-describing.
        'is_noised': False,
    }
    if with_labels:
        d['forces'] = atoms.get_forces().astype(np.float64)
        d['total_energy'] = float(atoms.get_potential_energy())
        d['stress'] = atoms.get_stress(voigt=False).reshape(1, 3, 3).astype(np.float64)
    return d


def interleaved_indices(n_total, duration_ps, interval_fs, offset_fs,
                        frame_interval_fs=10):
    """Two disjoint, interleaved frame grids within the first ``duration_ps``.

    MD frames sit on a grid of ``interval_fs`` starting at t=0; noised frames sit
    on the *same* grid shifted by ``offset_fs``. With the defaults (interval
    50 fs, offset 20 fs) that is MD at 0, 50, 100 ... fs and noised at 20, 70,
    120 ... fs, so the two sets never share a geometry.
    """
    last = min(int(duration_ps * 1000 / frame_interval_fs), n_total)
    step = round(interval_fs / frame_interval_fs)
    offset = round(offset_fs / frame_interval_fs)
    md_idx = list(range(0, last, step))
    noised_idx = list(range(offset, last, step))
    return md_idx, noised_idx


def read_md_and_noised(traj_path, duration_ps, interval_fs, offset_fs,
                       frame_interval_fs=10, no_noise=False):
    """Read the interleaved MD and to-be-noised frame sets from one trajectory.

    Returns ``(md_frames, noised_frames, md_has_labels)``. MD frames keep their
    stored UMA labels when the trajectory has them (``md_has_labels=True``);
    position-only trajectories yield label-less MD frames (``md_has_labels=False``)
    that the caller must send through UMA.

    The second (offset) grid is read geometry-only by default, because those
    frames get rattled — which changes their geometry — and then UMA-labeled. In
    ``no_noise=True`` mode (rattling disabled for the ablation) the offset frames
    stay at their true MD geometry, so their stored labels are still valid: read
    them WITH labels whenever the trajectory has them, so a fully pre-labeled
    trajectory needs no UMA pass at all.
    """
    print(f"Reading {traj_path}")
    with Trajectory(str(traj_path), 'r') as traj:
        total = len(traj)
        md_idx, noised_idx = interleaved_indices(
            total, duration_ps, interval_fs, offset_fs, frame_interval_fs)
        md_has_labels = traj[0].calc is not None
        # Offset frames get stored labels only when they keep their MD geometry
        # (no_noise) AND the trajectory carries labels; otherwise UMA labels them.
        noised_with_labels = md_has_labels and no_noise
        print(f"  Total frames in traj: {total}; using first {duration_ps} ps")
        print(f"  MD grid:     {len(md_idx)} frames every {interval_fs} fs from t=0 "
              f"({'pre-labeled' if md_has_labels else 'NEEDS UMA labeling'})")
        print(f"  {'Offset' if no_noise else 'Noised'} grid: {len(noised_idx)} "
              f"frames every {interval_fs} fs from t={offset_fs} fs "
              f"({'pre-labeled, un-rattled' if noised_with_labels else 'NEEDS UMA labeling'})")
        md_frames = [atoms_to_dict(traj[i], with_labels=md_has_labels)
                     for i in tqdm(md_idx, desc="Reading MD frames")]
        noised_frames = [atoms_to_dict(traj[i], with_labels=noised_with_labels)
                         for i in tqdm(noised_idx, desc="Reading offset frames")]

    print(f"  Got {len(md_frames)} MD + {len(noised_frames)} offset frames")
    return md_frames, noised_frames, md_has_labels


def sanitize_label(label):
    """Make a group label filesystem-safe, e.g. '0.1M' -> '0_1M'."""
    return label.replace('.', '_')


def save_simulation_start(traj_path, output_dir, start_ps, label,
                          frame_interval_fs=10):
    """Write a single trajectory frame to simulation_start_<label>.xyz to seed MD runs.

    Pulls the frame at `start_ps` (default 100 ps) from the trajectory. If the
    trajectory is shorter than that, falls back to the last available frame.
    `label` is the group label for this trajectory (e.g. '0.1M' or
    'naotf_dme_1M'); it is sanitized into the filename.
    """
    from ase.io import write

    frame_idx = int(start_ps * 1000 / frame_interval_fs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    xyz_path = out / f'simulation_start_{sanitize_label(label)}.xyz'

    with Trajectory(str(traj_path), 'r') as traj:
        total = len(traj)
        if frame_idx >= total:
            print(f"  Trajectory has only {total} frames; "
                  f"using last frame instead of the {start_ps} ps mark")
            frame_idx = total - 1
        atoms = traj[frame_idx]

    write(str(xyz_path), atoms)
    print(f"  Wrote {xyz_path} "
          f"(frame {frame_idx} ≈ {frame_idx * frame_interval_fs / 1000:.1f} ps)")


def rattle_frames(frames, stdev, rng):
    """Add Gaussian noise (sigma=`stdev` A) to the positions of every frame."""
    rattled = []
    for fr in tqdm(frames, desc="Rattling frames"):
        atoms = Atoms(
            numbers=fr['species'],
            positions=fr['coordinates'].copy(),
            cell=fr['cell'],
            pbc=True,
        )
        atoms.rattle(stdev=stdev, rng=rng)
        atoms.wrap()
        rattled_fr = dict(fr)
        rattled_fr['coordinates'] = atoms.positions.astype(np.float64)
        rattled_fr['is_noised'] = True
        rattled.append(rattled_fr)
    return rattled


def load_uma_calculator(checkpoint):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit, InferenceSettings

    settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=True,
        compile=True,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    predictor = load_predict_unit(
        checkpoint,
        device="cuda",
        inference_settings=settings,
        overrides={
            "_target_": "fairchem.core.models.base.HydraModel",
            "backbone": {"direct_forces": False},
        },
    )
    return FAIRChemCalculator(predictor, task_name="omol")


def label_frame(frame, calc):
    atoms = Atoms(
        numbers=frame['species'],
        positions=frame['coordinates'],
        cell=frame['cell'],
        pbc=True,
    )
    atoms.calc = calc

    try:
        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        stress = atoms.get_stress(voigt=False).reshape(1, 3, 3)
    except Exception as e:
        print(f"  UMA evaluation failed: {e}")
        torch.cuda.empty_cache()
        return None

    if np.any(np.isnan(forces)) or np.isnan(energy) or np.any(np.isnan(stress)):
        print("  UMA produced NaN — discarding frame")
        return None

    return {
        'species': frame['species'],
        'coordinates': frame['coordinates'],
        'cell': frame['cell'],
        'forces': forces.astype(np.float64),
        'total_energy': energy,
        'stress': stress.astype(np.float64),
        # Preserve provenance: label_frame rebuilds the dict from scratch, so
        # carry the flag through (rattled frames arrive here with is_noised=True).
        'is_noised': frame.get('is_noised', False),
    }


def label_frames(frames, calc, kind="frames"):
    print(f"\nLabeling {len(frames)} {kind} with UMA...")

    # Warmup
    print("  Warming up UMA...")
    fr0 = frames[0]
    warmup = Atoms(numbers=fr0['species'], positions=fr0['coordinates'],
                   cell=fr0['cell'], pbc=True)
    warmup.calc = calc
    _ = warmup.get_forces()
    torch.cuda.synchronize()
    print("  Warmup complete")

    labeled = []
    n_failed = 0
    for fr in tqdm(frames, desc="UMA labeling"):
        result = label_frame(fr, calc)
        if result is None:
            n_failed += 1
        else:
            labeled.append(result)

    print(f"  Labeled: {len(labeled)}, failed: {n_failed}")
    if labeled:
        max_forces = [np.max(np.linalg.norm(fr['forces'], axis=1)) for fr in labeled]
        print(f"  Max force — min: {min(max_forces):.1f}, "
              f"median: {np.median(max_forces):.1f}, max: {max(max_forces):.1f} eV/A")
    return labeled


def split_and_save(groups, output_dir, rng, no_noise=False):
    """Split 90/10 within each group, then pool the splits.

    `groups` is a list of (label, md_frames, rattled_frames) tuples, one per
    trajectory. Splitting each group's MD and noised frames independently
    guarantees the validation set holds a proportional ~10% slice of every
    trajectory and every source, rather than relying on a global random split.

    With ``no_noise=True`` the second grid is un-rattled pure MD too, so the
    MD/noised distinction is meaningless — only ``training_all.pkl`` is written
    (no ``training_md_data.pkl`` / ``training_noised.pkl``).
    """
    md_train, md_val = [], []
    rattled_train, rattled_val = [], []

    print("\nSplitting 90/10 within each group:")
    for label, md_frames, rattled_frames in groups:
        n_md = len(md_frames)
        n_rattled = len(rattled_frames)
        n_md_train = int(0.9 * n_md)
        n_rattled_train = int(0.9 * n_rattled)

        md_perm = rng.permutation(n_md)
        rattled_perm = rng.permutation(n_rattled)

        md_train.extend(md_frames[i] for i in md_perm[:n_md_train])
        md_val.extend(md_frames[i] for i in md_perm[n_md_train:])
        rattled_train.extend(rattled_frames[i] for i in rattled_perm[:n_rattled_train])
        rattled_val.extend(rattled_frames[i] for i in rattled_perm[n_rattled_train:])

        second = 'offset MD' if no_noise else 'noised'
        print(f"  {label}: MD {n_md_train}/{n_md - n_md_train} train/val, "
              f"{second} {n_rattled_train}/{n_rattled - n_rattled_train} train/val")

    all_train = md_train + rattled_train
    all_val = md_val + rattled_val

    print(f"  Pooled train: {len(md_train)} MD + {len(rattled_train)} rattled = {len(all_train)}")
    print(f"  Pooled val:   {len(md_val)} MD + {len(rattled_val)} rattled = {len(all_val)}")

    # Compute linref from ALL training frames
    print("\nComputing linear reference from all training frames...")
    coeff = compute_linear_reference(all_train)
    force_rms = compute_force_rms(all_train)
    stress_rms = compute_stress_rms(all_train)

    unique_z = sorted(int(z) for z in np.unique(
        np.concatenate([fr['species'] for fr in all_train])
    ))
    for z in unique_z:
        sym = Z_TO_SYMBOL.get(z, str(z))
        print(f"  {sym:2s} (Z={z:2d}): {coeff[z]:.6f} eV/atom")
    print(f"  force_rms:  {force_rms:.10f} eV/A")
    if stress_rms is not None:
        print(f"  stress_rms: {stress_rms:.10f} eV/A^3")

    # Apply deref to everything
    md_train_d = apply_deref(md_train, coeff)
    md_val_d = apply_deref(md_val, coeff)
    rattled_train_d = apply_deref(rattled_train, coeff)
    rattled_val_d = apply_deref(rattled_val, coeff)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets = {
        'training_all.pkl': {
            'training': md_train_d + rattled_train_d,
            'validation': md_val_d + rattled_val_d,
        },
    }
    # The MD/noised split only means something when the second grid is rattled.
    if not no_noise:
        datasets['training_md_data.pkl'] = {
            'training': md_train_d,
            'validation': md_val_d,
        }
        datasets['training_noised.pkl'] = {
            'training': rattled_train_d,
            'validation': rattled_val_d,
        }

    for name, data in datasets.items():
        path = out / name
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        n_train = len(data['training'])
        n_val = len(data['validation'])
        size_mb = os.path.getsize(path) / 1e6
        print(f"  Wrote {path} ({n_train} train + {n_val} val, {size_mb:.1f} MB)")

    linref = build_linref_dict(coeff, force_rms, stress_rms, all_train)
    linref_path = out / 'linref.json'
    with open(linref_path, 'w') as f:
        json.dump(linref, f, indent=2)
    print(f"  Wrote {linref_path}")

    # Sanity check
    e0 = md_train_d[0]['total_energy']
    print(f"\n  Deref sanity — train[0] energy: {e0:.6f} eV")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--traj', required=True, nargs='+',
                        help='Path(s) to UMA .traj file(s), one per group')
    parser.add_argument('--labels', '--concentrations', dest='labels',
                        required=True, nargs='+',
                        help='Group label for each --traj, in the same order. '
                             'May be concentrations of one system (0.1M 1M) or '
                             'distinct systems (naotf_dme_0.1M napf6_pc_1M). Used '
                             'as the per-trajectory split group and as the '
                             'simulation_start_<label>.xyz filename. '
                             '(--concentrations is a backward-compatible alias.)')
    parser.add_argument('--uma-checkpoint', required=True,
                        help='Path to UMA model checkpoint (.pt)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for pickles and linref.json')
    parser.add_argument('--duration-ps', type=float, default=100.0,
                        help='Use first N ps of trajectory (default: 100)')
    parser.add_argument('--sampling-interval-fs', type=float, default=50.0,
                        help='Spacing within each grid (MD and noised) in fs '
                             '(default: 50)')
    parser.add_argument('--noised-offset-fs', type=float, default=20.0,
                        help='Offset of the noised grid from the MD grid in fs, '
                             'e.g. MD at 0/50/100 and noised at 20/70/120 '
                             '(default: 20)')
    parser.add_argument('--frame-interval-fs', type=float, default=10.0,
                        help='Time between frames in the .traj file (default: 10)')
    parser.add_argument('--stdev', type=float, default=0.1,
                        help='Gaussian rattle sigma in Angstrom (default: 0.1)')
    parser.add_argument('--no-noise', action='store_true',
                        help='ABLATION: disable rattling. The exact same two '
                             'sampling grids are still read (so the frame count '
                             'and 90/10 split are unchanged), but the offset grid '
                             'is kept at its true MD geometry instead of being '
                             'rattled. Every frame is then pure MD (is_noised '
                             'False); for a pre-labeled trajectory no UMA pass '
                             'runs at all. --stdev is ignored.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--sim-start-ps', type=float, default=100.0,
                        help='Trajectory time (ps) to dump as '
                             'simulation_start_<label>.xyz (default: 100)')
    args = parser.parse_args()

    if len(args.traj) != len(args.labels):
        parser.error(
            f"got {len(args.traj)} --traj but {len(args.labels)} "
            f"--labels; they must match one-to-one"
        )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    rng = np.random.default_rng(args.seed)

    # Steps 1-2: per trajectory, read the interleaved MD / noised grids, then
    # rattle the noised set. Keep frames grouped by trajectory so the train/val
    # split stays proportional per group (see split_and_save).
    if args.no_noise:
        print("\n*** --no-noise: rattling DISABLED. Both sampling grids are kept "
              "as pure MD frames (ablation dataset). ***")

    md_by_group = []        # list of (label, md_frames, md_has_labels)
    rattled_by_group = []   # list of (label, offset_frames) — unlabeled for now
    for traj_path, label in zip(args.traj, args.labels):
        print(f"\n=== {label}: {traj_path} ===")
        md_frames, noised_frames, md_has_labels = read_md_and_noised(
            traj_path, args.duration_ps, args.sampling_interval_fs,
            args.noised_offset_fs, args.frame_interval_fs, no_noise=args.no_noise,
        )
        md_by_group.append((label, md_frames, md_has_labels))
        # Ablation: keep the offset grid at its true MD geometry (is_noised stays
        # False). Otherwise rattle it into the noised set (is_noised -> True).
        offset_frames = (noised_frames if args.no_noise
                         else rattle_frames(noised_frames, args.stdev, rng))
        rattled_by_group.append((label, offset_frames))

    # Step 3: Label the rattled frames with UMA, per trajectory so grouping is
    # preserved. The calculator is RE-LOADED for each trajectory: merge_mole=True
    # (see load_uma_calculator) merges the MoLE experts for one fixed composition
    # and then asserts every frame matches it. Each trajectory is internally a
    # single composition, but they differ across trajectories (whether by
    # concentration or by system), so one calculator cannot span them — a fresh,
    # freshly-merged calculator per trajectory keeps the merge speedup while
    # handling all trajectories correctly.
    groups = []
    for (label, md_frames, md_has_labels), (_, rattled_frames) in zip(md_by_group, rattled_by_group):
        print(f"\n=== Labeling frames for {label} ===")
        # What still needs UMA: the MD grid iff the trajectory is position-only,
        # and the offset grid iff it was rattled (--no-noise leaves it at its true
        # MD geometry, so a pre-labeled trajectory already carries valid labels).
        md_needs_uma = not md_has_labels
        offset_needs_uma = (not args.no_noise) or (not md_has_labels)
        if md_needs_uma or offset_needs_uma:
            print("Loading UMA calculator (fresh merge for this composition)...")
            calc = load_uma_calculator(args.uma_checkpoint)
        else:
            print("Trajectory is fully pre-labeled and rattling is off — "
                  "skipping UMA entirely.")
            calc = None
        # MD frames: keep stored UMA labels when present; otherwise (position-only
        # trajectory) UMA-label the unnoised geometries too — same model, so the
        # dataset stays single-model consistent.
        md_labeled = (md_frames if md_has_labels
                      else label_frames(md_frames, calc, kind=f"unnoised MD frames ({label})"))
        # Offset frames: reuse stored labels if they were read pre-labeled (only
        # possible with --no-noise on a labeled trajectory), else send to UMA.
        offset_kind = f"{'offset MD' if args.no_noise else 'noised'} frames ({label})"
        rattled_labeled = (rattled_frames if not offset_needs_uma
                           else label_frames(rattled_frames, calc, kind=offset_kind))
        groups.append((label, md_labeled, rattled_labeled))
        # Drop the merged calculator before the next composition.
        if calc is not None:
            del calc
            torch.cuda.empty_cache()

    # Step 4: Split (proportional per trajectory), deref, save
    split_and_save(groups, args.output_dir, rng, no_noise=args.no_noise)

    # Step 5: Dump one starting frame per trajectory for downstream simulations
    print("\nSaving simulation start frames...")
    for traj_path, label in zip(args.traj, args.labels):
        save_simulation_start(
            traj_path, args.output_dir, args.sim_start_ps, label,
            args.frame_interval_fs
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
