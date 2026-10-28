"""Generate and submit one FeNNol training job per system from a single experiment config.

FeNNol's `fennol_train` CLI takes only a single positional YAML path — it has no
CLI/env overrides. So instead of maintaining one near-identical config per system,
we keep ONE experiment template (src/train/fennol_configs/<exp>.yaml) and let this
script materialize a complete, standalone FeNNol config per system, then submit it.

For each system the script:
  1. Resolves the dataset dir ({data_root}/datasets/<system>_rattled) and reads
     its linref.json for the loss normalizer(s).
  2. Deep-merges the shared `base:` scheme + per-system fields (dspath,
     output_directory, wandb.name, loss normalizers) + any per-system overrides.
  3. Writes the resolved config to {output_directory}/config.yaml (co-located with
     the model, so it's reproducible and FeNNol can restart-from-directory).
  4. Submits the job. Default (Delta): one `sbatch run_fennol_train_delta.sh {config}`
     per system. With --nersc (Perlmutter): systems are batched 4-per-node and
     submitted as `sbatch run_fennol_train_nersc_4gpu.sh {cfg1..cfg4}`, one
     training run per GPU.

The output dir follows the README data layout:
    {data_root}/experiments/{exp_name}/{system}/model/

Usage:
    # Dry run — validate dataset paths, print the resolved plan, write/submit nothing
    python src/train/launch_fennol_experiment.py \
        --experiment src/train/fennol_configs/exp01_mixed_noise.yaml --dry-run

    # Generate the config.yaml files but do not submit
    python src/train/launch_fennol_experiment.py \
        --experiment src/train/fennol_configs/exp01_mixed_noise.yaml --no-submit

    # Submit all systems on Delta (one job per system)
    python src/train/launch_fennol_experiment.py \
        --experiment src/train/fennol_configs/exp01_mixed_noise.yaml

    # Submit all systems on Perlmutter (batched 4-per-node onto the 4-GPU script)
    python src/train/launch_fennol_experiment.py \
        --experiment src/train/fennol_configs/exp01_mixed_noise.yaml --nersc

    # Submit a subset
    python src/train/launch_fennol_experiment.py \
        --experiment src/train/fennol_configs/exp01_mixed_noise.yaml \
        --systems napf6_dme napf6_pc

CONDA ENV: any env with pyyaml (submission itself only needs `sbatch`).
"""
import argparse
import copy
import json
import pickle
import subprocess
import sys
from pathlib import Path

import yaml

# Per-cluster run script. The project root is this checkout wherever it sits, so
# the Slurm --chdir follows the repo rather than a fixed path. Default is Delta
# (one job per system); --nersc switches to Perlmutter, batching systems
# 4-per-node onto the 4-GPU script.
REPO_ROOT = Path(__file__).resolve().parents[2]
DELTA_ROOT = NERSC_ROOT = REPO_ROOT
DELTA_RUN_SCRIPT = REPO_ROOT / "src/train/run_fennol_train_delta.sh"
NERSC_RUN_SCRIPT = REPO_ROOT / "src/train/run_fennol_train_nersc_4gpu.sh"
NERSC_GPUS_PER_NODE = 4


# ── Config assembly ──────────────────────────────────────────────────────────


def deep_update(mapping, *updates):
    """Recursive dict merge (mirrors fennol.utils.deep_update)."""
    out = copy.deepcopy(mapping)
    for upd in updates:
        for k, v in upd.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = deep_update(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
    return out


def normalize_systems(systems):
    """Accept a list of `str` or `{name, overrides}` dicts; yield (name, overrides)."""
    out = []
    for s in systems:
        if isinstance(s, str):
            out.append((s, {}))
        elif isinstance(s, dict):
            out.append((s["name"], s.get("overrides", {})))
        else:
            raise ValueError(f"Bad system entry (expected str or dict): {s!r}")
    return out


def resolve_paths(exp, system):
    """Compute the dataset dir, dataset pickle, linref, and model output dir."""
    data_root = Path(exp["data_root"])
    datasets_root = data_root / "datasets"
    dataset_dir = Path(
        exp["dataset_dir_template"].format(
            data_root=data_root, datasets_root=datasets_root,
            system=system, exp_name=exp["name"],
        )
    )
    return {
        "dataset_dir": dataset_dir,
        "dspath": dataset_dir / exp.get("pkl", "training_all.pkl"),
        "linref": dataset_dir / "linref.json",
        "output_dir": data_root / "experiments" / exp["name"] / system / "model",
    }


def normalizers_from_linref(linref_path, rule):
    """Return (norm_e, norm_f, norm_s) per the chosen rule, read from linref.json."""
    linref = json.loads(linref_path.read_text())
    force_rms = linref["force_rms"]
    stress_rms = linref.get("stress_rms")
    if rule == "force_rms_all":
        return force_rms, force_rms, force_rms
    if rule == "force_stress_split":
        if stress_rms is None:
            raise ValueError(f"{linref_path} has no stress_rms (needed for force_stress_split)")
        return force_rms, force_rms, stress_rms
    raise ValueError(f"Unknown normalizer_rule: {rule!r}")


_TRAIN_SIZE_CACHE = {}


def training_set_size(dspath):
    """Number of training frames in a FeNNol dataset pickle.

    Cached per path: the pickles run to ~1 GB and several systems can share one.
    """
    key = str(dspath)
    if key not in _TRAIN_SIZE_CACHE:
        with open(dspath, "rb") as f:
            ds = pickle.load(f)
        _TRAIN_SIZE_CACHE[key] = (
            len(ds["training"]) if isinstance(ds, dict) and "training" in ds else len(ds)
        )
    return _TRAIN_SIZE_CACHE[key]


def nbatch_per_epoch(dspath, batch_size):
    """Batches for one full pass over the training set.

    This MUST be written into every generated config. FeNNol does not derive it:
    `training.py` does `training_parameters.get("nbatch_per_epoch", 200)`, so a
    config that omits the key silently trains on 200 batches per epoch regardless
    of dataset size. The only check on the FeNNol side (`io.py`) asserts the value
    equals n_train // batch_size, and it runs *only when the key is present*.

    exp07_no_noise was launched without it and did 200 batches/epoch (800 configs)
    against exp01's 900 (3600 configs) -- a 4.5x difference in gradient steps that
    silently confounded the noising ablation. Do not drop this key again.

    The value is per-system, not a constant: the pooled two-concentration datasets
    hold 7200 training frames (-> 1800 batches at batch_size 4) and the
    single-concentration ones 3600 (-> 900).
    """
    return training_set_size(dspath) // batch_size


def build_config(exp, base, system, overrides, paths, rule):
    """Assemble the complete FeNNol config dict for one system."""
    norm_e, norm_f, norm_s = normalizers_from_linref(paths["linref"], rule)
    per_system = {
        "output_directory": str(paths["output_dir"]),
        "training": {
            "dspath": str(paths["dspath"]),
            "wandb": {
                "project": exp["wandb_project"],
                "name": f"{exp['name']}_{system}",
            },
            "loss": {
                "e": {"normalizer_rmsd": norm_e},
                "f": {"normalizer_rmsd": norm_f},
                "s": {"normalizer_rmsd": norm_s},
            },
        },
    }
    config = deep_update(base, per_system, overrides)
    # After the merge, so an override that changes batch_size is respected.
    training = config.setdefault("training", {})
    training["nbatch_per_epoch"] = nbatch_per_epoch(
        paths["dspath"], training.get("batch_size", 16)
    )
    return config


# ── Submission ───────────────────────────────────────────────────────────────


def chunk(items, n):
    """Split a list into consecutive chunks of at most n."""
    return [items[i:i + n] for i in range(0, len(items), n)]


def _sbatch(project_root, run_script, job_name, config_paths):
    """sbatch run_script with one or more config paths; return job id or None."""
    cmd = [
        "sbatch",
        f"--chdir={project_root}",
        f"--job-name={job_name}",
        str(run_script),
        *[str(c) for c in config_paths],
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr.strip()}")
        return None
    job_id = result.stdout.strip().split()[-1]
    print(f"    Submitted: job {job_id} ({len(config_paths)} system(s))")
    return job_id


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fan a single FeNNol experiment config out over systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", required=True, help="Path to the experiment YAML")
    parser.add_argument("--systems", nargs="+", default=None,
                        help="Only process this subset of systems (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate + print the plan; write and submit nothing")
    parser.add_argument("--no-submit", action="store_true",
                        help="Write the config.yaml files but do not sbatch")
    parser.add_argument("--nersc", action="store_true",
                        help="Submit on Perlmutter: batch systems %d-per-node onto "
                             "run_fennol_train_nersc_4gpu.sh (default: Delta, one "
                             "job per system)" % NERSC_GPUS_PER_NODE)
    args = parser.parse_args()

    if args.nersc:
        project_root, run_script, batch_size = NERSC_ROOT, NERSC_RUN_SCRIPT, NERSC_GPUS_PER_NODE
    else:
        project_root, run_script, batch_size = DELTA_ROOT, DELTA_RUN_SCRIPT, 1

    # Catch a missing launcher here rather than at sbatch time.
    if not run_script.is_file():
        raise SystemExit(f"run script not found: {run_script}\n"
                         f"Available: "
                         f"{', '.join(sorted(p.name for p in run_script.parent.glob('run_fennol_train_*.sh')))}")

    spec = yaml.safe_load(Path(args.experiment).read_text())
    exp = spec["experiment"]
    base = spec["base"]
    rule = exp.get("normalizer_rule", "force_rms_all")
    systems = normalize_systems(spec["systems"])
    if args.systems:
        wanted = set(args.systems)
        systems = [(n, o) for (n, o) in systems if n in wanted]
        missing = wanted - {n for n, _ in systems}
        if missing:
            print(f"ERROR: --systems not found in experiment: {sorted(missing)}")
            sys.exit(1)

    cluster = "Perlmutter (NERSC)" if args.nersc else "Delta"
    print(f"Experiment: {exp['name']}  |  {len(systems)} system(s)  |  normalizer_rule={rule}")
    print(f"cluster: {cluster}  |  run script: {run_script.name}  |  "
          f"{'batch %d/node' % batch_size if args.nersc else '1 job/system'}")
    print(f"data_root: {exp['data_root']}")
    print(f"dataset variant: {exp.get('pkl', 'training_all.pkl')}\n")

    # Validate dataset/linref paths up front (mirrors launch_exp01_production.py).
    resolved = [(name, overrides, resolve_paths(exp, name)) for name, overrides in systems]
    errors = []
    for name, _, paths in resolved:
        if not paths["dspath"].exists():
            errors.append(f"  [{name}] dataset pickle missing: {paths['dspath']}")
        if not paths["linref"].exists():
            errors.append(f"  [{name}] linref.json missing: {paths['linref']}")
    if errors:
        print("ERRORS — datasets must be built first (see 'Building a dataset' in README):")
        print("\n".join(errors))
        if not args.dry_run:
            sys.exit(1)
        print("\n(continuing dry run; normalizers shown as <unresolved> where linref is missing)\n")

    # ── Phase 1: generate each system's config.yaml (or preview, for --dry-run) ──
    generated = []  # (name, config_path), in `systems` order
    for name, overrides, paths in resolved:
        out_dir = paths["output_dir"]
        config_path = out_dir / "config.yaml"
        print(f"  {name}")
        print(f"    dataset: {paths['dspath']}")
        print(f"    output:  {out_dir}")

        if args.dry_run:
            if paths["linref"].exists():
                ne, nf, ns = normalizers_from_linref(paths["linref"], rule)
                print(f"    normalizers (e/f/s): {ne:.6g} / {nf:.6g} / {ns:.6g}")
            else:
                print("    normalizers (e/f/s): <unresolved — linref missing>")
            # Worth the pickle read even on a dry run: a silently-wrong epoch
            # length is exactly the kind of thing a dry run exists to surface.
            if paths["dspath"].exists():
                bs = deep_update(base, {}, overrides).get("training", {}).get("batch_size", 16)
                n_train = training_set_size(paths["dspath"])
                print(f"    train frames: {n_train}  ->  nbatch_per_epoch: "
                      f"{n_train // bs} (batch_size {bs})")
            print(f"    [DRY RUN] would write {config_path}\n")
            generated.append((name, config_path))
            continue

        config = build_config(exp, base, name, overrides, paths, rule)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
        print(f"    Wrote {config_path}\n")
        generated.append((name, config_path))

    # ── Phase 2: submit. NERSC batches `batch_size` systems per node onto the
    # 4-GPU script; Delta submits one job per system. ──────────────────────────
    batches = chunk(generated, batch_size)
    submitted = []  # (job_name, job_id, [systems])
    if not args.dry_run and not args.no_submit:
        for bi, batch in enumerate(batches):
            names = [n for n, _ in batch]
            cfgs = [c for _, c in batch]
            job_name = f"{exp['name']}_g{bi}" if args.nersc else f"{exp['name']}_{names[0]}"
            print(f"  [{job_name}] {', '.join(names)}")
            jid = _sbatch(project_root, run_script, job_name, cfgs)
            if jid:
                submitted.append((job_name, jid, names))

    print("=" * 70)
    if args.dry_run:
        print(f"DRY RUN — {len(generated)} system(s) in {len(batches)} job(s) would be "
              f"generated + submitted via {run_script.name}:")
        for bi, batch in enumerate(batches):
            print(f"  job {bi}: {', '.join(n for n, _ in batch)}")
    elif args.no_submit:
        print(f"Generated {len(generated)} config(s); none submitted (--no-submit).")
    else:
        print(f"Submitted {len(submitted)} job(s):")
        for job_name, jid, names in submitted:
            print(f"  {jid}  {job_name}  [{', '.join(names)}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
