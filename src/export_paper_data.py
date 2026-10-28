"""Copy the paper's released inputs (trained students and simulation boxes) into ``paper_data/``.

``data/`` is the scratch tree everything is computed in; ``paper_data/`` is the
small, committed subset a reader needs to rerun the simulations:

    paper_data/
      models/<system>/best_model.fnx      the 9 trained PaiNN students (one per salt/solvent)
      models/<system>/config.yaml         its FeNNol training config (machine paths and
                                          the W&B account replaced, see below)
      boxes/initial/<system>/
        simulation_start_<conc>.xyz       teacher frame the first student runs started from
        equilibrated_<conc>_298K.xyz      student NPT-equilibrated frame; its density sets
                                          the molarity correction (README section 5.1)
      boxes/corrected/<system>/
        initial_<conc>.xyz                box repacked to the target molarity (+ packmol_<conc>.inp,
                                          paths made repo-relative)
        initial_relaxed_<conc>.xyz        after the fixed-cell UMA-S relaxation
        nvt_seed_<conc>_298K.xyz          NPT-equilibrated frame the 4 x 20 ns production runs start from
      MANIFEST.csv                        relative path, size and sha256 of every file

The per-replica analysis curves are exported separately, into ``results/``, by
``src.analysis.observables.export_results``. Trajectories and the distillation
datasets are not released; the pipeline in README.md regenerates them.

Run from the repository root (any env with PyYAML):

    python -m src.export_paper_data --dry-run   # list what would be copied
    python -m src.export_paper_data             # copy, rewrite configs, write MANIFEST.csv
    python -m src.export_paper_data --check     # exit nonzero if paper_data/ is stale
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
from pathlib import Path

import yaml

from src.analysis.observables import paths
from src.analysis.observables.systems import SYSTEMS

EXP01 = paths.DATA_DIR / "experiments" / "exp01_mixed_noise_all_systems"
EXP04 = paths.DATA_DIR / "experiments" / "exp04_correct_molarity_samebox"
DATASETS = paths.DATA_DIR / "datasets" / "version01_mixed_noise"
OUT = paths.PAPER_DATA


def _conc(name: str, prefix: str, suffix: str) -> str:
    """'initial_relaxed_0_1M.xyz', 'initial_relaxed_', '.xyz' -> '0_1M'."""
    return name[len(prefix):-len(suffix)]


def plan() -> list[tuple[Path, Path]]:
    """(source under data/, destination relative to paper_data/) for every file."""
    items = []
    for system in SYSTEMS:
        items += [(EXP01 / system / "model" / "best_model.fnx",
                   Path("models", system, "best_model.fnx")),
                  (EXP01 / system / "model" / "config.yaml",
                   Path("models", system, "config.yaml"))]
        for f in sorted((DATASETS / system).glob("simulation_start_*.xyz")):
            items.append((f, Path("boxes", "initial", system, f.name)))
        for f in sorted((EXP04 / system).glob("initial_*.xyz")):
            conc = _conc(f.name, "initial_relaxed_" if "relaxed" in f.name else "initial_", ".xyz")
            items.append((EXP01 / "nvt_equilibration_seeds" / f"{system}_{conc}_298K.xyz",
                          Path("boxes", "initial", system, f"equilibrated_{conc}_298K.xyz")))
        for pat in ("initial_*.xyz", "packmol_*.inp", "nvt_seed_*.xyz"):
            for f in sorted((EXP04 / system).glob(pat)):
                items.append((f, Path("boxes", "corrected", system, f.name)))
    # equilibrated_* is listed once per initial/initial_relaxed file; keep one
    seen, out = set(), []
    for src, dst in items:
        if dst not in seen:
            seen.add(dst)
            out.append((src, dst))
    return out


def released_config(src: Path, system: str) -> bytes:
    """The training config with machine-specific paths made relative and the
    W&B account removed. Hyperparameters are untouched."""
    cfg = yaml.safe_load(src.read_text())
    cfg["output_directory"] = f"paper_data/models/{system}"
    for block in cfg.values():
        if isinstance(block, dict):
            if "dspath" in block:
                block["dspath"] = (f"data/datasets/version01_mixed_noise/{system}/"
                                   "training_all.pkl")
            if isinstance(block.get("wandb"), dict):
                block["wandb"].pop("entity", None)
    header = ("# FeNNol training config of the released student. Paths are relative to the\n"
              "# repository root; the dataset itself is not released (README section 4).\n")
    return (header + yaml.safe_dump(cfg, sort_keys=False)).encode()


def released_packmol(src: Path) -> bytes:
    """Packmol input with absolute paths made repo-relative: molecules come from the
    in-repo copies (identical files), the output is written next to the input."""
    text = re.sub(r"^(structure\s+)\S*/([^/\s]+\.xyz)",
                  r"\1src/data_prep/box_creation/molecules/\2", src.read_text(), flags=re.M)
    text = re.sub(r"^(output\s+)\S*/([^/\s]+)", r"\1\2", text, flags=re.M)
    return text.encode()


def content(src: Path, dst: Path) -> bytes:
    if dst.name == "config.yaml":
        return released_config(src, dst.parts[1])
    if dst.suffix == ".inp":
        return released_packmol(src)
    return src.read_bytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="List what would be copied; write nothing.")
    ap.add_argument("--check", action="store_true",
                    help="Exit nonzero if paper_data/ differs from what would be written.")
    args = ap.parse_args()

    paths.require(paths.DATA_DIR)
    items = plan()
    missing = [s for s, _ in items if not s.is_file()]
    if missing:
        sys.exit("missing sources:\n  " + "\n  ".join(map(str, missing)))

    blobs = {dst: content(src, dst) for src, dst in items}
    manifest = io.StringIO()
    w = csv.writer(manifest, lineterminator="\n")
    w.writerow(["path", "bytes", "sha256"])
    for dst, b in blobs.items():
        w.writerow([dst.as_posix(), len(b), hashlib.sha256(b).hexdigest()])
    blobs[Path("MANIFEST.csv")] = manifest.getvalue().encode()

    stale = [d for d, b in blobs.items()
             if not (OUT / d).is_file() or (OUT / d).read_bytes() != b]
    total = sum(len(b) for b in blobs.values())
    print(f"{len(blobs)} files, {total / 1024**2:.1f} MB -> {OUT}  ({len(stale)} new or changed)")

    if args.dry_run:
        for d in stale:
            print(f"  {d}")
        return
    if args.check:
        if stale:
            sys.exit(f"paper_data/ is stale ({len(stale)} files); run python -m src.export_paper_data")
        print("paper_data/ is up to date")
        return
    for d in stale:
        (OUT / d).parent.mkdir(parents=True, exist_ok=True)
        (OUT / d).write_bytes(blobs[d])
    print(f"wrote {len(stale)} file(s)")


if __name__ == "__main__":
    main()
