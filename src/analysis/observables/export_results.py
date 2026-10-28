"""Publish the small analysis outputs from scratch into the tracked ``results/``.

``data/`` is a symlink to scratch and holds the full observable trees, hundreds
of MB of per-replica detail that cannot go in git. The figures, though, mostly
read a handful of summary CSVs and the RDF curves, and those come to about a
megabyte. This copies exactly those into ``results/``, which is committed, so
the figures still draw from a fresh clone where ``data/`` does not exist (see
the fallback in ``paths.py``).

Nothing else reads ``results/`` while ``data/`` is present, so this does not
change how the analysis or the figures behave locally. Rerun it after
recomputing an observable, then commit the diff:

    python -m src.analysis.observables.export_results
    git add results && git commit

``--check`` reports whether the tracked copy is stale without writing anything
and exits nonzero if it is, so it can gate a commit or a release.

Also exported: the ``per_run/`` trees of the transport observables (the
per-replica MSD, Green-Kubo and collective-conductivity curves the fits were made
on, ~180 MB of CSV/JSON; the diagnostic PNGs beside them are skipped). They let
the SI example-trace figures redraw from a clone and let a reader check every
transport number without rerunning the 20 ns trajectories.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

from .paths import DATA_DIR, RESULTS

# Files under data/ to publish. Globs are relative to DATA_DIR; every match is
# copied to the same relative location under results/.
PATTERNS = [
    # transport summaries: one row per replica, and per-box mean +/- s.d.
    "painn_transport_observables_replicas/*/*_master.csv",
    "painn_transport_observables_replicas/*/*_by_box.csv",
    "opls_transport_observables/*/*_master.csv",
    "opls_transport_observables/*/*_by_box.csv",
    # the settings sidecar that says how each number was produced
    "painn_transport_observables_replicas/*/*_config.yaml",
    "opls_transport_observables/*/*_config.yaml",
    # 293 K RDF curves (replica means) for the structural figures
    "painn_static_observables_NPT/rdfs_293K/**/*.csv",
    "uma_static_observables_1ns_NPT/rdfs_293K/**/*.csv",
    # student densities
    "painn_static_observables_NPT/*.csv",
    # per-run stability/UMA-agreement diagnostics behind the SI stability figure
    "experiments/exp04_correct_molarity_samebox/transport_diagnostics.csv",
    # per-replica curves behind every transport number (CSV/JSON only, no PNGs)
    "painn_transport_observables_replicas/*/per_run/**/*.csv",
    "painn_transport_observables_replicas/*/per_run/**/*.json",
    "opls_transport_observables/*/per_run/**/*.csv",
    "opls_transport_observables/*/per_run/**/*.json",
    # UMA reference densities for the figure-2 density panel (uma_densities.py)
    "UMA_298K_1ns/densities_298K_1ns.csv",
    "UMA_298K_1ns/densities_298K_1ns_config.yaml",
]


# Superseded or exploratory variants that sit next to the production outputs.
# Publishing them would put two conflicting *_master.csv in front of a reader.
EXCLUDE = ("_backup", "_no_eqcut", "_fixed_only", "natfsi")   # NaTFSI was dropped from the paper


def _matches() -> list[Path]:
    """Every file to publish, as paths relative to DATA_DIR."""
    seen: dict[Path, None] = {}
    for pat in PATTERNS:
        for p in sorted(DATA_DIR.glob(pat)):
            if not p.is_file():
                continue
            rel = p.relative_to(DATA_DIR)
            if any(tok in part for part in rel.parts for tok in EXCLUDE):
                continue
            seen.setdefault(rel, None)
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="Report staleness and exit nonzero if results/ differs; write nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be copied, write nothing")
    args = ap.parse_args()

    if not DATA_DIR.exists():
        sys.exit(f"{DATA_DIR} does not exist — nothing to export from. "
                 f"(On a machine without the scratch data, results/ is already "
                 f"the copy the figures read.)")

    rels = _matches()
    if not rels:
        sys.exit(f"No files matched under {DATA_DIR}. Has the analysis been run?")

    new, changed, same = [], [], []
    for rel in rels:
        dst = RESULTS / rel
        if not dst.exists():
            new.append(rel)
        elif not filecmp.cmp(DATA_DIR / rel, dst, shallow=False):
            changed.append(rel)
        else:
            same.append(rel)

    total = sum((DATA_DIR / r).stat().st_size for r in rels)
    print(f"{len(rels)} files, {total/1024**2:.1f} MB")
    print(f"  up to date {len(same)}   new {len(new)}   changed {len(changed)}")
    for label, group in (("new", new), ("changed", changed)):
        for rel in group[:20]:
            print(f"    {label:8s} {rel}")
        if len(group) > 20:
            print(f"    ... and {len(group) - 20} more {label}")

    if args.check:
        if new or changed:
            sys.exit(f"\nresults/ is stale: {len(new)} new, {len(changed)} changed. "
                     f"Run `python -m src.analysis.observables.export_results`.")
        print("\nresults/ is up to date")
        return

    if args.dry_run:
        print("\n(dry run, nothing written)")
        return

    for rel in new + changed:
        dst = RESULTS / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DATA_DIR / rel, dst)
    print(f"\nWrote {len(new) + len(changed)} file(s) to {RESULTS}")


if __name__ == "__main__":
    main()
