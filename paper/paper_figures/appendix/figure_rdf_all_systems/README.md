# Appendix figure — Na+ RDFs, all 8 salt/solvent systems

`python plot_figure_rdf_all_systems.py` writes
`paper/paper_electrolytes/Figures/appendix/figure_rdf_all_systems.png`;
`figure_rdf_all_systems.png` here is a symlink to that file (same pattern as
`main_paper/figure_2`).

## Figure

2 rows (NaPF$_6$ / NaOTf) x 4 columns (DME / DEGDME / PC / TEGDME), 1 M, 293 K.
PaiNN solid and coloured by partner element (O/F/P/S) with a ±1σ band across
replicas; UMA black dashed. Same visual style as panel a of the main-text figure
(`figure2_teacher_student_consistency.png`), but the full 8-system set rather
than the 4-system inset that fits in the main text — and with bands, which
figure 2 does not yet draw.

PaiNN is the **mean over 5 replicas** (r0 plus reruns r1–r4, differing only in
FeNNol's `random_seed`), with the band the sample std (ddof=1) across them. A
panel marked "†" contains replicas seeded from the 100 ps frame of the UMA
trajectory rather than the shared packing box — there the band mixes seed-to-seed
with start-to-start variation. It matters most for `napf6_pc` (4 of 5).

⚠️ UMA does not itself reach equilibrium within 1 ns for the PC systems (Na–F
coordination falls 4.1 → 1.7 and is still dropping). This figure shows that
**PaiNN tracks UMA**, not that either is the equilibrium liquid structure.

`lipf6_dme` is excluded: it exists only at 0.5 M, so it has no 1 M teacher
counterpart. The 8 systems with 1 M runs for both models are shown.

## Data

Both sides are 293 K, ~1 ns trajectories sampled/truncated with identical params
for a direct comparison:

* UMA: `data/uma_static_observables_1ns_NPT/rdfs_293K/<system>_1M/rdf.csv`
  (`compute_uma_1M_rdfs.py`) — a single trajectory, so no band
* PaiNN: `data/painn_static_observables_NPT/rdfs_293K/<system>_1M/replicas/rdf_r<N>.csv`
  (`compute_painn_replica_rdfs.py`)

This script reads the **per-replica** curves under `replicas/` so it can draw the
spread. The replica mean of those same curves is precomputed one level up in
`<system>_1M/rdf.csv` (as `g_<pair>`, with `sd_<pair>` beside it) — that is what
figure 2 reads.
