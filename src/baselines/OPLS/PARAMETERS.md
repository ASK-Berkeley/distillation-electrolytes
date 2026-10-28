# Force-field parameters: what is used and where it comes from

Every parameter in the OPLS-AA baseline traces to one of three published
sources. Nothing is hand-tuned. The combined force field GROMACS reads is
`forcefield_lpg.itp`, written per box by `lib/build_system_lpg.py`; the molecule
topologies are the `.itp` files under `data/`. `check_topologies.py` verifies
all of them (see the bottom of this file).

## Functional form (OPLS-AA)

| setting | value | meaning |
|---|---|---|
| nonbonded | Lennard-Jones 12-6 + Coulomb | `nbfunc 1` |
| LJ cross terms | geometric mean of sigma and epsilon | `comb-rule 3` |
| exclusions | pairs within 3 bonds | `nrexcl 3` in every molecule |
| 1-4 interactions | LJ and Coulomb both scaled by 0.5 | `gen-pairs yes`, `fudgeLJ 0.5`, `fudgeQQ 0.5` |
| bonds, angles | harmonic, V = k/2 (x - x0)^2 | funct 1 |
| dihedrals | Ryckaert-Bellemans (LigParGen) or OPLS Fourier (CL&P) | funct 3 / funct 5 |
| ion charges | full formal charge, no scaling | Na+ Li+ = +1, PF6- OTf- = -1 |

## Per-species provenance

| species | file | charges | Lennard-Jones | bonded terms | source |
|---|---|---|---|---|---|
| DME, DEGDME, TEGDME, PC | `data/molecules_lpg/<MOL>_out/<MOL>.gmx.itp` (atom types renamed by offset, see builder docstring) | 1.14*CM1A from LigParGen | OPLS-AA | OPLS-AA | LigParGen server: Dodda, Cabeza de Vaca, Tirado-Rives, Jorgensen, *Nucleic Acids Res.* 45, W331 (2017) |
| PF6- | `data/molecules/pf6.itp` | P +1.34, F -0.39 | P 3.74 A / 0.8368 kJ/mol; F 3.12 A / 0.2552 kJ/mol | P-F 1.606 A, 3100 kJ/mol/A^2; F-P-F 90 deg (cis) / 180 deg (trans), 1165 kJ/mol/rad^2 | CL&P `il.ff` "hexafluorophosphate": Canongia Lopes, Deschamps, Pádua, *J. Phys. Chem. B* 108, 2038 (2004), doi:10.1021/jp0362133; bonded terms from Kaminski, Jorgensen, *J. Chem. Soc., Perkin Trans. 2*, 2365 (1999), doi:10.1039/a905160k |
| OTf- (CF3SO3-) | `data/molecules/otf.itp` | C +0.35, S +1.02, O -0.63, F -0.16 | C 3.50/0.27614; S 3.55/1.046; O 3.15/0.83736; F 3.118/0.2554 (A, kJ/mol) | S-O 1.437 A 5331; C-S 1.818 A 1950; C-F 1.323 A 3698 (kJ/mol/A^2); O-S-O 118.5/969; C-S-O 102.6/870; F-C-F 107.1/781; F-C-S 111.7/694 (deg, kJ/mol/rad^2); O-S-C-F Fourier V3 = 1.451 kJ/mol | CL&P `il.ff` "triflate": Canongia Lopes, Pádua, *J. Phys. Chem. B* 108, 16893 (2004), doi:10.1021/jp0476545; O charge -0.63 from Gouveia et al., *Phys. Chem. Chem. Phys.* 19, 29617 (2017), doi:10.1039/C7CP06081E |
| Na+ | `data/molecules/na.itp` | +1 | 3.33045 A / 0.011598 kJ/mol | none | Åqvist, *J. Phys. Chem.* 94, 8021 (1990), doi:10.1021/j100384a009; values verbatim from GROMACS `oplsaa.ff/ffnonbonded.itp` (`opls_407`) |
| Li+ | `data/molecules/li.itp` | +1 | 2.12645 A / 0.0764793 kJ/mol | none | Åqvist (1990), as above; GROMACS `oplsaa.ff` (`opls_406`) |

CL&P (Canongia Lopes and Pádua) is built on the OPLS-AA functional form, with
the same combination rules and 1-4 scaling, so mixing it with the LigParGen
solvents is a consistent choice. The `il.ff` file the values were transcribed
from is published at <https://github.com/paduagroup/clandp> (version header
2026/04/16); it gives bond constants in kJ/mol/A^2, which the `.itp` files
multiply by 100 to get GROMACS' kJ/mol/nm^2, and angle constants in kJ/mol/rad^2,
used as is.

### Notes and judgement calls

* **PF6- trans angles.** `il.ff` lists one F-P-F angle type at 90 deg. An
  octahedron has three 180 deg (trans) F-P-F angles, and a harmonic potential
  centred at 90 deg cannot describe them. The topology therefore uses
  theta0 = 180 deg with the same force constant for the three trans angles.
  This is stated in the file header.
* **Triflate oxygen charge.** `il.ff` carries two oxygen types: -0.53 (type
  OBT, bistriflamide, 2004) and -0.63 (type OTF, triflate, 2017 revision). The
  triflate one is used.
* **LigParGen charge option.** The server offers 1.14*CM1A and 1.14*CM1A-LBCC. For these
  solvents the choice does not matter: on 2026-10-06 DME and PC were submitted to the
  LigParGen server (v2.1) with both options and returned identical charges, while a control
  (chlorobenzene) returned different ones. The LBCC bond corrections do not apply to ethers
  or carbonates, so the solvent charges are plain 1.14*CM1A. The vendored files were
  generated earlier; today's server output differs from them by at most 0.03 e per atom
  (conformer and version), and the vendored files are what the runs use.
* **No charge scaling.** Nonpolarizable force fields with full ionic charges
  are known to overestimate ion pairing and viscosity. This baseline is meant
  to be plain OPLS-AA, so charges are left at their formal values.
* **Masses.** Standard atomic masses as in the sources (Na 22.98977, Li 6.941,
  P 30.974, S 32.066, O 15.999, F 18.998, C 12.011, H 1.008).

### Superseded

`data/superseded/OTF_ligpargen/` holds the LigParGen (CM1A) triflate used for
the NaOTf runs before 2026-10-06. The earlier `pf6.itp` (nrexcl 1, F-F
[ pairs ], uncited bonded terms) is described in `PF6_TOPOLOGY_BUG.md`; its
content is in git history.

## Verification

`python check_topologies.py` (needs `GMXBIN`) builds each species as a single
molecule in vacuum through the same code path as the box builder and checks:
`nrexcl 3`; the exclusion list in the `.tpr` equals the pairs within three
bonds of the molecule's own bond graph; `[ pairs ]` equals exactly the 1-4 set;
integer net charge; masses match elements; and the minimised geometry
reproduces b0 and theta0 (strictly for molecules with only bonds and angles,
within 0.05 A / 8 deg for molecules whose 1-4 and dihedral terms legitimately
strain them). grompp runs with `-maxwarn 0` throughout. The last result is
committed as `topology_check.txt`.
