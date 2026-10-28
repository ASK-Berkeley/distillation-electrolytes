"""Physical constants and unit conversions for observables."""

import math

AMU_TO_KG = 1.66053906660e-27       # kg per amu
ANGSTROM3_TO_M3 = 1e-30             # m^3 per A^3
KB_J_PER_K = 1.380649e-23           # Boltzmann constant (J/K)
YH_XI = 2.837297                    # Yeh-Hummer self-term constant (cubic PBC, unitless)


def a2ps_to_m2s(d_a2ps: float) -> float:
    """Diffusion coefficient A^2/ps -> m^2/s."""
    return d_a2ps * 1e-20 / 1e-12


def yeh_hummer_correction_a2ps(T_K: float, viscosity_cp: float, box_length_a: float) -> float:
    """Yeh-Hummer finite-size correction to a self-diffusivity, in A^2/ps.

    Periodic boundaries systematically *suppress* self-diffusion via the
    hydrodynamic self-interaction; the leading correction to the infinite-box
    value is

        dD = k_B T xi / (6 pi eta L),   xi = 2.837297 (cubic box).

    It is species-independent (depends only on T, the shear viscosity, and the
    box edge), so the same dD is added to every species' D_PBC to estimate
    D_inf. ``viscosity_cp`` in cP (= mPa.s), ``box_length_a`` in A (use V^(1/3)
    for a non-cubic box), ``T_K`` in K. (Yeh & Hummer, J. Phys. Chem. B 2004.)
    """
    eta_pas = viscosity_cp * 1e-3                 # cP -> Pa.s
    L_m = box_length_a * 1e-10                    # A -> m
    dD_m2s = KB_J_PER_K * T_K * YH_XI / (6.0 * math.pi * eta_pas * L_m)
    return dD_m2s / a2ps_to_m2s(1.0)              # m^2/s -> A^2/ps


def density_g_cm3(mass_amu: float, vol_a3: float) -> float:
    """Mass density (g/cm^3) from total mass (amu) and cell volume (A^3)."""
    mass_kg = mass_amu * AMU_TO_KG
    vol_m3 = vol_a3 * ANGSTROM3_TO_M3
    return (mass_kg / vol_m3) / 1000.0   # kg/m^3 -> g/cm^3
