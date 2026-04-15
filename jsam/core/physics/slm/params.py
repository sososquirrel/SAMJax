"""
Scalar constants for the Simple Land Model.

Direct port of gSAM ``SRC/SLM/slm_params.f90``. Every constant below is a
verbatim copy; do not change the numeric values unless the Fortran source
is also changed. Values are held on a frozen dataclass so the whole object
is hashable and can be used as a ``static_argnames`` field in jax.jit.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SLMParams:
    # Soil water potential thresholds (m of water head)
    FC: float = -3.3          # field capacity (~-0.33 bar)
    WP: float = -150.0        # wilting point

    pii: float = 3.141592653589793

    # Stomatal resistance
    Rc_max: float = 5000.0
    T_opt:  float = 298.0

    # Radiation / misc
    sigma: float = 5.67e-8

    # Roughness lengths (m)
    z0_soil: float = 0.005
    z0_ice:  float = 0.001

    seaicedepth: float = 1.5
    tfriz:  float = 273.15    # fresh-water freezing
    tfrizs: float = 271.35    # sea-water freezing
    mws_mx0: float = 50.0     # max puddle storage (mm)

    # IR emissivities
    IR_emis_urban: float = 0.90
    IR_emis_leaf:  float = 0.97
    IR_emis_soil:  float = 0.96
    IR_emis_snow:  float = 0.985
    IR_emis_ice:   float = 0.97

    # Densities (kg/m³)
    rho_water: float = 998.0
    rho_ice:   float = 917.0
    rho_snow:  float = 100.0

    # Thermal conductivity (W/m/K)
    cond_water: float = 0.57
    cond_ice:   float = 2.2
    cond_snow:  float = 0.01

    # Heat capacities
    capa_water: float = 4182.0      # J/kg/K
    capa_ice:   float = 2030.0      # J/kg/K
    cp_water:   float = 4.1796e6    # J/m³/K

    # Vegetation
    LAI_min:        float = 0.1
    t_canop_max:    float = 343.0
    leaf_thickness: float = 1.0     # mm, used for heat capacity

    # Albedos (visible / NIR)
    albvis_snow: float = 0.98
    albnir_snow: float = 0.65
    albvis_ice:  float = 0.91
    albnir_ice:  float = 0.65
    albvis_soil: float = 0.208
    albnir_soil: float = 0.344
    albvis_urban: float = 0.15
    albnir_urban: float = 0.25
