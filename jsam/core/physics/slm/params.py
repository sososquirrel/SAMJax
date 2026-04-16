"""SLM constants (gSAM slm_params.f90). Frozen dataclass for use as static_argnames."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SLMParams:
    FC: float = -3.3
    WP: float = -150.0
    pii: float = 3.141592653589793
    Rc_max: float = 5000.0
    T_opt:  float = 298.0
    sigma: float = 5.67e-8
    z0_soil: float = 0.005
    z0_ice:  float = 0.001
    seaicedepth: float = 1.5
    tfriz:  float = 273.15
    tfrizs: float = 271.35
    mws_mx0: float = 50.0
    IR_emis_urban: float = 0.90
    IR_emis_leaf:  float = 0.97
    IR_emis_soil:  float = 0.96
    IR_emis_snow:  float = 0.985
    IR_emis_ice:   float = 0.97
    rho_water: float = 998.0
    rho_ice:   float = 917.0
    rho_snow:  float = 100.0
    cond_water: float = 0.57
    cond_ice:   float = 2.2
    cond_snow:  float = 0.01
    capa_water: float = 4182.0
    capa_ice:   float = 2030.0
    cp_water:   float = 4.1796e6
    LAI_min:        float = 0.1
    t_canop_max:    float = 343.0
    leaf_thickness: float = 1.0
    albvis_snow: float = 0.98
    albnir_snow: float = 0.65
    albvis_ice:  float = 0.91
    albnir_ice:  float = 0.65
    albvis_soil: float = 0.208
    albnir_soil: float = 0.344
    albvis_urban: float = 0.15
    albnir_urban: float = 0.25
