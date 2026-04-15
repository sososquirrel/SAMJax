"""
16-class IGBP land-cover lookup tables.

Verbatim port of the ``select case`` block in
``gSAM SRC/SLM/slm_vars.f90`` lines 440–731. Arrays are shape (16,), indexed
by ``landtype - 1`` (i.e. class 1 is index 0). Use
:func:`lookup_igbp_fields` to expand these tables into (ny, nx) fields
given an integer ``landtype`` array.

Special notes:
- Classes 13 (urban), 15 (ice), 16 (baresoil) have ``LAI=0`` by force.
- Class 13 uses urban soil albedo; class 15 uses ice soil albedo;
  everything else uses bare-soil albedo. Applied inside
  :func:`lookup_igbp_fields`.
- Seaice (no entry) is handled by ``slm_init.build_slm_static_and_state``
  with its own hard-coded parameter set.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jsam.core.physics.slm.params import SLMParams


# ---------------------------------------------------------------------------
# Per-class scalar tables (indexed by IGBP class 1..16 → array index 0..15)
# Order matches slm_vars.f90 cases 1..16.
# ---------------------------------------------------------------------------

# Vegetation (leaf) albedos
_ALB_VIS_V = np.array(
    [0.094, 0.086, 0.102, 0.056, 0.093, 0.069, 0.079, 0.071,
     0.061, 0.090, 0.081, 0.084, 0.102, 0.071, 0.000, 0.000], dtype=np.float32)
_ALB_NIR_V = np.array(
    [0.161, 0.146, 0.198, 0.151, 0.179, 0.121, 0.226, 0.085,
     0.227, 0.269, 0.180, 0.193, 0.164, 0.169, 0.000, 0.000], dtype=np.float32)

# Canopy top height [m]
_ZTOP = np.array(
    [20.0, 20.0, 20.0, 20.0, 20.0, 1.0, 1.0, 5.0,
     5.0, 0.5, 0.5, 0.5, 10.0, 0.5, 0.0, 0.0], dtype=np.float32)

# Surface roughness length [m] — class 15 overwritten with z0_ice,
# class 16 overwritten with z0_soil below.
_Z0_SFC = np.array(
    [0.5, 0.5, 0.5, 0.5, 0.3, 0.1, 0.1, 0.05,
     0.15, 0.04, 0.2, 0.03, 0.5, 0.04, 0.0, 0.0], dtype=np.float32)

# Leaf-angle distribution parameter χ_L
_KHAI_L = np.array(
    [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
     0.25, -0.3, -0.3, -0.3, -0.3, -0.3, 0.0, 0.0], dtype=np.float32)

# Root profile parameters (Zeng 2001)
_ROOT_L = np.array(
    [1.8, 3.0, 2.0, 2.0, 2.4, 2.5, 3.1, 1.7,
     2.4, 1.5, 1.5, 1.5, 1.5, 1.5, 0.0, 0.0], dtype=np.float32)
_ROOT_A = np.array(
    [6.706, 7.344, 7.066, 5.990, 4.453, 6.326, 7.718, 7.604,
     8.235, 10.74, 5.558, 5.558, 5.558, 5.558, 0.0, 0.0], dtype=np.float32)
_ROOT_B = np.array(
    [2.175, 1.303, 1.953, 1.955, 1.631, 1.567, 1.262, 2.300,
     1.627, 2.608, 2.614, 2.614, 2.614, 2.614, 0.0, 0.0], dtype=np.float32)

# Stomatal resistance parameters
_RC_MIN = np.array(
    [250.0, 250.0, 250.0, 250.0, 250.0, 220.0, 220.0, 180.0,
     100.0, 100.0, 100.0, 100.0, 0.0, 100.0, 0.0, 0.0], dtype=np.float32)
_RGL = np.array(
    [120.0, 120.0, 120.0, 120.0, 120.0, 100.0, 100.0, 100.0,
     100.0, 100.0, 100.0, 100.0, 0.0, 100.0, 0.0, 0.0], dtype=np.float32)
_HS_RC = np.array(
    [0.03, 0.03, 0.03, 0.03, 0.03, 0.01, 0.01, 0.02,
     0.01, 0.01, 0.01, 0.01, 0.00, 0.01, 0.00, 0.00], dtype=np.float32)

# Basal area index (ft²/acre, converted inside mw_mx formula)
_BAI = np.array(
    [200.0, 200.0, 200.0, 200.0, 200.0, 60.0, 60.0, 100.0,
     100.0, 20.0, 20.0, 60.0, 0.0, 20.0, 0.0, 0.0], dtype=np.float32)

# Impervious surface fraction
_IMPERV = np.array(
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
     0.00, 0.00, 0.00, 0.00, 0.75, 0.00, 0.00, 0.00], dtype=np.float32)

# Vegetated flag (1 if class produces a live canopy)
_VEGETATED = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 0, 1, 0, 0], dtype=np.int8)

# Classes that force LAI=0 (urban, ice, baresoil)
_LAI_FORCED_ZERO = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 0, 1, 1], dtype=np.int8)


def lookup_igbp_fields(
    landtype: np.ndarray,
    lai_raw: np.ndarray,
    params: SLMParams,
) -> dict:
    """
    Expand (ny, nx) integer ``landtype`` into per-cell land parameter fields.

    Parameters
    ----------
    landtype : (ny, nx) int
        IGBP class index 1..16. Non-land cells may be 0; callers mask those
        out separately via ``landmask``. Values are clipped to [1, 16] for
        the lookup; use the land mask to ignore the result on ocean.
    lai_raw : (ny, nx) float
        Raw LAI from the external file (monthly-climatology resolved to a
        single month, or a fixed field). Will be forced to 0 over
        urban/ice/baresoil and to ``max(LAI, LAI_min)`` over vegetated.
    params : SLMParams
        Scalar constants (supplies albedos, z0_soil, z0_ice, LAI_min).

    Returns
    -------
    dict of numpy arrays (ny, nx), keyed by field name. Cast to JAX arrays
    by the caller.
    """
    idx = np.clip(landtype.astype(np.int32) - 1, 0, 15)

    alb_vis_v = _ALB_VIS_V[idx]
    alb_nir_v = _ALB_NIR_V[idx]
    ztop      = _ZTOP[idx]
    z0_sfc    = _Z0_SFC[idx].copy()
    khai_L    = _KHAI_L[idx]
    rootL     = _ROOT_L[idx]
    root_a    = _ROOT_A[idx]
    root_b    = _ROOT_B[idx]
    Rc_min    = _RC_MIN[idx]
    Rgl       = _RGL[idx]
    hs_rc     = _HS_RC[idx]
    BAI       = _BAI[idx]
    IMPERV    = _IMPERV[idx]
    vegetated = _VEGETATED[idx].astype(bool)
    lai_force_zero = _LAI_FORCED_ZERO[idx].astype(bool)

    # z0 overrides for ice / baresoil classes (mirrors slm_vars.f90 cases 15,16)
    is_class15 = (landtype == 15)
    is_class16 = (landtype == 16)
    z0_sfc = np.where(is_class15, np.float32(params.z0_ice),  z0_sfc)
    z0_sfc = np.where(is_class16, np.float32(params.z0_soil), z0_sfc)

    # Soil albedos: urban uses urban, ice uses ice, else soil
    is_urban = (landtype == 13)
    is_ice   = (landtype == 15)
    alb_vis_s = np.where(is_urban, np.float32(params.albvis_urban),
                 np.where(is_ice, np.float32(params.albvis_ice),
                          np.float32(params.albvis_soil))).astype(np.float32)
    alb_nir_s = np.where(is_urban, np.float32(params.albnir_urban),
                 np.where(is_ice, np.float32(params.albnir_ice),
                          np.float32(params.albnir_soil))).astype(np.float32)

    # LAI: forced zero for class 13/15/16; else clipped to LAI_min
    LAI = np.where(lai_force_zero, np.float32(0.0),
                   np.maximum(lai_raw.astype(np.float32),
                              np.float32(params.LAI_min))).astype(np.float32)

    vege_YES = vegetated.astype(np.float32)

    # Derived: phi_1, phi_2, IR_emis_vege, mw_mx (slm_vars.f90:795-804)
    phi_1 = 0.5 - 0.633 * khai_L - 0.33 * khai_L ** 2
    phi_2 = 0.877 * (1.0 - 2.0 * phi_1)
    precip_extinc = phi_1 + phi_2
    IR_emis_vege = (1.0 - np.exp(-(phi_1 + phi_2) * LAI)) * np.float32(params.IR_emis_leaf)

    # mw_mx: 0.1*LAI + ztop * sqrt(4*pi*BAI/43560)
    mw_mx = 0.1 * LAI + ztop * np.sqrt(4.0 * np.pi * BAI / 43560.0)

    # IR_emis_grnd: urban blend vs soil (slm_vars.f90:822-828)
    IR_emis_grnd = np.where(
        is_urban,
        np.float32(params.IR_emis_urban) * IMPERV
            + np.float32(params.IR_emis_soil) * (1.0 - IMPERV),
        np.float32(params.IR_emis_soil),
    ).astype(np.float32)

    # landicemask: 1 where landtype == 15 (glacier)
    landicemask = is_class15.astype(np.int8)

    return dict(
        alb_vis_v=alb_vis_v.astype(np.float32),
        alb_nir_v=alb_nir_v.astype(np.float32),
        alb_vis_s=alb_vis_s,
        alb_nir_s=alb_nir_s,
        ztop=ztop.astype(np.float32),
        z0_sfc=z0_sfc.astype(np.float32),
        khai_L=khai_L.astype(np.float32),
        rootL=rootL.astype(np.float32),
        root_a=root_a.astype(np.float32),
        root_b=root_b.astype(np.float32),
        Rc_min=Rc_min.astype(np.float32),
        Rgl=Rgl.astype(np.float32),
        hs_rc=hs_rc.astype(np.float32),
        BAI=BAI.astype(np.float32),
        IMPERV=IMPERV.astype(np.float32),
        vegetated=vegetated,
        vege_YES=vege_YES,
        LAI=LAI,
        phi_1=phi_1.astype(np.float32),
        phi_2=phi_2.astype(np.float32),
        precip_extinc=precip_extinc.astype(np.float32),
        IR_emis_vege=IR_emis_vege.astype(np.float32),
        IR_emis_grnd=IR_emis_grnd,
        mw_mx=mw_mx.astype(np.float32),
        landicemask=landicemask,
    )
