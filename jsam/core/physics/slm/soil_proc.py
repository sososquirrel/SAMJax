"""
Soil water and soil temperature prognostics.

Direct port of gSAM ``SRC/SLM/soil_proc.f90``. Two routines:

* :func:`soil_water` — 9-layer implicit diffusion solve for volumetric
  soil wetness. Includes infiltration at the top (from ``precip_in``),
  per-layer transpiration sink (``evapo_dry * rootF``), gravitational
  drainage out of the bottom layer when wetness exceeds unity, and
  inter-layer diffusive+advective transport using the Cosby 1984
  hydraulics. Solved with a Thomas tridiagonal step. After the solve,
  any cell with wetness > 1 is capped and the excess is redistributed
  into unsaturated layers (top→bottom) preserving total water, with any
  residual rejected back to the caller as a negative ``precip_in_out``
  correction.

* :func:`soil_temperature` — 9-layer implicit diffusion solve for soil
  temperature. Top BC is ``grflux0`` (ground heat flux, positive out of
  soil). Bottom BC is zero-flux over land, clamped to sea-water freezing
  ``tfrizs`` over sea ice. Thermal conductivity uses Johansen (1975) with
  Kersten number ``Ke = log10(max(0.1, soilw)) + 1`` and volumetric heat
  capacity mixes the dry soil matrix with the liquid/ice fraction of the
  pore space depending on ``soilt`` relative to ``tfriz``. Both fields
  are returned as diagnostics (the main driver uses them for radiation
  and skin-temperature coupling).

Icemask (glacier / sea ice) cells short-circuit both routines:

* ``soilw`` is held at zero (no infiltration, no transpiration, no
  drainage — the Thomas solve degenerates to the identity).
* Thermal conductivity / heat capacity collapse to pure ice values
  ``cond_ice`` / ``rho_ice * capa_ice``.
* After the temperature solve, ``soilt`` is clamped to ≤ ``tfriz`` in
  icemask cells to prevent runaway warming.

Fortran features deliberately *not* ported here (they belong in the
snow / canopy / surface-water modules that drive this one):

* Snow melt / vapor deposition on ``snow_mass``.
* Puddle (``mws``) accounting and runoff generation.
* Wetland override (``landtype==11``) that saturates ``soilw`` to
  ``w_s_FC`` and zeroes surface water.
* Snow-top tridiagonal augmentation in :func:`soil_temperature` that
  prepends an ``alpha(0)/beta(0)`` row when ``snow_mass > rho_snow *
  0.01``. Callers should strip the snow layer before calling this
  routine and handle ``snowt`` separately.

The caller supplies ``precip_in`` (the already-clipped-to-``ks``
infiltration rate in kg/m²/s = mm/s) and ``evapo_dry`` (transpiration
demand in mm/s). Both are (ny, nx) arrays.

All loops over ``nsoil=9`` are statically unrolled — JAX traces Python
``for`` loops over fixed ranges without generating ``lax.scan``, and the
Thomas factorisation requires mutable state that is cleanest expressed
as a Python list of :class:`jax.Array` that is appended to in sequence.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from jsam.core.physics.slm.params import SLMParams
from jsam.core.physics.slm.state import NSOIL, SLMState, SLMStatic


# ---------------------------------------------------------------------------
# Thomas tridiagonal solver (static unroll, nsoil=9)
# ---------------------------------------------------------------------------
def _thomas_9(
    a: list[jax.Array],
    b: list[jax.Array],
    c: list[jax.Array],
    d: list[jax.Array],
) -> list[jax.Array]:
    """Solve a tridiagonal system ``T x = d`` with sub-diagonal ``a``,
    diagonal ``b``, super-diagonal ``c`` and right-hand-side ``d``.

    Each input is a length-``NSOIL`` Python list of ``(ny, nx)`` JAX
    arrays. ``a[0]`` and ``c[-1]`` are unused. The solve is vectorised
    across the horizontal plane and statically unrolled along the soil
    axis so JAX traces it without :func:`lax.scan`.
    """
    n = NSOIL
    # Working copies — never mutate the caller's lists.
    bw = list(b)
    dw = list(d)

    # Forward elimination.
    for k in range(1, n):
        w = a[k] / bw[k - 1]
        bw[k] = bw[k] - w * c[k - 1]
        dw[k] = dw[k] - w * dw[k - 1]

    # Back substitution.
    x: list[jax.Array] = [None] * n  # type: ignore[list-item]
    x[n - 1] = dw[n - 1] / bw[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = (dw[k] - c[k] * x[k + 1]) / bw[k]
    return x


# ---------------------------------------------------------------------------
# soil_water
# ---------------------------------------------------------------------------
def soil_water(
    state: SLMState,
    static: SLMStatic,
    params: SLMParams,
    precip_in: jax.Array,   # (ny, nx)  mm/s — caller-clipped to min(precip_sfc+mws/dt, ks[0])
    evapo_dry: jax.Array,   # (ny, nx)  mm/s — transpiration demand
    evp_soil: jax.Array,    # (ny, nx)  mm/s — bare-soil evaporation at layer 1
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """Advance volumetric soil wetness one time step.

    Returns:
        new_soilw: (nsoil, ny, nx) updated soil wetness (0..1).
        precip_in_out: (ny, nx) possibly-reduced infiltration rate
            (mm/s) — if the post-solve saturation fix-up rejected some
            water, the caller must route it to surface storage.
    """
    tfriz = params.tfriz

    soilw = state.soilw           # (nsoil, ny, nx)
    soilt = state.soilt
    icemask = static.icemask      # (ny, nx) int8
    ice = icemask == 1

    poro = static.poro_soil       # (nsoil, ny, nx)
    s_mm = static.s_depth * 1000.0  # Fortran uses s_depth_mm (mm) here
    Bc = static.Bconst
    mps = static.m_pot_sat
    ks = static.ks
    rootF = static.rootF
    w_s_WP = static.w_s_WP
    IMPERV = static.IMPERV

    # -----------------------------------------------------------------
    # Effective diffusion coefficient and advective velocity at each of
    # the (nsoil-1) internal interfaces. Depth-weighted average between
    # the two adjacent layers, from Cosby 1984.
    #   sh_eff_cond : mm²/s  (diffusion coefficient)
    #   sh_eff_vel  : mm/s   (gravity-driven advection)
    # Zeroed at any interface where either adjacent layer is frozen.
    # -----------------------------------------------------------------
    sh_eff_cond: list[jax.Array] = []
    sh_eff_vel: list[jax.Array] = []
    for k in range(NSOIL - 1):
        both_thawed = (soilt[k] >= tfriz) & (soilt[k + 1] >= tfriz)
        d_k = s_mm[k]
        d_kp1 = s_mm[k + 1]
        denom = d_k + d_kp1
        # |m_pot_sat| × ks × Bconst / poro factor, evaluated at layer k
        # (matches the Fortran which uses the upper-layer Bconst /
        # m_pot_sat / ks / poro_soil).
        base_cond = ks[k] * Bc[k] * jnp.abs(mps[k]) / poro[k]
        base_vel = ks[k] / poro[k]
        exp_c = Bc[k] + 2.0
        exp_v = 2.0 * Bc[k] + 2.0

        cond = (
            (d_k * soilw[k] ** exp_c + d_kp1 * soilw[k + 1] ** exp_c)
            / denom
            * base_cond
        )
        vel = (
            (d_k * soilw[k] ** exp_v + d_kp1 * soilw[k + 1] ** exp_v)
            / denom
            * base_vel
        )
        sh_eff_cond.append(jnp.where(both_thawed, cond, 0.0))
        sh_eff_vel.append(jnp.where(both_thawed, vel, 0.0))

    # -----------------------------------------------------------------
    # Transpiration gating: no root uptake from a layer below wilting
    # point. Matches `sw_wgt` in the Fortran.
    # -----------------------------------------------------------------
    sw_wgt = [jnp.where(soilw[k] < w_s_WP[k], 0.0, 1.0) for k in range(NSOIL)]

    # -----------------------------------------------------------------
    # Assemble the tridiagonal system for the implicit step.
    #   a_k * soilw_{k-1}^{n+1} + b_k * soilw_k^{n+1} + c_k * soilw_{k+1}^{n+1} = d_k
    # The Fortran factorises in-place; here we stack per-layer and hand
    # off to `_thomas_9`.
    # -----------------------------------------------------------------
    a_list: list[jax.Array] = []
    b_list: list[jax.Array] = []
    c_list: list[jax.Array] = []
    d_list: list[jax.Array] = []

    # --- layer 0 (top) -------------------------------------------------
    a0 = jnp.zeros_like(soilw[0])
    c0 = -2.0 * sh_eff_cond[0] * dt / (s_mm[0] * (s_mm[0] + s_mm[1]))
    b0 = 1.0 - c0 + sh_eff_vel[0] * dt / s_mm[0]
    # Top sink/source: bare-soil evap (>=0) + root uptake − infiltration.
    # Precip_in is added as a *negative* sink (source).
    evap_top = jnp.maximum(0.0, evp_soil)
    sink0 = (
        evap_top
        + rootF[0] * evapo_dry * sw_wgt[0]
        - precip_in
    )
    d0 = soilw[0] - sink0 * dt / poro[0] / s_mm[0]
    a_list.append(a0)
    b_list.append(b0)
    c_list.append(c0)
    d_list.append(d0)

    # --- interior layers 1..nsoil-2 ------------------------------------
    for k in range(1, NSOIL - 1):
        ak = (
            -dt
            / s_mm[k]
            * (
                sh_eff_vel[k - 1]
                + sh_eff_cond[k - 1] * 2.0 / (s_mm[k - 1] + s_mm[k])
            )
        )
        ck = -dt / s_mm[k] * 2.0 * sh_eff_cond[k] / (s_mm[k] + s_mm[k + 1])
        bk = (
            1.0
            - ck
            + dt / s_mm[k] * sh_eff_vel[k]
            + 2.0 * dt / s_mm[k] * sh_eff_cond[k - 1] / (s_mm[k - 1] + s_mm[k])
        )
        dk = soilw[k] - rootF[k] * evapo_dry * sw_wgt[k] * dt / poro[k] / s_mm[k]
        a_list.append(ak)
        b_list.append(bk)
        c_list.append(ck)
        d_list.append(dk)

    # --- bottom layer nsoil-1 ------------------------------------------
    nb = NSOIL - 1
    a_nb = (
        -dt
        / s_mm[nb]
        * (
            sh_eff_vel[nb - 1]
            + sh_eff_cond[nb - 1] * 2.0 / (s_mm[nb - 1] + s_mm[nb])
        )
    )
    c_nb = jnp.zeros_like(soilw[0])
    b_nb = 1.0 + (
        dt / s_mm[nb] * 2.0 * sh_eff_cond[nb - 1] / (s_mm[nb - 1] + s_mm[nb])
    )
    # Drainage flux: gravity-driven loss when bottom layer is supersaturated
    # (> 1). Evaluated from the *current* wetness, not the implicit solution.
    drainage_flux = (
        jnp.maximum(soilw[nb] - 1.0, 0.0) * poro[nb] * s_mm[nb] / dt
    )
    d_nb = soilw[nb] - (
        rootF[nb] * evapo_dry * sw_wgt[nb] + drainage_flux
    ) * dt / poro[nb] / s_mm[nb]
    a_list.append(a_nb)
    b_list.append(b_nb)
    c_list.append(c_nb)
    d_list.append(d_nb)

    # -----------------------------------------------------------------
    # Thomas solve and non-negativity clamp (Fortran does
    # `max(0, beta - alpha*x_{k+1})` during back-sub).
    # -----------------------------------------------------------------
    x = _thomas_9(a_list, b_list, c_list, d_list)
    x = [jnp.maximum(0.0, xk) for xk in x]

    # -----------------------------------------------------------------
    # Post-solve saturation fix-up. Any layer with w > 1 donates its
    # excess water to the first unsaturated layer encountered in
    # top→bottom order, mirroring the Fortran (changed from deepest-
    # first to top-first in 2025 for rain infiltration consistency).
    # Residual unabsorbed water becomes a negative correction to
    # precip_in that the caller must route to surface storage.
    # -----------------------------------------------------------------
    # dd = excess water pool (kg/m² = mm of water, since poro*s_mm is in mm).
    dd = jnp.zeros_like(soilw[0])
    new_w: list[jax.Array] = list(x)
    for k in range(NSOIL):
        over = jnp.maximum(new_w[k] - 1.0, 0.0)
        dd = dd + over * poro[k] * s_mm[k]
        new_w[k] = jnp.minimum(new_w[k], 1.0)

    # Distribute excess top-down into unsaturated layers.
    for k in range(NSOIL):
        capacity = jnp.maximum(0.0, (1.0 - new_w[k]) * poro[k] * s_mm[k])
        take = jnp.minimum(dd, capacity)
        new_w[k] = new_w[k] + take / (poro[k] * s_mm[k])
        dd = dd - take

    # `dd` is now the residual (mm of water) the profile could not
    # absorb; hand it back to the caller as a negative infiltration rate.
    precip_in_out = precip_in - dd / dt

    # -----------------------------------------------------------------
    # Icemask branch: clobber to zero.
    # -----------------------------------------------------------------
    new_soilw = jnp.stack(new_w, axis=0)
    new_soilw = jnp.where(ice[None, :, :], 0.0, new_soilw)
    precip_in_out = jnp.where(ice, 0.0, precip_in_out)

    return new_soilw, precip_in_out


# ---------------------------------------------------------------------------
# soil_temperature
# ---------------------------------------------------------------------------
def soil_temperature(
    state: SLMState,
    static: SLMStatic,
    params: SLMParams,
    grflux0: jax.Array,   # (ny, nx)  W/m² — positive = heat *leaves* the soil
    dt: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Advance 9-layer soil temperature one time step.

    Returns:
        new_soilt: (nsoil, ny, nx) soil temperature [K].
        st_cond:   (nsoil, ny, nx) total soil thermal conductivity
                   [W/m/K] (Johansen 1975). Diagnostic.
        st_capa:   (nsoil, ny, nx) volumetric soil heat capacity
                   [J/m³/K]. Diagnostic — used downstream for the
                   skin-temperature tendency.
    """
    tfriz = params.tfriz
    tfrizs = params.tfrizs
    cond_ice = params.cond_ice
    cond_water = params.cond_water
    rho_ice = params.rho_ice
    rho_water = params.rho_water
    capa_ice = params.capa_ice
    capa_water = params.capa_water

    soilt = state.soilt           # (nsoil, ny, nx)
    soilw = state.soilw
    icemask = static.icemask
    seaicemask = static.seaicemask
    ice = icemask == 1

    poro = static.poro_soil
    s_d = static.s_depth          # metres (temperature equation uses m)
    sst_cond = static.sst_cond
    sst_capa = static.sst_capa

    # -----------------------------------------------------------------
    # Per-layer thermal conductivity and heat capacity.
    # Soil branch follows Johansen (1975): weighted blend between dry
    # and saturated conductivities with Kersten number `Ke`. Saturated
    # conductivity uses water or ice in the pore space depending on
    # layer temperature. Heat capacity similarly mixes dry matrix with
    # water/ice contribution.
    # Icemask cells collapse to pure ice values.
    # -----------------------------------------------------------------
    st_cond_layers: list[jax.Array] = []
    st_capa_layers: list[jax.Array] = []
    for k in range(NSOIL):
        # Dry-soil density kg/m³ (empirical FIFE, 2700 kg/m³ solids).
        temp = (1.0 - poro[k]) * 2700.0
        k_dry = (0.135 * temp + 64.7) / (2700.0 - 0.947 * temp)

        # Saturated conductivity: pick water or ice in the pores.
        frozen = soilt[k] < tfriz
        k_sat_water = (sst_cond[k] ** (1.0 - poro[k])) * (cond_water ** poro[k])
        k_sat_ice = (sst_cond[k] ** (1.0 - poro[k])) * (cond_ice ** poro[k])
        k_sat = jnp.where(frozen, k_sat_ice, k_sat_water)

        # Kersten number; floor wetness at 0.1 to avoid log of zero.
        Ke = jnp.log10(jnp.maximum(0.1, soilw[k])) + 1.0
        cond_soil = Ke * (k_sat - k_dry) + k_dry

        capa_water_contrib = (1.0 - poro[k]) * sst_capa[k] + (
            rho_water * capa_water
        ) * soilw[k] * poro[k]
        capa_ice_contrib = (1.0 - poro[k]) * sst_capa[k] + (
            rho_ice * capa_ice
        ) * soilw[k] * poro[k]
        capa_soil = jnp.where(frozen, capa_ice_contrib, capa_water_contrib)

        # Icemask collapse: pure ice.
        cond_k = jnp.where(ice, cond_ice, cond_soil)
        capa_k = jnp.where(ice, rho_ice * capa_ice, capa_soil)
        st_cond_layers.append(cond_k)
        st_capa_layers.append(capa_k)

    # -----------------------------------------------------------------
    # Effective conductivity at interior interfaces — depth-weighted
    # mean of the two adjacent node conductivities.
    # -----------------------------------------------------------------
    st_eff_cond: list[jax.Array] = []
    for k in range(NSOIL - 1):
        eff = (
            st_cond_layers[k + 1] * s_d[k + 1] + st_cond_layers[k] * s_d[k]
        ) / (s_d[k + 1] + s_d[k])
        st_eff_cond.append(eff)

    # -----------------------------------------------------------------
    # Assemble the tridiagonal for the implicit temperature step.
    # -----------------------------------------------------------------
    a_list: list[jax.Array] = []
    b_list: list[jax.Array] = []
    c_list: list[jax.Array] = []
    d_list: list[jax.Array] = []

    # --- top layer -----------------------------------------------------
    # Upper BC: grflux0 is the ground heat flux (positive = *out of*
    # soil), applied as −grflux0*dt/(s_depth*st_capa) on the RHS.
    a0 = jnp.zeros_like(soilt[0])
    c0 = (
        -2.0
        * st_eff_cond[0]
        * dt
        / st_capa_layers[0]
        / (s_d[0] * (s_d[0] + s_d[1]))
    )
    b0 = 1.0 - c0
    d0 = soilt[0] - grflux0 * dt / s_d[0] / st_capa_layers[0]
    a_list.append(a0)
    b_list.append(b0)
    c_list.append(c0)
    d_list.append(d0)

    # --- interior layers ----------------------------------------------
    for k in range(1, NSOIL - 1):
        ak = (
            -2.0
            * st_eff_cond[k - 1]
            * dt
            / s_d[k]
            / st_capa_layers[k]
            / (s_d[k - 1] + s_d[k])
        )
        ck = (
            -2.0
            * st_eff_cond[k]
            * dt
            / s_d[k]
            / st_capa_layers[k]
            / (s_d[k] + s_d[k + 1])
        )
        bk = 1.0 - ck - ak
        dk = soilt[k]
        a_list.append(ak)
        b_list.append(bk)
        c_list.append(ck)
        d_list.append(dk)

    # --- bottom layer: zero-flux BC ------------------------------------
    nb = NSOIL - 1
    a_nb = (
        -2.0
        * st_eff_cond[nb - 1]
        * dt
        / s_d[nb]
        / st_capa_layers[nb]
        / (s_d[nb - 1] + s_d[nb])
    )
    c_nb = jnp.zeros_like(soilt[0])
    b_nb = 1.0 - a_nb
    d_nb = soilt[nb]
    a_list.append(a_nb)
    b_list.append(b_nb)
    c_list.append(c_nb)
    d_list.append(d_nb)

    # -----------------------------------------------------------------
    # Solve.
    # -----------------------------------------------------------------
    x = _thomas_9(a_list, b_list, c_list, d_list)

    # -----------------------------------------------------------------
    # Bottom-layer override for sea-ice cells: the deepest ice layer is
    # in contact with sea water, so pin it to tfrizs. This mirrors the
    # Fortran which overwrites beta(nsoil) directly before back-sub;
    # redoing the last back-sub step is equivalent because c_nb == 0.
    # -----------------------------------------------------------------
    sea = seaicemask == 1
    x[nb] = jnp.where(sea, tfrizs, x[nb])
    # Propagate the pinned bottom into the back-sub chain above.
    for k in range(nb - 1, -1, -1):
        # Recompute x[k] only for sea-ice cells; everywhere else x[k]
        # is already correct from `_thomas_9`. We rebuild the forward-
        # eliminated b,d on the fly — equivalent to rerunning Thomas
        # with a Dirichlet bottom, but only in sea-ice cells.
        # Simpler: linear chain using alpha/beta form. `_thomas_9`
        # doesn't expose them, so we just do the sea-ice solve inline.
        pass  # handled below with a second full solve.

    # Second solve only in sea-ice cells, using the same tridiagonal
    # but with the bottom row replaced by a Dirichlet row pinning
    # soilt[nb] = tfrizs. Cheap (9 unrolled ops) and avoids keeping
    # alpha/beta out of `_thomas_9`.
    a_list_sea = list(a_list)
    b_list_sea = list(b_list)
    c_list_sea = list(c_list)
    d_list_sea = list(d_list)
    a_list_sea[nb] = jnp.zeros_like(soilt[0])
    b_list_sea[nb] = jnp.ones_like(soilt[0])
    c_list_sea[nb] = jnp.zeros_like(soilt[0])
    d_list_sea[nb] = jnp.full_like(soilt[0], tfrizs)
    x_sea = _thomas_9(a_list_sea, b_list_sea, c_list_sea, d_list_sea)
    x = [jnp.where(sea, x_sea[k], x[k]) for k in range(NSOIL)]

    # -----------------------------------------------------------------
    # Icemask clamp: glacier / sea ice may not warm above freezing.
    # -----------------------------------------------------------------
    x = [jnp.where(ice, jnp.minimum(tfriz, xk), xk) for xk in x]

    new_soilt = jnp.stack(x, axis=0)
    st_cond = jnp.stack(st_cond_layers, axis=0)
    st_capa = jnp.stack(st_capa_layers, axis=0)
    return new_soilt, st_cond, st_capa
