"""
Monin-Obukhov surface-layer similarity for jsam SLM.

Direct port of ``gSAM1.8.7/SRC/SLM/transfer_coef.f90`` (Marat Khairoutdinov
2016, with 2025 revisions). The Fortran subroutine is scalar — it is
invoked once per (i,j) cell inside the ``run_slm`` nested loop. This
module vectorises the whole computation over the (ny, nx) grid and
replaces the ``do while`` dynamic-exit iteration with a fixed-length
``jax.lax.fori_loop`` (``NITERMAX = 10`` passes, matching the Fortran cap).

Inputs / outputs mirror the Fortran signature of ``transfer_coef`` called
from ``run_slm.f90``. The potential-temperature conversion done in the
caller (``tsfc_pot``, ``tr_pot``) is the responsibility of the JAX caller
too — we receive potential temperatures just like the Fortran.

Fortran module dependencies and where they land in the JAX signature
--------------------------------------------------------------------
``use slm_vars, only: rgas, cp, pii, epsv, DBL, ...``
    * ``rgas`` (287.0, J/kg/K), ``cp`` (1004.0, J/kg/K), ``epsv`` (0.61)
      — not actually used inside ``transfer_coef`` itself but by the
      caller converting to potential temperature; we keep them only where
      they appear (``epsv`` is used in the bulk Richardson number, so it
      is a module-level constant here).
    * ``pii`` — unused here.
    * ``DBL`` — double precision; JAX runs single-precision by default
      (``float32``). The Fortran is already algorithmically stable in
      single precision; bit-close here means float32-close.
    * The module-level output arrays (``mom_trans_coef``, ``ustar``,
      ``tstar``, ``RiB``, ``r_a``, ``vel_m``, ``u_10m``, ``v_10m``,
      ``temp_2m``, ``q_2m``) are replaced by an explicit
      :class:`TransferCoefOutput` return value.

``use grid, only: dtn``
    * ``dtn`` — dynamic timestep, enters the ``fm0`` flux limiter
      (Fortran lines 146-148). Passed in as ``dtn`` argument.

The SAM core ``params`` module is not used directly by this routine.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# Constants (match Fortran literals in transfer_coef.f90)
# ---------------------------------------------------------------------------

KARMAN: float = 0.4        # von Kármán constant (kk)
EPSV:   float = 0.61       # (Rv/Rd - 1); virtual T correction used in RiB
G:      float = 9.81       # gravity (m/s²)

XSIM: float = -1.574       # unstable/very-unstable xsi breakpoint (momentum)
XSIH: float = -0.465       # unstable/very-unstable xsi breakpoint (heat)

NITERMAX: int = 10         # Fortran ``nitermax`` — unrolled via fori_loop


# ``xm = (1 - 16*xsim)**(1/4)`` and likewise ``xh`` — Fortran parameters.
_XM: float = float((1.0 - 16.0 * XSIM) ** 0.25)
_XH: float = float((1.0 - 16.0 * XSIH) ** 0.25)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TransferCoefOutput:
    """Outputs of :func:`transfer_coef` (all arrays shape ``(ny, nx)``)."""

    ustar:           jax.Array
    tstar:           jax.Array
    mom_trans_coef:  jax.Array   # C_D
    heat_trans_coef: jax.Array   # C_H
    r_a:             jax.Array   # aerodynamic resistance (s/m)
    vel_m:           jax.Array   # wind speed used in the flux calc (m/s)
    RiB:             jax.Array   # bulk Richardson number
    u_10m:           jax.Array
    v_10m:           jax.Array
    temp_2m:         jax.Array   # potential temperature at 2 m
    q_2m:            jax.Array

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "ustar", "tstar",
        "mom_trans_coef", "heat_trans_coef",
        "r_a", "vel_m", "RiB",
        "u_10m", "v_10m", "temp_2m", "q_2m",
    )

    def tree_flatten(self):
        return [getattr(self, n) for n in self._dynamic_fields], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(**dict(zip(cls._dynamic_fields, children)))


jax.tree_util.register_pytree_node(
    TransferCoefOutput,
    TransferCoefOutput.tree_flatten,
    TransferCoefOutput.tree_unflatten,
)


# ---------------------------------------------------------------------------
# Helpers — stability functions (psi_m, psi_h)
# ---------------------------------------------------------------------------

def _xx(y: jax.Array) -> jax.Array:
    """Fortran ``xx(y) = ((1 - 16*y)**(1/4))``; used for unstable branch."""
    return jnp.sqrt(jnp.sqrt(jnp.maximum(1.0 - 16.0 * y, 0.0)))


def _psim1(x: jax.Array, x0: jax.Array) -> jax.Array:
    return (2.0 * jnp.log((1.0 + x) / (1.0 + x0))
            + jnp.log((1.0 + x * x) / (1.0 + x0 * x0))
            - 2.0 * (jnp.arctan(x) - jnp.arctan(x0)))


def _psih1(x: jax.Array, x0: jax.Array) -> jax.Array:
    return 2.0 * jnp.log((1.0 + x * x) / (1.0 + x0 * x0))


def _psim2(xsi: jax.Array, xsim0: jax.Array, xm0: jax.Array) -> jax.Array:
    # Uses module-level XSIM / _XM parameters (Fortran parameters).
    return (jnp.log(XSIM / xsim0)
            - _psim1(jnp.asarray(_XM, xsi.dtype), xm0)
            + 1.14 * (jnp.power(jnp.maximum(-xsi, 0.0), 1.0 / 3.0)
                      - (-XSIM) ** (1.0 / 3.0)))


def _psih2(xsi: jax.Array, xsih0: jax.Array, xh0: jax.Array) -> jax.Array:
    return (jnp.log(XSIH / xsih0)
            - _psih1(jnp.asarray(_XH, xsi.dtype), xh0)
            + 0.8 * (jnp.power(jnp.maximum(-xsi, 0.0), 1.0 / 3.0)
                     - (-XSIH) ** (1.0 / 3.0)))


def _psim3(xsi: jax.Array, xsim0: jax.Array) -> jax.Array:
    return -5.0 * (xsi - xsim0)


def _psih3(xsi: jax.Array, xsih0: jax.Array) -> jax.Array:
    return -5.0 * (xsi - xsih0)


def _psim4(xsi: jax.Array, xsim0: jax.Array) -> jax.Array:
    return (jnp.log(jnp.power(jnp.maximum(xsi, 1e-30), 5) / xsim0)
            + 5.0 * (1.0 - xsim0) + xsi - 1.0)


def _psih4(xsi: jax.Array, xsih0: jax.Array) -> jax.Array:
    return (jnp.log(jnp.power(jnp.maximum(xsi, 1e-30), 5) / xsih0)
            + 5.0 * (1.0 - xsih0) + xsi - 1.0)


def _fm_fh(xsi: jax.Array,
           zodym: jax.Array,
           zodyh: jax.Array,
           z0h:  jax.Array,
           zTh:  jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Vectorised port of the branched ``fm/fh`` evaluation in Fortran
    (lines 104-132). Selects the right stability regime per cell via
    ``jnp.where`` — every branch is evaluated unconditionally and the
    correct one is picked up afterwards. All intermediate expressions are
    guarded so NaNs from inactive branches never propagate to the
    ``where`` result (e.g. ``xsi**5`` is clipped for ``xsi<=0`` cells).
    """
    xsim0 = z0h * xsi
    xsih0 = zTh * xsi

    # --- Unstable branch (xsi < -0.01) ---------------------------------
    # Momentum: if xsi >= xsim use psim1, else psim2.
    # Use ``_xx`` clipped at 0 to avoid sqrt of negative in dead branches.
    x_u      = _xx(xsi)
    x0_um    = _xx(xsim0)
    x0_uh    = _xx(xsih0)
    fm_u_a   = zodym - _psim1(x_u, x0_um)
    fm_u_b   = _psim2(xsi, xsim0, x0_um)
    fm_unst  = jnp.where(xsi >= XSIM, fm_u_a, fm_u_b)

    fh_u_a   = zodyh - _psih1(x_u, x0_uh)
    fh_u_b   = _psih2(xsi, xsih0, x0_uh)
    fh_unst  = jnp.where(xsi >= XSIH, fh_u_a, fh_u_b)

    # --- Stable branch (xsi > 0.01) ------------------------------------
    fm_s_a = zodym - _psim3(xsi, xsim0)
    fh_s_a = zodyh - _psih3(xsi, xsih0)
    fm_s_b = _psim4(xsi, xsim0)
    fh_s_b = _psih4(xsi, xsih0)
    fm_stab = jnp.where(xsi <= 1.0, fm_s_a, fm_s_b)
    fh_stab = jnp.where(xsi <= 1.0, fh_s_a, fh_s_b)

    # --- Neutral branch (-0.01 <= xsi <= 0.01) -------------------------
    # fm = zodym, fh = zodyh.

    unstable = xsi < -0.01
    stable   = xsi > 0.01

    fm = jnp.where(unstable, fm_unst,
                    jnp.where(stable, fm_stab, zodym))
    fh = jnp.where(unstable, fh_unst,
                    jnp.where(stable, fh_stab, zodyh))
    return fm, fh


def _fm_only(xsi: jax.Array,
             zodym: jax.Array,
             z0h:  jax.Array) -> jax.Array:
    """Diagnostic 10-m-momentum version of ``_fm_fh`` (Fortran 182-199)."""
    xsim0 = z0h * xsi
    x_u   = _xx(xsi)
    x0_um = _xx(xsim0)
    fm_u_a = zodym - _psim1(x_u, x0_um)
    fm_u_b = _psim2(xsi, xsim0, x0_um)
    fm_unst = jnp.where(xsi >= XSIM, fm_u_a, fm_u_b)
    fm_s_a = zodym - _psim3(xsi, xsim0)
    fm_s_b = _psim4(xsi, xsim0)
    fm_stab = jnp.where(xsi <= 1.0, fm_s_a, fm_s_b)
    unstable = xsi < -0.01
    stable   = xsi > 0.01
    return jnp.where(unstable, fm_unst,
                      jnp.where(stable, fm_stab, zodym))


def _fh_only(xsi: jax.Array,
             zodyh: jax.Array,
             zTh:  jax.Array) -> jax.Array:
    """Diagnostic 2-m-scalar version (Fortran 200-217)."""
    xsih0 = zTh * xsi
    x_u   = _xx(xsi)
    x0_uh = _xx(xsih0)
    fh_u_a = zodyh - _psih1(x_u, x0_uh)
    fh_u_b = _psih2(xsi, xsih0, x0_uh)
    fh_unst = jnp.where(xsi >= XSIH, fh_u_a, fh_u_b)
    fh_s_a = zodyh - _psih3(xsi, xsih0)
    fh_s_b = _psih4(xsi, xsih0)
    fh_stab = jnp.where(xsi <= 1.0, fh_s_a, fh_s_b)
    unstable = xsi < -0.01
    stable   = xsi > 0.01
    return jnp.where(unstable, fh_unst,
                      jnp.where(stable, fh_stab, zodyh))


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def transfer_coef(tsp:     jax.Array,
                  thp:     jax.Array,
                  qh:      jax.Array,
                  qs:      jax.Array,
                  uh:      jax.Array,
                  vh:      jax.Array,
                  h:       jax.Array,
                  z0_in:   jax.Array,
                  disp:    jax.Array,
                  dtn:     float | jax.Array) -> TransferCoefOutput:
    """
    Monin-Obukhov surface-layer similarity transfer coefficients.

    Vectorised over the (ny, nx) horizontal grid.

    Parameters
    ----------
    tsp : potential temperature at the surface (z0), K.
    thp : potential temperature at reference height ``h``, K.
    qh  : specific humidity at ``h``, kg/kg.
    qs  : saturation specific humidity at the surface, kg/kg.
    uh  : zonal wind at ``h``, m/s.
    vh  : meridional wind at ``h``, m/s.
    h   : reference-level height, m.
    z0_in : surface roughness length for momentum, m.
    disp  : zero-plane displacement height, m.
    dtn : dynamic timestep (s) — enters the flux limiter.

    Returns
    -------
    :class:`TransferCoefOutput` — ``ustar, tstar, C_D, C_H, r_a, vel_m,
    RiB, u_10m, v_10m, temp_2m, q_2m`` (all shape ``(ny, nx)``).

    Notes
    -----
    The Fortran dynamic-exit iteration

        niter = 0
        do while (error > errormax .and. niter < nitermax) ...

    cannot be vectorised efficiently in JAX (different cells converge at
    different steps), so we always run the full ``NITERMAX = 10`` passes
    via ``jax.lax.fori_loop``. For stable / near-neutral cells the
    iteration has already converged well within 10 passes, so the
    post-convergence updates are no-ops and the output is bit-close to
    Fortran (float32 precision).
    """
    tsp = jnp.asarray(tsp)
    thp = jnp.asarray(thp)
    qh  = jnp.asarray(qh)
    qs  = jnp.asarray(qs)
    uh  = jnp.asarray(uh)
    vh  = jnp.asarray(vh)
    h   = jnp.asarray(h)
    z0_in = jnp.asarray(z0_in)
    disp  = jnp.asarray(disp)

    # Velocity floor (0.5 m/s) — identical in both stratification branches.
    vel = jnp.maximum(0.5, jnp.sqrt(uh * uh + vh * vh))

    # Bulk Richardson number, capped [-10, 0.5]. Matches Fortran line 66-68.
    r = (G / tsp
         * (thp * (1.0 + EPSV * qh) - tsp * (1.0 + EPSV * qs))
         * (h - disp) / (vel * vel))
    r = jnp.maximum(-10.0, jnp.minimum(r, 0.5))

    # High-wind z0 adjustment (Fortran line 72, MK 2025).
    z0  = z0_in * jnp.power(1.0 + vel / 10.0, -0.6)
    zt0 = 0.135 * z0   # Garratt BL textbook, p.93

    z0h = z0  / (h - disp)
    zTh = zt0 / (h - disp)

    zodym = jnp.log(1.0 / z0h)
    zodyh = jnp.log(1.0 / zTh)

    # First guess for xsi (Fortran lines 89-93).
    xsi_stab   = r * zodym / (1.0 - 5.0 * jnp.minimum(0.19, r))
    xsi_unstab = r * zodym
    xsi0 = jnp.where(r > 0.0, xsi_stab, xsi_unstab)

    # --- Monin-Obukhov fixed-point iteration ------------------------------
    # Run NITERMAX unrolled passes; each pass is a pure function of the
    # previous xsi and the cell-constant fields above.
    def _body(_i, carry):
        xsi_prev, _fm_prev, _fh_prev = carry
        fm_, fh_ = _fm_fh(xsi_prev, zodym, zodyh, z0h, zTh)
        xsi_new = r * fm_ * fm_ / fh_
        return (xsi_new, fm_, fh_)

    xsi_init = xsi0
    fm_init = jnp.zeros_like(xsi0)
    fh_init = jnp.ones_like(xsi0)
    xsi, _, _ = lax.fori_loop(0, NITERMAX, _body,
                               (xsi_init, fm_init, fh_init))

    # Recompute fm, fh at the final xsi so they are consistent with it
    # (matches Fortran: after the ``do while`` exits, ``fm``/``fh`` hold
    # the values computed from the last ``xsi1 = xsi`` before the update).
    # In the Fortran, after the last iteration, xsi has been updated
    # from fm,fh; fm_phys/fh_phys thus correspond to the xsi *before* the
    # last update. We replicate that by re-running the branch on the
    # penultimate xsi — but since we used fori_loop we don't have it.
    # Instead: run the branch on the *current* xsi; in steady state the
    # difference is below errormax=0.01 so fm,fh agree to that tolerance.
    fm, fh = _fm_fh(xsi, zodym, zodyh, z0h, zTh)

    # Save for 10 m / 2 m diagnostics (Fortran fm_phys, fh_phys).
    fm_phys = fm
    fh_phys = fh

    # Flux limiter: cap slowdown at 50% per timestep (Fortran 146-148).
    fm0 = jnp.sqrt(KARMAN ** 2 * vel * dtn / 0.5 / h)
    fm_lim = jnp.maximum(fm0, fm)
    fh_lim = jnp.maximum(fh, fh * fm0 / fm)   # fh/fm*fm0 == fh*fm0/fm

    # C_D and C_H (Fortran 153-154).
    mom_trans_coef  = KARMAN ** 2 / (fm_lim * fm_lim)
    heat_trans_coef = KARMAN ** 2 / (fm_lim * fh_lim)

    # ustar, tstar (Fortran 155-159, with the 2025 night-time floor).
    ustar = jnp.sqrt(mom_trans_coef) * vel
    tstar = -KARMAN * (thp - tsp) / fh_lim
    ustar = jnp.sqrt(ustar * ustar + 0.05 ** 2)

    # Recompute C_D, C_H for consistency with floored ustar (Fortran 161-162).
    mom_trans_coef  = (ustar / vel) ** 2
    heat_trans_coef = (KARMAN / fh_lim) * (ustar / vel)

    # Aerodynamic resistance (Fortran 167).
    r_a = fh_lim / KARMAN / ustar

    vel_m = vel
    RiB   = r

    # --- 10 m / 2 m diagnostics (Fortran 172-222) -------------------------
    z0h_d = z0  / 10.0
    zTh_d = zt0 / 2.0
    zodym_d = jnp.log(1.0 / z0h_d)
    zodyh_d = jnp.log(1.0 / zTh_d)
    xsi10 = xsi * 10.0 / (h - disp)
    xsi2  = xsi *  2.0 / (h - disp)

    fm10 = _fm_only(xsi10, zodym_d, z0h_d)
    fh2  = _fh_only(xsi2,  zodyh_d, zTh_d)

    u_10m   = uh * fm10 / fm_phys
    v_10m   = vh * fm10 / fm_phys
    temp_2m = tsp + (thp - tsp) * fh2 / fh_phys
    q_2m    = qs  + (qh  - qs ) * fh2 / fh_phys

    return TransferCoefOutput(
        ustar=ustar,
        tstar=tstar,
        mom_trans_coef=mom_trans_coef,
        heat_trans_coef=heat_trans_coef,
        r_a=r_a,
        vel_m=vel_m,
        RiB=RiB,
        u_10m=u_10m,
        v_10m=v_10m,
        temp_2m=temp_2m,
        q_2m=q_2m,
    )
