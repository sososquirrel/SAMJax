"""
Scalar nudging toward a frozen 1D reference profile.

Faithful port of the 1D-profile branch of gSAM SRC/nudging.f90 (lines 419-473).

gSAM relaxes the prognostic scalars toward a fixed reference profile
``tg0(k), qg0(k)`` on timescale ``tautqls``, within a vertical band
``[nudging_t_z1, nudging_t_z2]``::

    t(i,j,k) -= (t0(k) - tg0(k) - gamaz(k)) * dt / tautqls   in [z1, z2]
    qv(i,j,k) -= (q0(k) - qg0(k))            * dt / tautqls  in [z1, z2]

In gSAM ``t`` is the liquid-ice static energy (``T + gamaz``) and the target
is offset by ``gamaz`` so that the relaxation pulls ``T`` (not ``s``) toward
``tg0``.  Since jsam stores ``state.TABS`` directly, the offset is absorbed
and we just relax ``TABS -> tabs_ref``.

This is the standard way global CRMs handle stratospheric drift in short
runs: the free troposphere evolves normally, the stratosphere is held at
its initial profile, and energy imbalance at the cold pole does not runaway
because it has a local sink.

Used here primarily to stabilise the upper atmosphere in jsam IRMA runs
that use a prescribed tropical radiation profile instead of RRTMG.

References
----------
  gSAM SRC/nudging.f90    lines 419-473  (1D profile branch, donudging_tq)
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class NudgingParams:
    """Parameters for scalar nudging toward a 1D reference profile.

    Attributes
    ----------
    z1_m : float
        Bottom of nudging band (m).  Nudging is zero below this height.
    z2_m : float
        Top of nudging band (m).  Nudging is zero above this height.
    tau_s : float
        Relaxation timescale (seconds).  Typical values: 1800-7200 s.
    nudge_tabs : bool
        If True, nudge TABS toward ``tabs_ref``.
    nudge_qv : bool
        If True, nudge QV toward ``qv_ref``.
    """
    z1_m:       float = 15_000.0
    z2_m:       float = 50_000.0
    tau_s:      float = 3600.0
    nudge_tabs: bool  = True
    nudge_qv:   bool  = True


def _band_mask(z: jax.Array, z1: float, z2: float) -> jax.Array:
    """Return (nz,) float32/64 mask that is 1 inside [z1, z2], 0 outside."""
    return jnp.where((z >= z1) & (z <= z2), 1.0, 0.0).astype(z.dtype)


def nudge_scalar(
    phi:     jax.Array,      # (nz, ny, nx) prognostic field
    phi_ref: jax.Array,      # (nz,) frozen reference profile
    z:       jax.Array,      # (nz,) cell-centre heights
    dt:      float,
    z1_m:    float,
    z2_m:    float,
    tau_s:   float,
    cos_lat: jax.Array | None = None,  # (ny,) for domain-mean weighting
) -> jax.Array:
    """Relax ``phi`` toward ``phi_ref`` in band ``[z1_m, z2_m]``.

    C7 fix: matches gSAM nudging.f90 line 452 — nudges the DOMAIN MEAN
    ``phi0(k)`` toward ``phi_ref(k)``, not each individual grid point.
    This avoids damping horizontal variability::

        phi^{n+1}(i,j,k) = phi^n(i,j,k) - (phi0(k) - phi_ref(k)) * (dt / tau)

    where phi0(k) is the cos-lat-weighted horizontal mean of phi at level k.
    """
    coef = dt / tau_s                             # no clipping (matches gSAM)
    mask = _band_mask(z, z1_m, z2_m)            # (nz,)
    coef_k = coef * mask                        # (nz,)  zero outside band
    coef_3d = coef_k[:, None, None]             # (nz,1,1) broadcast
    ref_3d  = phi_ref[:, None, None]            # (nz,1,1) broadcast

    # Compute domain-mean profile phi0(k) using cos-lat weighting
    if cos_lat is not None:
        wgt = cos_lat / jnp.sum(cos_lat)        # (ny,) normalised weights
        phi0 = jnp.sum(jnp.mean(phi, axis=2) * wgt[None, :], axis=1)  # (nz,)
    else:
        phi0 = jnp.mean(phi, axis=(1, 2))       # (nz,) simple mean fallback

    phi0_3d = phi0[:, None, None]               # (nz,1,1) broadcast
    # Spatially uniform correction: nudge domain mean toward reference
    return phi - coef_3d * (phi0_3d - ref_3d)


def nudge_proc(
    state,                  # ModelState
    metric: dict,
    tabs_ref: jax.Array | None,   # (nz,) frozen reference TABS profile (K)
    qv_ref:   jax.Array | None,   # (nz,) frozen reference QV profile (kg/kg)
    dt:       float,
    params:   NudgingParams,
):
    """Apply scalar nudging and return a new ModelState.

    If ``tabs_ref`` is None and ``qv_ref`` is None, returns state unchanged.
    """
    from jsam.core.state import ModelState

    if tabs_ref is None and qv_ref is None:
        return state

    z = metric["z"]        # (nz,) cell-centre heights in metres
    cos_lat = metric.get("cos_lat", None)   # (ny,) for domain-mean weighting

    new_TABS = state.TABS
    new_QV   = state.QV

    if params.nudge_tabs and tabs_ref is not None:
        new_TABS = nudge_scalar(
            state.TABS, tabs_ref, z, dt,
            params.z1_m, params.z2_m, params.tau_s,
            cos_lat=cos_lat,
        )

    if params.nudge_qv and qv_ref is not None:
        new_QV = nudge_scalar(
            state.QV, qv_ref, z, dt,
            params.z1_m, params.z2_m, params.tau_s,
            cos_lat=cos_lat,
        )

    return ModelState(
        U=state.U, V=state.V, W=state.W,
        TABS=new_TABS,
        QV=new_QV,
        QC=state.QC, QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
        TKE=state.TKE,
        p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep,
        time=state.time,
    )
