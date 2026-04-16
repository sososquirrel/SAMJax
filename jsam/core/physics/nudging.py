"""Scalar nudging toward frozen 1D reference profiles (TABS, QV) within a band [z1,z2].
Prevents stratospheric drift by relaxing domain-mean toward reference on timescale tau."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class NudgingParams:
    """Nudging params: z1_m, z2_m (band), tau_s (timescale), nudge_tabs, nudge_qv."""
    z1_m:       float = 15_000.0
    z2_m:       float = 50_000.0
    tau_s:      float = 3600.0
    nudge_tabs: bool  = True
    nudge_qv:   bool  = True


def _band_mask(z: jax.Array, z1: float, z2: float) -> jax.Array:
    """Mask: 1 inside [z1,z2], 0 outside."""
    return jnp.where((z >= z1) & (z <= z2), 1.0, 0.0).astype(z.dtype)

def nudge_scalar(phi: jax.Array, phi_ref: jax.Array, z: jax.Array, dt: float,
                 z1_m: float, z2_m: float, tau_s: float,
                 cos_lat: jax.Array | None = None) -> jax.Array:
    """Relax domain-mean of phi toward phi_ref in band; cos_lat weights horizontal mean."""
    coef = dt / tau_s
    mask = _band_mask(z, z1_m, z2_m)
    coef_k = coef * mask
    coef_3d = coef_k[:, None, None]
    ref_3d  = phi_ref[:, None, None]

    if cos_lat is not None:
        wgt = cos_lat / jnp.sum(cos_lat)
        phi0 = jnp.sum(jnp.mean(phi, axis=2) * wgt[None, :], axis=1)
    else:
        phi0 = jnp.mean(phi, axis=(1, 2))

    phi0_3d = phi0[:, None, None]
    return phi - coef_3d * (phi0_3d - ref_3d)


def nudge_proc(state, metric: dict, tabs_ref: jax.Array | None,
               qv_ref: jax.Array | None, dt: float, params: NudgingParams):
    """Apply nudging to TABS/QV profiles if provided; return new ModelState."""
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
