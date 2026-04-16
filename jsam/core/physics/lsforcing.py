"""Large-scale forcing: horizontal advective tendencies (dtls, dqls) + subsidence (wsub).
Time-varying profiles interpolated bilinearly to model grid. First-order upwind for subsidence."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jsam.core.state import ModelState


@jax.tree_util.register_pytree_node_class
@dataclass
class LargeScaleForcing:
    """Time-varying profiles: dtls, dqls (K/s, kg/kg/s), wsub (m/s), z_prof, t_prof."""
    dtls:   jax.Array
    dqls:   jax.Array
    wsub:   jax.Array
    z_prof: jax.Array
    t_prof: jax.Array

    @classmethod
    def zeros(cls, nz_prof: int = 2) -> "LargeScaleForcing":
        """Zero forcing on trivial 2-level grid."""
        z = jnp.array([0.0, 1.0e5])
        q = jnp.zeros((2, nz_prof))
        return cls(
            dtls   = jnp.zeros((2, 2)),
            dqls   = jnp.zeros((2, 2)),
            wsub   = jnp.zeros((2, 2)),
            z_prof = jnp.array([0.0, 1.0e5]),
            t_prof = jnp.array([0.0, 1.0e30]),
        )

    @classmethod
    def constant(
        cls,
        dtls:   jax.Array,   # (nz_prof,) K/s
        dqls:   jax.Array,   # (nz_prof,) kg/kg/s
        wsub:   jax.Array,   # (nz_prof,) m/s
        z_prof: jax.Array,   # (nz_prof,) m
    ) -> "LargeScaleForcing":
        """Time-invariant forcing (held constant for all t)."""
        stack = lambda p: jnp.stack([p, p], axis=0)   # (2, nz_prof)
        return cls(
            dtls   = stack(dtls),
            dqls   = stack(dqls),
            wsub   = stack(wsub),
            z_prof = z_prof,
            t_prof = jnp.array([0.0, 1.0e30]),
        )

    # ---- pytree -----------------------------------------------------------

    def tree_flatten(self):
        return (self.dtls, self.dqls, self.wsub, self.z_prof, self.t_prof), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


# ---------------------------------------------------------------------------
# Shared interpolation (same pattern as radiation.py)
# ---------------------------------------------------------------------------

def _interp1d(x: jax.Array, xp: jax.Array, fp: jax.Array) -> jax.Array:
    """Linear 1-D interpolation of fp at xp onto scalar x; clamps at edges."""
    idx = jnp.searchsorted(xp, x, side="right")
    idx = jnp.clip(idx, 1, xp.shape[0] - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    dx = x1 - x0
    w  = jnp.where(dx > 0.0, (x - x0) / dx, 0.0)
    w  = jnp.clip(w, 0.0, 1.0)
    above = x > xp[-1]
    return jnp.where(above, 0.0, f0 + w * (f1 - f0))


def _interp_time(t: jax.Array, t_prof: jax.Array, table: jax.Array) -> jax.Array:
    """
    Linearly interpolate ``table`` (ntime, nz_prof) to time ``t``.
    Returns (nz_prof,).
    """
    idx = jnp.searchsorted(t_prof, t, side="right")
    idx = jnp.clip(idx, 1, t_prof.shape[0] - 1)
    t0, t1 = t_prof[idx - 1], t_prof[idx]
    p0, p1 = table[idx - 1, :], table[idx, :]
    dt = t1 - t0
    w  = jnp.where(dt > 0.0, (t - t0) / dt, 0.0)
    w  = jnp.clip(w, 0.0, 1.0)
    return p0 + w * (p1 - p0)   # (nz_prof,)


def _profile_on_model_grid(
    t:       jax.Array,
    t_prof:  jax.Array,
    table:   jax.Array,
    z_prof:  jax.Array,
    z_model: jax.Array,
) -> jax.Array:
    """Bilinear interpolation of a (ntime, nz_prof) table to (t, z_model)."""
    p = _interp_time(t, t_prof, table)                              # (nz_prof,)
    return jax.vmap(lambda z: _interp1d(z, z_prof, p))(z_model)    # (nz,)


# ---------------------------------------------------------------------------
# Subsidence: first-order upwind −w_ls · ∂φ/∂z
# (matches gSAM subsidence.f90)
# ---------------------------------------------------------------------------

def _subsidence_tend(phi: jax.Array, wsub: jax.Array, dz: jax.Array) -> jax.Array:
    """Compute −w_ls·∂φ/∂z with first-order upwind; edge-pad for BCs."""
    dz3 = dz[:, None, None]      # (nz, 1, 1)
    w3  = wsub[:, None, None]    # (nz, 1, 1)

    phi_lo = jnp.pad(phi, ((1, 0), (0, 0), (0, 0)), mode="edge")
    phi_hi = jnp.pad(phi, ((0, 1), (0, 0), (0, 0)), mode="edge")

    grad_bwd = (phi - phi_lo[:-1]) / dz3
    grad_fwd = (phi_hi[1:] - phi) / dz3

    dphi_dz  = jnp.where(w3 >= 0.0, grad_bwd, grad_fwd)
    return -w3 * dphi_dz   # (nz, ny, nx)


# ---------------------------------------------------------------------------
# Top-level large-scale forcing step
# ---------------------------------------------------------------------------

@jax.jit
def ls_proc(state: ModelState, metric: dict, forcing: LargeScaleForcing, dt: float) -> ModelState:
    """Apply horizontal advection (dtls, dqls) + subsidence (w_ls) in one step."""
    z_model = metric["z"]
    dz      = metric["dz"]
    t       = state.time

    def _on_grid(table):
        return _profile_on_model_grid(t, forcing.t_prof, table, forcing.z_prof, z_model)

    dtls_z = _on_grid(forcing.dtls)
    dqls_z = _on_grid(forcing.dqls)
    wsub_z = _on_grid(forcing.wsub)

    nz, ny, nx = state.TABS.shape
    TABS_new = state.TABS + dt * dtls_z[:, None, None]
    QV_new   = jnp.maximum(0.0, state.QV + dt * dqls_z[:, None, None])

    gamaz = metric["gamaz"][:, None, None]
    t_field = TABS_new + gamaz
    t_field = t_field + dt * _subsidence_tend(t_field, wsub_z, dz)
    TABS_new = t_field - gamaz

    QV_new   = jnp.maximum(0.0, QV_new + dt * _subsidence_tend(QV_new, wsub_z, dz))
    QC_new   = jnp.maximum(0.0, state.QC + dt * _subsidence_tend(state.QC, wsub_z, dz))
    QI_new   = jnp.maximum(0.0, state.QI + dt * _subsidence_tend(state.QI, wsub_z, dz))
    QR_new   = jnp.maximum(0.0, state.QR + dt * _subsidence_tend(state.QR, wsub_z, dz))
    QS_new   = jnp.maximum(0.0, state.QS + dt * _subsidence_tend(state.QS, wsub_z, dz))
    QG_new   = jnp.maximum(0.0, state.QG + dt * _subsidence_tend(state.QG, wsub_z, dz))

    return ModelState(
        U     = state.U,
        V     = state.V,
        W     = state.W,
        TABS  = TABS_new,
        QV    = QV_new,
        QC    = QC_new,
        QI    = QI_new,
        QR    = QR_new,
        QS    = QS_new,
        QG    = QG_new,
        TKE   = state.TKE,
        p_prev = state.p_prev, p_pprev = state.p_pprev,
        nstep = state.nstep,
        time  = state.time,
    )
