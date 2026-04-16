"""Adams-Bashforth 3rd-order timestepping. Matches gSAM adamsA.f90."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from jsam.core.dynamics.advection import advect_scalar, advect_momentum



@dataclass
class Tendencies:
    """Advective tendencies from one timestep (dφ/dt per scalar)."""

    TABS: jax.Array
    QV:   jax.Array
    QC:   jax.Array
    QI:   jax.Array
    QR:   jax.Array
    QS:   jax.Array
    QG:   jax.Array

    _fields: ClassVar[tuple[str, ...]] = (
        "TABS", "QV", "QC", "QI", "QR", "QS", "QG",
    )

    def tree_flatten(self):
        return [getattr(self, f) for f in self._fields], None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(**dict(zip(cls._fields, children)))

    @classmethod
    def zeros(cls, nz: int, ny: int, nx: int) -> "Tendencies":
        """Return zero tendencies (safe placeholder for AB2 bootstrap)."""
        return cls(**{f: jnp.zeros((nz, ny, nx)) for f in cls._fields})


jax.tree_util.register_pytree_node(
    Tendencies,
    Tendencies.tree_flatten,
    Tendencies.tree_unflatten,
)



def ab_coefs(
    nstep:    int,
    dt_curr:  float | None = None,
    dt_prev:  float | None = None,
    dt_pprev: float | None = None,
) -> tuple[float, float, float]:
    """Return (at, bt, ct) for AB update. Matches gSAM abcoefs.f90."""
    if nstep == 0 or dt_prev is None or dt_curr is None:
        return 1.0, 0.0, 0.0

    alpha = dt_prev / dt_curr
    if nstep == 1 or dt_pprev is None:
        at = (1.0 + 2.0 * alpha) / (2.0 * alpha)
        bt = -1.0 / (2.0 * alpha)
        return at, bt, 0.0

    beta = dt_pprev / dt_curr
    ct = (2.0 + 3.0 * alpha) / (6.0 * (alpha + beta) * beta)
    bt = -(1.0 + 2.0 * (alpha + beta) * ct) / (2.0 * alpha)
    at = 1.0 - bt - ct
    return at, bt, ct


def ab2_coefs(nstep, dt_curr=None, dt_prev=None):
    at, bt, _ = ab_coefs(nstep, dt_curr, dt_prev, None)
    return at, bt


def ab2_step(phi, tend_n, tend_nm1, dt, nstep, dt_prev=None):
    return ab_step(phi, tend_n, tend_nm1, jnp.zeros_like(tend_n),
                   dt, nstep, dt_prev=dt_prev, dt_pprev=None)


def ab_step(
    phi:      jax.Array,
    tend_n:   jax.Array,
    tend_nm1: jax.Array,
    tend_nm2: jax.Array,
    dt:       float,
    nstep:    int,
    dt_prev:  float | None = None,
    dt_pprev: float | None = None,
) -> jax.Array:
    """Advance array by one AB step (Euler/AB2/AB3)."""
    at, bt, ct = ab_coefs(nstep, dt, dt_prev, dt_pprev)
    return phi + dt * (at * tend_n + bt * tend_nm1 + ct * tend_nm2)



def advance_scalars(
    state,
    tends_nm1: Tendencies,
    tends_nm2: Tendencies,
    metric: dict,
    dt: float,
    dt_prev:  float | None = None,
    dt_pprev: float | None = None,
    U_old:    "jax.Array | None" = None,
    V_old:    "jax.Array | None" = None,
    W_old:    "jax.Array | None" = None,
) -> tuple:
    """Advance scalars (TABS, QV, QC, QI, QR, QS, QG) by one step."""
    from jsam.core.state import ModelState

    nstep = state.nstep
    if U_old is not None:
        U = 0.5 * (U_old + state.U)
        V = 0.5 * (V_old + state.V)
        W = 0.5 * (W_old + state.W)
    else:
        U, V, W = state.U, state.V, state.W
    gamaz = metric["gamaz"][:, None, None]
    from jsam.core.physics.microphysics import FAC_COND, FAC_SUB
    s_n = (state.TABS + gamaz
           - FAC_COND * (state.QC + state.QR)
           - FAC_SUB * (state.QI + state.QS + state.QG))
    def _adv(phi: jax.Array) -> jax.Array:
        return advect_scalar(phi, U, V, W, metric, dt, nstep=nstep)

    s_new  = _adv(s_n)
    qv_new = _adv(state.QV)
    qc_new = _adv(state.QC)
    qi_new = _adv(state.QI)
    qr_new = _adv(state.QR)
    qs_new = _adv(state.QS)
    qg_new = _adv(state.QG)

    tends_n = Tendencies(
        TABS=(s_new  - s_n)       / dt,
        QV  =(qv_new - state.QV)  / dt,
        QC  =(qc_new - state.QC)  / dt,
        QI  =(qi_new - state.QI)  / dt,
        QR  =(qr_new - state.QR)  / dt,
        QS  =(qs_new - state.QS)  / dt,
        QG  =(qg_new - state.QG)  / dt,
    )

    tabs_new = s_new - gamaz

    new_state = ModelState(
        U   =state.U,
        V   =state.V,
        W   =state.W,
        TABS=tabs_new,
        QV  =qv_new,
        QC  =qc_new,
        QI  =qi_new,
        QR  =qr_new,
        QS  =qs_new,
        QG  =qg_new,
        TKE =state.TKE,
        p_prev  =state.p_prev,
        p_pprev =state.p_pprev,
        nstep=state.nstep,
        time =state.time + dt,
    )

    return new_state, tends_n



@dataclass
class MomentumTendencies:
    """Momentum tendencies from one timestep (U, V, W)."""

    U: jax.Array
    V: jax.Array
    W: jax.Array

    def tree_flatten(self):
        return [self.U, self.V, self.W], None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @classmethod
    def zeros(cls, nz: int, ny: int, nx: int) -> "MomentumTendencies":
        """Return zero tendencies (safe placeholder for AB2 bootstrap)."""
        return cls(
            U=jnp.zeros((nz, ny, nx + 1)),
            V=jnp.zeros((nz, ny + 1, nx)),
            W=jnp.zeros((nz + 1, ny, nx)),
        )


jax.tree_util.register_pytree_node(
    MomentumTendencies,
    MomentumTendencies.tree_flatten,
    MomentumTendencies.tree_unflatten,
)



def advance_momentum(
    state,
    mom_tends_nm1: MomentumTendencies,
    mom_tends_nm2: MomentumTendencies,
    metric: dict,
    dt: float,
    dU_extra: "jax.Array | None" = None,
    dV_extra: "jax.Array | None" = None,
    dW_extra: "jax.Array | None" = None,
    dt_prev:  float | None       = None,
    dt_pprev: float | None       = None,
) -> tuple:
    """Advance U, V, W by one AB2 step with 3rd-order upwind advection."""
    from jsam.core.state import ModelState

    nstep = state.nstep
    U, V, W = state.U, state.V, state.W
    U_adv, V_adv, W_adv = advect_momentum(U, V, W, metric, dt)
    mom_tends_n = MomentumTendencies(
        U=(U_adv - U) / dt + (dU_extra if dU_extra is not None else 0.0),
        V=(V_adv - V) / dt + (dV_extra if dV_extra is not None else 0.0),
        W=(W_adv - W) / dt + (dW_extra if dW_extra is not None else 0.0),
    )

    def _step(phi, tn, tnm1, tnm2):
        return ab_step(phi, tn, tnm1, tnm2, dt, nstep,
                       dt_prev=dt_prev, dt_pprev=dt_pprev)
    U_new = _step(U, mom_tends_n.U, mom_tends_nm1.U, mom_tends_nm2.U)
    V_new = _step(V, mom_tends_n.V, mom_tends_nm1.V, mom_tends_nm2.V)
    W_new = _step(W, mom_tends_n.W, mom_tends_nm1.W, mom_tends_nm2.W)
    nz, ny, nx_p1 = U_new.shape
    nx = nx_p1 - 1
    U_new = U_new.at[:, :, nx].set(U_new[:, :, 0])   # periodic
    V_new = V_new.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)   # polar
    W_new = W_new.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)   # rigid lid

    new_state = ModelState(
        U   =U_new,
        V   =V_new,
        W   =W_new,
        TABS=state.TABS,
        QV  =state.QV,
        QC  =state.QC,
        QI  =state.QI,
        QR  =state.QR,
        QS  =state.QS,
        QG  =state.QG,
        TKE =state.TKE,
        p_prev  =state.p_prev,
        p_pprev =state.p_pprev,
        nstep=state.nstep,
        time =state.time,
    )

    return new_state, mom_tends_n
