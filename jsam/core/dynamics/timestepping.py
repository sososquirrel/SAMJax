"""
Adams-Bashforth 3rd-order timestepping for jsam scalar and momentum fields.

One AB3 step:
    phi_{n+1} = phi_n + dt * (at * F_n + bt * F_{n-1} + ct * F_{n-2})

where F = advective tendency (dφ/dt), and the coefficients are:
    nstep == 0  →  Euler:  at=1,    bt=0,    ct=0     (bootstrap, 1st-order)
    nstep == 1  →  AB2:    at=3/2,  bt=-1/2, ct=0     (bootstrap, 2nd-order)
    nstep >= 2  →  AB3:    at=23/12,bt=-16/12,ct=5/12 (3rd-order, equal dt)

This matches gSAM adamsA.f90 / abcoefs.f90 at nadams=3 — the default when
nadv_mom=2 (the setting used by IRMA and most gSAM CRM cases).  AB3 is the
lowest-order Adams-Bashforth scheme whose stability region touches the
imaginary axis, so it is marginally stable for pure gravity-wave oscillations.
AB2 is unconditionally unstable for such modes.

References
----------
  gSAM SRC/abcoefs.f90   — AB coefficient calculation (nadams=3 branch)
  gSAM SRC/adamsA.f90    — applies tendency to prognostic fields
  gSAM SRC/grid.f90:110  — nadams = 3 default
  Durran (2010) §2.3.1   — AB3 derivation and stability analysis
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from jsam.core.dynamics.advection import advect_scalar, advect_momentum


# ---------------------------------------------------------------------------
# Tendencies pytree — carries previous-step advective tendencies for AB2
# ---------------------------------------------------------------------------

@dataclass
class Tendencies:
    """
    Advective tendencies from one timestep (dφ/dt for each scalar field).

    All arrays (nz, ny, nx).  Registered as a JAX pytree so they can be
    passed through jit-compiled functions without re-tracing.
    """

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


# ---------------------------------------------------------------------------
# AB coefficient selection — Euler → AB2 → AB3 bootstrap, supports variable dt
# ---------------------------------------------------------------------------

def ab_coefs(
    nstep:    int,
    dt_curr:  float | None = None,
    dt_prev:  float | None = None,
    dt_pprev: float | None = None,
) -> tuple[float, float, float]:
    """
    Return (at, bt, ct) for the AB update, matching gSAM abcoefs.f90 exactly.

        phi_{n+1} = phi_n + dt_curr * (at*F_n + bt*F_{n-1} + ct*F_{n-2})

    This implements the gSAM nadams=3 semantics (the default; nadams=2 is a
    legacy CRM option not used by IRMA).  See gSAM SRC/abcoefs.f90 lines 14-30:

        if (nadams==3 .or. nadams==23) .and. (at/=0 .and. bt/=0 ...) then
            ! AB3 branch — both at and bt are nonzero from prior call
            alpha = dt3(nb)/dt3(na)        ! dt_{n-1}/dt_n
            beta  = dt3(nc)/dt3(na)        ! dt_{n-2}/dt_n
            ct = (2.+3.*alpha)/(6.*(alpha+beta)*beta)
            bt = -(1.+2.*(alpha+beta)*ct)/(2.*alpha)
            at = 1. - bt - ct
        else if (nadams==2 .and. ...) .or. (at/=0 .and. bt==0) then
            ! AB2 branch — bt was zero on entry (i.e. just past Euler)
            alpha = dt3(nb)/dt3(na)
            at = (1.+2*alpha)/(2.*alpha)
            bt = -1./(2.*alpha)
            ct = 0.
        else
            at = 1.; bt = 0.; ct = 0.       ! Euler bootstrap
        end if

    gSAM uses the (at, bt) globals as a state machine to decide the branch.
    jsam translates this into nstep-indexed semantics (nstep=0 is the very
    first call, buffers rotate AFTER the call):

        nstep == 0  (or dt_prev  is None)  →  Euler  (1, 0, 0)
        nstep == 1  (or dt_pprev is None)  →  AB2    (variable-dt)
        nstep >= 2                          →  AB3    (variable-dt)

    Note: gSAM's state machine would actually take the AB2 branch on the very
    first call (since at=1, bt=0 globals match the second branch), with
    dt_prev defaulting to dt3 initial value.  jsam returns Euler on nstep==0
    instead — a defensible simplification because no prior tendency exists,
    so falling back to Euler is safe and matches gSAM's behavior in the
    nrestart==0 cold-start path where F_{n-1} is uninitialized.

    gSAM convention:
        alpha = dt_prev / dt_curr    ( = dt3(nb)/dt3(na) )
        beta  = dt_pprev / dt_curr   ( = dt3(nc)/dt3(na) )

    AB2 (bootstrap, second call):
        at = (1 + 2α) / (2α),   bt = -1/(2α),   ct = 0

    AB3 (steady state, third call onward):
        ct = (2 + 3α) / (6 (α+β) β)
        bt = -(1 + 2(α+β) ct) / (2α)
        at = 1 - bt - ct

    Constant-dt (α=β=1) → AB3 = (23/12, -16/12, 5/12) ≈ (1.9167, -1.3333, 0.4167).
    """
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


# ---------------------------------------------------------------------------
# Per-field AB advance (Euler → AB2 → AB3, works on any JAX array shape)
# ---------------------------------------------------------------------------

# Back-compat shims — existing unit tests import ab2_coefs/ab2_step.
# NOTE: only valid for nstep <= 1 (Euler or AB2). For nstep >= 2 use ab_coefs
# directly with dt_pprev set — ab2_coefs passes dt_pprev=None which forces the
# AB2 branch and silently drops the AB3 ct term, giving wrong coefficients.
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
    """
    Advance one array by one Adams-Bashforth step (Euler/AB2/AB3 per nstep).

    Returns:
        phi_{n+1} = phi_n + dt * (at*tend_n + bt*tend_nm1 + ct*tend_nm2)
    """
    at, bt, ct = ab_coefs(nstep, dt, dt_prev, dt_pprev)
    return phi + dt * (at * tend_n + bt * tend_nm1 + ct * tend_nm2)


# ---------------------------------------------------------------------------
# Full scalar advance
# ---------------------------------------------------------------------------

def advance_scalars(
    state,
    tends_nm1: Tendencies,
    tends_nm2: Tendencies,
    metric: dict,
    dt: float,
    dt_prev:  float | None = None,
    dt_pprev: float | None = None,
    U_old:    "jax.Array | None" = None,   # C1: pre-dynamics U for half-step avg
    V_old:    "jax.Array | None" = None,   # C1: pre-dynamics V
    W_old:    "jax.Array | None" = None,   # C1: pre-dynamics W
) -> tuple:   # (ModelState, Tendencies)
    """
    Advance all scalar fields (TABS, QV, QC, QI, QR, QS, QG) by one step.

    Steps
    -----
    1. Compute advective tendency F_n = (phi_adv - phi) / dt for each scalar.
    2. Apply AB2 (Euler on nstep=0) to get phi_{n+1}.
    3. Return (new_state, tends_n) where tends_n becomes tends_prev next step.

    U, V, W are passed through unchanged (momentum advection not yet
    implemented).  nstep and time in the returned state are incremented.

    Args:
        state      : ModelState with current prognostic fields
        tends_prev : Tendencies from the previous step (F_{n-1})
        metric     : precomputed grid metrics dict (from build_metric)
        dt         : timestep in seconds

    Returns:
        (new_state, tends_n)
    """
    from jsam.core.state import ModelState

    nstep = state.nstep

    # C1 fix: gSAM advect_all_scalars uses time-averaged velocity:
    #   u_transport = 0.5*(u_old + u_new)  after the first dynamics step.
    # U_old is the pre-dynamics velocity saved before advance_momentum.
    if U_old is not None:
        U = 0.5 * (U_old + state.U)
        V = 0.5 * (V_old + state.V)
        W = 0.5 * (W_old + state.W)
    else:
        U, V, W = state.U, state.V, state.W

    # gSAM advects the liquid-ice static energy  s = TABS + gamaz
    # rather than TABS directly.  This removes the dry-adiabatic lapse-rate
    # from the vertical gradient, avoiding large flux-form tendencies in the
    # lowest cell.  After advection, TABS = s_adv - gamaz.
    gamaz = metric["gamaz"][:, None, None]   # (nz,1,1)  g*z/cp  K
    # D25 fix: gSAM advects full t = TABS + gamaz - fac_cond*(QC+QR) - fac_sub*(QI+QS+QG)
    from jsam.core.physics.microphysics import FAC_COND, FAC_SUB
    s_n = (state.TABS + gamaz
           - FAC_COND * (state.QC + state.QR)
           - FAC_SUB * (state.QI + state.QS + state.QG))

    # --- 1. Current advective tendencies ---
    # C5: pass nstep for MACHO direction ordering cycle
    def _tend(phi: jax.Array) -> jax.Array:
        """(phi_adv - phi) / dt  — one call to advect_scalar per field."""
        return (advect_scalar(phi, U, V, W, metric, dt, nstep=nstep) - phi) / dt

    tends_n = Tendencies(
        TABS=_tend(s_n),        # tendency of s = TABS + gamaz
        QV  =_tend(state.QV),
        QC  =_tend(state.QC),
        QI  =_tend(state.QI),
        QR  =_tend(state.QR),
        QS  =_tend(state.QS),
        QG  =_tend(state.QG),
    )

    # --- 2. AB3 advance (Euler → AB2 → AB3 bootstrap) ---
    def _step(phi: jax.Array, tn: jax.Array, tnm1: jax.Array, tnm2: jax.Array) -> jax.Array:
        return ab_step(phi, tn, tnm1, tnm2, dt, nstep,
                       dt_prev=dt_prev, dt_pprev=dt_pprev)

    s_new = _step(s_n, tends_n.TABS, tends_nm1.TABS, tends_nm2.TABS)

    new_state = ModelState(
        U   =state.U,
        V   =state.V,
        W   =state.W,
        TABS=s_new - gamaz,     # recover TABS from advected s
        QV  =_step(state.QV, tends_n.QV, tends_nm1.QV, tends_nm2.QV),
        QC  =_step(state.QC, tends_n.QC, tends_nm1.QC, tends_nm2.QC),
        QI  =_step(state.QI, tends_n.QI, tends_nm1.QI, tends_nm2.QI),
        QR  =_step(state.QR, tends_n.QR, tends_nm1.QR, tends_nm2.QR),
        QS  =_step(state.QS, tends_n.QS, tends_nm1.QS, tends_nm2.QS),
        QG  =_step(state.QG, tends_n.QG, tends_nm1.QG, tends_nm2.QG),
        TKE =state.TKE,
        p_prev  =state.p_prev,
        p_pprev =state.p_pprev,
        # D11 fix: nstep already incremented at top of step(); only advance time here
        nstep=state.nstep,
        time =state.time + dt,
    )

    return new_state, tends_n


# ---------------------------------------------------------------------------
# MomentumTendencies pytree — carries previous-step tendencies for U, V, W
# ---------------------------------------------------------------------------

@dataclass
class MomentumTendencies:
    """
    Advective tendencies for the momentum fields from one timestep.

    Shapes match the full staggered arrays (including BC rows/columns):
      U: (nz, ny, nx+1)
      V: (nz, ny+1, nx)
      W: (nz+1, ny, nx)

    BC rows are zero and remain zero after AB2 (BCs are re-applied inside
    advance_momentum), so storing them in the tendency is harmless and
    keeps the pytree shapes compatible with the prognostic fields.
    """

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


# ---------------------------------------------------------------------------
# Momentum advance
# ---------------------------------------------------------------------------

def advance_momentum(
    state,
    mom_tends_nm1: MomentumTendencies,
    mom_tends_nm2: MomentumTendencies,
    metric: dict,
    dt: float,
    dU_extra: "jax.Array | None" = None,   # (nz, ny, nx+1) — coriolis+buoyancy, AB'd
    dV_extra: "jax.Array | None" = None,   # (nz, ny+1, nx)
    dW_extra: "jax.Array | None" = None,   # (nz+1, ny, nx)
    dt_prev:  float | None       = None,
    dt_pprev: float | None       = None,
) -> tuple:   # (ModelState, MomentumTendencies)
    """
    Advance U, V, W by one AB2 (or Euler) step using the 3rd-order upwind
    C-grid momentum advection scheme.

    Steps
    -----
    1. Compute advective tendency F_adv = (phi_adv - phi) / dt for U, V, W.
    2. Accumulate total tendency: F_n = F_adv + dU_extra (coriolis + buoyancy).
       These extra tendencies are AB2'd together with advection, matching gSAM
       adamsA.f90 where all tendencies are accumulated before the AB2 step.
    3. Apply AB2 (Euler on nstep=0) to get U_{n+1}, V_{n+1}, W_{n+1}.
    4. Re-enforce BCs: U periodic, V polar=0, W rigid-lid=0.
    5. Return (new_state, mom_tends_n) — mom_tends_n becomes mom_tends_prev
       next step.

    Scalars (TABS, QV, …) and nstep/time are not modified here.

    Args:
        state          : ModelState with current prognostic fields
        mom_tends_prev : MomentumTendencies from the previous step (F_{n-1})
        metric         : precomputed grid metrics dict (from build_metric)
        dt             : timestep in seconds
        dU_extra       : optional additional tendency on U (e.g. Coriolis)
                         accumulated into the AB2 buffer (not just Euler-applied)
        dV_extra       : optional additional tendency on V
        dW_extra       : optional additional tendency on W (e.g. buoyancy)

    Returns:
        (new_state, mom_tends_n)
    """
    from jsam.core.state import ModelState

    nstep = state.nstep
    U, V, W = state.U, state.V, state.W

    # --- 1. Advect momentum (returns full staggered arrays with BCs applied) ---
    U_adv, V_adv, W_adv = advect_momentum(U, V, W, metric, dt)

    # --- 2. Total tendency: advection + optional coriolis/buoyancy (all AB'd) ---
    mom_tends_n = MomentumTendencies(
        U=(U_adv - U) / dt + (dU_extra if dU_extra is not None else 0.0),
        V=(V_adv - V) / dt + (dV_extra if dV_extra is not None else 0.0),
        W=(W_adv - W) / dt + (dW_extra if dW_extra is not None else 0.0),
    )

    # --- 3. AB3 advance (Euler → AB2 → AB3 bootstrap) ---
    def _step(phi, tn, tnm1, tnm2):
        return ab_step(phi, tn, tnm1, tnm2, dt, nstep,
                       dt_prev=dt_prev, dt_pprev=dt_pprev)

    U_new = _step(U, mom_tends_n.U, mom_tends_nm1.U, mom_tends_nm2.U)
    V_new = _step(V, mom_tends_n.V, mom_tends_nm1.V, mom_tends_nm2.V)
    W_new = _step(W, mom_tends_n.W, mom_tends_nm1.W, mom_tends_nm2.W)

    # --- 4. Re-enforce BCs (AB mix can drift slightly at boundary rows) ---
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
