"""
jsam step driver — one model timestep.

Orchestrates all physics and dynamics modules in gSAM order:

    ls_proc           large-scale horiz. advective tendencies + subsidence
    rad_proc          prescribed radiative heating
    bulk_surface_fluxes  ocean surface fluxes (→ SurfaceFluxes)
    sgs_mom_proc      Smagorinsky SGS on U,V,W only (before advance_momentum)
    [compute buoyancy tendency dW_buo]
    [compute coriolis tendency dU_cor, dV_cor]
    advance_momentum  momentum advection + coriolis + buoyancy AB2'd together
    pole_damping      polar velocity limiter
    adams_b           old pressure gradient correction (adamsB.f90)
    W clip            W Courant limiter (pre-pressure)
    pressure_step     project U,V,W to anelastic divergence-free
    W clip            W Courant limiter (post-pressure)
    advance_scalars   scalar (TABS,QV,…) advection + AB2  ← increments nstep, time
    sgs_scalars_proc  Smagorinsky SGS on scalars (after advance_scalars)
    micro_proc        SAM 1-moment microphysics

Each sub-function is JIT-compiled independently.  The step driver is a plain
Python function (not JIT-compiled as a unit) so that physics modules can be
switched on/off via Python-level None checks without JAX retracing.

To JIT-compile the full loop, wrap it in ``jax.lax.scan`` externally.

API
---
    from jsam.core.step import step, StepConfig, PhysicsForcing

    config  = StepConfig(sgs_params=SGSParams(), micro_params=MicroParams())
    forcing = PhysicsForcing(tabs0=t0, rad_forcing=rf, ls_forcing=lsf, sst=sst_xy)

    state, mom_tends, tends = step(
        state, mom_tends, tends, metric, grid, dt, config, forcing,
    )

StepConfig  — frozen dataclass; holds static (hashable) parameters.
               Pass as ``static_argnames`` when JIT-ting this function.
PhysicsForcing — JAX pytree; holds time-varying forcing arrays.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp


from jsam.core.state import ModelState
from jsam.core.dynamics.timestepping import (
    Tendencies, MomentumTendencies,
    advance_scalars, advance_momentum,
)
from jsam.core.dynamics.pressure import pressure_step, adams_b
from jsam.core.dynamics.damping import pole_damping, top_sponge, gsam_w_sponge
from jsam.core.dynamics.coriolis import coriolis_tend
from jsam.core.physics.sgs import (
    SGSParams, SurfaceFluxes, sgs_proc,
    sgs_mom_proc, sgs_scalars_proc,
    diffuse_momentum_horiz, diffuse_damping_mom_z,
    _sgs_coefs,
)
from jsam.core.physics.microphysics import MicroParams, micro_proc, CP
from jsam.core.physics.radiation import RadForcing, rad_proc
from jsam.core.physics.rad_rrtmg import RadRRTMGConfig, rad_rrtmg_proc, compute_qrad_rrtmg
from jsam.core.physics.lsforcing import LargeScaleForcing, ls_proc
from jsam.core.physics.surface import BulkParams, bulk_surface_fluxes
from jsam.core.physics.nudging import NudgingParams, nudge_proc
from jsam.core.physics.slm import (
    SLMParams, SLMStatic, SLMState, SLMRadInputs, slm_proc,
)
from jsam.core import debug_dump as _dd


def _stage_dump(state, stage_id: int, dt, force_nstep=None):
    """Emit an oracle-format dump if a ``DebugDumper`` is active."""
    if _dd.DUMPER is not None:
        _dd.DUMPER.dump(state, stage_id, dt, force_nstep=force_nstep)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepConfig:
    """
    Static (hashable) physics configuration.

    Set a params field to None to disable that physics module entirely.

    Attributes
    ----------
    sgs_params    : SGSParams or None  (None → skip SGS)
    micro_params  : MicroParams or None (None → skip microphysics)
    bulk_params   : BulkParams or None  (None → skip bulk surface fluxes)
    g             : gravitational acceleration (m/s²)
    epsv          : (Rv/Rd − 1) for virtual temperature
    """
    sgs_params:   Optional[SGSParams]     = field(default_factory=SGSParams)
    micro_params: Optional[MicroParams]   = field(default_factory=MicroParams)
    bulk_params:  Optional[BulkParams]    = field(default_factory=BulkParams)
    nudging_params: Optional[NudgingParams] = None  # None → skip scalar nudging
    rad_rrtmg:    Optional[RadRRTMGConfig] = None   # None → skip RRTMG_LW
    nrad:         int = 1                           # call RRTMG every nrad steps
    slm_params:   Optional[SLMParams]     = None    # scalar SLM constants;
                                                    # None → skip SLM (ocean only)
    g:             float = 9.79764  # gSAM consts.f90 ggr
    epsv:          float = 0.61
    damping_u_cu:  float = 0.3    # U/V polar velocity limiter (gSAM damping_u_cu)
    damping_w_cu:  float = 0.3    # W Courant-number damping threshold (gSAM damping_w_cu)
    sponge_z_frac: float = 0.6    # fraction of domain height where top sponge begins
    sponge_tau:    float = 10.0   # sponge damping time scale at model top (s); 0 = off
    w_sponge_nub:  float = 0.6    # gSAM nub — W sponge starts at (z-z0)/(ztop-z0) > nub
    w_sponge_max:  float = 0.333  # gSAM taudamp_max — max implicit W relax coeff per step
    polar_cool_tau: float = 0.0   # polar TABS Newtonian cooling time scale (s); 0 = off
                                  # Not in gSAM — only needed at coarse resolution (>1°)
                                  # where buoyancy feedback drives polar TABS instability.
    polar_avg_rows: int = 0       # number of polar rows to zonally average (each pole)
                                  # 0 = off; 1 = average j=0 and j=ny-1 only

    # Calendar day + year for the RRTMG_SW orbital geometry.  Passed through
    # compute_qrad_rrtmg as sw_aux["day_of_year"]/["iyear"] so the solar
    # declination and eccentricity factor match gSAM's shr_orb_decl call.
    # day0 is a fractional calendar day (1.xx .. 365.xx) at nstep=0, UT.
    # Set rad_day0=None (default) to disable SW entirely (LW-only, legacy).
    rad_day0:  Optional[float] = None
    rad_iyear: int = 2017


@jax.tree_util.register_pytree_node_class
@dataclass
class PhysicsForcing:
    """
    Time-varying forcing arrays (JAX pytree — passed through JIT).

    All fields are optional; set to None to disable the corresponding forcing.

    Attributes
    ----------
    tabs0       : (nz,) reference temperature profile for buoyancy (K).
                  Usually the initial column-mean TABS from ERA5.
                  If None, buoyancy step is skipped.
                  NOTE: step() recomputes this each step as the rolling
                  horizontal mean — do not use it as a frozen nudging target.
    qv0         : (nz,) reference moisture for buoyancy (kg/kg).
                  If None, qv0 = 0 (absolute virtual-temp buoyancy).
    tabs_ref    : (nz,) FROZEN reference TABS profile for scalar nudging.
                  Set once at init (e.g. from ERA5), never updated.  Used by
                  nudge_proc to restrain stratospheric drift.  None → nudging
                  is a no-op for TABS.
    qv_ref      : (nz,) FROZEN reference QV profile for scalar nudging.
    rad_forcing : RadForcing or None
    ls_forcing  : LargeScaleForcing or None
    sst         : (ny, nx) SST (K) or None — enables bulk surface fluxes
    """
    tabs0:      jax.Array | None             = None
    qv0:        jax.Array | None             = None
    # qn0 = <qcl+qci>, qp0 = <qpl+qpi>  — gSAM diagnose.f90:45-46 rolling
    # horizontal means.  Used only by _buoyancy_W for the
    # (1+epsv*qv0 - qn0 - qp0) thermal factor.  Zero at ERA5 init.
    qn0:        jax.Array | None             = None
    qp0:        jax.Array | None             = None
    tabs_ref:   jax.Array | None             = None
    qv_ref:     jax.Array | None             = None
    rad_forcing: RadForcing | None           = None
    ls_forcing:  LargeScaleForcing | None    = None
    sst:         jax.Array | None            = None
    o3vmr_rrtmg: jax.Array | None            = None   # (nz,) or (ncol,nz) vmr
    qrad_rrtmg:  jax.Array | None            = None   # (nz,ny,nx) K/s — stored RRTMG
                                                       # heating rates; recomputed every
                                                       # nrad steps, applied every step
                                                       # (matches gSAM qrad + dtn pattern)
    # --- SLM plumbing -------------------------------------------------------
    # slm_static : frozen SLMStatic pytree (same fields every step).
    # slm_state  : prognostic SLMState pytree, advanced by step() each call.
    # slm_rad    : SLMRadInputs (6 SW/LW/coszrs arrays) — supplied per step.
    # precip_ref : (ny,nx) precip flux at reference level from last micro step.
    # Any of these being None disables the SLM path.
    slm_static:  SLMStatic | None            = None
    slm_state:   SLMState  | None            = None
    slm_rad:     SLMRadInputs | None         = None
    precip_ref:  jax.Array | None            = None

    @classmethod
    def zeros(cls, nz: int, ny: int, nx: int) -> "PhysicsForcing":
        """Construct a PhysicsForcing with zero-effect forcing and buoyancy ref."""
        return cls(
            tabs0       = jnp.full(nz, 300.0),
            qv0         = jnp.zeros(nz),
            qn0         = jnp.zeros(nz),
            qp0         = jnp.zeros(nz),
            tabs_ref    = None,
            qv_ref      = None,
            rad_forcing  = RadForcing.zeros(nz_prof=2),
            ls_forcing   = LargeScaleForcing.zeros(nz_prof=2),
            sst          = None,
        )

    def tree_flatten(self):
        children = (self.tabs0, self.qv0, self.qn0, self.qp0,
                    self.tabs_ref, self.qv_ref,
                    self.rad_forcing, self.ls_forcing, self.sst, self.o3vmr_rrtmg,
                    self.qrad_rrtmg,
                    self.slm_static, self.slm_state, self.slm_rad, self.precip_ref)
        return children, None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (tabs0, qv0, qn0, qp0, tabs_ref, qv_ref,
         rad_forcing, ls_forcing, sst, o3vmr_rrtmg,
         qrad_rrtmg,
         slm_static, slm_state, slm_rad, precip_ref) = children
        return cls(
            tabs0=tabs0, qv0=qv0, qn0=qn0, qp0=qp0,
            tabs_ref=tabs_ref, qv_ref=qv_ref,
            rad_forcing=rad_forcing, ls_forcing=ls_forcing, sst=sst,
            o3vmr_rrtmg=o3vmr_rrtmg, qrad_rrtmg=qrad_rrtmg,
            slm_static=slm_static, slm_state=slm_state,
            slm_rad=slm_rad, precip_ref=precip_ref,
        )


# ---------------------------------------------------------------------------
# Buoyancy tendency
# ---------------------------------------------------------------------------

def _buoyancy_W(
    state:  ModelState,
    tabs0:  jax.Array,      # (nz,) reference temperature (K)
    qv0:    jax.Array,      # (nz,) reference moisture (kg/kg)
    dz:     jax.Array,      # (nz,) cell widths (m)
    g:      float,
    epsv:   float,
    qn0:    jax.Array | None = None,    # (nz,) reference cloud water (kg/kg)
    qp0:    jax.Array | None = None,    # (nz,) reference precip (kg/kg)
) -> jax.Array:
    """
    Compute buoyancy tendency on W-faces, shape (nz+1, ny, nx)  [m/s²].

    Exactly follows gSAM ``buoyancy.f90:48-53``:

        buo_cell[k] = (g/T0[k]) * {
              T0[k]*(epsv*(qv[k]-qv0[k]) - (qn[k]-qn0[k]) - (qp[k]-qp0[k]))
            + (T[k]-T0[k])*(1 + epsv*qv0[k] - qn0[k] - qp0[k])
        }

    where qn = qcl+qci and qp = qpl+qpi.  With qn0=qp0=0 this reduces to a
    pure virtual-temperature anomaly buoyancy.  The ``(1+epsv*qv0-qn0-qp0)``
    factor on the thermal term is ~1.01 at the surface → 1.00 aloft and was
    missing prior to 2026-04-15.

    Interpolated to W-faces with area-weighted averaging (non-uniform dz):
        b_w[k] = betu[k]*b_cell[k] + betd[k]*b_cell[k-1]
    where betu = dz[k-1]/(dz[k]+dz[k-1]), betd = dz[k]/(dz[k]+dz[k-1])
    (matches gSAM betu/betd in buoyancy.f90:41-42).

    Rigid-lid BCs: b_w[0] = b_w[nz] = 0.
    """
    # tabs0 may be (nz,) for global mean, (nz, ny) for latitude-varying,
    # or (nz, ny, nx) for fully 3D reference (buoyancy=0 at t=0 if tabs0=TABS_init)
    if tabs0.ndim == 1:
        tabs0_3d = tabs0[:, None, None]   # (nz,1,1)
    elif tabs0.ndim == 2:
        tabs0_3d = tabs0[:, :, None]      # (nz,ny,1)
    else:
        tabs0_3d = tabs0                  # (nz,ny,nx)
    if qv0.ndim == 1:
        qv0_3d = qv0[:, None, None]
    elif qv0.ndim == 2:
        qv0_3d = qv0[:, :, None]
    else:
        qv0_3d = qv0                      # (nz,ny,nx)

    # qn0 = <QC+QI>, qp0 = <QR+QS+QG> — cos-lat horizontal means supplied
    # by the diagnose block (step 14).  Default to zero if caller hasn't
    # plumbed them (matches fresh ERA5 init; cloud/precip ≈ 0).
    if qn0 is None:
        qn0_3d = jnp.zeros_like(tabs0_3d)
    else:
        qn0_3d = qn0[:, None, None] if qn0.ndim == 1 else (
            qn0[:, :, None] if qn0.ndim == 2 else qn0
        )
    if qp0 is None:
        qp0_3d = jnp.zeros_like(tabs0_3d)
    else:
        qp0_3d = qp0[:, None, None] if qp0.ndim == 1 else (
            qp0[:, :, None] if qp0.ndim == 2 else qp0
        )

    qn = state.QC + state.QI                           # gSAM qcl+qci
    qp = state.QR + state.QS + state.QG                # gSAM qpl+qpi (SAM1MOM lumps)

    thermal_factor = 1.0 + epsv * qv0_3d - qn0_3d - qp0_3d

    b = (g / tabs0_3d) * (
        tabs0_3d * (
            epsv * (state.QV - qv0_3d)
            - (qn - qn0_3d)
            - (qp - qp0_3d)
        )
        + (state.TABS - tabs0_3d) * thermal_factor
    )   # (nz, ny, nx)

    # Area-weighted interpolation to interior W-faces (k = 1..nz-1)
    dz_lo = dz[:-1][:, None, None]   # (nz-1, 1, 1)
    dz_hi = dz[1:][:, None, None]    # (nz-1, 1, 1)
    betu  = dz_lo / (dz_hi + dz_lo)  # weight for upper cell contribution
    betd  = dz_hi / (dz_hi + dz_lo)  # weight for lower cell contribution

    b_int = betu * b[1:] + betd * b[:-1]   # (nz-1, ny, nx) at w-faces 1..nz-1

    ny, nx = state.TABS.shape[1], state.TABS.shape[2]
    b_w = jnp.concatenate([
        jnp.zeros((1, ny, nx)),
        b_int,
        jnp.zeros((1, ny, nx)),
    ], axis=0)   # (nz+1, ny, nx)

    return b_w


# ---------------------------------------------------------------------------
# Step driver
# ---------------------------------------------------------------------------

def step(
    state:          ModelState,
    mom_tends_nm1:  MomentumTendencies,
    mom_tends_nm2:  MomentumTendencies,
    tends_nm1:      Tendencies,
    tends_nm2:      Tendencies,
    metric:         dict,
    grid,                           # LatLonGrid (passed to pressure_step)
    dt:             float,
    config:         StepConfig,
    forcing:        PhysicsForcing,
    dt_prev:        float | None = None,
    dt_pprev:       float | None = None,
) -> tuple[ModelState, MomentumTendencies, Tendencies, PhysicsForcing]:
    """
    Advance the model by one timestep ``dt``.

    Operator-splitting order (follows gSAM ``main.f90``):

    1.  ls_proc           — large-scale forcing (forcing + nudging)
    2.  rad_proc          — prescribed radiative heating
    3.  bulk_surface_fluxes → SurfaceFluxes
    4.  sgs_mom_proc      — SGS diffusion on U,V,W only  [≡ gSAM sgs.f90]
    5.  buoyancy (Euler)  — W += dt*b_W  [≡ gSAM buoyancy.f90; Euler, not AB2'd]
    6.  compute dU/dV_cor — Coriolis tendency (AB2'd with advection)
    7.  advance_momentum  — advection+coriolis AB2  [≡ gSAM adamsA()]
    8.  pole_damping      — polar velocity limiter  [≡ gSAM damping()]
    8b. adams_b           — old pressure gradient (disabled pending validation)
    9.  W clip            — W Courant limiter (pre-pressure)
    10. pressure_step     — project to anelastic divergence-free  [≡ gSAM pressure()]
    11. W clip            — W Courant limiter (post-pressure, prevents scalar blowup)
    12. advance_scalars   — scalar advection AB2   [increments nstep, time]
    12b.sgs_scalars_proc  — SGS diffusion on scalars  [≡ gSAM sgs_scalars after advection]
    13. micro_proc        — SAM 1-moment microphysics

    Parameters
    ----------
    state          : current ModelState
    mom_tends_nm1  : MomentumTendencies from step n-1 (AB3 buffer)
    mom_tends_nm2  : MomentumTendencies from step n-2 (AB3 buffer)
    tends_nm1      : scalar Tendencies from step n-1
    tends_nm2      : scalar Tendencies from step n-2
    metric         : precomputed grid metric dict (from build_metric)
    grid           : LatLonGrid (passed through to pressure_step)
    dt             : timestep (s)
    config         : StepConfig — static physics parameters
    forcing        : PhysicsForcing — time-varying forcing arrays

    Returns
    -------
    (new_state, new_mom_tends, new_tends)
    """

    # ------------------------------------------------------------------
    # Load p_prev, p_pprev before any operation (they will be lost in
    # intermediate ModelState constructions; we restore them at the very end).
    # ------------------------------------------------------------------
    p_prev_for_adamsb  = state.p_prev    # None on first step → adamsB skipped
    p_pprev_for_adamsb = state.p_pprev   # None on first two steps

    # ------------------------------------------------------------------
    # AB coefficients for adamsB: use the same (at, bt, ct) that
    # advance_momentum/scalars will use for the main AB step, so the old
    # pressure-gradient correction is consistent with the AB3 weighting.
    # ------------------------------------------------------------------
    from jsam.core.dynamics.timestepping import ab_coefs
    _nstep_py = int(state.nstep)
    _at_ab, _bt_ab, _ct_ab = ab_coefs(_nstep_py, dt, dt_prev, dt_pprev)

    # D11 fix: gSAM increments nstep at TOP of time loop, BEFORE any
    # physics.  Increment here so any nstep-conditional logic within the
    # step sees the correct value.
    state = ModelState(
        U=state.U, V=state.V, W=state.W,
        TABS=state.TABS, QV=state.QV, QC=state.QC,
        QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
        TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep + 1, time=state.time,
    )

    # ── Oracle-compatible stage dumps (19 gSAM stages per step) ──────
    _dump_nstep = int(state.nstep)
    _stage_dump(state, 0, dt, force_nstep=_dump_nstep)  # pre_step

    # ------------------------------------------------------------------
    # 1. Large-scale forcing (horizontal advection + subsidence)
    # ------------------------------------------------------------------
    if forcing.ls_forcing is not None:
        state = ls_proc(state, metric, forcing.ls_forcing, dt)

    _stage_dump(state, 1, dt, force_nstep=_dump_nstep)  # forcing
    _stage_dump(state, 2, dt, force_nstep=_dump_nstep)  # nudging

    # D9 fix: gSAM computes buoyancy BEFORE radiation, so buoyancy sees
    # pre-radiation TABS.  Save the state for buoyancy computation.
    _state_for_buoyancy = state

    _stage_dump(state, 3, dt, force_nstep=_dump_nstep)  # buoyancy

    # ------------------------------------------------------------------
    # 2. Prescribed radiation
    # ------------------------------------------------------------------
    if forcing.rad_forcing is not None:
        state = rad_proc(state, metric, forcing.rad_forcing, dt)

    # ------------------------------------------------------------------
    # 2a. RRTMG_LW — matches gSAM rad_full.f90 behaviour exactly:
    #     • qrad (K/s) is RECOMPUTED every nrad steps (when nradsteps >= nrad).
    #     • qrad is APPLIED every step as TABS += qrad * dtn.
    #     Both happen at stage 4 (radiation), consistent with gSAM main.f90.
    # ------------------------------------------------------------------
    if config.rad_rrtmg is not None and forcing.sst is not None:
        if int(state.nstep) % config.nrad == 0:
            # Build the SW orbital geometry payload (enabled whenever
            # rad_day0 is set — matches gSAM doshortwave=.true.).
            _sw_aux = None
            if config.rad_day0 is not None:
                _day_for_sw = float(config.rad_day0) + float(state.time) / 86400.0
                _sw_aux = {
                    "day_of_year": _day_for_sw,
                    "iyear":       int(config.rad_iyear),
                    "lat_rad":     metric["lat_rad"],
                    "lon_rad":     metric["lon_rad"],
                }
            # Recompute heating rates (gSAM: nradsteps >= nrad → new qrad)
            _new_qrad = compute_qrad_rrtmg(
                state, metric, config.rad_rrtmg, forcing.sst,
                o3vmr=(None if forcing.o3vmr_rrtmg is None
                       else jnp.asarray(forcing.o3vmr_rrtmg)),
                sw_aux=_sw_aux,
            )
            forcing = PhysicsForcing(
                tabs0=forcing.tabs0, qv0=forcing.qv0,
                qn0=forcing.qn0, qp0=forcing.qp0,
                tabs_ref=forcing.tabs_ref, qv_ref=forcing.qv_ref,
                rad_forcing=forcing.rad_forcing, ls_forcing=forcing.ls_forcing,
                sst=forcing.sst, o3vmr_rrtmg=forcing.o3vmr_rrtmg,
                qrad_rrtmg=_new_qrad,
                slm_static=forcing.slm_static, slm_state=forcing.slm_state,
                slm_rad=forcing.slm_rad, precip_ref=forcing.precip_ref,
            )
        # Apply stored qrad every step (gSAM: t += qrad * dtn, every step)
        if forcing.qrad_rrtmg is not None:
            state = ModelState(
                U=state.U, V=state.V, W=state.W,
                TABS=state.TABS + dt * forcing.qrad_rrtmg,
                QV=state.QV, QC=state.QC, QI=state.QI,
                QR=state.QR, QS=state.QS, QG=state.QG,
                TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
                nstep=state.nstep, time=state.time,
            )

    _stage_dump(state, 4, dt, force_nstep=_dump_nstep)  # radiation

    # ------------------------------------------------------------------
    # 2b. Scalar nudging toward frozen 1D reference profile
    #     [≡ gSAM nudging.f90 lines 419-473, 1D profile branch]
    #     Relaxes TABS, QV toward tabs_ref, qv_ref in the vertical band
    #     [z1, z2] on timescale tau.  For IRMA this replaces the missing
    #     RRTMG stratospheric LW balance by holding the upper atmosphere
    #     at its ERA5 init profile.
    # ------------------------------------------------------------------
    if config.nudging_params is not None and (
        forcing.tabs_ref is not None or forcing.qv_ref is not None
    ):
        state = nudge_proc(
            state, metric,
            tabs_ref=forcing.tabs_ref,
            qv_ref=forcing.qv_ref,
            dt=dt,
            params=config.nudging_params,
        )

    # ------------------------------------------------------------------
    # 3. Surface fluxes  (→ SurfaceFluxes for SGS BCs)
    #
    # Two paths:
    #   a) Ocean-only (today):  bulk_surface_fluxes drives SurfaceFluxes.
    #   b) SLM + ocean blend:   slm_proc runs the Simple Land Model at
    #      every land/seaice cell, bulk_surface_fluxes runs at every
    #      ocean cell, then SurfaceFluxes is blended per cell via
    #      ``jnp.where(landmask | seaicemask, land, ocean)``. The SLM
    #      skin temperature is written into forcing.sst for the next
    #      RRTMG call at land cells (gSAM order: radiation uses the
    #      skin temperature updated by the *previous* surface() call).
    # ------------------------------------------------------------------
    surface_fluxes: SurfaceFluxes | None = None
    new_slm_state: SLMState | None = forcing.slm_state

    ocean_fluxes: SurfaceFluxes | None = None
    if forcing.sst is not None and config.bulk_params is not None:
        ocean_fluxes = bulk_surface_fluxes(
            state, metric, forcing.sst, config.bulk_params,
        )

    if (
        config.slm_params is not None
        and forcing.slm_static is not None
        and forcing.slm_state is not None
        and forcing.slm_rad is not None
    ):
        precip_ref = (
            forcing.precip_ref
            if forcing.precip_ref is not None
            else jnp.zeros_like(state.TABS[0])
        )
        new_slm_state, land_fluxes = slm_proc(
            state, metric,
            forcing.slm_state,
            forcing.slm_static,
            config.slm_params,
            forcing.slm_rad,
            precip_ref,
            dt,
        )
        landmask = forcing.slm_static.landmask != 0
        seaicemask = forcing.slm_static.seaicemask != 0
        land_mask = landmask | seaicemask
        if ocean_fluxes is not None:
            surface_fluxes = SurfaceFluxes(
                shf   = jnp.where(land_mask, land_fluxes.shf,   ocean_fluxes.shf),
                lhf   = jnp.where(land_mask, land_fluxes.lhf,   ocean_fluxes.lhf),
                tau_x = jnp.where(land_mask, land_fluxes.tau_x, ocean_fluxes.tau_x),
                tau_y = jnp.where(land_mask, land_fluxes.tau_y, ocean_fluxes.tau_y),
            )
            # Feed the SLM skin temperature back into forcing.sst so the
            # NEXT RRTMG call sees the updated land surface.
            new_sst = jnp.where(land_mask, new_slm_state.t_skin, forcing.sst)
            forcing = PhysicsForcing(
                tabs0=forcing.tabs0, qv0=forcing.qv0,
                qn0=forcing.qn0, qp0=forcing.qp0,
                tabs_ref=forcing.tabs_ref, qv_ref=forcing.qv_ref,
                rad_forcing=forcing.rad_forcing, ls_forcing=forcing.ls_forcing,
                sst=new_sst, o3vmr_rrtmg=forcing.o3vmr_rrtmg,
                qrad_rrtmg=forcing.qrad_rrtmg,
                slm_static=forcing.slm_static, slm_state=new_slm_state,
                slm_rad=forcing.slm_rad, precip_ref=forcing.precip_ref,
            )
        else:
            surface_fluxes = land_fluxes
            forcing = PhysicsForcing(
                tabs0=forcing.tabs0, qv0=forcing.qv0,
                qn0=forcing.qn0, qp0=forcing.qp0,
                tabs_ref=forcing.tabs_ref, qv_ref=forcing.qv_ref,
                rad_forcing=forcing.rad_forcing, ls_forcing=forcing.ls_forcing,
                sst=forcing.sst, o3vmr_rrtmg=forcing.o3vmr_rrtmg,
                qrad_rrtmg=forcing.qrad_rrtmg,
                slm_static=forcing.slm_static, slm_state=new_slm_state,
                slm_rad=forcing.slm_rad, precip_ref=forcing.precip_ref,
            )
    else:
        surface_fluxes = ocean_fluxes

    # ------------------------------------------------------------------
    # 4-7. Assemble all momentum tendencies into a single AB buffer and
    #      apply one Adams-Bashforth advance.
    #
    #      gSAM main.f90 calls buoyancy → advect_mom → coriolis → sgs_mom,
    #      each accumulating into the SAME dudt/dvdt/dwdt(na) buffer.  That
    #      buffer is then AB-stepped in adamsA().  jsam previously applied
    #      buoyancy and SGS as in-place state mutations before advection,
    #      which mixes time levels and destabilises the momentum system.
    #      Here we follow the gSAM order literally: every tendency is
    #      computed from the SAME pre-step state and summed into F_n,
    #      which advance_momentum then AB-steps together with the adv
    #      tendency.
    # ------------------------------------------------------------------
    jax.debug.print(
        "  DIAG [{n:>3}] A_start   W=[{wn:.3f},{wx:.3f}] U=[{un:.2f},{ux:.2f}] T_min={tn:.2f}",
        n=state.nstep, wn=jnp.min(state.W), wx=jnp.max(state.W),
        un=jnp.min(state.U), ux=jnp.max(state.U), tn=jnp.min(state.TABS),
    )

    # gSAM stages 5 (surface), 6 (advect_mom), 7 (coriolis), 8 (sgs_proc),
    # 9 (sgs_mom) are all tendency-only in gSAM and state is unchanged.
    # jsam bundles them into the advance_momentum call below.  Dump the
    # unchanged state once per stage id to match the oracle's row count.
    for _sid in (5, 6, 7, 8, 9):
        _stage_dump(state, _sid, dt, force_nstep=_dump_nstep)

    # C1 fix: save pre-dynamics velocity for half-step averaging in scalar advection
    _U_old, _V_old, _W_old = state.U, state.V, state.W

    _tk = None   # will be set if SGS is active; reused by implicit solver in step 8
    if config.sgs_params is not None:
        _tk, _tkh = _sgs_coefs(
            state, metric, config.sgs_params, dt, tabs0=forcing.tabs0,
        )

    nz_s, ny_s, nx_s = state.TABS.shape
    dU_extra = jnp.zeros((nz_s, ny_s, nx_s + 1))
    dV_extra = jnp.zeros((nz_s, ny_s + 1, nx_s))
    dW_extra = jnp.zeros((nz_s + 1, ny_s, nx_s))

    # 4. Buoyancy tendency on W  [≡ gSAM buoyancy.f90 — adds to dwdt(na)]
    #    D9 fix: use pre-radiation state so buoyancy sees pre-radiation TABS
    if forcing.tabs0 is not None:
        qv0 = forcing.qv0 if forcing.qv0 is not None else jnp.zeros_like(forcing.tabs0)
        _dW_buo = _buoyancy_W(_state_for_buoyancy, forcing.tabs0, qv0,
                               metric["dz"], config.g, config.epsv,
                               qn0=forcing.qn0, qp0=forcing.qp0)
        dW_extra = dW_extra + _dW_buo

        # C13 fix: buoyancy energy correction (gSAM buoyancy.f90:58-62)
        # t(kb) -= 0.5*dtn/cp * buo * w;  t(k) -= 0.5*dtn/cp * buo * w
        # This conserves total (kinetic + thermal) energy when buoyancy
        # does work on the vertical velocity.  The correction is applied to
        # both mass cells adjacent to each interior w-face.
        _coef_buo = 0.5 * dt / CP
        # _dW_buo is at w-faces (nz+1, ny, nx); interior faces are 1..nz-1
        _buo_w_int = _dW_buo[1:-1, :, :]  # (nz-1, ny, nx)
        _w_int     = state.W[1:-1, :, :]   # (nz-1, ny, nx)
        _factor    = _coef_buo * _buo_w_int * _w_int   # (nz-1, ny, nx)
        # Apply to lower cell (kb = k-1, indices 0..nz-2) and upper cell (k, indices 1..nz-1)
        _TABS_corr = state.TABS
        _TABS_corr = _TABS_corr.at[:-1, :, :].add(-_factor)  # lower cells
        _TABS_corr = _TABS_corr.at[1:, :, :].add(-_factor)   # upper cells
        state = ModelState(
            U=state.U, V=state.V, W=state.W,
            TABS=_TABS_corr, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

        jax.debug.print(
            "  DIAG [{n:>3}] B_buoy    dW_buo_abs_max={m:.4e}  (dt*max={x:.4f})",
            n=state.nstep, m=jnp.max(jnp.abs(_dW_buo)),
            x=dt * jnp.max(jnp.abs(_dW_buo)),
        )

    # 5. Coriolis + metric tendencies  [≡ gSAM coriolis.f90 — adds to dudt/dvdt/dwdt(na)]
    #    coriolis_tend returns (dU, dV, dW) including the docoriolisz f'
    #    contribution coupling W↔U (gSAM coriolis.f90 lines 56-74).
    #    dU is (nz,ny,nx) at west faces of mass cells 0..nx-1.  Periodic
    #    x means face nx ≡ face 0, so both slots of the (nx+1)-staggered
    #    U buffer get the same tendency.
    _dU_cor_mass, _dV_cor, _dW_cor = coriolis_tend(
        state.U, state.V, state.W, metric,
    )
    dU_extra = dU_extra.at[:, :, :nx_s].add(_dU_cor_mass)
    dU_extra = dU_extra.at[:, :, nx_s].add(_dU_cor_mass[:, :, 0])
    dV_extra = dV_extra + _dV_cor
    dW_extra = dW_extra + _dW_cor

    # 6. SGS full-3D momentum diffusion  [≡ gSAM sgs_mom / diffuse_mom3D.f90
    #    — stored in dudtd/dvdtd/dwdtd, which adamsA adds to the AB3 update
    #    as a NON-AB fresh current-step tendency.  We apply it after
    #    advance_momentum as a direct Euler increment so the SGS flux
    #    divergence is NEVER put into the AB3 history buffer (otherwise the
    #    large bt=-16/12 weight on the lagged SGS term drives a fast
    #    exponential growth — see commit history).
    _dU_sgs = _dV_sgs = _dW_sgs = None
    if _tk is not None:
        from jsam.core.physics.sgs import diffuse_momentum
        _tx = None if surface_fluxes is None else surface_fluxes.tau_x
        _ty = None if surface_fluxes is None else surface_fluxes.tau_y
        _dU_sgs, _dV_sgs, _dW_sgs = diffuse_momentum(
            state.U, state.V, state.W, _tk, metric, tau_x=_tx, tau_y=_ty,
        )

    # 7. Momentum advance — Adams-Bashforth 3  [≡ gSAM adamsA(), nadams=3]
    #    advance_momentum computes the advective tendency from the same
    #    unmodified state and AB-sums it with dU/dV/dW_extra into one update.
    state, mom_tends_n = advance_momentum(
        state, mom_tends_nm1, mom_tends_nm2, metric, dt,
        dU_extra=dU_extra,
        dV_extra=dV_extra,
        dW_extra=dW_extra,
        dt_prev=dt_prev,
        dt_pprev=dt_pprev,
    )
    jax.debug.print(
        "  DIAG [{n:>3}] C_advmom  W=[{wn:.3f},{wx:.3f}] U=[{un:.2f},{ux:.2f}]",
        n=state.nstep, wn=jnp.min(state.W), wx=jnp.max(state.W),
        un=jnp.min(state.U), ux=jnp.max(state.U),
    )

    # Apply SGS momentum diffusion as a fresh current-step (non-AB) tendency
    # — equivalent to gSAM adamsA's `+ dudtd(i,j,k)` term.
    if _dU_sgs is not None:
        _nxp1 = state.U.shape[-1]
        _U_sgs_full = jnp.concatenate([_dU_sgs, _dU_sgs[:, :, :1]], axis=-1) \
                      if _dU_sgs.shape[-1] == _nxp1 - 1 else _dU_sgs
        _U_with_sgs = state.U + dt * _U_sgs_full
        _U_with_sgs = _U_with_sgs.at[:, :, -1].set(_U_with_sgs[:, :, 0])
        state = ModelState(
            U   =_U_with_sgs,
            V   =state.V + dt * _dV_sgs,
            W   =state.W + dt * _dW_sgs,
            TABS=state.TABS, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE =state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )
        jax.debug.print(
            "  DIAG [{n:>3}] D_sgsmom  W=[{wn:.3f},{wx:.3f}] dW_sgs_max={s:.4e}",
            n=state.nstep, wn=jnp.min(state.W), wx=jnp.max(state.W),
            s=jnp.max(jnp.abs(_dW_sgs)),
        )

    _stage_dump(state, 10, dt, force_nstep=_dump_nstep)  # adamsA

    # ------------------------------------------------------------------
    # 8. Implicit vertical SGS diffusion + damping
    #    [≡ gSAM diffuse_damping_mom_z.f90, doimplicitdiff=.true.]
    #    Replaces explicit damping() with a single implicit tridiagonal
    #    solve that combines:
    #      - SGS vertical diffusion (using tk from Smagorinsky)
    #      - Top sponge on W (dodamping, nub=0.6)
    #      - W CFL limiter (dodamping_w, below sponge only)
    #      - Polar U,V velocity limiter (dodamping_poles)
    #      - Upper-level U,V damping (pres < 70 hPa)
    #      - Surface momentum flux BCs
    # ------------------------------------------------------------------
    if _tk is not None:
        _fluxbu = None
        _fluxbv = None
        if surface_fluxes is not None:
            # Interpolate surface tau_x/tau_y to U/V face positions
            if surface_fluxes.tau_x is not None:
                _tx = surface_fluxes.tau_x   # (ny, nx) at cell centres
                _tx_ux = 0.5 * (jnp.roll(_tx, 1, axis=-1) + _tx)
                _fluxbu = jnp.concatenate([_tx_ux, _tx_ux[:, :1]], axis=-1)
            if surface_fluxes.tau_y is not None:
                _ty = surface_fluxes.tau_y
                _ty_yp = jnp.pad(_ty, ((1, 0), (0, 0)), mode='edge')
                _ty_v  = 0.5 * (_ty_yp[:-1, :] + _ty_yp[1:, :])
                _fluxbv = jnp.pad(_ty_v, ((0, 1), (0, 0)), mode='edge')

        _U_imp, _V_imp, _W_imp = diffuse_damping_mom_z(
            state.U, state.V, state.W, _tk, metric, dt,
            damping_u_cu=config.damping_u_cu,
            damping_w_cu=config.damping_w_cu,
            fluxbu=_fluxbu, fluxbv=_fluxbv,
        )
        # gSAM damping.f90 section 1 — W Rayleigh sponge at model top.
        # Suppresses gravity waves before they reflect off the rigid lid
        # and excite a runaway stratospheric jet via Reynolds-stress
        # convergence. W-only, implicit, nub=0.6, taudamp_max=0.333.
        if config.w_sponge_max > 0.0:
            _W_imp = gsam_w_sponge(
                _W_imp, metric["z"],
                nub=config.w_sponge_nub,
                taudamp_max=config.w_sponge_max,
                dtn=_at_ab * dt,   # D4: gSAM tau_max = dtn/dt
                dt=dt,
            )
        state = ModelState(
            U=_U_imp, V=_V_imp, W=_W_imp,
            TABS=state.TABS, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )
        jax.debug.print(
            "  DIAG [{n:>3}] E_dampz   W=[{wn:.3f},{wx:.3f}] U=[{un:.2f},{ux:.2f}]",
            n=state.nstep, wn=jnp.min(state.W), wx=jnp.max(state.W),
            un=jnp.min(state.U), ux=jnp.max(state.U),
        )

    _stage_dump(state, 11, dt, force_nstep=_dump_nstep)  # damping

    # gSAM has no spectral polar filter on velocity — only the implicit
    # CFL-based pole damping in damping.f90, already folded into
    # diffuse_damping_mom_z above.  The legacy jsam spectral filter has
    # been removed to match the Fortran pipeline exactly.

    # ------------------------------------------------------------------
    # 8e. adamsB: apply old pressure gradient  [≡ gSAM adamsB.f90]
    #    Corrects for the lagged pressure gradient in the AB2 scheme.
    #    bt = -1/(2*alpha) where alpha = dt_prev/dt_curr = 1 for const dt
    #    → bt = -0.5
    #    Skipped on the first step (p_prev is None).
    # ------------------------------------------------------------------
    if p_prev_for_adamsb is not None:
        state = adams_b(
            state, p_prev_for_adamsb, metric, dt,
            bt=_bt_ab,
            p_pprev=p_pprev_for_adamsb,
            ct=_ct_ab,
        )
        jax.debug.print(
            "  DIAG [{n:>3}] F_adamsB  W=[{wn:.3f},{wx:.3f}] U=[{un:.2f},{ux:.2f}]",
            n=state.nstep, wn=jnp.min(state.W), wx=jnp.max(state.W),
            un=jnp.min(state.U), ux=jnp.max(state.U),
        )

    _stage_dump(state, 12, dt, force_nstep=_dump_nstep)  # adamsB

    # ------------------------------------------------------------------
    # 9. Pressure correction → divergence-free U,V,W  [≡ gSAM pressure()]
    #    Also returns p_new to store for next step's adamsB.
    # ------------------------------------------------------------------
    state, p_new = pressure_step(state, grid, metric, dt, at=_at_ab)
    jax.debug.print(
        "  DIAG [{n:>3}] G_press   W=[{wn:.3f},{wx:.3f}] U=[{un:.2f},{ux:.2f}] p_max={pm:.2f}",
        n=state.nstep, wn=jnp.min(state.W), wx=jnp.max(state.W),
        un=jnp.min(state.U), ux=jnp.max(state.U),
        pm=jnp.max(jnp.abs(p_new)),
    )

    _stage_dump(state, 13, dt, force_nstep=_dump_nstep)  # pressure

    # ------------------------------------------------------------------
    # 11. Scalar advection — Adams-Bashforth 3  [increments nstep + time]
    # ------------------------------------------------------------------
    state, tends_n = advance_scalars(
        state, tends_nm1, tends_nm2, metric, dt,
        dt_prev=dt_prev, dt_pprev=dt_pprev,
    )
    jax.debug.print(
        "  DIAG [{n:>3}] H_advsc   W=[{wn:.3f},{wx:.3f}] T=[{tn:.2f},{tx:.2f}]",
        n=state.nstep, wn=jnp.min(state.W), wx=jnp.max(state.W),
        tn=jnp.min(state.TABS), tx=jnp.max(state.TABS),
    )

    _stage_dump(state, 14, dt, force_nstep=_dump_nstep)  # advect_scalars

    # ------------------------------------------------------------------
    # 11a. Latitude-dependent Newtonian cooling
    #     Not in gSAM (only needed at coarse resolution where the buoyancy
    #     feedback loop drives polar TABS instability).  Relaxes TABS toward
    #     the reference tabs0 at high latitudes with tau = sin²(lat)^4.
    #     Much gentler than spectral filtering, which creates artifacts.
    # ------------------------------------------------------------------
    if config.polar_cool_tau > 0.0 and forcing.tabs0 is not None:
        _lat     = metric["lat_rad"]                       # (ny,)
        _sin2    = jnp.sin(_lat) ** 2
        _tau_lat = _sin2 ** 4                              # ~0.5 at 60°, ~0.94 at 80°
        _factor  = 1.0 / (1.0 + _tau_lat * dt / config.polar_cool_tau)  # (ny,)

        _tabs0 = forcing.tabs0
        if _tabs0.ndim == 1:
            _tabs0 = _tabs0[:, None, None]
        elif _tabs0.ndim == 2:
            _tabs0 = _tabs0[:, :, None]

        _TABS_pc = _tabs0 + (state.TABS - _tabs0) * _factor[None, :, None]
        state = ModelState(
            U=state.U, V=state.V, W=state.W,
            TABS=_TABS_pc, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

    # ------------------------------------------------------------------
    # 11b. SGS scalar diffusion  [≡ gSAM sgs_scalars after advection]
    #     Gap 4 fix: scalars get SGS AFTER advection, not before.
    #     Full 3D explicit SGS + implicit vertical diffusion.
    #     gSAM: horiz explicit (diffuse_scalar3D) + vert implicit
    #     (diffuse_scalar_z). We do full explicit first, then implicit
    #     vertical on top — slightly more vert diffusion than gSAM but
    #     ensures unconditional vertical stability.
    # ------------------------------------------------------------------
    if config.sgs_params is not None:
        # Full 3D explicit SGS (horizontal + vertical explicit)
        state = sgs_scalars_proc(state, metric, config.sgs_params, dt,
                                 surface=surface_fluxes, tabs0=forcing.tabs0)

        # Implicit vertical diffusion on top (stabilises near-polar model top)
        from jsam.core.physics.sgs import diffuse_scalar_z_implicit
        _, _tkh_impl = _sgs_coefs(
            state, metric, config.sgs_params, dt, tabs0=forcing.tabs0,
        )
        _shf = None if surface_fluxes is None else surface_fluxes.shf
        _lhf = None if surface_fluxes is None else surface_fluxes.lhf

        state = ModelState(
            U=state.U, V=state.V, W=state.W,
            TABS=diffuse_scalar_z_implicit(state.TABS, _tkh_impl, metric, dt, fluxb=_shf),
            QV=diffuse_scalar_z_implicit(state.QV, _tkh_impl, metric, dt, fluxb=_lhf),
            QC=diffuse_scalar_z_implicit(state.QC, _tkh_impl, metric, dt),
            QI=diffuse_scalar_z_implicit(state.QI, _tkh_impl, metric, dt),
            QR=diffuse_scalar_z_implicit(state.QR, _tkh_impl, metric, dt),
            QS=diffuse_scalar_z_implicit(state.QS, _tkh_impl, metric, dt),
            QG=diffuse_scalar_z_implicit(state.QG, _tkh_impl, metric, dt),
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

    # Clip non-negative scalars to ≥ 0 (flux correction can drive near-zero
    # fields slightly negative near polar walls)
    state = ModelState(
        U=state.U, V=state.V, W=state.W,
        TABS=state.TABS,
        QV =jnp.maximum(state.QV,  0.0),
        QC =jnp.maximum(state.QC,  0.0),
        QI =jnp.maximum(state.QI,  0.0),
        QR =jnp.maximum(state.QR,  0.0),
        QS =jnp.maximum(state.QS,  0.0),
        QG =jnp.maximum(state.QG,  0.0),
        TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )

    _stage_dump(state, 15, dt, force_nstep=_dump_nstep)  # sgs_scalars
    # gSAM stage 16 "upperbound" is disabled for IRMA (doupperbound=.false.);
    # state is unchanged between 15 and 16.
    _stage_dump(state, 16, dt, force_nstep=_dump_nstep)  # upperbound

    # ------------------------------------------------------------------
    # 12. Microphysics
    # ------------------------------------------------------------------
    if config.micro_params is not None:
        state = micro_proc(state, metric, config.micro_params, dt)

    _stage_dump(state, 17, dt, force_nstep=_dump_nstep)  # micro

    # gSAM has no spectral polar filter on scalars either — the
    # implicit pole damping + SGS vertical diffusion handle grid-scale
    # noise near the poles.  The legacy jsam scalar spectral filter has
    # been removed to match the Fortran pipeline exactly.

    # ------------------------------------------------------------------
    # 13. Newtonian cooling in sponge zone
    #     Relax TABS toward the reference tabs0 in the sponge layer.
    #     This directly breaks the buoyancy-W feedback loop at the model
    #     top that otherwise amplifies any pressure-solver residual into
    #     an exponentially growing instability.
    # ------------------------------------------------------------------
    if config.sponge_tau > 0.0 and forcing.tabs0 is not None:
        _z        = metric["z"]
        _z_sponge = config.sponge_z_frac * float(_z[-1])
        _frac     = jnp.clip((_z - _z_sponge) / (float(_z[-1]) - _z_sponge),
                              0.0, 1.0)
        _alpha    = jnp.sin(0.5 * jnp.pi * _frac) ** 2
        _factor   = 1.0 / (1.0 + _alpha * dt / config.sponge_tau)

        # tabs0 may be (nz,), (nz,ny), or (nz,ny,nx)
        _tabs0 = forcing.tabs0
        if _tabs0.ndim == 1:
            _tabs0 = _tabs0[:, None, None]
        elif _tabs0.ndim == 2:
            _tabs0 = _tabs0[:, :, None]

        _TABS_damped = _tabs0 + (state.TABS - _tabs0) * _factor[:, None, None]
        state = ModelState(
            U=state.U, V=state.V, W=state.W,
            TABS=_TABS_damped, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

    # ------------------------------------------------------------------
    # 14. Update buoyancy reference profiles  [≡ gSAM diagnose()]
    #     gSAM recomputes tabs0(k) and qv0(k) as the GLOBAL HORIZONTAL
    #     MEAN of the current TABS and QV each timestep.  This keeps the
    #     buoyancy anomaly (TABS - tabs0) bounded — without it, tabs0 is
    #     fixed at the initial state and the anomaly grows without limit,
    #     creating an exponentially growing W instability.
    # ------------------------------------------------------------------
    if forcing.tabs0 is not None:
        # gSAM diagnose.f90:27-86 — cos-lat-weighted horizontal means.
        # Matches precisely:
        #   tabs0(k) = <tabs(i,j,k)>
        #   q0(k)    = <qv+qcl+qci>      qn0(k) = <qcl+qci>
        #                                qp0(k) = <qpl+qpi>
        #   qv0(k)   = q0(k) - qn0(k)     (diagnose.f90:86)
        # For flat-ocean IRMA terra(i,j,k)≡1 so the terrain mask is a no-op.
        _cos_lat = metric["cos_lat"]   # (ny,)
        _wgt = _cos_lat / jnp.sum(_cos_lat)  # normalised weights (ny,)

        def _hmean(field):
            # field: (nz, ny, nx) → (nz,)
            return jnp.sum(jnp.mean(field, axis=2) * _wgt[None, :], axis=1)

        # D10: gSAM diagnose.f90:37 recovers tabs from the conserved static
        # energy variable t: tabs = t - gamaz + fac_cond*(qcl+qpl) + fac_sub*(qci+qpi).
        # With D25 (advecting full t instead of s=TABS+gamaz), jsam's TABS is
        # now consistent with gSAM's recovered tabs, so averaging state.TABS
        # directly is correct.
        _tabs_mean = _hmean(state.TABS)
        _q0        = _hmean(state.QV + state.QC + state.QI)
        _qn0       = _hmean(state.QC + state.QI)
        _qp0       = _hmean(state.QR + state.QS + state.QG)
        _qv0       = _q0 - _qn0

        forcing = PhysicsForcing(
            tabs0=_tabs_mean,
            qv0=_qv0,
            qn0=_qn0,
            qp0=_qp0,
            tabs_ref=forcing.tabs_ref,
            qv_ref=forcing.qv_ref,
            rad_forcing=forcing.rad_forcing,
            ls_forcing=forcing.ls_forcing,
            sst=forcing.sst,
            o3vmr_rrtmg=forcing.o3vmr_rrtmg,
            qrad_rrtmg=forcing.qrad_rrtmg,
        )

    # ------------------------------------------------------------------
    # Rotate pressure buffer for next step's adamsB call:
    #   p_pprev ← p_prev (the p_{n-1} used in this step's adamsB)
    #   p_prev  ← p_new  (the p_n just computed by pressure_step)
    # ------------------------------------------------------------------
    state = ModelState(
        U=state.U, V=state.V, W=state.W,
        TABS=state.TABS, QV=state.QV, QC=state.QC,
        QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
        TKE=state.TKE,
        p_prev=p_new, p_pprev=p_prev_for_adamsb,
        nstep=state.nstep, time=state.time,
    )

    _stage_dump(state, 18, dt, force_nstep=_dump_nstep)  # diagnose

    return state, mom_tends_n, tends_n, forcing
