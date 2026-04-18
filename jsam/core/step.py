"""Step driver — one model timestep orchestrating all physics/dynamics modules.
Sub-functions JIT-compiled independently. Physics modules switched via None checks."""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


from jsam.core.state import ModelState
from jsam.core.dynamics.timestepping import (
    Tendencies, MomentumTendencies,
    advance_scalars, advance_momentum,
)
from jsam.core.dynamics.pressure import pressure_step, adams_b
from jsam.core.dynamics.damping import pole_damping, top_sponge, gsam_w_sponge, gsam_w_courant_damping, spectral_polar_filter
from jsam.core.dynamics.coriolis import coriolis_tend
from jsam.core.physics.sgs import (
    SGSParams, SurfaceFluxes, sgs_proc,
    sgs_mom_proc, sgs_scalars_proc,
    diffuse_momentum_horiz, diffuse_damping_mom_z,
    diffuse_scalar_horiz, diffuse_scalar_z_implicit,
    _sgs_coefs,
)
from jsam.core.physics.microphysics import MicroParams, micro_proc, CP, FAC_COND, FAC_SUB
from jsam.core.physics.radiation import RadForcing, rad_proc
from jsam.core.physics.rad_rrtmg import RadRRTMGConfig, rad_rrtmg_proc, compute_qrad_rrtmg, compute_qrad_and_lwds_rrtmg
from jsam.core.physics.lsforcing import LargeScaleForcing, ls_proc
from jsam.core.physics.surface import BulkParams, bulk_surface_fluxes
from jsam.core.physics.nudging import NudgingParams, nudge_proc
from jsam.core.physics.slm import (
    SLMParams, SLMStatic, SLMState, SLMRadInputs, slm_proc,
)
from jsam.core import debug_dump as _dd


# ---------------------------------------------------------------------------
# Batched SGS scalar diffusion — JIT + vmap over all 7 prognostic scalars
# ---------------------------------------------------------------------------

@jax.jit
def _diffuse_scalars_horiz_seq(
    fields:  list,         # list of N (nz, ny, nx) arrays
    tkh:     jax.Array,   # (nz, ny, nx)
    tk_max:  jax.Array,   # (nz, ny, 1)
    dx_lon:  jax.Array,
    dy_lat:  jax.Array,
    cos_lat: jax.Array,
    cos_v:   jax.Array,
    dy_ref:  jax.Array,
) -> list:                 # list of N (nz, ny, nx) arrays
    _m = {
        "dx_lon": dx_lon, "dy_lat": dy_lat, "cos_lat": cos_lat,
        "cos_v": cos_v,   "dy_lat_ref": dy_ref,
    }
    return [diffuse_scalar_horiz(f, tkh, _m, tk_max=tk_max) for f in fields]


def _diffuse_scalars_z_seq(
    fields:  list,         # list of N (nz, ny, nx) arrays
    fluxbs:  list,         # list of N (ny, nx) arrays
    tkh:     jax.Array,   # (nz, ny, nx)
    dt:      jax.Array,
    rho:     jax.Array,
    rhow:    jax.Array,
    adz:     jax.Array,
    adzw:    jax.Array,
    dz_ref:  jax.Array,
) -> list:                 # list of N (nz, ny, nx) arrays
    _m = {"rho": rho, "rhow": rhow, "adz": adz, "adzw": adzw, "dz_ref": dz_ref}
    return [diffuse_scalar_z_implicit(f, tkh, _m, dt, fluxb=fb)
            for f, fb in zip(fields, fluxbs)]


def _stage_dump(state, stage_id: int, dt, force_nstep=None):
    """Emit oracle-format dump if DebugDumper is active."""
    if _dd.DUMPER is not None:
        _dd.DUMPER.dump(state, stage_id, dt, force_nstep=force_nstep)

@dataclass(frozen=True)
class StepConfig:
    """Static physics configuration. Set params to None to disable module."""
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
    dodamping_w:   bool  = False  # gSAM dodamping_w: CFL-based W soft-clip below sponge
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
    rad_iyear: int = 1999


@jax.tree_util.register_pytree_node_class
@dataclass
class PhysicsForcing:
    """Time-varying forcing arrays (JAX pytree). None disables corresponding forcing."""
    tabs0:      jax.Array | None             = None
    qv0:        jax.Array | None             = None
    qn0:        jax.Array | None             = None  # qcl+qci
    qp0:        jax.Array | None             = None  # qpl+qpi
    tabs_ref:   jax.Array | None             = None
    qv_ref:     jax.Array | None             = None
    rad_forcing: RadForcing | None           = None
    ls_forcing:  LargeScaleForcing | None    = None
    sst:         jax.Array | None            = None
    o3vmr_rrtmg: jax.Array | None            = None   # (nz,) or (ncol,nz) vmr
    qrad_rrtmg:  jax.Array | None            = None   # (nz,ny,nx) K/s heating rates
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
    qn0:    jax.Array | None = None,    # (nz,)
    qp0:    jax.Array | None = None,    # (nz,)
    terraw: jax.Array | None = None,    # (ny, nx) or (nz, ny, nx) terrain mask
) -> jax.Array:
    """Buoyancy on W-faces (nz+1,ny,nx) [m/s²]. Mirrors gSAM buoyancy.f90 exactly.

    Fix A: for each W-face k (gSAM 1-based, k=2..nzm, kb=k-1), buoyancy is the
    sum of an upper term using tabs0(k)/bet(k) and a lower term using tabs0(kb)/bet(kb),
    weighted by betu=adz(kb)/(adz(k)+adz(kb)) and betd=adz(k)/(adz(k)+adz(kb)):
        buo = bet(k)*betu*(tabs0(k)*(epsv*(qv-qv0(k))-(qn-qn0(k)+qp-qp0(k)))
                           + (tabs-tabs0(k))*(1+epsv*qv0(k)-qn0(k)-qp0(k)))
            + bet(kb)*betd*(tabs0(kb)*(epsv*(qv-qv0(kb))-(qn-qn0(kb)+qp-qp0(kb)))
                            + (tabs-tabs0(kb))*(1+epsv*qv0(kb)-qn0(kb)-qp0(kb)))
    Each level's b is computed separately then combined, so non-uniform tabs0 profiles
    are handled correctly.

    terraw: applied at W-face level (after combining), matching gSAM's
    `if(terraw(i,j,k).gt.0.)` gate. 2D (ny,nx) is broadcast across all faces.
    """
    def _expand(a):
        if a.ndim == 1:
            return a[:, None, None]
        elif a.ndim == 2:
            return a[:, :, None]
        return a

    tabs0_3d = _expand(tabs0)                                           # (nz,1,1)
    qv0_3d   = _expand(qv0)                                            # (nz,1,1)
    qn0_3d   = _expand(qn0) if qn0 is not None else jnp.zeros_like(tabs0_3d)
    qp0_3d   = _expand(qp0) if qp0 is not None else jnp.zeros_like(tabs0_3d)

    qn = state.QC + state.QI              # qcl + qci  (cloud liquid + cloud ice)
    qp = state.QR + state.QS + state.QG  # qpl + qpi  (rain + snow + graupel)

    # Fix A: buoyancy at each mass level k using its own tabs0(k) and bet(k).
    # b[k] = bet(k) * (tabs0(k)*(epsv*(qv-qv0(k)) - (qn-qn0(k)+qp-qp0(k)))
    #                  + (tabs-tabs0(k))*(1+epsv*qv0(k)-qn0(k)-qp0(k)))
    # This is exactly gSAM buoyancy.f90 lines 48-53 per level.
    b = (g / tabs0_3d) * (
        tabs0_3d * (
            epsv * (state.QV - qv0_3d)
            - (qn - qn0_3d)
            - (qp - qp0_3d)
        )
        + (state.TABS - tabs0_3d) * (1.0 + epsv * qv0_3d - qn0_3d - qp0_3d)
    )   # (nz, ny, nx)

    # Fix A: interpolate to W-faces using gSAM's betu/betd weights.
    # W-face k (Python index k_py = k-1, range 1..nzm-1):
    #   adz(kb) = dz[k_py-1] = dz_lo[k_py-1]  (lower cell thickness)
    #   adz(k)  = dz[k_py]   = dz_hi[k_py-1]  (upper cell thickness)
    #   betu = adz(kb)/(adz(k)+adz(kb))   weights upper mass level b[k_py]
    #   betd = adz(k) /(adz(k)+adz(kb))   weights lower mass level b[k_py-1]
    #   buo_w[k_py-1] = betu * b[k_py] + betd * b[k_py-1]
    #                 = betu * b[1:][k_py-1] + betd * b[:-1][k_py-1]
    dz_lo = dz[:-1][:, None, None]   # adz(kb) for each interior W-face
    dz_hi = dz[1:][:, None, None]    # adz(k)  for each interior W-face
    betu  = dz_lo / (dz_hi + dz_lo)  # weight on upper level (k)
    betd  = dz_hi / (dz_hi + dz_lo)  # weight on lower level (kb)
    b_int = betu * b[1:] + betd * b[:-1]   # (nz-1, ny, nx)

    ny, nx = state.TABS.shape[1], state.TABS.shape[2]
    b_w = jnp.concatenate([
        jnp.zeros((1, ny, nx)),
        b_int,
        jnp.zeros((1, ny, nx)),
    ], axis=0)   # (nz+1, ny, nx)

    # Fix A (terraw): apply mask at W-face level, mirroring gSAM's
    # `if(terraw(i,j,k).gt.0.)` check before accumulating buo into dwdt.
    if terraw is not None:
        if terraw.ndim == 2:
            # 2D (ny, nx): broadcast across all W-face levels
            b_w = jnp.where(terraw[None, :, :] > 0.0, b_w, 0.0)
        else:
            # 3D (nz, ny, nx) mass-level mask: pad to W-face shape (nz+1, ny, nx)
            tw = jnp.concatenate([
                terraw[:1],
                terraw,
            ], axis=0)   # simple pad: top W-face inherits top mass-level value
            b_w = jnp.where(tw > 0.0, b_w, 0.0)

    return b_w


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
    dump_nstep:     int | None = None,
    polar_filter_masks: tuple | None = None,
) -> tuple[ModelState, MomentumTendencies, Tendencies, PhysicsForcing]:
    """Advance model one timestep. Operator-split following gSAM main.f90.

    dump_nstep: if not None, use this as the nstep value for all oracle stage
        dumps in this call.  Pass the outer-loop step counter (i) to align
        jsam's debug dump nstep with gSAM's outer-step nstep convention.
    """

    p_prev_for_adamsb  = state.p_prev
    p_pprev_for_adamsb = state.p_pprev

    from jsam.core.dynamics.timestepping import ab_coefs
    _nstep_py = int(state.nstep)
    _at_ab, _bt_ab, _ct_ab = ab_coefs(_nstep_py, dt, dt_prev, dt_pprev)
    _macho_order = int(_nstep_py % 6)   # nstep after increment = _nstep_py+1; macho=(nstep-1)%6=_nstep_py%6

    state = ModelState(
        U=state.U, V=state.V, W=state.W,
        TABS=state.TABS, QV=state.QV, QC=state.QC,
        QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
        TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep + 1, time=state.time,
    )

    # Enforce W=0 at rigid-lid boundaries (bottom and top)
    W_BC = state.W.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    state = ModelState(
        U=state.U, V=state.V, W=W_BC,
        TABS=state.TABS, QV=state.QV, QC=state.QC,
        QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
        TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )

    _dump_nstep = dump_nstep if dump_nstep is not None else int(state.nstep)
    _stage_dump(state, 0, dt, force_nstep=_dump_nstep)

    if forcing.ls_forcing is not None:
        state = ls_proc(state, metric, forcing.ls_forcing, dt)
    # gSAM forcing() also applies prescribed radiation at stage 1, before buoyancy.
    if forcing.rad_forcing is not None:
        state = rad_proc(state, metric, forcing.rad_forcing, dt)

    _stage_dump(state, 1, dt, force_nstep=_dump_nstep)  # forcing

    # gSAM main.f90: nudging() called immediately after forcing(), before buoyancy().
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

    _stage_dump(state, 2, dt, force_nstep=_dump_nstep)  # nudging

    # Buoyancy uses post-forcing state (includes prescribed radiation, pre-RRTMG),
    # matching gSAM: buoyancy() at stage 3 sees state from forcing() at stage 1.
    _state_for_buoyancy = state

    _stage_dump(state, 3, dt, force_nstep=_dump_nstep)  # buoyancy

    if config.rad_rrtmg is not None and forcing.sst is not None:
        if int(state.nstep) % config.nrad == 0:
            _sw_aux = None
            if config.rad_day0 is not None:
                _day_for_sw = float(config.rad_day0) + float(state.time) / 86400.0
                _sw_aux = {
                    "day_of_year": _day_for_sw,
                    "iyear":       int(config.rad_iyear),
                    "lat_rad":     metric["lat_rad"],
                    "lon_rad":     metric["lon_rad"],
                }
            # Fix 7.3: pass landmask so SW uses Briegleb land albedo over land.
            _rad_landmask = (
                np.asarray(forcing.slm_static.landmask)
                if forcing.slm_static is not None else None
            )
            # Compute both qrad (for main heating) and lwds (for SLM)
            _new_qrad, _lwds_new = compute_qrad_and_lwds_rrtmg(
                state, metric, config.rad_rrtmg, forcing.sst,
                o3vmr=(None if forcing.o3vmr_rrtmg is None
                       else jnp.asarray(forcing.o3vmr_rrtmg)),
                sw_aux=_sw_aux,
                landmask=_rad_landmask,
            )
            forcing = PhysicsForcing(
                tabs0=forcing.tabs0, qv0=forcing.qv0,
                qn0=forcing.qn0, qp0=forcing.qp0,
                tabs_ref=forcing.tabs_ref, qv_ref=forcing.qv_ref,
                rad_forcing=forcing.rad_forcing, ls_forcing=forcing.ls_forcing,
                sst=forcing.sst, o3vmr_rrtmg=forcing.o3vmr_rrtmg,
                qrad_rrtmg=_new_qrad,
                slm_static=forcing.slm_static, slm_state=forcing.slm_state,
                slm_rad=(forcing.slm_rad._replace(lwds=_lwds_new) if forcing.slm_rad is not None else forcing.slm_rad),
                precip_ref=forcing.precip_ref,
            )
        if forcing.qrad_rrtmg is not None:
            state = ModelState(
                U=state.U, V=state.V, W=state.W,
                TABS=state.TABS + dt * forcing.qrad_rrtmg,
                QV=state.QV, QC=state.QC, QI=state.QI,
                QR=state.QR, QS=state.QS, QG=state.QG,
                TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
                nstep=state.nstep, time=state.time,
            )

    _stage_dump(state, 4, dt, force_nstep=_dump_nstep)

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

    for _sid in (5, 6, 7, 8, 9):
        _stage_dump(state, _sid, dt, force_nstep=_dump_nstep)

    _U_old, _V_old, _W_old = state.U, state.V, state.W

    nz_s, ny_s, nx_s = state.TABS.shape
    dU_extra = jnp.zeros((nz_s, ny_s, nx_s + 1))
    dV_extra = jnp.zeros((nz_s, ny_s + 1, nx_s))
    dW_extra = jnp.zeros((nz_s + 1, ny_s, nx_s))

    # Fix 1.4: Compute AB-extrapolated advective velocities before calling
    # advance_momentum and advance_scalars (matching gSAM advect_all_scalars.f90).
    # Note: advective velocity uses fixed AB2 (0.5, 0.5) on step>1, NOT the
    # variable AB coefficients used for momentum/scalar advancement.
    mu     = metric["cos_lat"]
    muv    = metric["cos_v"]
    ady    = metric["ady"]
    rho    = metric["rho"]
    rhow   = metric["rhow"]
    dz     = metric["dz"]
    dx     = metric["dx_lon"]
    dy_ref = metric["dy_lat_ref"]

    dz_ref = dz[0]
    adz    = dz / dz_ref

    dtdx = dt / dx
    dtdy = dt / dy_ref
    dtdz = dt / dz_ref

    # Current step's mass-flux-weighted velocities (u1, v1, w1 in gSAM)
    u1_curr = state.U * (rho[:, None, None] * dtdx * adz[:, None, None] * ady[None, :, None])
    v1_curr = state.V * (rho[:, None, None] * dtdy * adz[:, None, None] * muv[None, :, None])
    w1_curr = state.W * (rhow[:, None, None] * dtdz * ady[None, :, None] * mu[None, :, None])

    # AB2-extrapolate advective velocity (fixed coefficients, not variable like AB3)
    # At nstep=1: a1=1, a2=0 (Euler)
    # At nstep>1: a1=0.5, a2=0.5 (AB2)
    _nstep_int = int(state.nstep)
    a1_adv = jnp.where(_nstep_int == 1, 1.0, 0.5)
    a2_adv = jnp.where(_nstep_int == 1, 0.0, 0.5)

    u1_adv = a1_adv * u1_curr + a2_adv * mom_tends_nm1.U_adv
    v1_adv = a1_adv * v1_curr + a2_adv * mom_tends_nm1.V_adv
    w1_adv = a1_adv * w1_curr + a2_adv * mom_tends_nm1.W_adv

    # Convert back to velocities (undo mass-flux weighting)
    U_adv_for_advection = jnp.where(
        rho[:, None, None] * dtdx * adz[:, None, None] * ady[None, :, None] != 0,
        u1_adv / (rho[:, None, None] * dtdx * adz[:, None, None] * ady[None, :, None]),
        state.U
    )
    V_adv_for_advection = jnp.where(
        rho[:, None, None] * dtdy * adz[:, None, None] * muv[None, :, None] != 0,
        v1_adv / (rho[:, None, None] * dtdy * adz[:, None, None] * muv[None, :, None]),
        state.V
    )
    W_adv_for_advection = jnp.where(
        rhow[:, None, None] * dtdz * ady[None, :, None] * mu[None, :, None] != 0,
        w1_adv / (rhow[:, None, None] * dtdz * ady[None, :, None] * mu[None, :, None]),
        state.W
    )

    if forcing.tabs0 is not None:
        qv0 = forcing.qv0 if forcing.qv0 is not None else jnp.zeros_like(forcing.tabs0)
        _dW_buo = _buoyancy_W(_state_for_buoyancy, forcing.tabs0, qv0,
                               metric["dz"], config.g, config.epsv,
                               qn0=forcing.qn0, qp0=forcing.qp0,
                               terraw=metric.get("terraw", None))
        dW_extra = dW_extra + _dW_buo

        # Fix B: gSAM buoyancy.f90 energy conservation (lines 35, 58-60).
        # For each W-face k (gSAM 1-based, k=2..nzm), kb=k-1:
        #   coef   = 0.5 * dtn / cp       (dtn = dt in non-subcycled operation)
        #   factor = coef * buo(i,j,k) * w(i,j,k)
        #   t(i,j,kb) -= factor            (lower mass cell)
        #   t(i,j,k)  -= factor            (upper mass cell)
        # _dW_buo[1:-1] are the interior W-faces (Python 1..nz-1 = gSAM k=2..nzm).
        # _state_for_buoyancy.W is the W field at the time buoyancy is computed
        # (before any tendency is applied), matching gSAM's w(i,j,k) usage.
        # Because Fix A applied terraw at W-face level, _dW_buo[1:-1] is already
        # zero where terraw <= 0, so the energy correction is automatically skipped
        # there — matching gSAM's `if(terraw(i,j,k).gt.0.)` gate.
        _buo_faces = _dW_buo[1:-1]                          # (nz-1, ny, nx)
        _W_faces   = _state_for_buoyancy.W[1:-1]            # (nz-1, ny, nx)
        _factor    = 0.5 * dt / CP * _buo_faces * _W_faces  # (nz-1, ny, nx)
        # Apply to lower cell (kb = k-1, Python index 0..nz-2)
        _TABS_buo_corrected = state.TABS
        _TABS_buo_corrected = _TABS_buo_corrected.at[:-1, :, :].add(-_factor)
        # Apply to upper cell (k, Python index 1..nz-1)
        _TABS_buo_corrected = _TABS_buo_corrected.at[1:, :, :].add(-_factor)
        state = ModelState(
            U=state.U, V=state.V, W=state.W,
            TABS=_TABS_buo_corrected, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )


    _dU_cor_mass, _dV_cor, _dW_cor = coriolis_tend(
        state.U, state.V, state.W, metric,
    )
    dU_extra = dU_extra.at[:, :, :nx_s].add(_dU_cor_mass)
    dU_extra = dU_extra.at[:, :, nx_s].add(_dU_cor_mass[:, :, 0])
    dV_extra = dV_extra + _dV_cor
    dW_extra = dW_extra + _dW_cor

    # gSAM sgs_proc(): compute SGS coefficients from the pre-advection state
    # (sgs_proc is called after advect_mom/coriolis but before adamsA, so velocities
    # are still U_n).  sgs_mom() writes horizontal SGS into dudtd which adamsA applies
    # with coefficient 1 (Euler), NOT into dudt(na) which is AB-stepped with at.
    _tk = None
    _tkh = None
    _tk_max = None
    _dUh_sgs = None
    _dVh_sgs = None
    _dWh_sgs = None
    _fluxbt_sgs = None if surface_fluxes is None else surface_fluxes.shf
    _fluxbq_sgs = None if surface_fluxes is None else surface_fluxes.lhf
    if config.sgs_params is not None:
        _tk, _tkh, _tk_max = _sgs_coefs(
            state, metric, config.sgs_params, dt, tabs0=forcing.tabs0,
            fluxbt=_fluxbt_sgs, fluxbq=_fluxbq_sgs,
        )
        _dUh_sgs, _dVh_sgs, _dWh_sgs = diffuse_momentum_horiz(
            state.U, state.V, state.W, _tk, metric,
        )
        # DO NOT add to dU_extra — horizontal SGS is Euler-stepped (dudtd), not AB-stepped

    state, mom_tends_n = advance_momentum(
        state, mom_tends_nm1, mom_tends_nm2, metric, dt,
        dU_extra=dU_extra,
        dV_extra=dV_extra,
        dW_extra=dW_extra,
        dt_prev=dt_prev,
        dt_pprev=dt_pprev,
        U_adv=U_adv_for_advection,
        V_adv=V_adv_for_advection,
        W_adv=W_adv_for_advection,
    )

    # gSAM sgs_mom() → adamsA: apply horizontal SGS as forward Euler step using the
    # pre-advance tendency (dudtd coefficient 1).  Must happen before terrain masking
    # so that terrau multiplies the full adamsA result: terrau*(u_AB + dt*dudtd).
    if _dUh_sgs is not None:
        state = ModelState(
            U=state.U + dt * _dUh_sgs,
            V=state.V + dt * _dVh_sgs,
            W=state.W + dt * _dWh_sgs,
            TABS=state.TABS, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

    # Apply terrain masks (gSAM adamsA.f90: terrau multiplies the combined AB+dudtd result)
    if metric.get("terrau") is not None:
        state = ModelState(
            U=state.U * metric["terrau"],
            V=state.V * metric["terrav"],
            W=state.W * metric["terraw"],
            TABS=state.TABS, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

    _stage_dump(state, 10, dt, force_nstep=_dump_nstep)

    # gSAM diffuse_damping_mom_z(): implicit vertical SGS + velocity limiter,
    # applied after adamsA.
    if _tk is not None:
        _fluxbu = None
        _fluxbv = None
        if surface_fluxes is not None:
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
        # gSAM dodamping branch: W-only Rayleigh sponge at model top.
        if config.w_sponge_max > 0.0:
            _W_imp = gsam_w_sponge(
                _W_imp, metric["zi"],
                nub=config.w_sponge_nub,
                taudamp_max=config.w_sponge_max,
                dtn=_at_ab * dt,
                dt=dt,
            )
        # gSAM dodamping_w branch: CFL-based W soft-clip below the sponge.
        # Independent of dodamping — in gSAM these are separate if-blocks sharing taudamp.
        # gsam_w_courant_damping recomputes the sponge mask (nu > nub) internally to
        # identify levels where taudamp==0 (below sponge), matching gSAM exactly.
        if config.dodamping_w:
            _W_imp = gsam_w_courant_damping(
                _W_imp, metric,
                dtn=_at_ab * dt,
                damping_w_cu=config.damping_w_cu,
                tau_max=_at_ab,         # gSAM: tau_max = dtn/dt = at_ab
                nub=config.w_sponge_nub,
            )
        state = ModelState(
            U=_U_imp, V=_V_imp, W=_W_imp,
            TABS=state.TABS, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

    _stage_dump(state, 11, dt, force_nstep=_dump_nstep)

    if p_prev_for_adamsb is not None:
        state = adams_b(
            state, p_prev_for_adamsb, metric, dt,
            bt=_bt_ab,
            p_pprev=p_pprev_for_adamsb,
            ct=_ct_ab,
        )

    _stage_dump(state, 12, dt, force_nstep=_dump_nstep)

    if polar_filter_masks is not None:
        mask_u, mask_v = polar_filter_masks
        U_f, V_f = spectral_polar_filter(state.U, state.V, mask_u, mask_v)
        state = ModelState(
            U=U_f, V=V_f, W=state.W,
            TABS=state.TABS, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

    state, p_new = pressure_step(state, grid, metric, dt, at=_at_ab)

    # Normalize pressure for adamsB: scale by 1/at so that the AB coefficients
    # (bt*dt, ct*dt) in adamsB combine correctly across steps with different at values.
    # This matches gSAM's convention where press_rhs divides by at and press_grad
    # multiplies by at, making the stored pressure independent of at scaling.
    p_for_storage = p_new / _at_ab if _at_ab != 0.0 else p_new

    _stage_dump(state, 13, dt, force_nstep=_dump_nstep)

    # gSAM advect_all_scalars.f90: scalar advection velocity is
    #   u_scalar = a1*u(n+1) + a2*u(n)
    # where u(n+1) = post-pressure velocity (state.U now) and u(n) = _U_old.
    # At nstep=1: a1=1, a2=0 (Euler).  At nstep≥2: a1=0.5, a2=0.5 (centered).
    # This is different from the momentum AB-extrapolation (which uses lagged
    # u(n)/u(n-1) computed before advance_momentum).
    _a1_s = 1.0 if _nstep_int == 1 else 0.5
    _a2_s = 0.0 if _nstep_int == 1 else 0.5
    U_adv_scalar = _a1_s * state.U + _a2_s * _U_old
    V_adv_scalar = _a1_s * state.V + _a2_s * _V_old
    W_adv_scalar = _a1_s * state.W + _a2_s * _W_old

    _TABS_before_adv = state.TABS
    _omp_f11 = None
    _qp_f11 = None
    if config.micro_params is not None:
        _gamaz_3d_f11 = metric["gamaz"][:, None, None]
        _a_pr_f11 = 1.0 / (config.micro_params.tprmax - config.micro_params.tprmin)
        _omp_f11  = jnp.clip((_TABS_before_adv - config.micro_params.tprmin) * _a_pr_f11, 0.0, 1.0)
        _qp_f11   = state.QR + state.QS + state.QG
        _t_static = (_TABS_before_adv + _gamaz_3d_f11
                     - FAC_COND * (state.QC + _qp_f11 * _omp_f11)
                     - FAC_SUB  * (state.QI + _qp_f11 * (1.0 - _omp_f11)))
        state = ModelState(
            U=state.U, V=state.V, W=state.W,
            TABS=_t_static, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

    state, tends_n = advance_scalars(
        state, tends_nm1, tends_nm2, metric, dt,
        dt_prev=dt_prev, dt_pprev=dt_pprev,
        U_old=_U_old, V_old=_V_old, W_old=_W_old,
        is_f11=(config.micro_params is not None),
        U_adv=U_adv_scalar, V_adv=V_adv_scalar, W_adv=W_adv_scalar,
        macho_order=_macho_order,
    )

    # F11 mode: convert advected static energy back to absolute temperature
    if config.micro_params is not None and _omp_f11 is not None:
        _gamaz_3d_f11 = metric["gamaz"][:, None, None]
        _TABS_phys = (state.TABS - _gamaz_3d_f11
                      + FAC_COND * (state.QC + _qp_f11 * _omp_f11)
                      + FAC_SUB  * (state.QI + _qp_f11 * (1.0 - _omp_f11)))
        state = ModelState(
            U=state.U, V=state.V, W=state.W,
            TABS=_TABS_phys, QV=state.QV, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )


    _stage_dump(state, 14, dt, force_nstep=_dump_nstep)

    if config.polar_cool_tau > 0.0 and forcing.tabs0 is not None:
        _lat     = metric["lat_rad"]
        _sin2    = jnp.sin(_lat) ** 2
        _tau_lat = _sin2 ** 4
        _factor  = 1.0 / (1.0 + _tau_lat * dt / config.polar_cool_tau)

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

    if _tkh is not None:
        # gSAM sgs_scalars(): use the same tk/tkh computed from pre-advection state
        # (gSAM's sgs_proc() runs once per step and its coefficients are reused here).
        _shf = _fluxbt_sgs
        _lhf = _fluxbq_sgs

        # Step 1: horizontal SGS — one scalar at a time to avoid vmap memory explosion
        _fields_h = [state.TABS, state.QV, state.QC, state.QI,
                     state.QR, state.QS, state.QG]
        _dfdts = _diffuse_scalars_horiz_seq(
            _fields_h, _tkh, _tk_max,
            metric["dx_lon"], metric["dy_lat"], metric["cos_lat"],
            metric["cos_v"],  metric["dy_lat_ref"],
        )
        state = ModelState(
            U=state.U, V=state.V, W=state.W,
            TABS=state.TABS + dt * _dfdts[0],
            QV  =state.QV   + dt * _dfdts[1],
            QC  =state.QC   + dt * _dfdts[2],
            QI  =state.QI   + dt * _dfdts[3],
            QR  =state.QR   + dt * _dfdts[4],
            QS  =state.QS   + dt * _dfdts[5],
            QG  =state.QG   + dt * _dfdts[6],
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

        # Step 2: implicit vertical SGS — one scalar at a time
        _zero_flux = jnp.zeros_like(state.TABS[0])
        _shf_arr   = _shf if _shf is not None else _zero_flux
        _lhf_arr   = _lhf if _lhf is not None else _zero_flux
        _fields_z  = [state.TABS, state.QV, state.QC, state.QI,
                      state.QR, state.QS, state.QG]
        _fluxbs_z  = [_shf_arr, _lhf_arr, _zero_flux, _zero_flux,
                      _zero_flux, _zero_flux, _zero_flux]
        _updated = _diffuse_scalars_z_seq(
            _fields_z, _fluxbs_z, _tkh, dt,
            metric["rho"], metric["rhow"], metric["adz"], metric["adzw"], metric["dz_ref"],
        )
        state = ModelState(
            U=state.U, V=state.V, W=state.W,
            TABS=_updated[0], QV =_updated[1], QC =_updated[2], QI =_updated[3],
            QR  =_updated[4], QS =_updated[5], QG =_updated[6],
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

    _tke_out = _tk if _tk is not None else state.TKE
    _hydros = jnp.maximum(
        jnp.stack([state.QV, state.QC, state.QI, state.QR, state.QS, state.QG]), 0.0
    )
    state = ModelState(
        U=state.U, V=state.V, W=state.W,
        TABS=state.TABS,
        QV=_hydros[0], QC=_hydros[1], QI=_hydros[2],
        QR=_hydros[3], QS=_hydros[4], QG=_hydros[5],
        TKE=_tke_out, p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )

    _stage_dump(state, 15, dt, force_nstep=_dump_nstep)

    # gSAM upperbound.f90: nudge the two highest scalar levels toward the
    # reference sounding when dolargescale is active (i.e. tabs_ref is set).
    # tau_nudging = 3600 s (hardcoded in gSAM), coef = dtn / tau_nudging.
    # Applied BEFORE microphysics (matching gSAM main.f90 stage 16 → stage 17).
    if forcing.tabs_ref is not None:
        _ub_coef = dt / 3600.0
        _gamaz_ub = metric["gamaz"]  # (nz,)
        # tabs_ref and qv_ref are (nz,) reference profiles
        _tabs_ref_ub = forcing.tabs_ref          # tg0
        _qv_ref_ub   = forcing.qv_ref            # qg0 (may be None)
        # Nudge top 2 mass levels: indices -2 and -1
        _TABS_ub = state.TABS
        _TABS_ub = _TABS_ub.at[-2, :, :].add(
            -(_TABS_ub[-2, :, :] - _tabs_ref_ub[-2] - _gamaz_ub[-2]) * _ub_coef
        )
        _TABS_ub = _TABS_ub.at[-1, :, :].add(
            -(_TABS_ub[-1, :, :] - _tabs_ref_ub[-1] - _gamaz_ub[-1]) * _ub_coef
        )
        _QV_ub = state.QV
        if _qv_ref_ub is not None:
            _QV_ub = _QV_ub.at[-2, :, :].add(
                -(_QV_ub[-2, :, :] - _qv_ref_ub[-2]) * _ub_coef
            )
            _QV_ub = _QV_ub.at[-1, :, :].add(
                -(_QV_ub[-1, :, :] - _qv_ref_ub[-1]) * _ub_coef
            )
        state = ModelState(
            U=state.U, V=state.V, W=state.W,
            TABS=_TABS_ub, QV=_QV_ub, QC=state.QC,
            QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
            TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
            nstep=state.nstep, time=state.time,
        )

    _stage_dump(state, 16, dt, force_nstep=_dump_nstep)  # upperbound stage

    if config.micro_params is not None:
        _micro_landmask = (
            (forcing.slm_static.landmask != 0)
            if forcing.slm_static is not None else None
        )
        state = micro_proc(state, metric, config.micro_params, dt,
                           tabs_phys=_TABS_before_adv,
                           landmask=_micro_landmask)

    _stage_dump(state, 17, dt, force_nstep=_dump_nstep)

    if config.sponge_tau > 0.0 and forcing.tabs0 is not None:
        _z        = metric["z"]
        _z_sponge = config.sponge_z_frac * float(_z[-1])
        _frac     = jnp.clip((_z - _z_sponge) / (float(_z[-1]) - _z_sponge), 0.0, 1.0)
        _alpha    = jnp.sin(0.5 * jnp.pi * _frac) ** 2
        _factor   = 1.0 / (1.0 + _alpha * dt / config.sponge_tau)

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
        _cos_lat = metric["cos_lat"]
        _ady     = metric["ady"]
        _wgt = (_cos_lat * _ady) / jnp.sum(_cos_lat * _ady)

        def _hmean(field):
            return jnp.sum(jnp.mean(field, axis=2) * _wgt[None, :], axis=1)

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

    state = ModelState(
        U=state.U, V=state.V, W=state.W,
        TABS=state.TABS, QV=state.QV, QC=state.QC,
        QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
        TKE=state.TKE,
        p_prev=p_for_storage, p_pprev=p_prev_for_adamsb,
        nstep=state.nstep, time=state.time,
    )

    _stage_dump(state, 18, dt, force_nstep=_dump_nstep)

    return state, mom_tends_n, tends_n, forcing
