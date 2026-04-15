"""
IRMA case driver for jsam.

Initialises from ERA5 Sep 5 2017 00UTC, runs step() in a Python for-loop,
and writes 3D_atm snapshots every --output-interval steps.

Typical usage
-------------
# Short test (1 hour, output every 30 min):
python scripts/run_irma.py --nsteps 360 --output-interval 180

# Full day:
python scripts/run_irma.py --nsteps 8640 --output-interval 180 --out-dir /path/to/out

PBS: see run_irma.pbs
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root → tests/ importable

import jax
import jax.numpy as jnp
import numpy as np

from jsam.core.state import ModelState

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

DT          = 10.0                           # seconds — matches gSAM IRMA 10s
START_TIME  = datetime(2017, 9, 5, 0, 0, 0)
RDA_ROOT    = "/glade/campaign/collections/rda/data/d633000"

# gSAM defaults (params.f90): cfl_max = 0.7, ncycle_max = 4.
CFL_MAX     = 0.7
NCYCLE_MAX  = 4


def _parse_args():
    p = argparse.ArgumentParser(description="jsam IRMA case driver")
    p.add_argument("--nsteps",          type=int,  default=360,
                   help="total number of steps to run (default 360 = 1h)")
    p.add_argument("--output-interval", type=int,  default=180,
                   help="write 3D_atm every N steps (default 180 = 30min)")
    p.add_argument("--ls-forcing-hours",type=int,  default=6,
                   help="hours of ERA5 large-scale forcing to load (default 6)")
    p.add_argument("--out-dir",         type=str,
                   default="/glade/derecho/scratch/sabramian/jsam_IRMA",
                   help="output directory for 3D_atm files")
    p.add_argument("--casename",        type=str,  default="jsam_IRMA",
                   help="output file prefix")
    p.add_argument("--dlat",            type=float, default=None,
                   help="horizontal resolution in degrees (default: use IRMALoader grid ~0.25°). "
                        "E.g. --dlat 2.0 for a coarser dev run.")
    p.add_argument("--lat-max",         type=float, default=89.0,
                   help="maximum absolute latitude (default 89°). Use e.g. 70 to avoid "
                        "polar singularity when running without a polar filter.")
    p.add_argument("--float32",         action="store_true", default=False,
                   help="use float32 state arrays (default: float64). Halves GPU memory; "
                        "microphysics/pressure still compute in float64 internally.")
    p.add_argument("--nudge-strato",    action="store_true", default=False,
                   help="enable scalar nudging (TABS+QV) toward the ERA5 init profile "
                        "in the upper atmosphere.  Faithful port of gSAM nudging.f90 "
                        "1D-profile branch.  Needed when using prescribed radiation "
                        "(not RRTMG) to prevent stratospheric drift at the cold pole.")
    p.add_argument("--nudge-z1",        type=float, default=15_000.0,
                   help="nudging band bottom (m). Default 15 km (tropopause).")
    p.add_argument("--nudge-z2",        type=float, default=60_000.0,
                   help="nudging band top (m). Default 60 km (model top).")
    p.add_argument("--nudge-tau",       type=float, default=3600.0,
                   help="nudging relaxation timescale (s). Default 3600 = 1 hour.")
    p.add_argument("--rad",             type=str,   default="prescribed",
                   choices=["prescribed", "rrtmg", "none"],
                   help="radiation scheme: prescribed (default), rrtmg, or none")
    p.add_argument("--nrad",            type=int,   default=90,
                   help="call RRTMG every nrad steps (default 90 = every ~15 min at dt=10s)")
    p.add_argument("--co2",             type=float, default=400.0,
                   help="CO2 mixing ratio in ppmv (default 400)")
    p.add_argument("--debug-dump-dir",  type=str, default=None,
                   help="Enable gSAM-oracle-compatible per-stage dumps "
                        "(U,V,W,TABS,QC,QV,QI) into this directory. Matches "
                        "the on-disk format of gSAM debug_dump.f90 so output "
                        "can be diffed against the 500-step IRMA oracle at "
                        "/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug. "
                        "Expect ~5x runtime slowdown from per-stage host sync.")
    p.add_argument("--o3file",          type=str,
                   default="/glade/u/home/sabramian/gSAM1.8.7/GLOBAL_DATA/BIN_D/"
                           "ozone_era5_monthly_201709-201709_GLOBAL.bin",
                   help="gSAM o3file for RRTMG ozone climatology. "
                        "Empty string → analytic Gaussian fallback.")
    p.add_argument("--no-polar-filter", action="store_true", default=False,
                   help="Disable the spectral polar filter (velocity + scalar). "
                        "gSAM has no spectral polar filter — only the implicit "
                        "CFL-based pole damping in damping.f90.  Required for "
                        "oracle-comparable debug runs (--debug-dump-dir).")
    p.add_argument("--sponge-tau",      type=float, default=10.0,
                   help="TABS Newtonian damping timescale (s) in the top sponge "
                        "layer (default 10 s).  Set to 0 to disable, which is "
                        "required for oracle-comparable debug runs because gSAM "
                        "has no equivalent sponge on TABS.")
    p.add_argument("--slm", action="store_true", default=False,
                   help="Enable the Simple Land Model (port of gSAM SRC/SLM). "
                        "Reads static binaries from --slm-data-root, initialises "
                        "soil/canopy/snow state, and blends land/ocean surface "
                        "fluxes cell-by-cell into the SurfaceFluxes handed to "
                        "the SGS module.  Required for IRMA debug500 parity.")
    p.add_argument("--slm-data-root", type=str,
                   default="/glade/u/home/sabramian/gSAM1.8.7/GLOBAL_DATA/BIN_D",
                   help="Directory holding landtype/soil/lai/landmask/soil_init/"
                        "snow/snowt binary files (lat_720_dyvar layout).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_zi(z: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Reconstruct nz+1 interface heights from cell centres and thicknesses."""
    zi = np.empty(len(z) + 1)
    zi[0]  = z[0] - 0.5 * dz[0]
    zi[1:] = zi[0] + np.cumsum(dz)
    return zi



def _prescribed_rad_forcing(z: np.ndarray) -> np.ndarray:
    """
    Tropical prescribed radiative cooling profile (K/s).

    -1.5 K/day in the troposphere (z < 15 km), tapered to 0 above.
    Standard CRM approach for cases without interactive radiation.
    """
    rate_trop = -1.5 / 86400.0          # K/s
    taper_base = 12_000.0               # m
    taper_top  = 15_000.0              # m
    taper = np.clip((taper_top - z) / (taper_top - taper_base), 0.0, 1.0)
    return (rate_trop * taper).astype(np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    jax.config.update("jax_enable_x64", not args.float32)

    print(f"jsam IRMA driver")
    print(f"  nsteps={args.nsteps}  dt={DT}s  output every {args.output_interval} steps")
    print(f"  out_dir={args.out_dir}")
    if args.no_polar_filter:
        print(f"  Spectral polar filter: OFF (--no-polar-filter; matches gSAM)")
    if args.sponge_tau == 0.0:
        print(f"  TABS sponge: OFF (--sponge-tau 0; matches gSAM)")
    elif args.sponge_tau != 10.0:
        print(f"  TABS sponge tau: {args.sponge_tau} s")
    print()

    # ── 1. Load IRMA grid from gSAM reference ──────────────────────────────
    print("Loading IRMA grid from gSAM reference...")
    from jsam.utils.IRMALoader import IRMALoader
    loader = IRMALoader()
    g = loader.grid
    z   = g["z"]
    dz  = g["dz"]
    zi  = _build_zi(z, dz)

    if args.dlat is not None:
        # Coarsened horizontal grid — ERA5 init interpolates to whatever lat/lon we pass.
        # NOTE: this is UNIFORM spacing, NOT the gSAM lat_720_dyvar non-uniform grid.
        # Only use for dev/debugging at resolutions coarser than native.  Do NOT pass
        # --dlat 0.25 expecting to reproduce gSAM IRMA native — that would give a uniform
        # 0.25° grid instead of the dyvar grid the gSAM IRMA inputs were built on.
        dlat = args.dlat
        dlon = args.dlat  # keep isotropic
        lat_max = args.lat_max
        lat = np.arange(-lat_max + dlat / 2, lat_max, dlat)
        lon = np.arange(0.0, 360.0, dlon)
        print(f"  Grid (coarsened {dlat}°, lat ±{lat_max}°, UNIFORM): "
              f"{len(lat)} lat × {len(lon)} lon × {len(z)} z")
        if dlat <= 0.3:
            print(f"  WARNING: uniform {dlat}° lat grid ≠ gSAM lat_720_dyvar. "
                  f"For native-resolution IRMA, omit --dlat.")
    else:
        # Native IRMA grid: lat comes from gSAM IRMA 10-day NC output, which was run
        # with latlonfile = ./GRIDS/lat_720_dyvar (non-uniform, -89.4° to +89.4°).
        # This MUST stay consistent with the gSAM setup — the ERA5 input binaries and
        # the advection/pressure metric factors (ady(j) in gSAM kurant.f90) are built
        # against the dyvar grid.  Using a uniform lat here would reproduce the Apr 12
        # 2026 crash: CFL blowup at step 1, pressure solver fails to converge, NaN at
        # step 2.  See feedback_gsam_latgrid_match memory.
        lat = g["lat"]
        lon = g["lon"]
        _dlat = np.diff(lat)
        assert not np.allclose(_dlat, _dlat[0], rtol=1e-3), (
            "IRMALoader returned a uniformly-spaced lat grid — expected gSAM lat_720_dyvar "
            "(non-uniform). Check /glade/derecho/scratch/sabramian/gSAM_IRMA_10days/OUT_NC."
        )
        assert abs(lat[0] + 89.4) < 1e-3 and abs(lat[-1] - 89.4) < 1e-3, (
            f"Expected lat_720_dyvar range ±89.4°, got [{lat[0]:.3f}, {lat[-1]:.3f}]"
        )
        print(f"  Grid (gSAM lat_720_dyvar): {len(lat)} lat × {len(lon)} lon × {len(z)} z  "
              f"(lat ∈ [{lat[0]:.2f}, {lat[-1]:.2f}], dlat ∈ [{_dlat.min():.3f}, {_dlat.max():.3f}])")

    # ── 2. ERA5 initialisation ─────────────────────────────────────────────
    print("Running era5_init (this reads ERA5 from RDA — takes ~1 min)...")
    t0_io = time.perf_counter()

    from jsam.io.era5 import era5_init
    ls_times = [START_TIME + timedelta(hours=h)
                for h in range(args.ls_forcing_hours + 1)]

    out = era5_init(lat, lon, z, zi, START_TIME,
                    ls_forcing_times=ls_times,
                    rda_root=RDA_ROOT,
                    polar_filter=not args.no_polar_filter)

    grid   = out["grid"]
    metric = out["metric"]
    state  = out["state"]
    sst    = out["sst"]
    ls_forcing = out["ls_forcing"]

    print(f"  ERA5 init done in {time.perf_counter()-t0_io:.1f}s")

    # Downcast state + metric to float32 for GPU memory (--float32 flag)
    if args.float32:
        _f32 = jnp.float32
        state = ModelState(
            U=state.U.astype(_f32), V=state.V.astype(_f32), W=state.W.astype(_f32),
            TABS=state.TABS.astype(_f32), QV=state.QV.astype(_f32),
            QC=state.QC.astype(_f32), QI=state.QI.astype(_f32),
            QR=state.QR.astype(_f32), QS=state.QS.astype(_f32),
            QG=state.QG.astype(_f32), TKE=state.TKE.astype(_f32),
            p_prev=state.p_prev.astype(_f32),
            nstep=state.nstep, time=state.time,
        )
        for k, v in metric.items():
            if hasattr(v, 'dtype') and v.dtype == jnp.float64:
                metric[k] = v.astype(_f32)
        print(f"  Downcast state+metric to float32")

    # ── 3. Physics forcing ─────────────────────────────────────────────────
    from jsam.core.physics.radiation import RadForcing
    from jsam.core.step import PhysicsForcing

    # ── Radiation scheme selection ────────────────────────────────────────
    rad_forcing = None
    rad_rrtmg_cfg = None
    o3vmr_rrtmg = None
    if args.rad == "prescribed":
        qrad_profile = _prescribed_rad_forcing(z)
        rad_forcing  = RadForcing.constant(
            qrad_profile=jnp.array(qrad_profile),
            z_prof=jnp.array(z),
        )
        print(f"  Radiation: prescribed tropical cooling (-1.5 K/d in trop)")
    elif args.rad == "rrtmg":
        from jsam.core.physics.rad_rrtmg import (
            RadRRTMGConfig, GSAMOzoneClimo, build_o3vmr_for_metric,
        )
        rad_rrtmg_cfg = RadRRTMGConfig(co2_vmr=args.co2 * 1e-6)
        # Build (ncol,nz) ozone climatology on the jsam grid (latitude-varying)
        metric_with_ll = dict(metric)
        metric_with_ll["lat_deg"] = np.asarray(lat)
        metric_with_ll["lon_deg"] = np.asarray(lon)
        if args.o3file:
            print(f"  Loading gSAM o3file: {args.o3file}")
            climo = GSAMOzoneClimo.from_file(args.o3file)
            o3_np = build_o3vmr_for_metric(metric_with_ll, climo)
        else:
            o3_np = build_o3vmr_for_metric(metric_with_ll, None)
            print(f"  RRTMG ozone: analytic Gaussian fallback")
        o3vmr_rrtmg = jnp.array(o3_np, dtype=jnp.float32 if args.float32 else jnp.float64)
        print(f"  Radiation: RRTMG_LW  (nrad={args.nrad} steps, "
              f"CO2={args.co2:.1f} ppmv, o3 shape={o3_np.shape})")
    else:
        print(f"  Radiation: off")

    # Buoyancy reference: cos(lat)-weighted horizontal mean of initial state.
    # gSAM's diagnose() recomputes tabs0(k) = area-weighted horiz mean of TABS
    # every timestep.  We initialise tabs0 the same way, and step() updates it
    # each step via the diagnose block (step 14).
    cos_lat = np.cos(np.deg2rad(lat))           # (ny,)
    wgt = cos_lat / cos_lat.sum()               # normalised area weights
    tabs0 = jnp.array(
        np.sum(np.mean(np.array(state.TABS), axis=2) * wgt[None, :], axis=1)
    )  # (nz,)
    qv0 = jnp.array(
        np.sum(np.mean(np.array(state.QV), axis=2) * wgt[None, :], axis=1)
    )  # (nz,)

    # Frozen t=0 reference profiles for scalar nudging (gSAM nudging.f90).
    # These are identical to tabs0/qv0 at init, but tabs0/qv0 are rolling
    # (updated each step by diagnose), whereas tabs_ref/qv_ref are frozen.
    tabs_ref = tabs0 if args.nudge_strato else None
    qv_ref   = qv0   if args.nudge_strato else None
    if args.nudge_strato:
        print(f"  Stratospheric nudging: TABS+QV → ERA5 init profile, "
              f"band [{args.nudge_z1/1000:.1f}, {args.nudge_z2/1000:.1f}] km, "
              f"tau={args.nudge_tau:.0f}s")

    # ── Optional: build SLMStatic + initial SLMState ──────────────────────
    slm_static = None
    slm_state  = None
    slm_params = None
    if args.slm:
        from jsam.io.slm_init import build_slm_static_and_state
        from jsam.core.physics.slm import SLMParams
        print(f"  SLM: reading binaries from {args.slm_data_root}")
        slm_params = SLMParams()
        slm_static, slm_state = build_slm_static_and_state(
            grid=grid, metric=metric, state=state,
            date_month=START_TIME.month,
            data_root=args.slm_data_root,
            params=slm_params,
        )
        print(
            f"  SLM: landmask={int(slm_static.landmask.sum())} land cells, "
            f"LAI range [{float(slm_static.LAI.min()):.2f}, "
            f"{float(slm_static.LAI.max()):.2f}], "
            f"soilt[0] range [{float(slm_state.soilt[0].min()):.1f}, "
            f"{float(slm_state.soilt[0].max()):.1f}] K"
        )

    # gSAM IRMA prm: dolargescale = .false. — no large-scale forcing applied
    forcing = PhysicsForcing(
        tabs0=tabs0,
        qv0=qv0,
        tabs_ref=tabs_ref,
        qv_ref=qv_ref,
        rad_forcing=rad_forcing,
        ls_forcing=None,   # matches gSAM dolargescale=.false.
        sst=sst,
        o3vmr_rrtmg=o3vmr_rrtmg,
        slm_static=slm_static,
        slm_state=slm_state,
        slm_rad=None,       # will be populated each step from RRTMG + zenith
        precip_ref=None,
    )

    # ── 4. Step configuration ──────────────────────────────────────────────
    from jsam.core.step import StepConfig, step
    from jsam.core.physics.sgs import SGSParams
    from jsam.core.physics.microphysics import MicroParams
    from jsam.core.physics.surface import BulkParams
    from jsam.core.physics.nudging import NudgingParams

    _nudging = (
        NudgingParams(z1_m=args.nudge_z1, z2_m=args.nudge_z2,
                      tau_s=args.nudge_tau,
                      nudge_tabs=True, nudge_qv=True)
        if args.nudge_strato else None
    )

    config = StepConfig(
        sgs_params=SGSParams(),
        micro_params=MicroParams(),
        bulk_params=BulkParams(),
        nudging_params=_nudging,
        rad_rrtmg=rad_rrtmg_cfg,
        nrad=args.nrad,
        slm_params=slm_params,
        g=9.81,
        epsv=0.61,
        damping_u_cu=0.25,    # matches gSAM IRMA prm (doglobalpresets)
        damping_w_cu=0.3,     # matches gSAM IRMA prm (doglobalpresets)
        sponge_tau=args.sponge_tau,  # 0 for oracle comparison; 10s for production
        polar_avg_rows=1,     # zonally average j=0 and j=ny-1 (pole convergence)
    )

    # ── 5. AB3 bootstrap state (3-level rotating tendency buffers) ─────────
    from jsam.core.dynamics.timestepping import MomentumTendencies, Tendencies
    nz, ny, nx = len(z), len(lat), len(lon)
    mom_tends_nm1 = MomentumTendencies.zeros(nz, ny, nx)
    mom_tends_nm2 = MomentumTendencies.zeros(nz, ny, nx)
    tends_nm1     = Tendencies.zeros(nz, ny, nx)
    tends_nm2     = Tendencies.zeros(nz, ny, nx)

    # ── 5b. Optional: enable per-stage oracle dump ────────────────────────
    if args.debug_dump_dir:
        from jsam.core import debug_dump as _dd
        _dd.DUMPER = _dd.DebugDumper(
            debug_dir=args.debug_dump_dir,
            lat=np.asarray(lat),
            lon=np.asarray(lon),
            z_len=len(z),
        )
        print(f"  Stage dumper enabled → {args.debug_dump_dir}")

    # ── 6. Output directory ────────────────────────────────────────────────
    from jsam.io.writer import write_3d_atm
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write t=0 snapshot
    sim_time = START_TIME
    write_3d_atm(state, grid, metric, sim_time, out_dir, casename=args.casename)
    print(f"  Wrote t=0 snapshot")

    # ── 7. Time loop ───────────────────────────────────────────────────────
    print(f"\nStarting time loop: {args.nsteps} steps × {DT}s = "
          f"{args.nsteps * DT / 3600:.2f} h of simulation")

    loop_start = time.perf_counter()
    step_times = []

    # Kurant state — previous-step dt(s) for variable-dt AB3 coefficients.
    # Both None on the very first call so that the Euler → AB2 → AB3
    # bootstrap happens automatically inside ab_coefs.
    from jsam.core.dynamics.kurant import compute_cfl

    _dt_dtype    = jnp.float32 if args.float32 else jnp.float64
    dt_prev_jnp  = None
    dt_pprev_jnp = None
    total_substeps = 0

    for i in range(1, args.nsteps + 1):
        t_step = time.perf_counter()
        target_time = i * DT
        substeps_this = 0
        cfl_peak_this = 0.0
        dtn_min_this  = DT

        # Subcycle until we reach the next 10-s output tick.
        # Worst case: NCYCLE_MAX substeps of DT/NCYCLE_MAX each.
        while True:
            _cur_time = float(state.time)
            if _cur_time >= target_time - 1e-6:
                break

            remaining = target_time - _cur_time

            # Global-max advective CFL at dt=DT given current state.
            cfl_adv_j = compute_cfl(state.U, state.V, state.W, metric, DT)
            cfl_adv   = float(cfl_adv_j)
            cfl_peak_this = max(cfl_peak_this, cfl_adv)

            # gSAM formula: dtn = min(dt, dt * cfl_max / (cfl_adv + eps))
            dtn_py = min(DT, DT * CFL_MAX / (cfl_adv + 1e-10))
            dtn_py = min(dtn_py, remaining)
            dtn_min_this = min(dtn_min_this, dtn_py)

            # Pass dt and dt_prev as JAX scalars so sub-JITs see them as
            # dynamic (traced) inputs — no recompilation on each new dtn.
            dtn_j = jnp.asarray(dtn_py, dtype=_dt_dtype)
            dt_prev_arg  = dt_prev_jnp    # None on the very first call
            dt_pprev_arg = dt_pprev_jnp   # None on the first two calls

            # ── SLM per-step radiative / precip forcing ──────────────────
            if args.slm:
                from jsam.io.slm_forcing import build_slm_rad_inputs
                from jsam.core.physics.rad_rrtmg import (
                    compute_qrad_and_lwds_rrtmg,
                )
                from jsam.core.physics.microphysics import micro_proc_with_precip

                _cur_dt_from_start = float(state.time)
                _cur_sim_time = START_TIME + timedelta(seconds=_cur_dt_from_start)

                if forcing.sst is not None and config.rad_rrtmg is not None:
                    _, _lwds = compute_qrad_and_lwds_rrtmg(
                        state, metric, config.rad_rrtmg,
                        forcing.sst,
                        o3vmr=(None if forcing.o3vmr_rrtmg is None
                               else jnp.asarray(forcing.o3vmr_rrtmg)),
                    )
                else:
                    _lwds = jnp.full_like(state.TABS[0], 350.0)

                _lat_rad = jnp.asarray(grid.lat * (np.pi / 180.0))
                _lon_rad = jnp.asarray(grid.lon * (np.pi / 180.0))
                _slm_rad_inputs = build_slm_rad_inputs(
                    _cur_sim_time, _lat_rad, _lon_rad, _lwds,
                )
                forcing = PhysicsForcing(
                    tabs0=forcing.tabs0, qv0=forcing.qv0,
                    tabs_ref=forcing.tabs_ref, qv_ref=forcing.qv_ref,
                    rad_forcing=forcing.rad_forcing,
                    ls_forcing=forcing.ls_forcing,
                    sst=forcing.sst, o3vmr_rrtmg=forcing.o3vmr_rrtmg,
                    slm_static=forcing.slm_static,
                    slm_state=forcing.slm_state,
                    slm_rad=_slm_rad_inputs,
                    precip_ref=(forcing.precip_ref
                                if forcing.precip_ref is not None
                                else jnp.zeros_like(state.TABS[0])),
                )

            state, mom_tends_n, tends_n, forcing = step(
                state, mom_tends_nm1, mom_tends_nm2, tends_nm1, tends_nm2,
                metric, grid, dtn_j, config, forcing,
                dt_prev=dt_prev_arg,
                dt_pprev=dt_pprev_arg,
            )

            # Rotate AB3 tendency buffers: nm2 ← nm1, nm1 ← n
            mom_tends_nm2 = mom_tends_nm1
            mom_tends_nm1 = mom_tends_n
            tends_nm2     = tends_nm1
            tends_nm1     = tends_n

            # Rotate variable-dt buffer: pprev ← prev, prev ← curr
            dt_pprev_jnp = dt_prev_jnp
            dt_prev_jnp  = dtn_j
            substeps_this  += 1
            total_substeps += 1

            if substeps_this >= NCYCLE_MAX:
                # Forced exit — pin to target_time so we don't livelock.
                # Warn loudly; this shouldn't happen in a healthy run.
                print(f"  *** kurant: {NCYCLE_MAX} substeps exhausted at "
                      f"outer step {i} (cfl_peak={cfl_peak_this:.2f}, "
                      f"dtn_min={dtn_min_this:.3f}) — continuing anyway")
                break

        dt_wall = time.perf_counter() - t_step
        step_times.append(dt_wall)
        sim_time = START_TIME + timedelta(seconds=i * DT)

        # Per-step diagnostics — reduce on-device, pull only scalars to host
        _nan_interval = 1 if args.nsteps <= 100 else 10
        if i % _nan_interval == 0:
            _T_min = float(jnp.min(state.TABS)); _T_max = float(jnp.max(state.TABS))
            _W_min = float(jnp.min(state.W));    _W_max = float(jnp.max(state.W))
            _U_min = float(jnp.min(state.U));    _U_max = float(jnp.max(state.U))
            print(f"  [{i:3d}] TABS=[{_T_min:.1f}, {_T_max:.1f}]  "
                  f"W=[{_W_min:.2f}, {_W_max:.2f}]  "
                  f"U=[{_U_min:.1f}, {_U_max:.1f}]  "
                  f"cfl={cfl_peak_this:.2f} nsub={substeps_this} "
                  f"dtn={dtn_min_this:.2f}s  dt_wall={dt_wall:.1f}s")

            if not (np.isfinite(_T_min) and np.isfinite(_W_min) and np.isfinite(_U_min)):
                print(f"  *** NaN/Inf detected in TABS/W/U at step {i} "
                      f"(sim {sim_time.strftime('%H:%M:%S')})")
                break

        if i % args.output_interval == 0:
            write_3d_atm(state, grid, metric, sim_time, out_dir, casename=args.casename)
            elapsed = time.perf_counter() - loop_start
            sdpd = (i * DT / 86400.0) / (elapsed / 86400.0)
            print(f"  step {i:6d}  sim={sim_time.strftime('%H:%M:%S')}  "
                  f"wall={elapsed:.0f}s  SDPD={sdpd:.4f}  "
                  f"step_mean={np.mean(step_times[-args.output_interval:]):.2f}s")

    # ── 8. Summary ─────────────────────────────────────────────────────────
    total_wall = time.perf_counter() - loop_start
    sim_seconds = args.nsteps * DT
    sdpd_final = (sim_seconds / 86400.0) / (total_wall / 86400.0)

    print(f"\n=== Run complete ===")
    print(f"  Simulated:   {sim_seconds/3600:.2f} h")
    print(f"  Wall time:   {total_wall:.1f} s  ({total_wall/3600:.2f} h)")
    print(f"  SDPD:        {sdpd_final:.4f}")
    print(f"  Mean step:   {np.mean(step_times):.2f} s/step")
    print(f"  Output dir:  {out_dir}")

    if args.debug_dump_dir:
        from jsam.core import debug_dump as _dd
        if _dd.DUMPER is not None:
            _dd.DUMPER.finalize()
            _dd.DUMPER = None


if __name__ == "__main__":
    main()
