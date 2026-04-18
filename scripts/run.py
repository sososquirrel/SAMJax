"""
General jsam driver — works with any gSAM binary initialisation.

Initialises from gSAM binary files, runs step() in a Python for-loop,
and writes 3D_atm snapshots every --output-interval steps.

Typical usage
-------------
# Short test (1 hour, output every 30 min):
python scripts/run.py --nsteps 360 --output-interval 180 \
    --gsam-root /glade/u/home/sabramian/gSAM1.8.7 \
    --start-time 2017-09-05T00:00:00

# Full day:
python scripts/run.py --nsteps 8640 --output-interval 180 \
    --gsam-root /glade/u/home/sabramian/gSAM1.8.7 \
    --start-time 2017-09-05T00:00:00 \
    --out-dir /path/to/out

PBS: see run_irma_debug500_all_params.pbs
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from jsam.core.state import ModelState


def _parse_args():
    p = argparse.ArgumentParser(description="jsam general case driver")

    # ── Run identity ──────────────────────────────────────────────────────────
    p.add_argument("--gsam-root",         type=str,
                   default="/glade/u/home/sabramian/gSAM1.8.7",
                   help="Root directory of the gSAM installation (contains "
                        "GLOBAL_DATA/BIN_D and the binary init file).")
    p.add_argument("--start-time",        type=str,
                   default="2017-09-05T00:00:00",
                   help="Simulation start time as ISO-8601 (e.g. 2017-09-05T00:00:00).")
    p.add_argument("--dt",                type=float, default=10.0,
                   help="Model timestep in seconds (default 10).")
    p.add_argument("--cfl-max",           type=float, default=0.7,
                   help="CFL limiter for adaptive sub-stepping (default 0.7).")
    p.add_argument("--ncycle-max",        type=int,   default=4,
                   help="Maximum sub-steps per outer step (default 4).")

    # ── Time integration ──────────────────────────────────────────────────────
    p.add_argument("--nsteps",            type=int,   default=360,
                   help="Total number of steps to run (default 360 = 1 h at dt=10s).")
    p.add_argument("--output-interval",   type=int,   default=180,
                   help="Write 3D_atm every N steps (default 180 = 30 min at dt=10s).")

    # ── SST indexing ──────────────────────────────────────────────────────────
    p.add_argument("--sst-file-start",    type=str,   default=None,
                   help="Start datetime of the SST binary file (ISO-8601). "
                        "Defaults to 24 h before --start-time.")
    p.add_argument("--sst-interval-hours", type=int,  default=6,
                   help="Temporal spacing (hours) between SST snapshots in the "
                        "binary file (default 6).")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--out-dir",           type=str,
                   default="/glade/derecho/scratch/sabramian/jsam_out",
                   help="Output directory for 3D_atm files.")
    p.add_argument("--casename",          type=str,   default="jsam",
                   help="Output file prefix.")

    # ── Numerics ──────────────────────────────────────────────────────────────
    p.add_argument("--float32",           action="store_true", default=False,
                   help="Use float32 state arrays (default: float64). "
                        "Halves GPU memory; microphysics/pressure still compute "
                        "in float64 internally.")
    p.add_argument("--no-polar-filter",   action="store_true", default=False,
                   help="Disable the spectral polar filter (velocity + scalar). "
                        "Required for oracle-comparable debug runs (--debug-dump-dir).")
    p.add_argument("--sponge-tau",        type=float, default=10.0,
                   help="TABS Newtonian damping timescale (s) in the top sponge "
                        "layer (default 10 s). Set to 0 to match gSAM.")

    # ── Radiation ─────────────────────────────────────────────────────────────
    p.add_argument("--rad",               type=str,   default="prescribed",
                   choices=["prescribed", "rrtmg", "none"],
                   help="Radiation scheme: prescribed (default), rrtmg, or none.")
    p.add_argument("--nrad",              type=int,   default=90,
                   help="Call RRTMG every nrad steps (default 90 ≈ 15 min at dt=10s).")
    p.add_argument("--co2",               type=float, default=400.0,
                   help="CO2 mixing ratio in ppmv (default 400).")
    p.add_argument("--o3file",            type=str,   default=None,
                   help="gSAM o3file for RRTMG ozone climatology. "
                        "Defaults to <gsam-root>/GLOBAL_DATA/BIN_D/"
                        "ozone_era5_monthly_201709-201709_GLOBAL.bin.")
    p.add_argument("--rad-day0",          type=float, default=None,
                   help="Fractional calendar day-of-year (1-based, UT) at nstep=0 "
                        "for RRTMG_SW solar geometry (e.g. 248.0 for Sep 5). "
                        "Defaults to the day derived from --start-time.")
    p.add_argument("--rad-iyear",         type=int,   default=None,
                   help="Calendar year for RRTMG orbital geometry "
                        "(defaults to the year from --start-time).")

    # ── Stratospheric nudging ─────────────────────────────────────────────────
    p.add_argument("--nudge-strato",      action="store_true", default=False,
                   help="Enable scalar nudging (TABS+QV) toward the init profile "
                        "in the upper atmosphere.")
    p.add_argument("--nudge-z1",          type=float, default=15_000.0,
                   help="Nudging band bottom (m). Default 15 km.")
    p.add_argument("--nudge-z2",          type=float, default=60_000.0,
                   help="Nudging band top (m). Default 60 km.")
    p.add_argument("--nudge-tau",         type=float, default=3600.0,
                   help="Nudging relaxation timescale (s). Default 3600.")

    # ── Microphysics ──────────────────────────────────────────────────────────
    p.add_argument("--qci0",              type=float, default=1.0e-4,
                   help="Ice autoconversion threshold (kg/kg). "
                        "gSAM IRMA prm_debug500 uses 1e-5; jsam default is 1e-4.")

    # ── Simple Land Model ─────────────────────────────────────────────────────
    p.add_argument("--slm",               action="store_true", default=False,
                   help="Enable the Simple Land Model (port of gSAM SRC/SLM).")
    p.add_argument("--slm-data-root",     type=str,   default=None,
                   help="Directory holding landtype/soil/lai/landmask/soil_init/"
                        "snow/snowt binary files. "
                        "Defaults to <gsam-root>/GLOBAL_DATA/BIN_D.")

    # ── Debug ─────────────────────────────────────────────────────────────────
    p.add_argument("--debug-dump-dir",    type=str,   default=None,
                   help="Enable gSAM-oracle-compatible per-stage dumps "
                        "(U,V,W,TABS,QC,QV,QI) into this directory.")
    p.add_argument("--ls-forcing-hours",  type=int,   default=6,
                   help="Hours of ERA5 large-scale forcing to load (default 6, "
                        "currently unused — kept for CLI compatibility).")

    return p.parse_args()


def _prescribed_rad_forcing(z: np.ndarray) -> np.ndarray:
    """Tropical prescribed radiative cooling profile (K/s)."""
    rate_trop  = -1.5 / 86400.0
    taper_base = 12_000.0
    taper_top  = 15_000.0
    taper = np.clip((taper_top - z) / (taper_top - taper_base), 0.0, 1.0)
    return (rate_trop * taper).astype(np.float64)


def main():
    args = _parse_args()
    jax.config.update("jax_enable_x64", not args.float32)

    START_TIME = datetime.fromisoformat(args.start_time)
    DT         = args.dt
    CFL_MAX    = args.cfl_max
    NCYCLE_MAX = args.ncycle_max
    GSAM_ROOT  = args.gsam_root

    # Derived defaults
    o3file = args.o3file or (
        f"{GSAM_ROOT}/GLOBAL_DATA/BIN_D/"
        "ozone_era5_monthly_201709-201709_GLOBAL.bin"
    )
    slm_data_root = args.slm_data_root or f"{GSAM_ROOT}/GLOBAL_DATA/BIN_D"
    sst_file_start = (
        datetime.fromisoformat(args.sst_file_start)
        if args.sst_file_start
        else START_TIME - timedelta(hours=24)
    )

    print(f"jsam general driver")
    print(f"  start_time={START_TIME.isoformat()}  dt={DT}s")
    print(f"  nsteps={args.nsteps}  output every {args.output_interval} steps")
    print(f"  out_dir={args.out_dir}")
    print(f"  gsam_root={GSAM_ROOT}")
    if args.no_polar_filter:
        print(f"  Spectral polar filter: OFF (--no-polar-filter)")
    if args.sponge_tau == 0.0:
        print(f"  TABS sponge: OFF (--sponge-tau 0)")
    elif args.sponge_tau != 10.0:
        print(f"  TABS sponge tau: {args.sponge_tau} s")
    print()

    # ── 1. Load gSAM binary initial state ────────────────────────────────────
    print("Loading gSAM binary initial state ...")
    t0_io = time.perf_counter()

    from jsam.io.gsam_binary import load_gsam_init, build_gsam_grid
    from jsam.core.grid.latlon import LatLonGrid
    from jsam.core.dynamics.pressure import build_metric
    from jsam.core.dynamics.damping import build_polar_filter_masks

    raw = load_gsam_init(gsam_root=GSAM_ROOT, convert_omega=True)
    g      = raw["grid"]
    lat    = g["lat"]
    lon    = g["lon"]
    z      = g["z"]
    zi     = g["zi"]
    adz    = g["adz"]
    nzm    = g["nzm"]
    pres   = g["pres"]
    pres0  = g["pres0"]

    GGR = 9.81
    CP  = 1004.0
    RD  = 287.04

    tabs0_np = np.mean(np.array(raw["TABS"]), axis=(1, 2))
    presi = np.empty(nzm + 1, dtype=np.float64)
    presi[0] = pres0
    presr = (pres0 / 1000.0) ** (RD / CP)
    for k in range(nzm):
        prespot = (1000.0 / pres[k]) ** (RD / CP)
        t0k = tabs0_np[k] * prespot
        presr -= GGR / CP / t0k * (zi[k + 1] - zi[k])
        presi[k + 1] = 1000.0 * presr ** (CP / RD)
    rho = (presi[:-1] - presi[1:]) / (zi[1:] - zi[:-1]) / GGR * 100.0

    grid   = LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)
    metric = build_metric(grid, polar_filter=not args.no_polar_filter)
    _pf_masks = None
    if not args.no_polar_filter:
        _pf_masks = build_polar_filter_masks(np.deg2rad(lat), len(lon))
        print(f"  Spectral polar filter: ON")

    # ── 2. Build ModelState ───────────────────────────────────────────────────
    _arr = lambda a: jnp.array(a, dtype=jnp.float32 if args.float32 else jnp.float64)

    state = ModelState(
        U=_arr(raw["U"]),
        V=_arr(raw["V"]),
        W=_arr(raw["W"]),
        TABS=_arr(raw["TABS"]),
        QV=_arr(raw["QV"]),
        QC=_arr(raw["QC"]),
        QI=_arr(raw["QI"]),
        QR=_arr(raw["QR"]),
        QS=_arr(raw["QS"]),
        QG=jnp.zeros(raw["TABS"].shape, dtype=jnp.float32 if args.float32 else jnp.float64),
        TKE=jnp.zeros(raw["TABS"].shape, dtype=jnp.float32 if args.float32 else jnp.float64),
        p_prev=jnp.zeros(raw["TABS"].shape, dtype=jnp.float32 if args.float32 else jnp.float64),
        p_pprev=jnp.zeros(raw["TABS"].shape, dtype=jnp.float32 if args.float32 else jnp.float64),
        nstep=jnp.int32(0),
        time=jnp.float64(0.0),
    )

    _sst_idx = int((START_TIME - sst_file_start).total_seconds() / 3600 / args.sst_interval_hours)
    sst = _arr(raw["sst"][_sst_idx])

    print(f"  Binary init done in {time.perf_counter()-t0_io:.1f}s")
    print(f"  Grid: {len(lat)} lat × {len(lon)} lon × {nzm} z")

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

    # ── 3. Physics forcing ────────────────────────────────────────────────────
    from jsam.core.physics.radiation import RadForcing
    from jsam.core.step import PhysicsForcing

    rad_forcing    = None
    rad_rrtmg_cfg  = None
    o3vmr_rrtmg    = None

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
        metric_with_ll = dict(metric)
        metric_with_ll["lat_deg"] = np.asarray(lat)
        metric_with_ll["lon_deg"] = np.asarray(lon)
        if o3file:
            print(f"  Loading gSAM o3file: {o3file}")
            climo   = GSAMOzoneClimo.from_file(o3file)
            o3_np   = build_o3vmr_for_metric(metric_with_ll, climo)
        else:
            o3_np   = build_o3vmr_for_metric(metric_with_ll, None)
            print(f"  RRTMG ozone: analytic Gaussian fallback")
        o3vmr_rrtmg = jnp.array(o3_np, dtype=jnp.float32 if args.float32 else jnp.float64)
        print(f"  Radiation: RRTMG_LW  (nrad={args.nrad} steps, "
              f"CO2={args.co2:.1f} ppmv, o3 shape={o3_np.shape})")
    else:
        print(f"  Radiation: off")

    cos_lat = np.cos(np.deg2rad(lat))
    wgt     = cos_lat / cos_lat.sum()

    def _hmean_init(field3d_np):
        return np.sum(np.mean(field3d_np, axis=2) * wgt[None, :], axis=1)

    TABS_np = np.array(state.TABS)
    QV_np   = np.array(state.QV)
    QC_np   = np.array(state.QC)
    QI_np   = np.array(state.QI)
    QR_np   = np.array(state.QR)
    QS_np   = np.array(state.QS)
    QG_np   = np.array(state.QG)

    tabs0    = jnp.array(_hmean_init(TABS_np))
    qn0_init = _hmean_init(QC_np + QI_np)
    qp0_init = _hmean_init(QR_np + QS_np + QG_np)
    q0_init  = _hmean_init(QV_np + QC_np + QI_np)
    qv0      = jnp.array(q0_init - qn0_init)
    qn0      = jnp.array(qn0_init)
    qp0      = jnp.array(qp0_init)

    tabs_ref = tabs0 if args.nudge_strato else None
    qv_ref   = qv0   if args.nudge_strato else None
    if args.nudge_strato:
        print(f"  Stratospheric nudging: TABS+QV → init profile, "
              f"band [{args.nudge_z1/1000:.1f}, {args.nudge_z2/1000:.1f}] km, "
              f"tau={args.nudge_tau:.0f}s")

    slm_static = None
    slm_state  = None
    slm_params = None
    if args.slm:
        from jsam.io.slm_init import build_slm_static_and_state
        from jsam.core.physics.slm import SLMParams
        print(f"  SLM: reading binaries from {slm_data_root}")
        slm_params = SLMParams()
        slm_static, slm_state = build_slm_static_and_state(
            grid=grid, metric=metric, state=state,
            date_month=START_TIME.month,
            data_root=slm_data_root,
            params=slm_params,
        )
        print(
            f"  SLM: landmask={int(slm_static.landmask.sum())} land cells, "
            f"LAI range [{float(slm_static.LAI.min()):.2f}, "
            f"{float(slm_static.LAI.max()):.2f}], "
            f"soilt[0] range [{float(slm_state.soilt[0].min()):.1f}, "
            f"{float(slm_state.soilt[0].max()):.1f}] K"
        )

    forcing = PhysicsForcing(
        tabs0=tabs0,
        qv0=qv0,
        qn0=qn0,
        qp0=qp0,
        tabs_ref=tabs_ref,
        qv_ref=qv_ref,
        rad_forcing=rad_forcing,
        ls_forcing=None,
        sst=sst,
        o3vmr_rrtmg=o3vmr_rrtmg,
        slm_static=slm_static,
        slm_state=slm_state,
        slm_rad=None,
        precip_ref=None,
    )

    # ── 4. Step configuration ─────────────────────────────────────────────────
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

    _day0_frac_auto = (
        START_TIME.timetuple().tm_yday
        + (START_TIME.hour * 3600 + START_TIME.minute * 60 + START_TIME.second) / 86400.0
    )
    _rad_day0  = args.rad_day0  if args.rad_day0  is not None else (_day0_frac_auto if args.rad == "rrtmg" else None)
    _rad_iyear = args.rad_iyear if args.rad_iyear is not None else START_TIME.year

    config = StepConfig(
        sgs_params=SGSParams(),
        micro_params=MicroParams(qci0=args.qci0),
        bulk_params=BulkParams(),
        nudging_params=_nudging,
        rad_rrtmg=rad_rrtmg_cfg,
        nrad=args.nrad,
        rad_day0=_rad_day0,
        rad_iyear=_rad_iyear,
        slm_params=slm_params,
        g=9.81,
        epsv=0.61,
        damping_u_cu=0.25,
        damping_w_cu=0.3,
        sponge_tau=args.sponge_tau,
        polar_avg_rows=1,
    )

    # ── 5. AB3 bootstrap state ────────────────────────────────────────────────
    from jsam.core.dynamics.timestepping import MomentumTendencies, Tendencies
    nz, ny, nx = len(z), len(lat), len(lon)
    mom_tends_nm1 = MomentumTendencies.zeros(nz, ny, nx)
    mom_tends_nm2 = MomentumTendencies.zeros(nz, ny, nx)
    tends_nm1     = Tendencies.zeros(nz, ny, nx)
    tends_nm2     = Tendencies.zeros(nz, ny, nx)

    if args.debug_dump_dir:
        from jsam.core import debug_dump as _dd
        _dd.DUMPER = _dd.DebugDumper(
            debug_dir=args.debug_dump_dir,
            lat=np.asarray(lat),
            lon=np.asarray(lon),
            z_len=len(z),
        )
        print(f"  Stage dumper enabled → {args.debug_dump_dir}")
        # Dump initial state as nstep=0 so init can be compared against gSAM
        # oracle nstep=1/stage=pre_step (gSAM's pre-loop state).
        _dd.DUMPER.dump(state, stage_id=0, dtn=0.0, force_nstep=0)
        print(f"  Stage dumper: wrote init dump (nstep=0/pre_step)")

    # ── 6. Output directory ───────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 7. Time loop ──────────────────────────────────────────────────────────
    print(f"\nStarting time loop: {args.nsteps} steps × {DT}s = "
          f"{args.nsteps * DT / 3600:.2f} h of simulation")

    loop_start  = time.perf_counter()
    step_times  = []

    from jsam.core.dynamics.kurant import compute_cfl
    from jsam.io.writer import write_3d_atm

    _dt_dtype    = jnp.float32 if args.float32 else jnp.float64
    dt_prev_jnp  = None
    dt_pprev_jnp = None
    total_substeps = 0

    # Pre-compute SLM grid constants (lat/lon are fixed; import once outside loop)
    _build_slm_rad = None
    _slm_lat_rad   = None
    _slm_lon_rad   = None
    if args.slm:
        from jsam.io.slm_forcing import build_slm_rad_inputs as _build_slm_rad
        _slm_lat_rad = jnp.asarray(grid.lat * (np.pi / 180.0))
        _slm_lon_rad = jnp.asarray(grid.lon * (np.pi / 180.0))

    # Track simulation time in Python to avoid a GPU sync (float(state.time))
    # on every inner substep iteration.  _sim_time_py stays in lock-step with
    # state.time because dtn_j = jnp.asarray(dtn_py) and step() increments
    # state.time by exactly dtn_j.
    _sim_time_py = 0.0

    for i in range(1, args.nsteps + 1):
        t_step = time.perf_counter()
        target_time      = i * DT
        substeps_this    = 0
        cfl_peak_this    = 0.0
        dtn_min_this     = DT

        # Build SLM radiation inputs once per outer step.  Solar geometry
        # changes on hourly timescales; recomputing every 10 s substep is
        # wasted work and costs two extra GPU syncs per substep.
        if _build_slm_rad is not None:
            _cur_sim_time = START_TIME + timedelta(seconds=_sim_time_py)
            _lwds = (
                forcing.slm_rad.lwds
                if forcing.slm_rad is not None and hasattr(forcing.slm_rad, 'lwds')
                else jnp.full_like(state.TABS[0], 350.0)
            )
            _slm_rad_inputs = _build_slm_rad(
                _cur_sim_time, _slm_lat_rad, _slm_lon_rad, _lwds,
            )
            forcing = PhysicsForcing(
                tabs0=forcing.tabs0, qv0=forcing.qv0,
                qn0=forcing.qn0, qp0=forcing.qp0,
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

        while _sim_time_py < target_time - 1e-6:
            remaining = target_time - _sim_time_py

            cfl_adv_j = compute_cfl(state.U, state.V, state.W, metric, DT)
            cfl_adv   = float(cfl_adv_j)
            cfl_peak_this = max(cfl_peak_this, cfl_adv)

            dtn_py = min(DT, DT * CFL_MAX / (cfl_adv + 1e-10), remaining)
            dtn_min_this = min(dtn_min_this, dtn_py)

            dtn_j        = jnp.asarray(dtn_py, dtype=_dt_dtype)
            dt_prev_arg  = dt_prev_jnp
            dt_pprev_arg = dt_pprev_jnp

            state, mom_tends_n, tends_n, forcing = step(
                state, mom_tends_nm1, mom_tends_nm2, tends_nm1, tends_nm2,
                metric, grid, dtn_j, config, forcing,
                dt_prev=dt_prev_arg,
                dt_pprev=dt_pprev_arg,
                dump_nstep=i,
                polar_filter_masks=_pf_masks,
            )

            mom_tends_nm2 = mom_tends_nm1
            mom_tends_nm1 = mom_tends_n
            tends_nm2     = tends_nm1
            tends_nm1     = tends_n

            dt_pprev_jnp  = dt_prev_jnp
            dt_prev_jnp   = dtn_j
            _sim_time_py  += dtn_py
            substeps_this  += 1
            total_substeps += 1

            if substeps_this >= NCYCLE_MAX:
                print(f"  *** kurant: {NCYCLE_MAX} substeps exhausted at "
                      f"outer step {i} (cfl_peak={cfl_peak_this:.2f}, "
                      f"dtn_min={dtn_min_this:.3f}) — continuing anyway")
                break

        dt_wall = time.perf_counter() - t_step
        step_times.append(dt_wall)
        sim_time = START_TIME + timedelta(seconds=i * DT)

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

    # ── 8. Summary ────────────────────────────────────────────────────────────
    total_wall  = time.perf_counter() - loop_start
    sim_seconds = args.nsteps * DT
    sdpd_final  = (sim_seconds / 86400.0) / (total_wall / 86400.0)

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
