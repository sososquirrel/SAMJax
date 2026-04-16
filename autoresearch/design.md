# Design — gSAM (Fortran) vs jsam (JAX)

## gSAM — Oracle (Fortran, MPI)

**Root**: `/glade/u/home/sabramian/gSAM1.8.7/SRC/`

| File | Role |
|------|------|
| `main.f90` | Time loop, calls 19 stages per step, calls `dump_stage` |
| `vars.f90` | All prognostic arrays: u, v, w, t, qv, qcl, qci, qpl, qpi, ... |
| `params.f90` | Physical constants, grid params |
| `consts.f90` | Derived constants (gamaz, fac_cond, fac_sub, ...) |
| `grid.f90` / `setgrid.f90` | Grid setup (lat-lon, dx/dy/dz, adz/adzw) |
| `init.f90` / `setdata.f90` | IC/BC initialization |
| `forcing.f90` | Large-scale forcing |
| `nudging.f90` | Nudging to reanalysis |
| `buoyancy.f90` | Buoyancy term for w |
| `radiation.f90` / `RAD_RRTM/` | Radiation (RRTMG wrapper) |
| `surface.f90` / `oceflx.f90` / `SLM/` | Surface fluxes, land model |
| `advect_mom.f90` / `ADV_UM5/` | Momentum advection (5th-order upwind) |
| `coriolis.f90` | Coriolis + metric terms |
| `SGS_TKE/` | SGS TKE closure |
| `adamsA.f90` / `adamsB.f90` | Adams-Bashforth time integration |
| `damping.f90` | Sponge layer damping |
| `pressure.f90` / `press_rhs.f90` / `press_grad.f90` | Pressure solver (Poisson) |
| `advect_all_scalars.f90` / `ADV_MPDATA/` | Scalar advection (MPDATA/FCT) |
| `MICRO_SAM1MOM/` | Microphysics (1-moment) |
| `upperbound.f90` | Upper boundary condition |
| `diagnose.f90` | Diagnostic variables (TABS from t) |
| `debug_dump.f90` | Dumps oracle tensors |

## jsam — JAX Port (Python/JAX, single-GPU)

**Root**: `/glade/u/home/sabramian/SAMJax/jsam/`

| File | Maps to (gSAM) |
|------|-----------------|
| `core/step.py` | `main.f90` — main time step, calls 19 stages |
| `core/state.py` | `vars.f90` — State dataclass (u, v, w, t, qv, qcl, ...) |
| `core/grid/base.py` | `grid.f90` — Grid base class |
| `core/grid/latlon.py` | `setgrid.f90` — Lat-lon grid impl |
| `core/dynamics/timestepping.py` | `adamsA.f90`, `adamsB.f90` |
| `core/dynamics/advection.py` | `advect_mom.f90`, `advect_all_scalars.f90` |
| `core/dynamics/coriolis.py` | `coriolis.f90` |
| `core/dynamics/damping.py` | `damping.f90` |
| `core/dynamics/pressure.py` | `pressure.f90`, `press_rhs.f90`, `press_grad.f90` |
| `core/dynamics/kurant.py` | `kurant.f90` |
| `core/dynamics/polar_filter.py` | (no gSAM equivalent — polar filter) |
| `core/physics/lsforcing.py` | `forcing.f90` |
| `core/physics/nudging.py` | `nudging.f90` |
| `core/physics/radiation.py` | `radiation.f90` |
| `core/physics/rad_rrtmg.py` | `RAD_RRTM/` |
| `core/physics/surface.py` | `surface.f90`, `oceflx.f90` |
| `core/physics/sgs.py` | `SGS_TKE/` |
| `core/physics/microphysics.py` | `MICRO_SAM1MOM/` |
| `core/physics/slm/` | `SLM/` |
| `core/debug_dump.py` | `debug_dump.f90` |
| `io/era5.py` | ERA5 initialization |
| `io/gsam_binary.py` | Read gSAM binary restart |

## 19 Stages (identical order in both)

```
 0 pre_step       7 coriolis      14 advect_scalars
 1 forcing        8 sgs_proc      15 sgs_scalars
 2 nudging        9 sgs_mom       16 upperbound
 3 buoyancy      10 adamsA        17 micro
 4 radiation     11 damping       18 diagnose
 5 surface       12 adamsB
 6 advect_mom    13 pressure
```

## Oracle Tensors

See `.claude/Oracle_Tensor_Structure.md` for full schema. Key points:
- 7 fields per stage: U, V, W, TABS, QC, QV, QI
- `globals.csv`: rank-0 min/max/mean per (step, stage) — 9538 rows
- `rank_NNNNNN.bin`: per-rank IRMA-box 3D data (float32)
- QC/QV/QI only valid at stages 17-18
- Stage 12 W spike is intentional
