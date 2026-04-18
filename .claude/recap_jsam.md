# jSAM Architecture Recap

## Oracle Rule
jSAM is a **JAX port of gSAM** (Fortran). 

REALLY IMPORTANT : Fortran source at `/glade/u/home/sabramian/gSAM1.8.7/SRC/` is the oracle for all behavior questions. Exact numerical output matching is required. **Never modify Fortran source — only jsam code.**

## Directory Map

```
/glade/u/home/sabramian/SAMJax/
├── jsam/
│   ├── core/
│   │   ├── state.py          — ModelState pytree (U,V,W,TABS,QV,QC,QI,QR,QS,QG,TKE,...)
│   │   ├── step.py           — single timestep driver, JIT-compiles physics modules
│   │   ├── dynamics/
│   │   │   ├── timestepping.py  — scalar/momentum advance, Tendencies structs
│   │   │   ├── pressure.py      — pressure solver + Adams-Bashforth correction
│   │   │   ├── advection.py     — scalar & momentum advection
│   │   │   ├── damping.py       — top sponge, pole damping, w-sponge
│   │   │   ├── coriolis.py      — Coriolis forcing
│   │   │   ├── kurant.py        — Courant diagnostics
│   │   │   └── polar_filter.py  — polar filtering
│   │   ├── physics/
│   │   │   ├── microphysics.py  — cloud microphysics (MicroParams, qc/qi/qr/qs/qg)
│   │   │   ├── sgs.py           — SGS turbulence (SGSParams, diffusion, stress)
│   │   │   ├── surface.py       — bulk surface fluxes (BulkParams)
│   │   │   ├── radiation.py     — radiation forcing (RadForcing)
│   │   │   ├── rad_rrtmg.py     — RRTMG scheme (RadRRTMGConfig)
│   │   │   ├── lsforcing.py     — large-scale forcing
│   │   │   ├── nudging.py       — nudging/relaxation
│   │   │   └── slm/             — Soil-Land-Model (state, params, run_slm, soil_proc, ...)
│   │   └── grid/
│   │       ├── base.py          — base grid class
│   │       └── latlon.py        — lat-lon grid
│   ├── io/
│   │   ├── era5.py / era5_binary.py  — ERA5 init (grib or binary cache, 50-100x faster)
│   │   ├── gsam_binary.py            — read gSAM binary dumps
│   │   ├── gsam_era5_init.py         — ERA5→gSAM initialization
│   │   ├── writer.py                 — NetCDF output
│   │   ├── restart.py                — restart state I/O
│   │   └── slm_init.py / slm_forcing.py
│   └── utils/
│       └── IRMALoader.py             — IRMA case helper
├── matching_tests/             — regression tests (jsam vs gSAM), one dir per process
│   ├── common/                 — shared: bin_io.py, compare.py (L2 norms, rel diffs)
│   └── test_<name>/            — dump_inputs.py + verify_*.py per component
├── autoresearch/               — automated loop: submit PBS -> poll -> compare -> commit
├── scripts/
│   ├── run_irma.py             — full IRMA case driver (ERA5 init -> step loop)
│   └── compare_oracle.py / compare_debug_globals.py
└── verify_binary_caching.py
```

## Key Patterns
- **ModelState** is a JAX pytree; passed through all core functions immutably.
- **step.py** orchestrates: SGS -> advection -> pressure -> microphysics -> surface -> radiation -> timestepping.
- Each `matching_tests/test_<X>/` validates one Fortran subroutine equivalent; `dump_inputs.py` extracts gSAM binary inputs, `verify_*.py` checks output.
- ERA5 binary cache lives in scratch; see `io/era5_binary.py`.
- Fortran oracle files in `/glade/u/home/sabramian/gSAM1.8.7/SRC/` map by process name (e.g., `SGS.f90`, `MICRO_M2005.f90`, `PRESSURE.f90`, `ADV_SCALAR.f90`).
