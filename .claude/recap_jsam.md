# jsam recap — compact context for agent handoff

**Goal**: differentiable + accelerated JAX port of gSAM. Reproduce gSAM output bit-for-bit on IRMA `lat_720_dyvar`. **No** Koopman / TC / squall objectives.

Repo: `/glade/u/home/sabramian/jsam/` · env: `/glade/u/home/sabramian/.conda/envs/jsam/` · py3.12 · JAX 0.9 cuda12

---

## 1. Architecture

### Modules
- **dynamics/** `pressure.py` (direct-LU spherical Helmholtz per rfft mode, float64), `advection.py` (`_flux3` 3rd-upwind mom, `_face5` 5th ULTIMATE scalar +positivity), `timestepping.py` (Euler→AB2→AB3 variable-dt), `coriolis.py`, `damping.py`, `kurant.py`, `polar_filter.py` (legacy, unused).
- **physics/** `microphysics.py` (SAM1MOM + ice_fall), `sgs.py` (Smagorinsky + `diffuse_damping_mom_z`), `surface.py` (Large&Pond 1981 ocean), `radiation.py` (prescribed), `rad_rrtmg.py` (RRTMG_LW via f2py + pure_callback), `lsforcing.py`, `nudging.py` (1D profile).
- **io/** `era5.py` (RDA `d633000`), `writer.py` (gSAM 3D_atm.nc), `restart.py` (NetCDF, byte-identical).
- **core/state.py** `ModelState` pytree. **core/step.py** `StepConfig` (frozen) + `PhysicsForcing` (pytree) + `step()`.

### `build_metric` keys
`imu, cos_v (ny+1, ady-weighted), dx_lon, dy_lat (ny,), dy_lat_ref, ady (ny,), rho (nz), rhow (nz+1), dz, cos_lat, lat_rad, dlon_rad, dlat_rad, nx, pres (Pa), gamaz (K=g·z/cp), z (m)`. Run-time adds `pfmask_u/v, lat_deg, lon_deg`.

### step() order (matches gSAM main.f90)
1. `ls_proc`
2. `rad_proc` / 2a `rad_rrtmg_proc` every `nrad` steps / 2b `nudge_proc`
3. `bulk_surface_fluxes` → `SurfaceFluxes`
4. `diffuse_momentum` (explicit 3D SGS horizontal, Euler-applied post-advection as non-AB fresh tendency; NOT in AB3 buffer — lagged SGS term drives exp growth via `bt=-16/12`)
5. Buoyancy on W (Euler, accumulated into `dW_extra`)
6. Coriolis (→ `dU_extra`, `dV_extra`)
7. `advance_momentum` — AB3 on `F_adv + dU/V/W_extra` (≡ gSAM `adamsA`)
8. `diffuse_damping_mom_z` — ONE Thomas solve per component: implicit vertical SGS + polar UV limiter + upper-level (pres<70 hPa) UV limiter + surface τ BCs
8b. `gsam_w_sponge` — W-only Rayleigh sponge (nub=0.6, taudamp_max=0.333)
8c. `spectral_polar_filter` — rfft per lat row, zero m > ⌊nx/2·cos φⱼ⌋ (jsam-only; not in gSAM)
8d. `adams_b` — lagged PGF correction using `state.p_prev`, AB3 `(at,bt,ct)`
9. `pressure_step` — anelastic projection, returns `p_new` → `state.p_prev`
10. `advance_scalars` — AB3 on advection of `s = TABS + gamaz` (increments `nstep,time`)
10b. `diffuse_scalar` (sgs_scalars_proc) — scalar SGS AFTER advection (gSAM order)
11. `micro_proc` — SAM1MOM

### Driver (`scripts/run_irma.py`)
Outer Python loop. Per big step: `compute_cfl` → `dtn = min(DT, 0.7·DT/cfl)` → sub-cycle up to `NCYCLE_MAX=4`. `dtn_j = jnp.asarray(dtn_py)` passed as traced input → no retrace per dtn.

---

## 2. Hard-won design decisions

- **Pressure solver**: direct LU per rfft mode; `metric["nx"]` = `len(grid.lon)` (never from `dlon_rad` — LAM wrong by ×10⁵); float64 mandatory (`conftest.py`); m=0 pinned to zero mean; skip mode_energy < 1e-14·rhs_energy; Hm−1e-7·I regularisation **removed** in Gap 8 (was dominating on coarse grids). Round-trip residual 2.7e-10 on LAM test.
- **Gap 8 metric**: `dy_lat` is `(ny,)`, `ady = dy_per_row/dy_ref`, `cos_v` ady-weighted. Flux-form Ly in Helmholtz. Uniform grid: `ady=1` → no-op. **Critical** for `lat_720_dyvar` (tropical dy wrong by ~5× with scalar).
- **AB3**: full gSAM `abcoefs` (variable-dt α, β); Euler@nstep=0, AB2@1, AB3@≥2; constant-dt collapses to `(23/12, -16/12, 5/12)`.
- **ModelState pytree**: `nstep`, `time`, `p_prev`, `p_pprev` must be **dynamic** leaves (`jnp.int32(0)`, `jnp.zeros`). Static → pytree structure change → full JIT retrace → OOM.
- **SAM1MOM mapping**: `omn/omp/omg` from T clamps → bulk `qn, qp` split into `QC/QI/QR/QS/QG`.
- **Lat grid**: `IRMALoader` reads `lat` from gSAM OUT_NC (Mar 22 run, `lat_720_dyvar`, −89.4→89.4°, dlat ∈ [0.133, 0.986]). Input binaries `*_dyvar_*.bin` **must** match `lat_720_dyvar` — mismatch with uniform `lat_720` → CFL 1.45→3.06 at step 1.
- **QV units**: jsam = kg/kg; gSAM 3D NC output = g/kg. Multiply by 1e-3 when feeding jsam from gSAM NC.
- **GPU recipe**: A100-40GB (no 80 GB variant). `--float32`, `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, `spectral_scalar_filter` on all 7 scalars (float32 polar NaN prevention). Float64 OOMs on single A100-40GB at 0.25°.

---

## 3. Known problem: dynamics blowup ~step 30-50 (UNRESOLVED)

### Evidence
Both `run_irma_norad_native_gpu.log` (RRTMG **off**, nudging on τ=3600s) and `run_irma_rrtmg_gpu.log` (RRTMG on, nrad=180, nudging off) blow up **identically**:
```
[10] TABS_min=179.2  W=[-2.3, 1.5]
[20] TABS_min=158.2  W=[-8.2, 7.1]
[30] TABS_min=110.8  W=[-19.6, 21.1]
[40] NaN at (k=0, j=0, i=0)
```
CFL ~0.5 throughout — **not** advection-limited.

### Conclusion
**Radiation is NOT causal.** ~5 K/step cooling at a single cell is 2000× faster than any atmospheric LW process — it's pure dynamics. Same ERA5 init on same `lat_720_dyvar` grid runs stable in gSAM. Bug is jsam-specific dynamics.

### Candidates (ordered by likelihood)
1. **`s = TABS + gamaz` scalar advection**. `gamaz` ~293 K at 30 km, ~580 K at 60 km. jsam advects `s` in `advance_scalars`; gSAM advects `t` differently. ULTIMATE 5th-order scheme not designed for high-mean fields. Dispersive error in `s` → error in `TABS = s − gamaz`.
2. **`_buoyancy_W` reference profile `tabs0`**. If not hydrostatic w.r.t. init state, spurious W from step 1 → AB3/pressure amplification.
3. **`adams_b` at rigid-lid top**. PGF correction vs pressure-solver Neumann BC mismatch → one-cell top error growing every step. NaN location `k=0` is consistent.
4. **South-pole corner (`j=0, i=0`)**. `lat_720_dyvar` southernmost row at −89.4°, cos=0.010. ady-weighted `cos_v` edge term + V=0 wall interaction.
5. **`spectral_scalar_filter` Gibbs** on sharp strat gradients if enabled.

### Debug plan for next agent
1. Print `argmin(TABS)` with `(k,j,i)` every step in norad run. Stuck cell → boundary bug; migrating → advection.
2. `nsteps=1` norad snapshot, componentwise `log10(|jsam − gSAM|)` vs gSAM 1-step. **Debug step 1, not step 30** — step-30 blowup drowns in secondary effects.
3. Bisect modules via `StepConfig` toggles: buoyancy off, `adams_b` off, spectral filter off, each isolating.
4. Passive TABS test: U=V=W=0, step 100x, verify TABS invariant (tests `s=TABS+gamaz` round-trip).

---

## 4. Damping layers (currently in step() path)

### What runs
- **`diffuse_damping_mom_z`** (sgs.py:700): Thomas solve bundling vertical SGS diffusion + polar UV limiter `tauy=(1−cos²)^200` + upper-level (pres<70 hPa) `tau=1/dt` + surface τ BCs. All-in-one per U,V,W component.
- **`gsam_w_sponge`** (damping.py:238): exact port of gSAM `damping.f90` §1. W-only. `nu=(z−z[0])/(z[-1]−z[0])`, `taudamp=taudamp_max·zzz/(1+zzz)` for `nu>nub=0.6`. Implicit `W/(1+taudamp)`.
- **`spectral_polar_filter`** (damping.py:138): rfft per lat row, zero `m > ⌊nx/2·cos φⱼ⌋`. Jsam-only addition for direct-LU pole conditioning.

### Dead code (imported but never called)
- **`pole_damping`** (damping.py:32) — has `u_max_phys=150 m/s` hard clip NOT in gSAM (would clip UTLS jet maxes).
- **`top_sponge`** (damping.py:273) — alternative sponge formulation, superseded.

step.py docstring still lists "Step 9 pole_damping" — wrong, not called. Cleanup TODO.

### gSAM damping.f90 sections (for comparison)
1. `dodamping` W sponge (→ jsam `gsam_w_sponge` ✓)
2. `dodamping_w` W CFL limiter below sponge (→ jsam: **verify** W branch of `diffuse_damping_mom_z` carries it)
3. `dodamping_poles` UV polar (→ jsam bundled in `diffuse_damping_mom_z` ✓)
4. `dodamping_u` UV pres<70 hPa (→ jsam bundled in `diffuse_damping_mom_z` ✓)

**Damping is momentum-only — cannot fix TABS runaway.**

---

## 5. RRTMG_LW wrapper (rad_rrtmg.py)

### Build
`/glade/work/sabramian/jsam_rrtmg_build/`. `build.sh` → f2py → `jsam_rrtmg_lw.cpython-312-x86_64-linux-gnu.so`. Only `rrtmg_lw_ini` + `rrtmg_lw` (nomcica). `patch_kinds.py` strips `rb=>kind_rb`/`im=>kind_im` aliases, rewrites literals `1.0_rb → 1.0_kind_rb` — **must re-run after cp from gSAM**. `.f2py_f2cmap`: `kind_rb→double`, `kind_im→int`. `rrtmg_lw_rad_nomcica.f90` output arrays explicit-shape `(ncol, nlay+1/nlay)`. k-distribution from `rrtmg_lw_k_g.constants`.

### Python bridge
`RadRRTMGConfig` (frozen): `cpdair, emis=0.98 (matches gSAM emis_water), co2/ch4/n2o/o2 vmr, CFCs, ccl4`. `rad_rrtmg_proc` uses `jax.pure_callback(vmap_method="sequential")`. Host chunks `_CHUNK_NCOL=4096`. ~230 µs/col on CPU → 1M cols ≈ 4 min serial. `JAX_PLATFORMS=cpu` before import on login nodes.

- **Clouds**: binary fraction from QC/QI. Liq Re=14 µm (CAM ocean default). Ice Re from CAM hexagonal retab LUT (180-274 K). Clamped liq 2.5-60 µm, ice 5-131 µm.
- **Ozone**: `GSAMOzoneClimo.from_file` parses gSAM o3file. Header `(nx1,ny1,nz1)`, `nobs`, `days`, per-time `lonr,latr,zr,pr` + `nz1` real(4) records. **Values are mass mixing ratio** → multiply by `mw_air/mw_o3 ≈ 0.6034` for VMR. `build_o3vmr_for_metric` bilinear(lat,lon) + log-p vertical.
- **SST**: per-column `(ny,nx)` required; land fallback `tsfc_col = TABS[0]`. Uniform SST → +33 K/d spurious polar heating.
- **tlev layout** (matches gSAM rad.f90:444): `tlev[:,0]=tsfc`, `tlev[:,1:-1]=0.5·(TABS[:-1]+TABS[1:])`, `tlev[:,-1]=2·TABS[-1]-tlev[:,-2]`.
- **Extra top layer** (gSAM rad.f90:465-479): RRTMG called with `nlay+1`. Extra isothermal layer between real top and `min(1e-4 hPa, 0.25·p_top)`, zero clouds, h2o/o3 from top real layer. Without it, k=73 LW cooling is 3× too strong.
- **Heating clamp**: ±50 K/day before conversion to dTABS/dt.

### Validated vs gSAM job 3128602 (IRMA 1-day, lat_720_dyvar)
Clear-sky LW heating, 4° coarsened:
| k | p (hPa) | jsam | gSAM |  |
|---|---------|------|------|---|
| 73 | 1.8 | −8.45 | −5.08 | TOA overcool (trace gas) |
| 45-5 | 207-984 | within 5% | | ✓ mid-trop validated |
| 0 | 1011 | +0.39 | −2.52 | **sign flip unresolved** |

### Known defects
- **Trace gas profile**: jsam scalar CO2/CH4/N2O; gSAM reads altitude-dependent from `rrtmg_lw.nc` via `tracesini`. Explains ~60% TOA overcooling. Fix: port `tracesini` → per-level VMR.
- **k=0 near-surface sign flip**: 2-3 K/d clear-sky residual. Likely `p_surf=1013.25 hPa` fixed (jsam single hydrostatic column); revisit at native resolution.
- **Caching**: `rad_rrtmg_proc` applies `dt·nrad` in ONE kick on the trigger step, then zero for `nrad-1` steps. **Not gSAM pattern** (gSAM stores `qrad` module-level, applies `dtn·qrad` every step between refreshes). **This is a real code defect** but NOT the blowup cause (norad run blows up identically). Fix later: add `qrad` as dynamic `ModelState` leaf, refresh every `nrad`, apply `TABS += dt·qrad` unconditionally.

---

## 6. Known technical debt

### `ab_coefs` Python branches → blocks `lax.scan`
`timestepping.py:85-127` uses `if nstep==0 / elif nstep==1 / else` with `int(state.nstep)` in step.py:316. Causes:
- 3 JIT retraces at bootstrap (nstep=0,1,2)
- Cannot wrap `step()` in single `lax.scan`
- Blocks clean `jax.grad` through fixed-length rollouts

Fix: rewrite with `jnp.where(nstep<1, ..., jnp.where(nstep<2, ..., ...))` on traced `state.nstep`, drop `int(...)`. Guard div-by-zero in α,β for bootstrap.

### test debt
- `test_advection.py` 4 fails: positivity floor clips synthetic negative-valued test fields (not physics bug — real scalars are ≥0). Expose `positivity=False` flag for tests.
- `test_era5.py`, `test_radiation.py`: unknown state.
- ERA5 integration tests (9) deselected: pending Casper allocation.

### Misc cleanup
- step.py docstring lists "Step 9 pole_damping" — wrong.
- `pole_damping`, `top_sponge` dead imports in damping.py.

---

## 7. Gap vs gSAM status

| Gap | Status |
|---|---|
| 1 `adams_b` | ✅ with AB3 coeffs |
| 2 Coriolis AB'd | ✅ via `dU/V_extra` into `advance_momentum` |
| 3 AB3 | ✅ (full variable-dt) |
| 4 SGS scalar after advection | ✅ |
| 5 Adaptive dt via kurant | ✅ (driver-level) |
| 6 Direct LU vs GMG | won't fix (LU is more correct) |
| 7 `igam2` on W | deferred (anelastic γ≈1) |
| 8 `dy_lat (ny,)` | ✅ 2026-04-12 |

Priority-1 gaps all closed. No known discrepancy that would cause the step-30 blowup — bug is elsewhere in jsam dynamics.

---

## 8. PBS / compute

Derecho `main`, account **UCLB0065** (ERA5 tests UCLB0054). All A100-40GB.
- `run_irma.pbs` — 0.25° CPU 36 steps (JIT compile >60 min; OOM at 211 GB)
- `run_irma_gpu.pbs` — 0.25° GPU 36 steps, float32
- `run_irma_norad_native_gpu.pbs` — **canonical debug run** (no radiation, nudging on)
- `run_irma_rrtmg_2deg.pbs` — 2° 6h RRTMG nrad=60
- `run_irma_rrtmg_gpu.pbs` — 0.25° native RRTMG nrad=180
- `run_era5_tests.pbs` — integration tests (UCLB0054)

---

## 9. Primary next action

**Do not chase radiation, damping, or microphysics.** The norad run is the reference: it blows up at step ~40 with pure dynamics. Steps:

1. `nsteps=1` snapshot, componentwise diff vs gSAM step-1 on identical init.
2. Track `argmin(TABS)` location every step in norad run.
3. Bisect modules via `StepConfig` toggles.
4. Passive-flow TABS invariance test (`s = TABS + gamaz` round-trip).

**Goal metric**: componentwise agreement jsam vs gSAM to ~1e-6 at `lat_720_dyvar` IRMA nstep=1 (modulo LU vs GMG pressure solver).
