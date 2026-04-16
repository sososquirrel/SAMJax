# jsam vs gSAM: Exhaustive Pipeline Difference Report

**Date**: 2026-04-16
**Context**: Comparing the JAX port (`jsam/`) against the Fortran oracle (`gSAM1.8.7/`) for the IRMA debug500 pipeline.
**Goal**: Identify every difference that could cause tensor mismatches at 1e-5 precision.

---

## TABLE OF CONTENTS

1. [Architecture & State](#1-architecture--state)
2. [Constants](#2-constants)
3. [Grid & Metric](#3-grid--metric)
4. [Timestepping & Adams-Bashforth](#4-timestepping--adams-bashforth)
5. [Advection (Scalars)](#5-advection-scalars)
6. [Advection (Momentum)](#6-advection-momentum)
7. [Buoyancy](#7-buoyancy)
8. [Coriolis](#8-coriolis)
9. [Pressure Solver](#9-pressure-solver)
10. [SGS (Subgrid-Scale Turbulence)](#10-sgs-subgrid-scale-turbulence)
11. [Damping & Sponge Layers](#11-damping--sponge-layers)
12. [Microphysics](#12-microphysics)
13. [Surface Fluxes](#13-surface-fluxes)
14. [Radiation (RRTMG)](#14-radiation-rrtmg)
15. [SLM (Simple Land Model)](#15-slm-simple-land-model)
16. [Forcing, Nudging, Diagnose](#16-forcing-nudging-diagnose)
17. [Kurant / Sub-cycling](#17-kurant--sub-cycling)
18. [Missing Modules](#18-missing-modules)

---

## 1. Architecture & State

### 1.1 Prognostic variable choice
- **gSAM** evolves `t` (liquid/ice water static energy = TABS + gamaz - fac_cond*qcl - fac_sub*qci) as the thermodynamic prognostic.
- **jsam** evolves `TABS` (absolute temperature) directly.
- jsam converts TABS->static energy before scalar advection ("F11 mode" in step.py:566-579), then converts back after microphysics. This double-conversion must be exact.
- **RISK**: The buoyancy-work energy correction (step.py:421-427) is applied to `TABS`, whereas gSAM applies it to `t` (static energy). This is a real physics difference.

### 1.2 Moisture species
- **gSAM**: Carries `q` (total non-precip = QV+QC+QI) and `qp` (total precip = QR+QS+QG) as prognostic; individual species are diagnostic via `micro_diagnose()`.
- **jsam**: Carries all 6 species independently (QV, QC, QI, QR, QS, QG).
- Phase repartitioning (`qrr = qp*omp`) gives different results than using QR directly when species are not in exact phase equilibrium.

### 1.3 TKE
- **gSAM**: TKE is diagnostic (computed each step inside SGS with `dosmagor=.true.`).
- **jsam**: TKE is carried as a prognostic field on ModelState.

### 1.4 Tendency buffers
- **gSAM**: 3-level rotating buffer `dudt(:,:,:,na/nb/nc)` plus separate `dudtd` (non-AB-stepped direct tendency).
- **jsam**: Explicit `MomentumTendencies` nm1/nm2 passed as arguments. No `dudtd` equivalent — all extra tendencies (buoyancy, coriolis, SGS) go into `dU_extra` which IS AB-filtered. In gSAM these bypass AB.
- **CRITICAL**: `dudtd` contributions (buoyancy, coriolis, SGS diffusion) are applied with coefficient 1.0 in gSAM (not AB-weighted). In jsam, they go through the AB filter via `dU_extra`.

### 1.5 Pressure storage
- **gSAM**: 3-level buffer `p(:,:,:,na/nb/nc)`.
- **jsam**: `p_prev` and `p_pprev` on ModelState (2 fields). Equivalent for AB3.

### 1.6 Index ordering
- **gSAM**: Fortran column-major `(x, y, z)`.
- **jsam**: C-major (JAX) `(z, y, x)`. Transposed but correct.

---

## 2. Constants

### 2.1 Gravity inconsistency
- `run_irma.py` line 411 sets `g = 9.81`, overriding the correct `StepConfig` default of `g = 9.79764`.
- All other code (gamaz, pressure, microphysics, hydrostatic balance) uses `G_GRAV = 9.79764`.
- **BUG**: Buoyancy sees 9.81, everything else sees 9.79764.

### 2.2 EPS rounding
- gSAM: `eps = Rd/Rv = 287.04/461.5 = 0.621886...`
- jsam: `EPS = 0.622` (hardcoded). ~0.02% difference.

### 2.3 All other constants match
- cp=1004.64, lcond=2.501e6, lfus=0.337e6, lsub=2.834e6, Rd=287.04, Rv=461.5, ggr=9.79764, epsv=0.61, sigma_SB=5.670373e-8, earth_radius=6371229. All consistent.

---

## 3. Grid & Metric

### 3.1 Terrain
- **gSAM**: Full terrain support (`doterrain=.true.`). `terra(i,j,k)`, `terrau`, `terrav`, `terraw` masks modify advection, pressure, averaging, SGS everywhere.
- **jsam**: No terrain support whatsoever. No masks, no elevation arrays, no terrain-modified stencils.
- **Impact**: prm has `doterrain=.true.` — jsam ignores terrain entirely.

### 3.2 Pressure units
- gSAM: `pres(nzm)` in hPa (millibar).
- jsam: `metric["pres"]` in Pa. Factor of 100 at boundaries.

### 3.3 Missing metric fields
- jsam does not store: `adyv(j)` (computed locally), `imuv(j)` (computed locally), `presi(nz)` (interface pressure), `prespot/prespoti` (Exner function), averaging weights `wgt/wgtu/wgtv/wgtw`.

### 3.4 earth_factor
- gSAM supports `earth_factor` for planet-radius experiments. jsam does not. Fine for Earth runs.

---

## 4. Timestepping & Adams-Bashforth

### 4.1 AB coefficients
- Both use the same formulas for `at`, `bt`, `ct`.
- jsam does not contain the AB coefficient computation internally — caller must supply them. Default `bt=-0.5` matches AB2 with constant dt.
- gSAM supports `nadams=2` (AB2-only mode); jsam always tries AB3 after nstep>=2.

### 4.2 dudtd bypass (CRITICAL)
- gSAM: `dudtd` contributions (buoyancy, coriolis, SGS) are applied as `dt * dudtd` (Euler-stepped, NOT AB-filtered).
- jsam: All extra tendencies go through `dU_extra` which enters the AB filter.
- **This changes the effective time integration of buoyancy, Coriolis, and SGS momentum diffusion.**

### 4.3 gamma_RAVE
- gSAM `adamsA`: W update is multiplied by `1/gamma_RAVE^2`.
- gSAM momentum advection: W tendency multiplied by `gamma_RAVE^2`.
- jsam: No `gamma_RAVE` anywhere. For default `gamma_RAVE=1.0` this is a no-op.

### 4.4 Terrain masks in adamsA
- gSAM: `u = terrau*(u + dt*(…))`. Velocities zeroed inside topography.
- jsam: No terrain masks.

### 4.5 Sub-cycling
- gSAM: Full icycle/ncycle sub-cycling when CFL > 1. Multiple dynamics passes per nstep.
- jsam: No sub-cycling at all. Single pass per step.

### 4.6 Scalar advection velocity
- gSAM: Blends mass-weighted fluxes: `u1 = 0.5*u*rhox + 0.5*u1_prev*dtn`.
- jsam: Simple velocity average: `U = 0.5*(U_old + U)`, then applies metrics inside advection.
- Different because gSAM blends metric-weighted mass fluxes; jsam averages raw velocities.

### 4.7 Static energy double-conversion risk
- step.py:566 converts TABS → static energy via omega-partition before calling `advance_scalars`.
- `advance_scalars` itself also constructs `s_n = TABS + gamaz - FAC_COND*(QC+QR) - FAC_SUB*(QI+QS+QG)`.
- If the state already holds static energy (from step.py), this is a double conversion.
- Need to verify that advance_scalars recognizes the already-converted field.

---

## 5. Advection (Scalars)

### 5.1 Vertical face reconstruction assumes uniform grid
- gSAM `face_5th_z`: Full non-uniform grid corrections using `adz/adzw` ratios in the 5th-order stencil.
- jsam `_face5`: Same uniform-grid 5th-order formula for all directions including vertical.
- **CRITICAL for stretched grids.**

### 5.2 No stencil-order degradation near z-boundaries
- gSAM: 2nd-order at levels k=2,nzm; 3rd-order at k=3,nzm-1; 5th-order only at k=4..nzm-2.
- jsam: 5th-order everywhere with edge-padding.

### 5.3 Courant number construction
- gSAM: Strips mass-weighting before computing face values: `cu = u1 * irho`.
- jsam: Keeps `adz * ady` in the Courant number fed to `_face5`.
- Different Courant numbers yield different face values in the 5th-order reconstruction.

### 5.4 FCT scale factor normalization
- gSAM FCT: Includes `iadz(k)` and `irho(k)` in the outflow/inflow scale factor denominators.
- jsam FCT: No `iadz` or `irho` in scale factor computation.
- **Vertical anti-diffusive flux contribution is incorrectly weighted in jsam.**

### 5.5 dosubtr precision offset
- gSAM: Subtracts 250 from static energy before advection for numerical precision.
- jsam: No such offset.

### 5.6 Positivity floor
- Both apply `max(0, ...)` to final scalar update. Match.

---

## 6. Advection (Momentum)

### 6.1 Scheme
- gSAM: Hybrid 2nd-order centered / 3rd-order upwind with terrain blending (`alphah`). Falls back to 2nd-order near terrain (`alphah → 1`).
- jsam: Pure 3rd-order upwind everywhere (`_flux3`). No terrain blending.

### 6.2 gamma_RAVE on W advection
- gSAM: `dwdt *= gamma_RAVE^2` in `advect23_mom_xy/z`.
- jsam: No gamma_RAVE.

### 6.3 muv clamping
- gSAM: `gv = max(1e-5, muv(j)) * rho * adyv * adz`. Prevents division by zero at poles.
- jsam: `gv = muv * rho * adyv * adz`. No clamp. Division by zero at poles.

### 6.4 Boundary conditions
- gSAM: `dowallx`/`dowally` terrain-aware wall BCs.
- jsam: `jnp.roll` for periodic x, `pad(mode='edge')` for y. No wall BCs.

### 6.5 Vertical advection is 3rd-order in jsam, 2nd-order in gSAM
- gSAM `advect2_mom_z`: 2nd-order centered vertical flux.
- jsam `_flux3`: 3rd-order upwind for all directions including vertical.

---

## 7. Buoyancy

### 7.1 Formula match
- Both use the same virtual-temperature buoyancy expression. Interpolation weights match (cross-weighted by dz).

### 7.2 Terrain mask missing
- gSAM: `if(terraw(i,j,k) > 0)` — buoyancy only above terrain.
- jsam: No terrain mask. Applied everywhere.

### 7.3 Energy correction applied to TABS vs static energy
- gSAM: `t(i,j,k) -= 0.5*dt/cp * buo * W` (corrects static energy `t`).
- jsam: `TABS -= 0.5*dt/cp * buo * W` (corrects TABS directly).
- **Different physical variable being corrected.**

### 7.4 Buoyancy enters AB vs direct application
- gSAM: Buoyancy accumulates into `dwdt(na)` which is AB-stepped.
- jsam: Buoyancy goes into `dW_extra` which also enters AB (via advance_momentum).
- However, in gSAM buoyancy is also in `dwdtd` (direct, non-AB). Need to verify which path gSAM actually uses — if it's in `dwdt(na)`, the AB treatment matches.

---

## 8. Coriolis

### 8.1 Only spherical branch implemented
- gSAM has 4 branches (spherical, f-plane, metric-only, 2D). jsam only implements spherical.

### 8.2 dV tendency uses wrong latitude for southern row (HIGH SEVERITY)
- gSAM: Southern pair evaluated at latitude `j-1` with `fcory(j-1)`.
- jsam: `_q()` evaluated at cell-center latitude `j` for ALL points. The southern row `q_s` uses `fcory(j)`, NOT `fcory(j-1)`.
- **Incorrect latitude in the V Coriolis stencil.**

### 8.3 imuv computation
- gSAM: `imuv = 1/muv` where `muv` is an `ady`-weighted average of adjacent `cos(lat)`.
- jsam: `imuv = 1/cos_v` directly. Different for non-uniform grids.

### 8.4 Vertical Coriolis dzw approximation
- gSAM: `adzw(k)` from actual z-coordinates.
- jsam: `dzw = 0.5*(dz[:-1] + dz[1:])` (average of adjacent cells). Different on stretched grids.

### 8.5 Polar boundary handling
- jsam: Artificial mirroring `q_s[:,0,:] = q_n[:,0,:]` and zero-padding at pole V-faces. Not in gSAM.

### 8.6 adyv polar padding
- gSAM: `jb`/`jc` clamping at poles.
- jsam: Pads first/last entries with `ady[0]`/`ady[-1]`.

---

## 9. Pressure Solver

### 9.1 Solver method
- gSAM (lat-lon): FFT in x, then Geometric Multigrid (GMG) in (y,z) per zonal mode. Iterative to `gmg_precision`.
- jsam: FFT in x, then sparse LU factorization of the (ny×nz) Helmholtz matrix per mode. Direct solve (exact to machine precision).

### 9.2 gamma_RAVE in solver
- gSAM GMG: Scales zi by `gamma_RAVE` inside the elliptic solve.
- jsam: Does not scale zi. Only uses `igam2` in the pressure gradient.
- **Different when gamma_RAVE != 1.**

### 9.3 mu^2 RHS scaling
- gSAM GMG: Premultiplies RHS by `cos^2(lat)`.
- jsam: No such scaling. Different equation formulation (both correct but different numerical properties).

### 9.4 Singular mode handling
- gSAM: Tridiagonal structure pins the null space.
- jsam: `1e-10*I` shift + global mean removal. Both yield same physical result.

### 9.5 Per-level vs global mean removal
- gSAM: Area-weighted, per-level, terrain-masked mean removal.
- jsam: Single scalar global mean removal.

### 9.6 Richardson iteration
- gSAM: Iterates with terrain masking of velocity between iterations.
- jsam: Pure Richardson iteration on residual divergence. No terrain masking.

---

## 10. SGS (Subgrid-Scale Turbulence)

### 10.1 Only Smagorinsky
- jsam only implements the Smagorinsky branch (`dosmagor=.true.`). No prognostic TKE equation.
- This matches the prm setting `dosmagor=.true.`.

### 10.2 tk_factor / tkmin floor missing
- gSAM: `tkmin(j,k) = tk_factor * tkmax(j,k)` provides a minimum background diffusion.
- jsam: No tkmin floor at all. Only upper CFL cap.
- With `tk_factor=0.001` in the prm, gSAM has a nonzero diffusion floor that jsam lacks.

### 10.3 Per-variable CFL caps missing
- gSAM: Separate `tkmaxu/v/w` for staggered grids, using `cfl_diffsc_max=0.46`.
- jsam: Single `tkmax` with constant `0.09`. Different cap formula.

### 10.4 Y-direction spherical metric missing in diffusion
- gSAM scalar diffusion: Uses `mu(j)`, `muv(j)`, `ady(j)`, `adyv(j)` for the spherical divergence operator.
- jsam scalar diffusion: Uses `1/dy_lat` without `mu(j)` weighting.
- gSAM momentum diffusion: Same issue.
- **Missing `1/cos(lat)` divergence factor in both scalar and momentum horizontal diffusion.**

### 10.5 Scalar diffusion split (explicit vs implicit)
- gSAM with `doimplicitdiff=.true.`: horizontal-explicit + vertical-implicit.
- jsam `sgs_scalars_proc`: Calls `diffuse_scalar` which does fully explicit (horiz+vert). Then separately calls `diffuse_scalar_z_implicit`.
- The explicit vertical pass in `diffuse_scalar` is redundant when the implicit pass follows.

### 10.6 W vertical diffusion missing rho factor
- gSAM: Flux includes `rho(k)/adz(k)` weighting.
- jsam: No rho in the W vertical flux.

### 10.7 fwz(:,:,2)=0 off-by-one
- gSAM: Zeros flux at lowest interior W face (Fortran k=2).
- jsam: `fz_Wint[1]` corresponds to Fortran k=3. Off by one.

### 10.8 Saturated buoyancy temperature
- gSAM: Uses cell-centre `tabs(i,j,k)` in the precipitation drag terms.
- jsam: Uses interface-average `0.5*(TABS[kc]+TABS[kb])`.

### 10.9 dtn vs dt throughout
- gSAM SGS uses `dtn` (AB sub-timestep). jsam uses `dt` (base timestep).
- Affects: surface flux injection, velocity caps, CFL caps, damping timescales.

---

## 11. Damping & Sponge Layers

### 11.1 W sponge zi_top computation
- gSAM: `zi(nzm)` = top of nzm-th cell.
- jsam `gsam_w_sponge`: `zi_top = 0.5*(z[-1]+z[-2])` — the interface BELOW the top cell.
- **BUG**: Sponge normalization uses the wrong top height, making sponge activate earlier.

### 11.2 top_sponge formula
- jsam `top_sponge`: Uses `sin(pi/2*frac)^2` profile with `1/(1+alpha*dt/tau)`.
- gSAM: Uses `zzz/(1+zzz)` with `zzz=100*((nu-nub)/(1-nub))^2`.
- Completely different sponge profiles. Only `gsam_w_sponge` matches gSAM.

### 11.3 W CFL damping
- gSAM: `wmax = damping_w_cu * dz*adzw(k) / dtn`.
- jsam: `wmax = damping_w_cu * dzw / dt`. Different timestep and possibly different spacing.

### 11.4 Pole damping V-grid treatment
- gSAM: Uses mass-cell `mu(j)` directly for both U and V.
- jsam: Interpolates to v-faces for V. Different near poles.

---

## 12. Microphysics

### 12.1 3D vs 1D pressure in saturation
- gSAM: Uses full 3D pressure `pp(i,j,k)` for all saturation calls.
- jsam: Uses 1D reference pressure `pres(k)`. Ignores pressure perturbation.

### 12.2 Bulk vs per-species sedimentation (CRITICAL)
- gSAM: Sediments bulk `qp` with a mass-weighted bulk terminal velocity.
- jsam: Sediments QR, QS, QG independently, each with its own terminal velocity.
- **Fundamentally different precipitation fall architecture.**

### 12.3 No CFL subcycling in precip_fall
- gSAM: Subcycles when `prec_cfl > 0.9`.
- jsam: Caps `wp = min(1, vt*dt/dz)`. Non-conservative when CFL > 1.

### 12.4 gamma_RAVE in precip_fall
- gSAM: Divides terminal velocity by `gamma_RAVE`.
- jsam: Does not (only applies in ice_fall).

### 12.5 Latent heat from sedimentation
- gSAM: Flux-divergence form `-(lfac(kc)*fz(kc) - lfac(k)*fz(k))`.
- jsam: Local approximation `lfac * (dQR+dQS+dQG)`.
- Differs when lfac varies across the melting level.

### 12.6 KK autoconversion missing
- prm has `doKKauto=.true.`. gSAM uses Khairoutdinov-Kogan 2000 autoconversion.
- jsam only has Kessler autoconversion. **Wrong scheme for this run.**

### 12.7 rh_homo in mixed-phase Newton
- gSAM: `rh_homo` NOT applied in the mixed-phase Newton iteration.
- jsam: Applies `rh_homo` to `qsati` in all branches including mixed-phase.

### 12.8 Newton convergence
- gSAM: While-loop, tolerance `|dtabs| > 0.001`, max 100 iterations.
- jsam: Fixed 20 iterations, no convergence check.

### 12.9 cloud_fall missing
- gSAM has cloud water sedimentation (`cloud_fall.f90`). jsam does not.

### 12.10 Temperature update in precip_proc
- gSAM: Does NOT update TABS in precip_proc; modifies q and qp only.
- jsam: Explicitly updates TABS. Different approach.

### 12.11 rhow vs rho in precip_fall
- gSAM: Uses cell-face `rhow(k)`. jsam uses cell-center `rho(k)`.

---

## 13. Surface Fluxes

### 13.1 No gustiness parameter
- gSAM `surface.f90`: Wind includes `wd` gustiness term (from convective scheme).
- jsam: No gustiness in wind calculation. For `docup=.false.` this is likely zero anyway.

### 13.2 No terrain weighting
- gSAM: Wind averaged with terrain masks. jsam: No terrain weighting.

### 13.3 No sea-ice qsati branch
- gSAM: Uses `qsati` over sea ice. jsam: Always uses `qsatw`. Benign for warm IRMA SST.

### 13.4 No diagnostic outputs
- gSAM: Accumulates 2m temperature, 10m wind, aerodynamic resistance, etc.
- jsam: None of these diagnostics.

---

## 14. Radiation (RRTMG)

### 14.1 Trace gas profiles
- gSAM: Vertically varying profiles from CAM3 reference sounding.
- jsam: Uniform (column-constant) values.
- Affects upper troposphere/stratosphere heating rates.

### 14.2 Heating rate clip
- jsam: Clips heating rates to +/-50 K/day. gSAM does not.

### 14.3 Layer mass computation
- gSAM: `rho*adz*dz` (density-based).
- jsam: `100*dp/g` (pressure-based). Small numerical differences on stretched grids.

### 14.4 CO2 and iyear
- Must verify `co2=400.0` and `iyear=2017` are passed correctly in the jsam run script.
- The run script has `--co2 400.0` (correct).

---

## 15. SLM (Simple Land Model)

### 15.1 Snow processes entirely missing
- No accumulation, melt, sublimation, snowt evolution, snow-layer heat conduction.
- prm has `readsnow=.true.`, `readsnowt=.true.` — initial snow is loaded but never evolves.

### 15.2 Puddle (mws) not evolved
- Standing water frozen at initial values. No drain, no update.

### 15.3 Soil infiltration condition differs
- gSAM: Requires `any(soilw < 1)` AND `soilt(1) >= tfriz`.
- jsam: Always allows infiltration capped at `ks[0]`.

### 15.4 Wetland override missing
- gSAM: Sets `soilw = w_s_FC` for landtype=11 (wetland).
- jsam: No wetland special case.

### 15.5 No soil/temperature nudging
- gSAM: `dosoilwnudging`, `dosoiltnudging`.
- jsam: Neither implemented.

### 15.6 No runoff module
- gSAM: `move_water()` shallow-water flood model. jsam: Missing.

---

## 16. Forcing, Nudging, Diagnose

### 16.1 Large-scale forcing
- `dolargescale=.false.` in prm. Both should skip. jsam: skips when `ls_forcing is None`.
- If ever activated: jsam subsidence missing `adzw` stretch factor, boundary skip, and U/V subsidence.

### 16.2 Nudging
- `donudging_uv=.false.`, `donudging_tq=.false.` in prm. Both should skip.
- If ever activated: jsam only nudges TABS/QV (not U/V), no 3D nudging, no spectral nudging.

### 16.3 Diagnose horizontal means
- gSAM: Per-level, area-weighted, terrain-masked mean via `wgt(j,k)*terra(i,j,k)`.
- jsam: Level-independent `cos_lat * ady` weighting, no terrain mask.
- **Different tabs0/qv0 when terrain is present.**

### 16.4 Diagnose recomputes TABS
- gSAM: Diagnoses `tabs` from `t - gamaz + latent_heat_terms` each step.
- jsam: Uses TABS directly (it's the prognostic). No reconstruction needed.

---

## 17. Kurant / Sub-cycling

### 17.1 No sub-cycling in jsam
- gSAM: Full icycle/ncycle sub-stepping. Multiple dynamics passes when CFL > cfl_max.
- jsam: Single pass per `step()`. `kurant_dt` exists but no loop to execute sub-cycles.
- **If CFL > 1, jsam goes unstable; gSAM adapts.**

### 17.2 CFL computation differences
- gSAM: Uses cell-center velocity and `dtn` (adaptive timestep).
- jsam: Uses max of adjacent staggered faces and `dt_ref` (reference timestep).
- gSAM: Includes `ady(j)` metric in y-direction CFL. jsam: No ady metric.
- gSAM: Includes `kurant_sgs` (SGS CFL). jsam: No SGS CFL.

---

## 18. Missing Modules

These gSAM modules are entirely absent from jsam:

| Module | gSAM file | Impact for IRMA |
|--------|-----------|-----------------|
| Terrain fill | `terrain_fill.f90` | Active (`doterrain=.true.`) |
| Buildings | `buildings.f90` | Inactive (`dobuildings=.false.`) |
| Tracers | `tracers.f90` | Inactive |
| Cup (convection) | `cup.f90` | Inactive (`docup=.false.`) |
| Dynamic ocean | `dyn_ocean.f90` | Inactive (`dodynamicocean=.false.`) |
| Cloud fall | `cloud_fall.f90` | Potentially active |
| Upperbound | `upperbound.f90` | Inactive (`doupperbound=.false.`) |
| 2D diagnostics/statistics | `stat_2Dinit`, `stepout`, `hbuf` | No impact on dynamics |
| Coarse-graining | `collect_coars` | No impact on dynamics |

---

## SEVERITY RANKING (for 1e-5 matching)

### Will definitely cause mismatches:
1. **dudtd vs dU_extra AB treatment** — buoyancy/coriolis/SGS enter AB in jsam, bypass it in gSAM
2. **KK autoconversion** — prm has `doKKauto=.true.`, jsam uses Kessler
3. **Bulk vs per-species precipitation fall** — fundamentally different
4. **Coriolis dV wrong latitude** — uses fcory(j) instead of fcory(j-1) for southern row
5. **Vertical face reconstruction uniform grid** — ignores adz/adzw on stretched grid
6. **Missing terrain masks** — active in prm, missing in jsam
7. **Y-direction spherical metric in SGS diffusion** — missing 1/cos(lat) factor
8. **FCT scale factor normalization** — missing iadz/irho in denominators
9. **Gravity inconsistency** — buoyancy uses 9.81, everything else uses 9.79764
10. **tk_factor=0.001 floor missing** — no background diffusion in jsam

### Will likely cause mismatches:
11. Buoyancy-work correction on TABS vs static energy
12. Scalar advection velocity blending (metric-weighted vs raw average)
13. Courant number construction for face reconstruction
14. 3D vs 1D pressure in saturation calls
15. rh_homo in mixed-phase Newton
16. rhow vs rho in precip_fall
17. W sponge zi_top wrong height
18. muv clamping (division by zero at poles)
19. imuv computation (direct cos_v vs ady-weighted average)
20. dtn vs dt throughout SGS

### Minor but detectable:
21. EPS rounding (0.622 vs 0.62189)
22. Trace gas profiles (uniform vs vertically varying)
23. Layer mass computation (dp/g vs rho*adz*dz)
24. No stencil-order degradation near z-boundaries
25. Newton convergence (fixed 20 iter vs while-loop)
26. W vertical diffusion missing rho factor
27. fwz off-by-one
28. Snow/puddle not evolved in SLM
29. Diagnose terrain weighting
30. Heating rate 50 K/day clip
