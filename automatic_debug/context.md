# jsam Debug Context

**Updated**: 2026-04-17 09:55:00
**Git commit tested**: `cdf8b51` *(uncommitted changes present)*
**Status**: OK
**jsam records compared**: 19
**Max relative error**: 0.0000e+00

## Result: MATCH

All compared stages are within 0.1% of oracle.
Pipeline will re-run automatically when code changes.

## Recent Code Changes

### Git log
```
b42dc7c fix: remove float() conversion in jitted precip_proc
4565494 fix: restore float() conversion on metric values
9c630da fix: restore evaporation coefficient caching (micro_proc not JIT-compiled)
459e370 fix: remove traced array conversions in microphysics JAX JIT calls
d4c757f fix: remove float() conversions causing JAX ConcretizationTypeError
1cb1e2a fix: boundary-aware y-face interpolation for per-face tkmax clamping
b65a32f NICE JOB START FROM HERE
15affbf WIP: W initialization scaling attempts (jnp.where didn't work)
94a3e53 fix: use jnp.where for W scaling (JAX JIT compatible)
4eed901 fix: move W scaling to after state creation, before dump
```

### Uncommitted diff (jsam/)
```diff
diff --git a/jsam/core/dynamics/pressure.py b/jsam/core/dynamics/pressure.py
index 41eb01d..db1a2c8 100644
--- a/jsam/core/dynamics/pressure.py
+++ b/jsam/core/dynamics/pressure.py
@@ -321,7 +321,7 @@ def _pcg_solve(
     op,
     b: jax.Array,
     precond,
-    tol: float = 1e-5,
+    tol: float = 1e-7,  # gSAM gmg_precision typical value
     maxiter: int = 200,
 ) -> jax.Array:
     """Preconditioned Conjugate Gradient solver (eager execution)."""
@@ -864,6 +864,10 @@ def pressure_step(
         )
         U_cur = U_cur.at[:, :, -1].set(U_cur[:, :, 0])
 
+    # gSAM pressure.f90:83-84: clamp pressure perturbation to ±15% of reference
+    pres_ref = metric["pres"][:, None, None]   # (nz, 1, 1)
+    p_total = jnp.clip(p_total, -0.15 * pres_ref, 0.15 * pres_ref)
+
     new_state = ModelState(
         U=U_cur, V=V_cur, W=W_cur,
         TABS=state.TABS, QV=state.QV, QC=state.QC,
diff --git a/jsam/core/physics/microphysics.py b/jsam/core/physics/microphysics.py
index 118e2f7..16d72bb 100644
--- a/jsam/core/physics/microphysics.py
+++ b/jsam/core/physics/microphysics.py
@@ -1047,21 +1047,54 @@ def ice_fall(
     """
     Gravitational sedimentation of cloud ice with MC flux limiter.
 
-    Port of gSAM MICRO_SAM1MOM/ice_fall.f90 (scalar icefall_fudge branch;
-    Heymsfield 2003 terminal velocity).  Latent heat of sublimation is
-    released into TABS following the same static-energy convention as
-    precip_fall: ΔTABS = -fac_sub · ΔQI.
+    Port of gSAM MICRO_SAM1MOM/ice_fall.f90.  When metric contains the grid
+    spacing arrays (dx_lon, cos_lat, dy_lat_ref, ady) the per-row fudge factor
+    follows gSAM ice_fall.f90:51-61 (doglobalpresets branch); otherwise the
+    scalar params.icefall_fudge is used uniformly.
+
+    Latent heat of sublimation is released into TABS following the same
+    static-energy convention as precip_fall: ΔTABS = -fac_sub · ΔQI.
     """
     rho_1d = np.array(metric["rho"])
     dz_1d  = np.array(metric["dz"])
 
-    def _col(qi_col):
+    # gSAM ice_fall.f90:51-61: per-row fudge based on effective grid spacing.
+    # Uses doglobalpresets formula with fudge1km=0.8, fudge4km=0.3.
+    #   delta(j) = min(8, 0.001*sqrt(dx*mu(j)*dy*ady(j)))   [km]
+    #   fudge(j) = max(0.2, min(1, 0.3333*((fudge4km-fudge1km)*delta
+    #                                        + 4*fudge1km - fudge4km)))
+    if (
+        "dx_lon" in metric
+        and "cos_lat" in metric
+        and "dy_lat_ref" in metric
+        and "ady" in metric
+    ):
+        dx     = float(metric["dx_lon"])           # scalar, m
+        dy     = float(metric["dy_lat_ref"])        # scalar reference dy, m
+        mu_j   = jnp.asarray(metric["cos_lat"])    # (ny,)
+        ady_j  = jnp.asarray(metric["ady"])        # (ny,)
+        _delta = jnp.minimum(8.0, 0.001 * jnp.sqrt(dx * mu_j * dy * ady_j))  # (ny,) km
+        _fudge1km = 0.8
+        _fudge4km = 0.3
+        _fudge = jnp.maximum(
+            0.2,
+            jnp.minimum(
+                1.0,
+                0.3333 * ((_fudge4km - _fudge1km) * _delta + 4.0 * _fudge1km - _fudge4km),
+            ),
+        )  # (ny,)
+    else:
+        ny = QI.shape[1]
+        _fudge = jnp.full((ny,), params.icefall_fudge)  # (ny,) uniform fallback
+
+    def _col(qi_col, fudge_j):
         return _ice_fall_col(qi_col, rho_1d, dz_1d,
-                             params.icefall_fudge, params.gamma_rave, dt)
... (164 more lines)
```

## Files to Read
1. `automatic_debug/instructions.md`
2. `automatic_debug/knowledge.md`
3. Source file listed in **Source file** above
4. `/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/DEBUG_DUMP_SCHEMA.md`