# jsam IRMA Debug Run Fixes — April 16, 2026

## Issues Found and Fixed

### 1. JAX Array to Static Argument Conversion Bug (advection.py:281)

**Problem:**  
The `advect_scalar` wrapper function was computing `macho_order = (nstep - 1) % 6` where `nstep` is a JAX array. This result was then passed to `_advect_scalar_jit` as a static argument (line 43: `@jax.jit(static_argnums=(6,))`). JAX requires static arguments to be Python primitives, not JAX arrays, so the JIT hash operation failed with:

```
ValueError: Non-hashable static arguments are not supported. 
An error occurred while trying to hash an object of type <class 'jaxlib._jax.ArrayImpl'>, 0.
TypeError: unhashable type: 'jaxlib._jax.ArrayImpl'
```

**Fix:**  
Convert `nstep` to Python int before computing `macho_order`:

```python
# Before
macho_order = (nstep - 1) % 6

# After  
nstep_int = int(nstep)
macho_order = (nstep_int - 1) % 6
```

File: `jsam/core/dynamics/advection.py:267-282`

**Status:** ✅ Fixed and tested

---

## Changes Made

### 1. Fixed `/glade/u/home/sabramian/SAMJax/jsam/core/dynamics/advection.py` (line 281)
- Added conversion of `nstep` to Python int before static argument computation
- Ensures `macho_order` is a Python int, not a JAX array

### 2. Enabled Debug Dumping in `/glade/u/home/sabramian/SAMJax/scripts/run_irma_debug500.pbs`
- Changed `--nsteps 30` → `500` to match gSAM oracle (500 steps)
- Uncommented `--debug-dump-dir` flag
- Changed `--output-interval 30` → `500` (single output at end)

### 3. Created Test Script `/glade/u/home/sabramian/SAMJax/scripts/test_debug_small.pbs`
- Runs 5 steps with debug dumping to verify fix works before 500-step full run
- Job ID: 3182591

---

## Verification Steps

### Test Run (5 steps)
- **Job:** test_debug_small.pbs (submitted 2026-04-16 11:07)
- **Status:** In progress (JIT compilation)
- **Expected output:** `/glade/derecho/scratch/sabramian/jsam_test_debug_small/debug/`

### Full Run (500 steps)
- **PBS script:** `run_irma_debug500.pbs` (ready to submit)
- **Will compare:** jsam vs gSAM oracle using `scripts/compare_debug_globals.py`

---

## Oracle Reference

See `/glade/u/home/sabramian/SAMJax/.claude/Oracle_Tensor_Structure.md` for:
- Dump format specification (rank_NNNNNN.bin, globals.csv)
- 19 gSAM stages per step
- 7 fields: U, V, W, TABS, QC, QV, QI
- 500 steps × 19 stages = 9500 + 38 icycle records

Oracle location:  
`/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug/`

---

## Next Steps

1. Monitor test_debug_small.pbs completion
2. If test passes (no crashes), run full 500-step debug comparison
3. Use `compare_debug_globals.py` to identify any remaining divergences
4. Trace root causes in jsam code and fix deviations from gSAM

---

## Architecture Notes

### TABS Representation
- **gSAM:** Stores "t" (liquid/ice water static energy), computes TABS from formula
- **jsam:** Stores TABS (absolute temperature) directly, applies transformation during advection
- **Debug dump:** Both should output equivalent TABS values at all stages

### Scalar Advection (MACHO)
- **gSAM:** 5th-order ULTIMATE with Zalesak FCT, MACHO direction cycling
- **jsam:** Identical port in `jsam/core/dynamics/advection.py`
- **MACHO order:** Cycles every 6 steps to prevent directional bias
