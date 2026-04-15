"""
verify_no_hard_clip.py — Damping hard clip fix (2026-04-15 follow-up audit).

gSAM damping.f90:76 uses   umax(j) = damping_u_cu*dx*mu(j)/dtn
with NO physical velocity cap and NO post-damping hard clip.

jsam previously applied  umax = min(cu*dx*cos(lat)/dt, 150)  and also
a post-solve `U,V = clip(U,V, -150, 150)`.  Both have been removed to
match gSAM exactly.

This script:
  1. Confirms `pole_damping` has no `u_max_phys` kwarg.
  2. Feeds a 200 m/s U field at moderate latitude where tau_lat ≈ 0
     and verifies that U comes back unchanged (pure gSAM: no clip, no
     relaxation because tau≈0, CFL umax>U so no tau_vel triggering).
  3. Feeds the same field at 85° where tau_lat ≈ 1 and checks the
     result matches the literal gSAM formula

         u_new = (u + clip(u,-umax,umax)*tau) / (1+tau)

     evaluated on the inputs — no 150 m/s cap anywhere.

Run:
    PYTHONPATH=. python matching_tests/test_damping/verify_no_hard_clip.py
"""
from __future__ import annotations

import inspect
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jsam.core.dynamics.damping import pole_damping


def _gsam_formula(u, umax, tau):
    clipped = np.clip(u, -umax, umax)
    return (u + clipped * tau) / (1.0 + tau)


def main() -> int:
    # (1) signature check: u_max_phys must be gone
    sig = inspect.signature(pole_damping)
    assert "u_max_phys" not in sig.parameters, (
        f"pole_damping still accepts u_max_phys: {list(sig.parameters)}"
    )
    print("[1] signature clean — no u_max_phys kwarg")

    R = 6.371e6
    dx = R * np.deg2rad(1.0)
    dt = 30.0
    cu = 0.3

    # (2) Mid-latitude 30°: tau_lat = sin²(30°)^200 ≈ 0 → U unchanged.
    nz, ny, nx = 2, 4, 8
    lat = np.deg2rad(np.full(ny, 30.0))
    U = jnp.full((nz, ny, nx + 1), 200.0, dtype=jnp.float64)
    V = jnp.zeros((nz, ny + 1, nx), dtype=jnp.float64)
    U_out, V_out = pole_damping(
        U, V, jnp.asarray(lat), dx=dx, dy=1.0, dt=dt, cu=cu,
    )
    # At 30° the CFL umax = 0.3*111e3*cos(30°)/30 ≈ 962 m/s — bigger than 200,
    # so tau_vel not triggered.  tau_lat ≈ 0 so relaxation is a no-op.
    # Without the hard clip at 150 m/s, U must survive unchanged.
    assert np.allclose(np.asarray(U_out), 200.0, atol=1e-10), (
        f"mid-lat U mutated: max|U_out-200| = {np.max(np.abs(np.asarray(U_out)-200))}"
    )
    print("[2] 30° N: U=200 m/s passes through unchanged (hard clip removed)")

    # (3) High-lat 85°: replicate gSAM formula and check equality.
    lat85 = np.deg2rad(np.full(ny, 85.0))
    U = jnp.full((nz, ny, nx + 1), 50.0, dtype=jnp.float64)
    U_out, _ = pole_damping(
        U, V, jnp.asarray(lat85), dx=dx, dy=1.0, dt=dt, cu=cu,
    )

    cos_lat = np.cos(lat85)
    umax = cu * dx * cos_lat / dt            # no min(...,150)
    tau_lat = (1.0 - cos_lat ** 2) ** 200    # ≈ 1 at 85°
    u_ref = _gsam_formula(np.asarray(U), umax[None, :, None], tau_lat[None, :, None])

    diff = np.max(np.abs(np.asarray(U_out) - u_ref))
    assert diff < 1e-10, f"85° N: jsam mismatch vs gSAM formula (max={diff:g})"
    print(f"[3] 85° N: max|jsam − gSAM formula| = {diff:.3e}  (OK)")

    print("\nverify_no_hard_clip.py: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
