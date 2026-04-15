"""
verify_rhow_interp.py — Item #10 fix.

Pins the `rhow` interior-face formula to gSAM setdata.f90:475-484 exact
(dolatlon branch).  Uses a synthetic non-uniform vertical grid to show
the difference from the naive midpoint average used before 2026-04-15.

Run:
    python matching_tests/test_pressure/verify_rhow_interp.py
"""

from __future__ import annotations

import numpy as np

from jsam.core.grid.latlon import LatLonGrid
from jsam.core.dynamics.pressure import build_metric


def _gsam_rhow(rho, adz):
    """Port of gSAM setdata.f90:475-484 — adz cross-weighted."""
    nz = len(rho)
    rhow = np.zeros(nz + 1)
    for k_f in range(1, nz):
        rhow[k_f] = (rho[k_f - 1] * adz[k_f] + rho[k_f] * adz[k_f - 1]) \
                    / (adz[k_f - 1] + adz[k_f])
    rhow[0]    = 2.0 * rho[0]     - rhow[1]
    rhow[-1]   = 2.0 * rho[nz-1]  - rhow[-2]
    return rhow


def main():
    # Non-uniform grid: 5 × 200 m bottom layers then stretched exponentially.
    nz = 30
    zi = np.zeros(nz + 1)
    zi[1:6]  = 200.0 * np.arange(1, 6)                        # 200..1000 m
    stretch  = 1.15 ** np.arange(nz - 5)
    zi[6:]   = 1000.0 + 200.0 * np.cumsum(stretch)
    z   = 0.5 * (zi[:-1] + zi[1:])
    dz  = np.diff(zi)
    adz = dz / dz[0]

    # Artificial stratified rho so the difference shows up.
    rho = 1.225 * np.exp(-z / 8500.0)

    grid = LatLonGrid(
        lat = np.linspace(-89.0, 89.0, 20),
        lon = np.linspace(0.0, 358.0, 40),
        z=z, zi=zi, rho=rho,
    )
    metric = build_metric(grid)

    jsam_rhow = np.array(metric["rhow"])
    gsam_rhow = _gsam_rhow(rho, adz)

    max_abs = np.max(np.abs(jsam_rhow - gsam_rhow))
    max_rel = max_abs / np.max(np.abs(gsam_rhow))

    print(f"  max |jsam-gSAM| rhow diff = {max_abs:.3e}")
    print(f"  max rel diff              = {max_rel:.3e}")

    # What the pre-fix midpoint would have given
    rhow_naive = np.zeros(nz + 1)
    rhow_naive[1:-1] = 0.5 * (rho[:-1] + rho[1:])
    rhow_naive[0]    = 2.0 * rhow_naive[1]  - rhow_naive[2]
    rhow_naive[-1]   = 2.0 * rhow_naive[-2] - rhow_naive[-3]
    naive_err = np.max(np.abs(rhow_naive - gsam_rhow)) / np.max(np.abs(gsam_rhow))
    print(f"  (pre-fix naive midpoint max rel diff = {naive_err:.3e})")

    # build_metric stores rhow as jnp arrays in the default dtype (float32
    # without x64), which caps agreement at ~1e-7.  The naive midpoint was
    # wrong by ~2e-3 on this grid, so anything below 1e-6 demonstrates the
    # fix.  With jax_enable_x64 the error collapses to ~1e-15.
    assert max_rel < 1e-6, \
        f"jsam rhow does NOT match gSAM within tolerance: rel={max_rel:.3e}"
    assert naive_err > 1e-4, \
        "pre-fix error too small — check test grid"
    print("PASS — item #10 rhow adz-cross-weighted formula matches gSAM")


if __name__ == "__main__":
    main()
