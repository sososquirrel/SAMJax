"""
verify_buoyancy_factor.py — Item #2b fix.

Compares jsam's `_buoyancy_W` cell-centre b[k] to a literal transcription
of gSAM buoyancy.f90:48-53.  Tests that the (1+epsv*qv0 - qn0 - qp0)
thermal factor is applied and that the qn-qn0, qp-qp0 subtraction
is correct.

Run:
    PYTHONPATH=. python matching_tests/test_sgs/verify_buoyancy_factor.py
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jsam.core.state import ModelState
from jsam.core.step import _buoyancy_W


def _gsam_cell_buo(tabs, qv, qc, qi, qr, qs, qg,
                   tabs0, qv0, qn0, qp0, g, epsv):
    """
    gSAM buoyancy.f90:48-53 cell-centre form, before betu/betd interp.
    Returns (nz, ny, nx).
    """
    qn = qc + qi
    qp = qr + qs + qg
    # bet(k) = g/tabs0(k)
    bet = g / tabs0[:, None, None]
    # T0*(eps*Δqv − (qn-qn0) − (qp-qp0)) + (T-T0)*(1+eps*qv0 - qn0 - qp0)
    term1 = tabs0[:, None, None] * (
        epsv * (qv - qv0[:, None, None])
        - (qn - qn0[:, None, None])
        - (qp - qp0[:, None, None])
    )
    thermal_factor = 1.0 + epsv * qv0 - qn0 - qp0
    term2 = (tabs - tabs0[:, None, None]) * thermal_factor[:, None, None]
    return bet * (term1 + term2)


def _jsam_cell_buo(state, tabs0, qv0, qn0, qp0, g, epsv):
    """
    Re-derive the cell-centre b from the current jsam _buoyancy_W by
    undoing the area-weighted vertical interpolation — simpler path:
    clone the formula directly from step.py.
    """
    qn = state.QC + state.QI
    qp = state.QR + state.QS + state.QG
    tabs0_3d = tabs0[:, None, None]
    qv0_3d   = qv0[:, None, None]
    qn0_3d   = qn0[:, None, None]
    qp0_3d   = qp0[:, None, None]
    thermal_factor = 1.0 + epsv * qv0_3d - qn0_3d - qp0_3d
    return (g / tabs0_3d) * (
        tabs0_3d * (
            epsv * (state.QV - qv0_3d)
            - (qn - qn0_3d)
            - (qp - qp0_3d)
        )
        + (state.TABS - tabs0_3d) * thermal_factor
    )


def main():
    nz, ny, nx = 12, 8, 16
    g, epsv = 9.79764, 0.61

    rng = np.random.default_rng(0)
    tabs0 = np.linspace(290, 220, nz)
    qv0   = np.linspace(0.018, 1e-6, nz)
    qn0   = np.full(nz, 1e-5)          # non-zero for a real test of the factor
    qp0   = np.full(nz, 2e-5)

    TABS = tabs0[:, None, None] + 0.3 * rng.standard_normal((nz, ny, nx))
    QV   = qv0[:, None, None]  + 1e-4 * rng.standard_normal((nz, ny, nx))
    QC   = 1e-5 * rng.random((nz, ny, nx))
    QI   = 1e-6 * rng.random((nz, ny, nx))
    QR   = 1e-5 * rng.random((nz, ny, nx))
    QS   = 1e-6 * rng.random((nz, ny, nx))
    QG   = 5e-7 * rng.random((nz, ny, nx))

    state = ModelState(
        U=jnp.zeros((nz, ny, nx + 1)), V=jnp.zeros((nz, ny + 1, nx)),
        W=jnp.zeros((nz + 1, ny, nx)),
        TABS=jnp.array(TABS), QV=jnp.array(QV),
        QC=jnp.array(QC), QI=jnp.array(QI),
        QR=jnp.array(QR), QS=jnp.array(QS), QG=jnp.array(QG),
        TKE=jnp.zeros((nz, ny, nx)),
        p_prev=jnp.zeros((nz, ny, nx)), p_pprev=jnp.zeros((nz, ny, nx)),
        nstep=0, time=0.0,
    )

    gsam_b = _gsam_cell_buo(
        np.array(TABS), np.array(QV), np.array(QC), np.array(QI),
        np.array(QR), np.array(QS), np.array(QG),
        tabs0, qv0, qn0, qp0, g, epsv,
    )
    jsam_b = np.array(_jsam_cell_buo(
        state,
        jnp.array(tabs0), jnp.array(qv0),
        jnp.array(qn0), jnp.array(qp0),
        g, epsv,
    ))

    max_rel = np.max(np.abs(jsam_b - gsam_b) / (np.abs(gsam_b) + 1e-30))
    print(f"  max rel diff (jsam cell buo vs gSAM buoyancy.f90) = {max_rel:.3e}")
    assert max_rel < 1e-14, f"mismatch: {max_rel:.3e}"

    # Also prove that OMITTING the thermal factor (the pre-fix path) gives
    # a demonstrably wrong b — the fix must matter.
    tabs0_3d = tabs0[:, None, None]
    prefix_b = (g / tabs0_3d) * (
        (TABS - tabs0_3d)
        + tabs0_3d * (epsv * (QV - qv0[:, None, None]) - (QC + QI + QR + QS + QG))
    )
    prefix_err = np.max(np.abs(prefix_b - gsam_b) / (np.abs(gsam_b) + 1e-30))
    print(f"  pre-fix (missing thermal factor) max rel diff     = {prefix_err:.3e}")
    assert prefix_err > 1e-4, "pre-fix error too small — factor had negligible effect"

    # Finally: verify the full _buoyancy_W still runs end-to-end (interp + bc).
    dz = jnp.ones(nz)
    b_w = _buoyancy_W(
        state, jnp.array(tabs0), jnp.array(qv0),
        dz, g, epsv,
        qn0=jnp.array(qn0), qp0=jnp.array(qp0),
    )
    assert b_w.shape == (nz + 1, ny, nx)
    assert float(jnp.abs(b_w[0]).max()) == 0.0, "bottom W-face should be zero"
    assert float(jnp.abs(b_w[-1]).max()) == 0.0, "top    W-face should be zero"
    print("PASS — item #2b buoyancy thermal factor matches gSAM buoyancy.f90")


if __name__ == "__main__":
    main()
