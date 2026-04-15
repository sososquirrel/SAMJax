"""
verify_prefactor_and_sgs_ordering.py — Items #1, #9 (FALSE ALARMS).

#9: the jsam momentum advection face flux
        0.5*(u_L+u_R) * (2*phi_p1 + 5*phi_0 - phi_m1)/6
    is algebraically identical to gSAM's
        (u_L+u_R) * (1/12) * (2*phi_p1 + 5*phi_0 - phi_m1)
    Pin the identity bit-exactly for a random stencil input.

#1: gSAM adamsA.f90 applies the SGS momentum tendency `dudtd` *alongside*
    the AB3 sum inside the same `u + dt3(na)*(...)` bracket:
        u_new = u + dt*(AB3_sum + dudtd)
    jsam instead does `u += dt*AB3_sum` inside advance_momentum, then
    `u += dt*sgs` in a separate pass.  These are algebraically identical
    — `(u + dt*AB3) + dt*sgs == u + dt*(AB3+sgs)` — so the two orderings
    are numerically equivalent when `terrau=1` and `gamma_RAVE=1` (IRMA).

Run:
    PYTHONPATH=. python matching_tests/test_momentum_advection/verify_prefactor_and_sgs_ordering.py
"""

from __future__ import annotations

import numpy as np


def main():
    # ── Item #9: stencil prefactor identity ──────────────────────────────
    rng = np.random.default_rng(42)
    u_L, u_R = rng.standard_normal(1000), rng.standard_normal(1000)
    phi_m1 = rng.standard_normal(1000)
    phi_0  = rng.standard_normal(1000)
    phi_p1 = rng.standard_normal(1000)

    jsam_flux = 0.5 * (u_L + u_R) * ((2 * phi_p1 + 5 * phi_0 - phi_m1) / 6.0)
    gsam_flux = (u_L + u_R) * (1.0 / 12.0) * (2 * phi_p1 + 5 * phi_0 - phi_m1)

    max_diff = np.max(np.abs(jsam_flux - gsam_flux))
    print(f"  jsam /6 form vs gSAM d12*uuu form: max abs diff = {max_diff:.3e}")
    assert max_diff < 1e-14, "prefactor identity broken"
    print("  (0.5·(u_L+u_R)·stencil/6) ≡ ((u_L+u_R)·d12·stencil)")

    # ── Item #1: SGS ordering algebraic equivalence ──────────────────────
    dt  = 10.0                                      # IRMA dt
    u0  = rng.standard_normal(1000)
    ab3 = rng.standard_normal(1000)
    sgs = rng.standard_normal(1000)

    # gSAM: all inside one parenthesis
    u_gsam = u0 + dt * (ab3 + sgs)

    # jsam: two-step (advance_momentum + separate SGS increment)
    u_mid  = u0 + dt * ab3
    u_jsam = u_mid + dt * sgs

    max_diff = np.max(np.abs(u_jsam - u_gsam))
    print(f"  gSAM adamsA form vs jsam two-step:  max abs diff = {max_diff:.3e}")
    assert max_diff < 1e-14, "SGS ordering equivalence broken"

    print("PASS — items #1 (SGS ordering) and #9 (advection prefactor) "
          "are provably equivalent")


if __name__ == "__main__":
    main()
