"""Fixture builder for test_adams.

Produces inputs.bin + jsam_out.bin for a single case. Each case is
small (nx, ny, nz handful) so the Fortran driver can hard-code
dimensions and avoid buffer gymnastics.

Usage (from run.sh):
    python dump_inputs.py <case>

Cases
-----
adamsA_zero     — identity: all tendencies zero
adamsA_ab3      — 3-slot AB3 update, random-ish tendencies
adamsA_diff     — same as ab3 plus a non-zero diffusion (dudtd) slot

adamsB_noop     — p_prev is None → jsam returns state unchanged
adamsB_const_p  — spatially uniform p → zero gradient → identity

Binary layout (matches driver.f90 reader)
-----------------------------------------
adamsA_* cases:
    i4 nz, i4 ny, i4 nx
    f4 at, f4 bt, f4 ct, f4 dt
    f4 phi(nz,ny,nx)          # single 3D field (we test per-component)
    f4 tend_n   (nz,ny,nx)
    f4 tend_nm1 (nz,ny,nx)
    f4 tend_nm2 (nz,ny,nx)
    f4 tend_d   (nz,ny,nx)    # dudtd / dvdtd / dwdtd (=0 except adamsA_diff)

adamsB_* cases:
    i4 nz, i4 ny, i4 nx
    f4 dt
    f4 dx, f4 dy, f4 dz       # uniform spacing in the test
    f4 bt, f4 ct
    i4 has_pprev              # 0 or 1
    f4 U(nz, ny, nx+1)
    f4 V(nz, ny+1, nx)
    f4 W(nz+1, ny, nx)
    f4 p_prev(nz, ny, nx)
    f4 p_pprev(nz, ny, nx)    # only present if has_pprev=1
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
SAMJAX_ROOT = MT_ROOT.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, str(SAMJAX_ROOT))

from common.bin_io import write_bin  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny test grid (shared across all cases)
# ---------------------------------------------------------------------------

NZ, NY, NX = 4, 6, 8
DT = 10.0


# ---------------------------------------------------------------------------
# adamsA — Adams-Bashforth momentum update (ab_step)
# ---------------------------------------------------------------------------

def _ab_case_arrays(case: str):
    """Deterministic inputs for each adamsA case."""
    rng = np.random.default_rng(seed={"adamsA_zero": 0,
                                      "adamsA_ab3":  1,
                                      "adamsA_diff": 2}[case])
    phi = rng.standard_normal((NZ, NY, NX)).astype(np.float32) * 3.0

    if case == "adamsA_zero":
        tend_n   = np.zeros_like(phi)
        tend_nm1 = np.zeros_like(phi)
        tend_nm2 = np.zeros_like(phi)
        tend_d   = np.zeros_like(phi)
    elif case == "adamsA_ab3":
        tend_n   = rng.standard_normal((NZ, NY, NX)).astype(np.float32) * 0.2
        tend_nm1 = rng.standard_normal((NZ, NY, NX)).astype(np.float32) * 0.2
        tend_nm2 = rng.standard_normal((NZ, NY, NX)).astype(np.float32) * 0.2
        tend_d   = np.zeros_like(phi)
    elif case == "adamsA_diff":
        tend_n   = rng.standard_normal((NZ, NY, NX)).astype(np.float32) * 0.2
        tend_nm1 = rng.standard_normal((NZ, NY, NX)).astype(np.float32) * 0.2
        tend_nm2 = rng.standard_normal((NZ, NY, NX)).astype(np.float32) * 0.2
        tend_d   = rng.standard_normal((NZ, NY, NX)).astype(np.float32) * 0.1
    else:
        raise ValueError(case)

    # Use gSAM const-dt AB3 weights: (at, bt, ct) = (23/12, -16/12, 5/12)
    at = np.float32(23.0 / 12.0)
    bt = np.float32(-16.0 / 12.0)
    ct = np.float32(5.0 / 12.0)
    return phi, tend_n, tend_nm1, tend_nm2, tend_d, at, bt, ct


def _write_adamsA(case: str, workdir: Path):
    import jax
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp
    from jsam.core.dynamics.timestepping import ab_step

    phi, tn, tnm1, tnm2, td, at, bt, ct = _ab_case_arrays(case)

    # Fortran adamsA formula (gSAM SRC/adamsA.f90):
    #   phi_new = phi + dt * ( at*tn + bt*tnm1 + ct*tnm2 + td )
    #
    # Note: the diffusion term (td, i.e. dudtd) is added WITHOUT any
    # AB weight — it is the current-step SGS diffusion tendency that
    # gSAM accumulates through setsgs / diffuse_mom / etc. before
    # adamsA runs.
    #
    # jsam ab_step only does the AB3 sum:
    #   phi_ab = phi + dt * ( at*tn + bt*tnm1 + ct*tnm2 )
    #
    # So we reproduce gSAM by calling ab_step on the pure advective
    # tendencies and then adding dt*td on top — which is what step.py
    # does via its `_dU_sgs / dU_extra` branches. Calling ab_step with
    # nstep=2 + dt_prev=dt_pprev=dt forces the constant-dt (23,-16,5)/12
    # coefs so we're testing the AB formula itself.
    phi_j   = jnp.asarray(phi)
    phi_ab  = ab_step(
        phi_j, jnp.asarray(tn), jnp.asarray(tnm1), jnp.asarray(tnm2),
        dt=DT, nstep=2, dt_prev=DT, dt_pprev=DT,
    )
    phi_new = np.asarray(phi_ab, dtype=np.float32) + DT * td

    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", NZ, NY, NX))
        f.write(struct.pack("ffff", float(at), float(bt), float(ct), DT))
        f.write(phi.astype(np.float32).tobytes(order="C"))
        f.write(tn.astype(np.float32).tobytes(order="C"))
        f.write(tnm1.astype(np.float32).tobytes(order="C"))
        f.write(tnm2.astype(np.float32).tobytes(order="C"))
        f.write(td.astype(np.float32).tobytes(order="C"))

    write_bin(workdir / "jsam_out.bin", phi_new.ravel(order="C"))


# ---------------------------------------------------------------------------
# adamsB — lagged pressure gradient correction (adams_b)
# ---------------------------------------------------------------------------

def _make_simple_state_and_metric():
    """Minimal ModelState + metric for adamsB on a tiny uniform grid."""
    import jax.numpy as jnp
    from jsam.core.state import ModelState
    from jsam.core.grid.latlon import LatLonGrid
    from jsam.core.dynamics.pressure import build_metric

    lat = np.linspace(-0.3, 0.3, NY)
    lon = np.linspace(0.0,  0.8, NX)
    z   = np.linspace(200.0, 4000.0, NZ)
    dz  = np.diff(np.concatenate([[0.0], z + (z[1] - z[0]) / 2]))
    zi  = np.concatenate([[0.0], z + (z[1] - z[0]) / 2])
    rho = 1.2 * np.exp(-z / 8000.0)
    grid = LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)
    metric = build_metric(grid)

    zeros_s = jnp.zeros((NZ, NY, NX))
    state = ModelState(
        U=jnp.ones((NZ, NY, NX + 1)) * 2.0,
        V=jnp.ones((NZ, NY + 1, NX)) * -1.0,
        W=jnp.zeros((NZ + 1, NY, NX)),
        TABS=jnp.full((NZ, NY, NX), 290.0),
        QV=zeros_s, QC=zeros_s, QI=zeros_s,
        QR=zeros_s, QS=zeros_s, QG=zeros_s,
        TKE=zeros_s,
        p_prev=None, p_pprev=None,
        nstep=2, time=20.0,
    )
    return state, metric


def _write_adamsB(case: str, workdir: Path):
    import jax
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp
    from jsam.core.dynamics.pressure import adams_b

    state, metric = _make_simple_state_and_metric()

    if case == "adamsB_noop":
        # jsam skips adams_b entirely when p_prev is None; we model that
        # by asserting the identity. Write dummy p_prev anyway.
        p_prev = np.zeros((NZ, NY, NX), dtype=np.float32)
        has_pprev = 0
        expected_U = np.asarray(state.U)
        expected_V = np.asarray(state.V)
        expected_W = np.asarray(state.W)
    elif case == "adamsB_const_p":
        # Uniform p → ∇p = 0 → identity within rounding.
        p_prev = np.full((NZ, NY, NX), 1000.0, dtype=np.float32)
        has_pprev = 0
        new_state = adams_b(state, jnp.asarray(p_prev), metric, dt=DT, bt=-0.5)
        expected_U = np.asarray(new_state.U)
        expected_V = np.asarray(new_state.V)
        expected_W = np.asarray(new_state.W)
    else:
        raise ValueError(case)

    # Uniform spacing derived from the metric for the Fortran driver.
    dx = float(metric["dx_lon"])
    dy = float(metric["dy_lat_ref"])
    dz = float(np.asarray(metric["dz"])[0])

    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", NZ, NY, NX))
        f.write(struct.pack("f",   DT))
        f.write(struct.pack("fff", dx, dy, dz))
        f.write(struct.pack("ff",  -0.5, 0.0))   # bt, ct (AB2 bootstrap)
        f.write(struct.pack("i",   has_pprev))
        f.write(np.asarray(state.U, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(state.V, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(state.W, dtype=np.float32).tobytes(order="C"))
        f.write(p_prev.tobytes(order="C"))
        if has_pprev:
            f.write(np.zeros((NZ, NY, NX), dtype=np.float32).tobytes(order="C"))

    out = np.concatenate([
        np.asarray(expected_U, dtype=np.float32).ravel(order="C"),
        np.asarray(expected_V, dtype=np.float32).ravel(order="C"),
        np.asarray(expected_W, dtype=np.float32).ravel(order="C"),
    ])
    write_bin(workdir / "jsam_out.bin", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    case = sys.argv[1]
    if case.startswith("adamsA_"):
        _write_adamsA(case, workdir)
    elif case.startswith("adamsB_"):
        _write_adamsB(case, workdir)
    else:
        raise SystemExit(f"unknown case: {case}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
