"""Dump inputs and jsam outputs for test_pressure_rhs_div.

Cases
-----
div_uniform_u  — U=1 everywhere, V=W=0. RHS should be ~0.
div_linear_u   — U = a*i. RHS = const * imu(j).
div_v_with_ady — V = cos(lat). Tests muv/ady metric.
div_w_with_rho — W linear in z. Tests rhow/rho density weighting.
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


def _small_metric(nz=4, ny=6, nx=8):
    """Build a small synthetic spherical metric matching gSAM conventions."""
    lat_deg = np.linspace(-60, 60, ny, dtype=np.float64)
    lat_rad = np.deg2rad(lat_deg)
    cos_lat = np.cos(lat_rad).astype(np.float32)
    imu = (1.0 / cos_lat).astype(np.float32)

    # ady = non-uniform dy stretching (simulate dyvar grid)
    ady = (0.5 + 0.5 * cos_lat).astype(np.float32)  # smaller at poles

    # muv = cos(lat) at V-faces (ny+1)
    lat_v = np.zeros(ny + 1, dtype=np.float64)
    lat_v[0] = lat_rad[0] - 0.5 * (lat_rad[1] - lat_rad[0])
    lat_v[-1] = lat_rad[-1] + 0.5 * (lat_rad[-1] - lat_rad[-2])
    for j in range(1, ny):
        lat_v[j] = 0.5 * (lat_rad[j - 1] + lat_rad[j])
    muv = np.cos(lat_v).astype(np.float32)

    # Vertical metric
    dz = np.float32(500.0)
    adz = np.ones(nz, dtype=np.float32)  # uniform dz
    rho = np.linspace(1.2, 0.3, nz, dtype=np.float32)
    rhow = np.zeros(nz + 1, dtype=np.float32)
    rhow[0] = rho[0]
    rhow[-1] = rho[-1]
    for k in range(1, nz):
        rhow[k] = 0.5 * (rho[k - 1] + rho[k])

    dx = np.float32(25000.0)
    dy = np.float32(25000.0)

    return dict(nz=nz, ny=ny, nx=nx, dx=dx, dy=dy, dz=dz,
                imu=imu, ady=ady, muv=muv, adz=adz, rho=rho, rhow=rhow,
                cos_lat=cos_lat, lat_rad=lat_rad.astype(np.float32))


def _build_state(case: str):
    m = _small_metric()
    nz, ny, nx = m["nz"], m["ny"], m["nx"]
    dt_at = np.float32(10.0)  # dt * at, at=1 for Euler step

    U = np.zeros((nz, ny, nx + 1), dtype=np.float32)
    V = np.zeros((nz, ny + 1, nx), dtype=np.float32)
    W = np.zeros((nz + 1, ny, nx), dtype=np.float32)

    if case == "div_uniform_u":
        U[:] = 1.0
    elif case == "div_linear_u":
        for i in range(nx + 1):
            U[:, :, i] = float(i) * 0.5
    elif case == "div_v_with_ady":
        for j in range(ny + 1):
            V[:, j, :] = m["muv"][j] * 3.0
    elif case == "div_w_with_rho":
        for k in range(nz + 1):
            W[k, :, :] = float(k) * 0.1
    else:
        raise ValueError(f"unknown case: {case}")

    return m, dt_at, U, V, W


def _jsam_press_rhs(m, dt_at, U, V, W):
    """Compute pressure RHS using jsam's approach."""
    import jax
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp

    nz, ny, nx = m["nz"], m["ny"], m["nx"]
    imu = jnp.asarray(m["imu"])
    ady = jnp.asarray(m["ady"])
    muv = jnp.asarray(m["muv"])
    cos_lat = jnp.asarray(m["cos_lat"])
    adz = jnp.asarray(m["adz"])
    rho = jnp.asarray(m["rho"])
    rhow = jnp.asarray(m["rhow"])

    Uj = jnp.asarray(U)
    Vj = jnp.asarray(V)
    Wj = jnp.asarray(W)

    dx, dy, dz = float(m["dx"]), float(m["dy"]), float(m["dz"])
    dta = 1.0 / float(dt_at)

    # jsam-style divergence (pressure.py approach)
    div_u = imu[None, :, None] * (Uj[:, :, 1:] - Uj[:, :, :-1]) / dx

    # V divergence: using muv and ady
    div_v = imu[None, :, None] / (dy * ady[None, :, None]) * \
            (muv[None, 1:, None] * Vj[:, 1:, :] - muv[None, :-1, None] * Vj[:, :-1, :])

    # W divergence: with density weighting
    rdz = 1.0 / (adz[:, None, None] * dz)
    rup = rhow[1:, None, None] / rho[:, None, None] * rdz
    rdn = rhow[:-1, None, None] / rho[:, None, None] * rdz
    div_w = Wj[1:, :, :] * rup - Wj[:-1, :, :] * rdn

    rhs = (div_u + div_v + div_w) * dta
    return np.asarray(rhs, dtype=np.float32)


def main() -> int:
    case = sys.argv[1]
    m, dt_at, U, V, W = _build_state(case)
    nz, ny, nx = m["nz"], m["ny"], m["nx"]

    # Write inputs.bin
    with open("inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(struct.pack("ffff", float(m["dx"]), float(m["dy"]),
                            float(m["dz"]), float(dt_at)))
        f.write(m["imu"].astype(np.float32).tobytes())
        f.write(m["ady"].astype(np.float32).tobytes())
        f.write(m["muv"].astype(np.float32).tobytes())
        f.write(m["adz"].astype(np.float32).tobytes())
        f.write(m["rho"].astype(np.float32).tobytes())
        f.write(m["rhow"].astype(np.float32).tobytes())
        f.write(U.astype(np.float32).tobytes(order="C"))
        f.write(V.astype(np.float32).tobytes(order="C"))
        f.write(W.astype(np.float32).tobytes(order="C"))

    # jsam side
    rhs = _jsam_press_rhs(m, dt_at, U, V, W)
    write_bin("jsam_out.bin", rhs.ravel(order="C"))

    print(f"[pressure_rhs] case={case}  nz={nz}  ny={ny}  nx={nx}  "
          f"rhs range=[{rhs.min():.4e}, {rhs.max():.4e}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
