"""Dump inputs and jsam outputs for test_buoyancy matching tests.

Cases
-----
buoy_neutral    — state == reference → zero buoyancy
buoy_qv_anom    — qv = qv0 + 1e-3 kg/kg; else reference
buoy_T_anom     — tabs = tabs0 + 1 K;   else reference
buoy_cloud      — qcl/qci non-zero anomaly; else reference
buoy_native_dz  — like buoy_T_anom but with gSAM lat_720_dyvar dz(74)

Binary `inputs.bin` layout (little-endian stream, no record markers)
-------------------------------------------------------------------
    i4 nz, ny, nx
    f4 g, epsv
    f4 tabs0(nz), qv0(nz), qn0(nz), qp0(nz)
    f4 dz(nz)
    f4 TABS(nz, ny, nx)            [C order]
    f4 QV  (nz, ny, nx)
    f4 QC  (nz, ny, nx)
    f4 QI  (nz, ny, nx)
    f4 QR  (nz, ny, nx)
    f4 QS  (nz, ny, nx)
    f4 QG  (nz, ny, nx)

Output (both jsam_out.bin and fortran_out.bin):
    f4 buoy_face(nz+1, ny, nx)     [C order]
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
# Physical constants (gSAM params.f90 values)
# ---------------------------------------------------------------------------

G    = np.float32(9.81)
EPSV = np.float32(0.61)        # Rv/Rd - 1


# ---------------------------------------------------------------------------
# Reference profiles + small grids
# ---------------------------------------------------------------------------

def _uniform_grid(nz: int = 8):
    ny, nx = 4, 4
    # Uniform dz = 200 m for the first 4 cases; gSAM still requires
    # betu/betd interpolation so we expose the formula on trivial dz.
    dz = np.full(nz, 200.0, dtype=np.float32)
    # Tropical reference profile (roughly ERA5 sounding shape).
    tabs0 = np.linspace(295.0, 210.0, nz, dtype=np.float32)
    qv0   = np.linspace(15e-3, 1e-6, nz, dtype=np.float32)
    qn0   = np.zeros(nz, dtype=np.float32)
    qp0   = np.zeros(nz, dtype=np.float32)
    return ny, nx, dz, tabs0, qv0, qn0, qp0


def _native_dz_grid():
    """Use gSAM lat_720_dyvar dz(74) from IRMALoader."""
    from jsam.utils.IRMALoader import IRMALoader
    g = IRMALoader().grid
    dz = np.asarray(g["dz"], dtype=np.float32)
    nz = len(dz)
    ny, nx = 4, 4
    # Build tabs0, qv0 from a simple tropical profile at the 74-level z
    z = np.asarray(g["z"])
    tabs0 = (295.0 - 6.5e-3 * z).clip(min=200.0).astype(np.float32)
    qv0   = (15e-3 * np.exp(-z / 2500.0)).astype(np.float32)
    qn0   = np.zeros(nz, dtype=np.float32)
    qp0   = np.zeros(nz, dtype=np.float32)
    return ny, nx, dz, tabs0, qv0, qn0, qp0


def _broadcast(profile: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """(nz,) → (nz, ny, nx)."""
    return np.broadcast_to(profile[:, None, None], (profile.size, ny, nx)).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-case state builders
# ---------------------------------------------------------------------------

def _build_state(case: str):
    if case == "buoy_native_dz":
        ny, nx, dz, tabs0, qv0, qn0, qp0 = _native_dz_grid()
    else:
        ny, nx, dz, tabs0, qv0, qn0, qp0 = _uniform_grid()
    nz = dz.size

    TABS = _broadcast(tabs0, ny, nx).copy()
    QV   = _broadcast(qv0,   ny, nx).copy()
    QC   = np.zeros((nz, ny, nx), dtype=np.float32)
    QI   = np.zeros((nz, ny, nx), dtype=np.float32)
    QR   = np.zeros((nz, ny, nx), dtype=np.float32)
    QS   = np.zeros((nz, ny, nx), dtype=np.float32)
    QG   = np.zeros((nz, ny, nx), dtype=np.float32)

    if case == "buoy_neutral":
        pass  # already at reference
    elif case == "buoy_qv_anom":
        QV += np.float32(1e-3)
    elif case in ("buoy_T_anom", "buoy_native_dz"):
        TABS += np.float32(1.0)
    elif case == "buoy_cloud":
        QC += np.float32(0.5e-3)
        QI += np.float32(0.3e-3)
    else:
        raise ValueError(f"unknown case: {case}")

    return (nz, ny, nx, dz, tabs0, qv0, qn0, qp0,
            TABS, QV, QC, QI, QR, QS, QG)


# ---------------------------------------------------------------------------
# jsam side — call _buoyancy_W and return (nz+1, ny, nx) float32
# ---------------------------------------------------------------------------

def _jsam_buoy(nz, ny, nx, dz, tabs0, qv0, qn0, qp0,
               TABS, QV, QC, QI, QR, QS, QG):
    import jax
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp

    from jsam.core.state import ModelState
    from jsam.core.step import _buoyancy_W

    zeros_s = jnp.zeros((nz, ny, nx))
    state = ModelState(
        U=jnp.zeros((nz, ny, nx + 1)),
        V=jnp.zeros((nz, ny + 1, nx)),
        W=jnp.zeros((nz + 1, ny, nx)),
        TABS=jnp.asarray(TABS),
        QV=jnp.asarray(QV), QC=jnp.asarray(QC), QI=jnp.asarray(QI),
        QR=jnp.asarray(QR), QS=jnp.asarray(QS), QG=jnp.asarray(QG),
        TKE=zeros_s,
        p_prev=None, p_pprev=None,
        nstep=0, time=0.0,
    )

    buo = _buoyancy_W(
        state,
        tabs0=jnp.asarray(tabs0),
        qv0  =jnp.asarray(qv0),
        dz   =jnp.asarray(dz),
        g=float(G), epsv=float(EPSV),
        qn0=jnp.asarray(qn0),
        qp0=jnp.asarray(qp0),
    )
    return np.asarray(buo, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    case = sys.argv[1]
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    (nz, ny, nx, dz, tabs0, qv0, qn0, qp0,
     TABS, QV, QC, QI, QR, QS, QG) = _build_state(case)

    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(struct.pack("ff", float(G), float(EPSV)))
        f.write(tabs0.astype(np.float32).tobytes())
        f.write(qv0  .astype(np.float32).tobytes())
        f.write(qn0  .astype(np.float32).tobytes())
        f.write(qp0  .astype(np.float32).tobytes())
        f.write(dz   .astype(np.float32).tobytes())
        for a in (TABS, QV, QC, QI, QR, QS, QG):
            f.write(a.astype(np.float32).tobytes(order="C"))

    buo = _jsam_buoy(nz, ny, nx, dz, tabs0, qv0, qn0, qp0,
                     TABS, QV, QC, QI, QR, QS, QG)
    write_bin(workdir / "jsam_out.bin", buo.ravel(order="C"))
    print(f"[buoyancy] case={case}  nz={nz}  ny={ny}  nx={nx}  "
          f"n_out={(nz+1)*ny*nx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
