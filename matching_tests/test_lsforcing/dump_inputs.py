"""
Dump inputs and jsam outputs for test_lsforcing matching tests.

Cases (subsidence mode):
  subsidence.zero_w         — wsub=0 → tend=0
  subsidence.uniform_phi    — uniform phi → tend=0
  subsidence.magnitude      — nz=4, dz=500, T[k]=300+k, wsub=+1 → tend[1:-1]=-0.002

Cases (ls_proc_direct mode):
  ls_proc_direct.dtls_magnitude — dtls=2e-4, dt=30, wsub=0 → TABS += 0.006

inputs.bin for subsidence:
  int32 nz, int32 ny, int32 nx
  float32 phi(nz,ny,nx), wsub(nz), dz(nz)

inputs.bin for ls_proc_direct:
  int32 nz, int32 ny, int32 nx
  float32 TABS(nz,ny,nx), QV(nz,ny,nx)
  float32 dtls(nz), dqls(nz), wsub(nz), dz(nz)
  float32 dt
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, "/glade/u/home/sabramian/jsam")

from common.bin_io import write_bin                              # noqa: E402
from jsam.core.physics.lsforcing import (                       # noqa: E402
    _subsidence_tend, LargeScaleForcing, ls_proc,
)
from jsam.core.state import ModelState                          # noqa: E402
from jsam.core.dynamics.pressure import build_metric            # noqa: E402
from jsam.core.grid.latlon import LatLonGrid                    # noqa: E402

WORKDIR = HERE / "work"


def _write_subsidence_inputs(nz, ny, nx, phi, wsub, dz):
    with open(WORKDIR / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(np.asarray(phi, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(wsub, dtype=np.float32).tobytes())
        f.write(np.asarray(dz, dtype=np.float32).tobytes())


def _write_ls_proc_inputs(nz, ny, nx, TABS, QV, dtls, dqls, wsub, dz, dt):
    with open(WORKDIR / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(np.asarray(TABS, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(QV, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(dtls, dtype=np.float32).tobytes())
        f.write(np.asarray(dqls, dtype=np.float32).tobytes())
        f.write(np.asarray(wsub, dtype=np.float32).tobytes())
        f.write(np.asarray(dz, dtype=np.float32).tobytes())
        f.write(struct.pack("f", float(dt)))


def _make_trivial_state(nz, ny, nx, TABS_arr, QV_arr):
    return ModelState(
        U=jnp.zeros((nz, ny, nx + 1)),
        V=jnp.zeros((nz, ny + 1, nx)),
        W=jnp.zeros((nz + 1, ny, nx)),
        TABS=jnp.array(TABS_arr),
        QV=jnp.array(QV_arr),
        QC=jnp.zeros((nz, ny, nx)),
        QI=jnp.zeros((nz, ny, nx)),
        QR=jnp.zeros((nz, ny, nx)),
        QS=jnp.zeros((nz, ny, nx)),
        QG=jnp.zeros((nz, ny, nx)),
        TKE=jnp.zeros((nz, ny, nx)),
        nstep=0,
        time=0.0,
    )


def _trivial_metric(nz, ny, nx, dz_val=500.0):
    """Minimal flat metric for ls_proc (only needs 'z', 'dz', 'nx', 'imu')."""
    lat = np.linspace(-0.1, 0.1, ny)
    lon = np.linspace(0.0, 359.0 / nx * nx, nx)
    z   = np.arange(nz, dtype=float) * dz_val + dz_val / 2
    zi  = np.concatenate([[0.0], 0.5 * (z[:-1] + z[1:]), [z[-1] + dz_val / 2]])
    rho = 1.2 * np.exp(-z / 8000.0)
    grid   = LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)
    return build_metric(grid)


def main() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    case = sys.argv[1]

    # -----------------------------------------------------------------------
    # subsidence cases
    # -----------------------------------------------------------------------

    if case == "subsidence.zero_w":
        nz, ny, nx = 8, 4, 6
        phi  = np.ones((nz, ny, nx), dtype=np.float32) * 300.0
        wsub = np.zeros(nz, dtype=np.float32)
        dz   = np.ones(nz, dtype=np.float32) * 500.0
        _write_subsidence_inputs(nz, ny, nx, phi, wsub, dz)
        tend = np.asarray(_subsidence_tend(
            jnp.array(phi), jnp.array(wsub), jnp.array(dz)
        ), dtype=np.float32)
        write_bin(WORKDIR / "jsam_out.bin", tend)

    elif case == "subsidence.uniform_phi":
        nz, ny, nx = 8, 4, 6
        phi  = np.ones((nz, ny, nx), dtype=np.float32) * 295.0
        wsub = np.linspace(-2.0, 2.0, nz).astype(np.float32)
        dz   = np.ones(nz, dtype=np.float32) * 500.0
        _write_subsidence_inputs(nz, ny, nx, phi, wsub, dz)
        tend = np.asarray(_subsidence_tend(
            jnp.array(phi), jnp.array(wsub), jnp.array(dz)
        ), dtype=np.float32)
        write_bin(WORKDIR / "jsam_out.bin", tend)

    elif case == "subsidence.magnitude":
        nz, ny, nx = 4, 2, 3
        k_arr = np.arange(nz, dtype=np.float32)
        phi  = (300.0 + k_arr)[:, None, None] * np.ones((1, ny, nx), dtype=np.float32)
        wsub = np.ones(nz, dtype=np.float32) * 1.0
        dz   = np.ones(nz, dtype=np.float32) * 500.0
        _write_subsidence_inputs(nz, ny, nx, phi, wsub, dz)
        tend = np.asarray(_subsidence_tend(
            jnp.array(phi), jnp.array(wsub), jnp.array(dz)
        ), dtype=np.float32)
        write_bin(WORKDIR / "jsam_out.bin", tend)

    # -----------------------------------------------------------------------
    # ls_proc case
    # -----------------------------------------------------------------------

    elif case == "ls_proc_direct.dtls_magnitude":
        nz, ny, nx = 8, 4, 6
        dt   = 30.0
        rate = 2e-4
        TABS = np.full((nz, ny, nx), 300.0, dtype=np.float32)
        QV   = np.full((nz, ny, nx), 1e-2, dtype=np.float32)
        dtls = np.full(nz, rate, dtype=np.float32)
        dqls = np.zeros(nz, dtype=np.float32)
        wsub = np.zeros(nz, dtype=np.float32)
        dz   = np.ones(nz, dtype=np.float32) * 500.0
        _write_ls_proc_inputs(nz, ny, nx, TABS, QV, dtls, dqls, wsub, dz, dt)

        # jsam: use LargeScaleForcing.constant with no subsidence
        from jsam.core.physics.lsforcing import LargeScaleForcing
        metric = _trivial_metric(nz, ny, nx, dz_val=500.0)
        state  = _make_trivial_state(
            nz, ny, nx,
            TABS_arr=jnp.full((nz, ny, nx), 300.0),
            QV_arr=jnp.full((nz, ny, nx), 1e-2),
        )
        z_prof  = jnp.array([0.0, 20000.0])
        forcing = LargeScaleForcing.constant(
            dtls=jnp.array([rate, rate]),
            dqls=jnp.zeros(2),
            wsub=jnp.zeros(2),
            z_prof=z_prof,
        )
        new_st = ls_proc(state, metric, forcing, dt=dt)
        TABS_new = np.asarray(new_st.TABS, dtype=np.float32)
        QV_new   = np.asarray(new_st.QV,   dtype=np.float32)
        out = np.concatenate([TABS_new.ravel(), QV_new.ravel()])
        write_bin(WORKDIR / "jsam_out.bin", out)

    else:
        raise SystemExit(f"unknown case: {case}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
