"""Re-uses the fixtures from jsam/tests/unit/test_kurant.py and dumps:
   - inputs.bin    (in the format the Fortran driver expects)
   - jsam_out.bin  (jsam-side answer for the same case)

Called once per case from run.sh:

   python dump_inputs.py compute_cfl  <case_name>
   python dump_inputs.py ab2_coefs

For compute_cfl, <case_name> picks one of the test fixtures so the runner
covers every numerical assertion in the python test file.
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
sys.path.insert(0, "/glade/u/home/sabramian/jsam")     # jsam package + tests/

from common.bin_io import write_bin  # noqa: E402

# Re-use the existing python test's helpers verbatim.
from tests.unit.test_kurant import _mock_metric, _zero_velocity  # noqa: E402
from jsam.core.dynamics.kurant import compute_cfl                # noqa: E402
from jsam.core.dynamics.timestepping import ab2_coefs            # noqa: E402


def _mass_centred(U, V, W):
    """Mirror the staggered → mass-centre max that compute_cfl does."""
    U_abs = np.maximum(np.abs(np.asarray(U[:, :, :-1])),
                       np.abs(np.asarray(U[:, :, 1:])))
    V_abs = np.maximum(np.abs(np.asarray(V[:, :-1, :])),
                       np.abs(np.asarray(V[:, 1:, :])))
    W_abs = np.maximum(np.abs(np.asarray(W[:-1, :, :])),
                       np.abs(np.asarray(W[1:, :, :])))
    return U_abs.astype(np.float32), V_abs.astype(np.float32), W_abs.astype(np.float32)


def _write_cfl_inputs(U, V, W, m, dt: float) -> None:
    """Serialize a compute_cfl case in the layout the driver reads."""
    U_a, V_a, W_a = _mass_centred(U, V, W)
    nz, ny, nx = U_a.shape
    dx = float(m["dx_lon"])
    dy = np.asarray(m["dy_lat"], dtype=np.float32)
    dz = np.asarray(m["dz"], dtype=np.float32)
    cl = np.asarray(m["cos_lat"], dtype=np.float32)

    with open(HERE / "work" / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(U_a.tobytes(order="C"))
        f.write(V_a.tobytes(order="C"))
        f.write(W_a.tobytes(order="C"))
        f.write(struct.pack("f", dx))
        f.write(dy.tobytes())
        f.write(dz.tobytes())
        f.write(cl.tobytes())
        f.write(struct.pack("f", float(dt)))


def _make_cfl_case(name: str):
    """Each case mirrors one assertion in test_kurant.py."""
    if name == "zero_velocity":
        nz, ny, nx = 4, 6, 8
        U, V, W = _zero_velocity(nz, ny, nx)
        return U, V, W, _mock_metric(nz, ny, nx), 10.0

    if name == "pure_zonal":
        nz, ny, nx = 4, 6, 8
        U, V, W = _zero_velocity(nz, ny, nx)
        U = U.at[:].set(50.0)
        return U, V, W, _mock_metric(nz, ny, nx), 10.0

    if name == "pure_vertical":
        nz, ny, nx = 4, 6, 8
        U, V, W = _zero_velocity(nz, ny, nx)
        W = W.at[:].set(20.0)
        return U, V, W, _mock_metric(nz, ny, nx), 10.0

    if name == "combines_axes":
        nz, ny, nx = 4, 6, 8
        U, V, W = _zero_velocity(nz, ny, nx)
        U = U.at[:].set(50.0)
        W = W.at[:].set(20.0)
        return U, V, W, _mock_metric(nz, ny, nx), 10.0

    if name == "latitude_narrowing":
        nz, ny, nx = 2, 4, 4
        m = {
            "dx_lon":  10_000.0,
            "dy_lat":  jnp.full((ny,), 10_000.0),
            "dz":      jnp.full((nz,), 1_000.0),
            "cos_lat": jnp.array([1.0, 0.5, 0.1, 1.0]),
        }
        U = jnp.zeros((nz, ny, nx + 1)).at[:].set(50.0)
        V = jnp.zeros((nz, ny + 1, nx))
        W = jnp.zeros((nz + 1, ny, nx))
        return U, V, W, m, 10.0

    raise ValueError(f"unknown compute_cfl case: {name}")


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]

    if mode == "compute_cfl":
        case = sys.argv[2]
        U, V, W, m, dt = _make_cfl_case(case)
        _write_cfl_inputs(U, V, W, m, dt)
        cfl = float(compute_cfl(U, V, W, m, dt=dt))
        write_bin(workdir / "jsam_out.bin", np.array([cfl], dtype=np.float32))
        return 0

    if mode == "ab2_coefs":
        # Cover every (nstep, dt_curr, dt_prev) the python test asserts on.
        cases = [
            (1, 10.0, 10.0),    # constant dt → (1.5, -0.5)
            (5, 10.0, 10.0),
            (0, 10.0, 10.0),    # Euler
            (0, 3.0, 10.0),     # Euler regardless of dt
            (1, 5.0, 10.0),     # r=0.5  → (1.25, -0.25)
            (1, 20.0, 10.0),    # r=2.0  → (2.0, -1.0)
        ]
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", len(cases)))
            for nstep, dc, dp in cases:
                f.write(struct.pack("iff", nstep, float(dc), float(dp)))
        out = []
        for nstep, dc, dp in cases:
            at, bt = ab2_coefs(nstep=nstep, dt_curr=dc, dt_prev=dp)
            out.extend([float(at), float(bt)])
        write_bin(workdir / "jsam_out.bin", np.array(out, dtype=np.float32))
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
