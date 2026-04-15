"""Re-uses the fixtures from jsam/tests/unit/test_radiation.py and dumps:
   - inputs.bin    (in the format the Fortran driver expects)
   - jsam_out.bin  (jsam-side answer for the same case)

Called once per case from run.sh:

   python dump_inputs.py interp1d_midpoint
   python dump_inputs.py interp1d_clamp_below
   python dump_inputs.py interp1d_clamp_above
   python dump_inputs.py rad_proc_magnitude
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

from jsam.core.physics.radiation import _interp1d, RadForcing, rad_proc  # noqa: E402


def _make_rad_proc_state(nz, ny, nx):
    """Minimal ModelState with TABS=300 and zero velocities."""
    from jsam.core.state import ModelState
    return ModelState(
        U     = jnp.zeros((nz, ny, nx + 1)),
        V     = jnp.zeros((nz, ny + 1, nx)),
        W     = jnp.zeros((nz + 1, ny, nx)),
        TABS  = jnp.full((nz, ny, nx), 300.0),
        QV    = jnp.zeros((nz, ny, nx)),
        QC    = jnp.zeros((nz, ny, nx)),
        QI    = jnp.zeros((nz, ny, nx)),
        QR    = jnp.zeros((nz, ny, nx)),
        QS    = jnp.zeros((nz, ny, nx)),
        QG    = jnp.zeros((nz, ny, nx)),
        TKE   = jnp.zeros((nz, ny, nx)),
        nstep = 0,
        time  = 0.0,
    )


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]

    # ------------------------------------------------------------------
    # interp1d_midpoint: xp=[0,2], fp=[0,4], x=1.0 -> 2.0
    # ------------------------------------------------------------------
    if mode == "interp1d_midpoint":
        xp = np.array([0.0, 2.0], dtype=np.float32)
        fp = np.array([0.0, 4.0], dtype=np.float32)
        x  = np.float32(1.0)
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", 2))        # n = len(xp)
            f.write(xp.tobytes())
            f.write(fp.tobytes())
            f.write(struct.pack("f", float(x)))
        result = float(_interp1d(jnp.array(x), jnp.array(xp), jnp.array(fp)))
        write_bin(workdir / "jsam_out.bin", np.array([result], dtype=np.float32))
        return 0

    # ------------------------------------------------------------------
    # interp1d_clamp_below: xp=[1,2], fp=[5,10], x=0.0 -> 5.0
    # ------------------------------------------------------------------
    if mode == "interp1d_clamp_below":
        xp = np.array([1.0, 2.0], dtype=np.float32)
        fp = np.array([5.0, 10.0], dtype=np.float32)
        x  = np.float32(0.0)
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", 2))
            f.write(xp.tobytes())
            f.write(fp.tobytes())
            f.write(struct.pack("f", float(x)))
        result = float(_interp1d(jnp.array(x), jnp.array(xp), jnp.array(fp)))
        write_bin(workdir / "jsam_out.bin", np.array([result], dtype=np.float32))
        return 0

    # ------------------------------------------------------------------
    # interp1d_clamp_above: xp=[1,2], fp=[5,10], x=5.0 -> 10.0
    # ------------------------------------------------------------------
    if mode == "interp1d_clamp_above":
        xp = np.array([1.0, 2.0], dtype=np.float32)
        fp = np.array([5.0, 10.0], dtype=np.float32)
        x  = np.float32(5.0)
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", 2))
            f.write(xp.tobytes())
            f.write(fp.tobytes())
            f.write(struct.pack("f", float(x)))
        result = float(_interp1d(jnp.array(x), jnp.array(xp), jnp.array(fp)))
        write_bin(workdir / "jsam_out.bin", np.array([result], dtype=np.float32))
        return 0

    # ------------------------------------------------------------------
    # rad_proc_magnitude: TABS=300, q_val=5e-4 K/s, dt=30 -> TABS=315 everywhere
    #   Uses nz=4, ny=3, nx=5 (small, arbitrary)
    # ------------------------------------------------------------------
    if mode == "rad_proc_magnitude":
        nz, ny, nx = 4, 3, 5
        dt    = 30.0
        q_val = 5.0e-4   # K/s
        # z_model: cell centres (doesn't matter much, just needs to be inside z_prof)
        z_model = np.linspace(500.0, 4000.0, nz).astype(np.float32)
        # q_profile: constant q_val everywhere
        q_profile = np.full(nz, q_val, dtype=np.float32)

        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("iii", nz, ny, nx))
            f.write(np.full((nz, ny, nx), 300.0, dtype=np.float32).tobytes(order="C"))
            f.write(q_profile.tobytes())      # q_profile(nz) K/s
            f.write(z_model.tobytes())        # z_model(nz) m
            f.write(struct.pack("f", dt))

        # jsam side: build RadForcing with constant q_val, run rad_proc
        z_prof  = jnp.array([0.0, 20000.0])
        profile = jnp.array([q_val, q_val])
        forcing = RadForcing.constant(profile, z_prof)
        state   = _make_rad_proc_state(nz, ny, nx)
        metric  = {"z": jnp.array(z_model)}
        new_st  = rad_proc(state, metric, forcing, dt=dt)
        write_bin(workdir / "jsam_out.bin",
                  np.array(new_st.TABS, dtype=np.float32).ravel(order="C"))
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
