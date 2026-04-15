"""Re-uses the fixtures from jsam/tests/unit/test_nudging.py and dumps:
   - inputs.bin    (in the format the Fortran driver expects)
   - jsam_out.bin  (jsam-side answer for the same case)

Called once per case from run.sh:

   python dump_inputs.py band_mask_full
   python dump_inputs.py band_mask_partial
   python dump_inputs.py nudge_decay
   python dump_inputs.py nudge_zero_outside
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

from jsam.core.physics.nudging import _band_mask, nudge_scalar  # noqa: E402


def _linear_z(nz: int, dz: float = 500.0) -> jnp.ndarray:
    """Cell-centre heights for a uniform-dz column (matches test_nudging.py)."""
    return jnp.arange(nz, dtype=jnp.float32) * dz + 0.5 * dz


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]

    # ------------------------------------------------------------------
    # band_mask_full: nz=10, dz=500, z=[250..4750], z1=0, z2=1e9 -> all 1
    # ------------------------------------------------------------------
    if mode == "band_mask_full":
        nz = 10
        z  = _linear_z(nz, dz=500.0)
        z1, z2 = 0.0, 1.0e9
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", nz))
            f.write(np.array(z, dtype=np.float32).tobytes())
            f.write(struct.pack("ff", z1, z2))
        mask = _band_mask(z, z1, z2)
        write_bin(workdir / "jsam_out.bin", np.array(mask, dtype=np.float32))
        return 0

    # ------------------------------------------------------------------
    # band_mask_partial: nz=10, dz=500, z1=1000, z2=3000 -> 0,0,1,1,1,1,0,0,0,0
    # ------------------------------------------------------------------
    if mode == "band_mask_partial":
        nz = 10
        z  = _linear_z(nz, dz=500.0)
        z1, z2 = 1000.0, 3000.0
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", nz))
            f.write(np.array(z, dtype=np.float32).tobytes())
            f.write(struct.pack("ff", z1, z2))
        mask = _band_mask(z, z1, z2)
        write_bin(workdir / "jsam_out.bin", np.array(mask, dtype=np.float32))
        return 0

    # ------------------------------------------------------------------
    # nudge_decay: nz=5,ny=2,nx=3, phi=300, ref=250, dt=60, tau=600,
    #              z=[250..2250], z1=0, z2=1e9 -> expected 295 everywhere
    # ------------------------------------------------------------------
    if mode == "nudge_decay":
        nz, ny, nx = 5, 2, 3
        z   = _linear_z(nz, dz=500.0)
        phi = jnp.full((nz, ny, nx), 300.0, dtype=jnp.float32)
        ref = jnp.full((nz,), 250.0, dtype=jnp.float32)
        dt, tau = 60.0, 600.0
        z1, z2 = 0.0, 1.0e9
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("iii", nz, ny, nx))
            f.write(np.array(phi, dtype=np.float32).tobytes(order="C"))
            f.write(np.array(ref, dtype=np.float32).tobytes())
            f.write(np.array(z, dtype=np.float32).tobytes())
            f.write(struct.pack("ffff", dt, z1, z2, tau))
        out = nudge_scalar(phi, ref, z, dt=dt, z1_m=z1, z2_m=z2, tau_s=tau)
        write_bin(workdir / "jsam_out.bin",
                  np.array(out, dtype=np.float32).ravel(order="C"))
        return 0

    # ------------------------------------------------------------------
    # nudge_zero_outside: nz=6,ny=2,nx=2, dz=1000, band [2000,4000]
    #   inside (indices 2,3): 295; outside: 300
    # ------------------------------------------------------------------
    if mode == "nudge_zero_outside":
        nz, ny, nx = 6, 2, 2
        z   = _linear_z(nz, dz=1000.0)       # 500,1500,2500,3500,4500,5500
        phi = jnp.full((nz, ny, nx), 300.0, dtype=jnp.float32)
        ref = jnp.full((nz,), 250.0, dtype=jnp.float32)
        dt, tau = 60.0, 600.0
        z1, z2 = 2000.0, 4000.0
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("iii", nz, ny, nx))
            f.write(np.array(phi, dtype=np.float32).tobytes(order="C"))
            f.write(np.array(ref, dtype=np.float32).tobytes())
            f.write(np.array(z, dtype=np.float32).tobytes())
            f.write(struct.pack("ffff", dt, z1, z2, tau))
        out = nudge_scalar(phi, ref, z, dt=dt, z1_m=z1, z2_m=z2, tau_s=tau)
        write_bin(workdir / "jsam_out.bin",
                  np.array(out, dtype=np.float32).ravel(order="C"))
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
