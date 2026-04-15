"""Dump inputs/outputs for AB3 coefficient and step tests.

Cases:
  ab3_coefs  — nstep=0 (Euler), nstep=1 (AB2), nstep=2 (AB3, equal dt)
  ab3_step   — single phi column, apply AB3 step at nstep=2 (equal dt)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, "/glade/u/home/sabramian/jsam")

from common.bin_io import write_bin  # noqa: E402
from jsam.core.dynamics.timestepping import ab_coefs, ab_step  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]

    # ------------------------------------------------------------------
    # ab3_coefs: batch (nstep, dt_curr, dt_prev, dt_pprev) -> (at, bt, ct)
    #
    # Three cases:
    #   nstep=0 -> Euler:  (1, 0, 0)
    #   nstep=1 -> AB2:    alpha=1 -> (1.5, -0.5, 0)
    #   nstep=2 -> AB3:    alpha=beta=1 -> (23/12, -16/12, 5/12)
    # ------------------------------------------------------------------
    if mode == "ab3_coefs":
        dt = 30.0
        # (nstep, dt_curr, dt_prev, dt_pprev)
        cases = [
            (0, dt, dt, dt),     # Euler
            (1, dt, dt, dt),     # AB2
            (2, dt, dt, dt),     # AB3 equal-dt
        ]
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", len(cases)))
            for nstep, dc, dp, dpp in cases:
                f.write(struct.pack("ifff", nstep, float(dc), float(dp), float(dpp)))
        out = []
        for nstep, dc, dp, dpp in cases:
            at, bt, ct = ab_coefs(nstep=nstep, dt_curr=dc, dt_prev=dp, dt_pprev=dpp)
            out.extend([float(at), float(bt), float(ct)])
        write_bin(workdir / "jsam_out.bin", np.array(out, dtype=np.float32))
        return 0

    # ------------------------------------------------------------------
    # ab3_step: phi(nz=8) array, apply AB3 step at nstep=2 (equal dt)
    #
    # phi     = sin(pi * k / nzm)
    # tend_n  = 0.1 * cos(pi * k / nzm)
    # tend_nm1= 0.05 * cos(2*pi * k / nzm)
    # tend_nm2= 0.02 * sin(2*pi * k / nzm)
    # dt = 30.0 s, nstep=2, dt_prev=dt_pprev=dt
    # ------------------------------------------------------------------
    if mode == "ab3_step":
        nzm = 16
        k = np.arange(1, nzm + 1, dtype=np.float32)
        phi      = np.sin(np.pi * k / nzm).astype(np.float32)
        tend_n   = (0.1  * np.cos(np.pi   * k / nzm)).astype(np.float32)
        tend_nm1 = (0.05 * np.cos(2*np.pi * k / nzm)).astype(np.float32)
        tend_nm2 = (0.02 * np.sin(2*np.pi * k / nzm)).astype(np.float32)
        dt = np.float32(30.0)
        nstep = 2

        # Write inputs: nzm, phi, tend_n, tend_nm1, tend_nm2, dt, nstep
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", nzm))
            f.write(phi.tobytes())
            f.write(tend_n.tobytes())
            f.write(tend_nm1.tobytes())
            f.write(tend_nm2.tobytes())
            f.write(struct.pack("fi", float(dt), nstep))

        phi_new = ab_step(
            jnp.array(phi, dtype=jnp.float32),
            jnp.array(tend_n, dtype=jnp.float32),
            jnp.array(tend_nm1, dtype=jnp.float32),
            jnp.array(tend_nm2, dtype=jnp.float32),
            float(dt), nstep,
            dt_prev=float(dt), dt_pprev=float(dt),
        )
        write_bin(workdir / "jsam_out.bin", np.array(phi_new, dtype=np.float32))
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
