"""Re-uses the fixtures from jsam/tests/unit/test_timestepping.py and dumps:
   - inputs.bin    (in the format the Fortran driver expects)
   - jsam_out.bin  (jsam-side answer for the same case)

Called once per case from run.sh:

   python dump_inputs.py ab2_coefs
   python dump_inputs.py ab2_step
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

from jsam.core.dynamics.timestepping import ab2_coefs, ab2_step  # noqa: E402


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]

    # ------------------------------------------------------------------
    # ab2_coefs: batch of (nstep, dt_curr, dt_prev) -> (at, bt)
    #
    # Cases:
    #   (0, 10.0, 10.0) -> Euler: (1.0, 0.0)
    #   (1, 10.0, 10.0) -> AB2 equal dt: (1.5, -0.5)
    # ------------------------------------------------------------------
    if mode == "ab2_coefs":
        # Cover the two key assertions from the python test:
        #   ab2_coefs_euler:    nstep=0 -> (1.0, 0.0)
        #   ab2_coefs_ab2:      nstep=1, equal dt -> (1.5, -0.5)
        cases = [
            (0, 10.0, 10.0),   # Euler (nstep=0)
            (1, 10.0, 10.0),   # AB2 equal-dt
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

    # ------------------------------------------------------------------
    # ab2_step: batch of (phi, tend_n, tend_nm1, dt, nstep, dt_prev) -> phi_new
    #
    # Cases:
    #   euler:  phi=5.0, tend_n=2.0, tend_nm1=999.0, dt=0.1, nstep=0, dt_prev=0.1
    #           -> 5.0 + 0.1*(1.0*2.0) = 5.2
    #   ab2:    phi=5.0, tend_n=2.0, tend_nm1=1.0, dt=0.1, nstep=1, dt_prev=0.1
    #           -> 5.0 + 0.1*(1.5*2.0 - 0.5*1.0) = 5.0 + 0.25 = 5.25
    # ------------------------------------------------------------------
    if mode == "ab2_step":
        # Each record: (phi, tend_n, tend_nm1, dt, nstep, dt_prev)
        cases = [
            (5.0, 2.0, 999.0, 0.1, 0, 0.1),   # Euler
            (5.0, 2.0, 1.0,   0.1, 1, 0.1),   # AB2
        ]
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", len(cases)))
            for phi, fn, fnm1, dt, nstep, dp in cases:
                f.write(struct.pack("fffff", phi, fn, fnm1, dt, float(dp)))
                f.write(struct.pack("i", nstep))
        out = []
        for phi, fn, fnm1, dt, nstep, dp in cases:
            result = float(ab2_step(
                jnp.array(phi), jnp.array(fn), jnp.array(fnm1),
                dt, nstep, dt_prev=dp,
            ))
            out.append(result)
        write_bin(workdir / "jsam_out.bin", np.array(out, dtype=np.float32))
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
