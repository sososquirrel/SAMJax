"""Re-uses the fixtures from jsam/tests/unit/test_timestepping.py and dumps:
   - inputs.bin    (in the format the Fortran driver expects)
   - jsam_out.bin  (jsam-side answer for the same case)

Called once per case from run.sh:

   python dump_inputs.py ab2_coefs
   python dump_inputs.py ab2_step

Both modes exercise the full AB3 state machine
(jsam.core.dynamics.timestepping.ab_coefs / ab_step) — see gSAM
SRC/abcoefs.f90 for the original.
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

from common.bin_io import write_bin  # noqa: E402

from jsam.core.dynamics.timestepping import ab_coefs, ab_step  # noqa: E402


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]

    # ------------------------------------------------------------------
    # ab2_coefs: batch of (nstep, dt_curr, dt_prev, dt_pprev) -> (at, bt, ct)
    # ------------------------------------------------------------------
    if mode == "ab2_coefs":
        cases = [
            # nstep=0 → Euler
            (0, 10.0, 10.0, 10.0),
            (0,  3.0, 10.0,  7.0),
            # nstep=1 → AB2 bootstrap (constant + variable)
            (1, 10.0, 10.0, 10.0),
            (1,  5.0, 10.0, 10.0),
            (1, 20.0, 10.0, 10.0),
            # nstep>=2 → AB3 — constant dt should give (23/12, -16/12, 5/12)
            (2, 10.0, 10.0, 10.0),
            (3, 10.0, 10.0, 10.0),
            (5, 10.0, 10.0, 10.0),
            # variable-dt AB3 (3+ cases exercising the alpha/beta branch)
            (4, 10.0,  8.0,  6.0),
            (6,  4.0,  5.0,  6.0),
            (7,  9.0,  3.0,  2.0),
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
    # ab2_step: batch of (phi, tend_n, tend_nm1, tend_nm2, dt_curr,
    #                     dt_prev, dt_pprev, nstep) -> phi_new
    # ------------------------------------------------------------------
    if mode == "ab2_step":
        cases = [
            # Euler: phi=5, tend_n=2, others ignored
            (5.0, 2.0, 999.0, 999.0, 0.1, 0.1, 0.1, 0),
            # AB2 (constant dt): 5 + 0.1*(1.5*2 - 0.5*1) = 5.25
            (5.0, 2.0, 1.0,   999.0, 0.1, 0.1, 0.1, 1),
            # AB3 (constant dt): 5 + 0.1*((23/12)*2 + (-16/12)*1 + (5/12)*3)
            (5.0, 2.0, 1.0,   3.0,   0.1, 0.1, 0.1, 2),
            (5.0, 2.0, 1.0,   3.0,   0.1, 0.1, 0.1, 3),
            # AB3 variable-dt
            (1.0, 0.5, 0.25,  0.125, 0.2, 0.18, 0.16, 4),
        ]
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", len(cases)))
            for phi, fn, fnm1, fnm2, dt, dp, dpp, nstep in cases:
                f.write(struct.pack(
                    "fffffff",
                    float(phi), float(fn), float(fnm1), float(fnm2),
                    float(dt), float(dp), float(dpp),
                ))
                f.write(struct.pack("i", nstep))
        out = []
        for phi, fn, fnm1, fnm2, dt, dp, dpp, nstep in cases:
            result = float(ab_step(
                jnp.array(phi), jnp.array(fn),
                jnp.array(fnm1), jnp.array(fnm2),
                dt, nstep, dt_prev=dp, dt_pprev=dpp,
            ))
            out.append(result)
        write_bin(workdir / "jsam_out.bin", np.array(out, dtype=np.float32))
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
