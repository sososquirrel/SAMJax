"""Dump inputs and jsam outputs for test_microphysics matching tests.

Called from run.sh:
    python dump_inputs.py qsatw_at_20C_1000mb
    python dump_inputs.py qsati_below_freezing
    python dump_inputs.py qsatw_monotone

inputs.bin layout (all cases):
    int32  nT         (number of T values)
    float32 T(nT)     (K)
    float32 p         (mb, scalar)

jsam uses Buck (1981) formulas from microphysics.py:
    esatw(T) = 6.1121 * exp(17.502*(T-273.16)/(T-32.18))   mb
    esati(T) = 6.1121 * exp(22.587*(T-273.16)/(T+0.7))     mb
    EPS = RGAS/RV = 287.04/461.5 = 0.6220
    qsatw = EPS * esatw / max(p-esatw, 1e-3)

Output for qsatw cases:   qsatw(nT)
Output for qsati case:    [qsatw(nT), qsati(nT)] concatenated
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, "/glade/u/home/sabramian/jsam")

from common.bin_io import write_bin  # noqa: E402
from jsam.core.physics.microphysics import qsatw, qsati  # noqa: E402


def _write_inputs(workdir, T_arr, p_mb):
    T = np.asarray(T_arr, dtype=np.float32)
    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("i", len(T)))
        f.write(T.tobytes())
        f.write(struct.pack("f", float(p_mb)))


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]

    if mode == "qsatw_at_20C_1000mb":
        T_arr = jnp.array([293.16])
        p_mb  = 1000.0
        _write_inputs(workdir, T_arr, p_mb)
        qs = qsatw(T_arr, jnp.array(p_mb))
        write_bin(workdir / "jsam_out.bin", np.asarray(qs, dtype=np.float32))
        return 0

    if mode == "qsatw_monotone":
        T_arr = jnp.linspace(270.0, 300.0, 13)
        p_mb  = 1000.0
        _write_inputs(workdir, T_arr, p_mb)
        qs = qsatw(T_arr, jnp.full_like(T_arr, p_mb))
        write_bin(workdir / "jsam_out.bin", np.asarray(qs, dtype=np.float32))
        return 0

    if mode == "qsati_below_freezing":
        T_arr = jnp.array([263.16])
        p_mb  = 500.0
        _write_inputs(workdir, T_arr, p_mb)
        qw = qsatw(T_arr, jnp.array(p_mb))
        qi = qsati(T_arr, jnp.array(p_mb))
        combined = np.concatenate([
            np.asarray(qw, dtype=np.float32).ravel(),
            np.asarray(qi, dtype=np.float32).ravel(),
        ])
        write_bin(workdir / "jsam_out.bin", combined)
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
