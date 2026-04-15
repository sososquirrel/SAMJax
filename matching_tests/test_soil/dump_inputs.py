"""Dump inputs and jsam outputs for test_soil matching tests.

Called from run.sh:
    python dump_inputs.py sat_qsatw
    python dump_inputs.py sat_qsati
    python dump_inputs.py cosby_1984
    python dump_inputs.py fh_calc

Each case writes:
    work/inputs.bin   — case-specific raw float32 stream (consumed by driver.f90)
    work/jsam_out.bin — jsam output via common.bin_io.write_bin

The common comparator (common.compare) then diffs fortran_out.bin
against jsam_out.bin.

inputs.bin layout per case: see driver.f90 header.
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
sys.path.insert(0, "/glade/u/home/sabramian/SAMJax")

from common.bin_io import write_bin  # noqa: E402

from jsam.core.physics.slm.sat import qsatw, qsati  # noqa: E402
from jsam.core.physics.slm.vapor_fluxes import fh_calc  # noqa: E402
from jsam.io.slm_init import _cosby_1984  # noqa: E402


# ---------------------------------------------------------------------------
# Sample grids
# ---------------------------------------------------------------------------
def _sat_samples():
    """Representative (T, P) grid for qsatw/qsati: spans the full physical
    range the SLM will ever see, including cold stratosphere, warm tropics
    and mid-troposphere."""
    T = np.array([
        220.0, 240.0, 250.0, 260.0, 270.0,
        273.15, 275.0, 280.0, 290.0, 295.0,
        300.0, 305.0, 310.0, 315.0, 320.0,
    ], dtype=np.float32)
    P = np.array([
        200.0, 300.0, 400.0, 500.0, 600.0,
        700.0, 800.0, 850.0, 900.0, 950.0,
        1000.0, 1010.0, 1013.25, 1020.0, 1025.0,
    ], dtype=np.float32)
    return T, P


def _cosby_samples():
    """Range of SAND/CLAY percentages covering every texture class in
    the USDA soil triangle (sandy through silty through clayey soils)."""
    SAND = np.array([
        90.0, 80.0, 65.0, 55.0, 50.0,
        40.0, 33.0, 25.0, 20.0, 17.0,
        10.0,  5.0,  1.0, 45.0, 60.0,
    ], dtype=np.float32)
    CLAY = np.array([
         5.0, 10.0, 15.0, 20.0, 25.0,
        30.0, 33.0, 35.0, 40.0, 45.0,
        50.0, 55.0, 60.0, 25.0, 10.0,
    ], dtype=np.float32)
    return SAND, CLAY


def _fh_samples():
    """Representative (T, mps, sw, B) tuples for fh_calc spanning dry,
    near-WP and near-FC conditions at typical soil temperatures."""
    # 15 samples; each row hits a different combination
    T   = np.array([280., 285., 290., 295., 300., 280., 285., 290.,
                    295., 300., 275., 278., 302., 305., 310.],
                   dtype=np.float32)
    mps = np.array([-150., -200., -300., -500., -800., -1000.,
                    -1500., -180., -250., -400., -600., -900.,
                    -1200., -1800., -160.], dtype=np.float32)
    sw  = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                    0.60, 0.70, 0.12, 0.22, 0.33, 0.44, 0.08],
                   dtype=np.float32)
    B   = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                    8.5, 4.2, 5.2, 6.2, 7.2, 3.8], dtype=np.float32)
    return T, mps, sw, B


# ---------------------------------------------------------------------------
# inputs.bin writers
# ---------------------------------------------------------------------------
def _write_two(workdir, a, b):
    n = len(a)
    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("i", n))
        f.write(np.asarray(a, dtype=np.float32).tobytes())
        f.write(np.asarray(b, dtype=np.float32).tobytes())


def _write_four(workdir, a, b, c, d):
    n = len(a)
    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("i", n))
        f.write(np.asarray(a, dtype=np.float32).tobytes())
        f.write(np.asarray(b, dtype=np.float32).tobytes())
        f.write(np.asarray(c, dtype=np.float32).tobytes())
        f.write(np.asarray(d, dtype=np.float32).tobytes())


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]

    if mode == "sat_qsatw":
        T, P = _sat_samples()
        _write_two(workdir, T, P)
        q = np.asarray(qsatw(jnp.asarray(T), jnp.asarray(P)), dtype=np.float32)
        write_bin(workdir / "jsam_out.bin", q)
        return 0

    if mode == "sat_qsati":
        T, P = _sat_samples()
        _write_two(workdir, T, P)
        q = np.asarray(qsati(jnp.asarray(T), jnp.asarray(P)), dtype=np.float32)
        write_bin(workdir / "jsam_out.bin", q)
        return 0

    if mode == "cosby_1984":
        SAND, CLAY = _cosby_samples()
        _write_two(workdir, SAND, CLAY)
        cosby = _cosby_1984(SAND.astype(np.float32), CLAY.astype(np.float32))
        # Order MUST match driver.f90 run_cosby output layout:
        order = ["ks", "Bconst", "poro_soil", "m_pot_sat",
                 "sst_capa", "sst_cond",
                 "theta_FC", "theta_WP", "w_s_FC", "w_s_WP"]
        out = np.concatenate([
            np.asarray(cosby[k], dtype=np.float32).ravel() for k in order
        ])
        write_bin(workdir / "jsam_out.bin", out)
        return 0

    if mode == "fh_calc":
        T, mps, sw, B = _fh_samples()
        _write_four(workdir, T, mps, sw, B)
        fh = np.asarray(
            fh_calc(jnp.asarray(T), jnp.asarray(mps), jnp.asarray(sw),
                    jnp.asarray(B)),
            dtype=np.float32,
        )
        write_bin(workdir / "jsam_out.bin", fh)
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
