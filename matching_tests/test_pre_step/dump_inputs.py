"""Produce inputs.bin + jsam_out.bin for the saturation-function test.

This is the small-brick test for the QC/QI init mismatch seen in the
IRMA debug500 run at step 1 stage 0 (pre_step):

  * QC_max  rel diff ~3.8e-1  (gSAM 7.67e-4, jsam 2.79e-3)
  * QC_mean rel diff ~4.5e-1
  * QI_mean rel diff ~3.1e-1

Every init-time QC/QI is ultimately a function of the four primitives
below applied to the ERA5 (T, p, qv) column. If the primitives don't
match bit-for-bit, the downstream init certainly won't.

  esatw(T)        — saturation vapour pressure over liquid  (hPa)
  esati(T)        — saturation vapour pressure over ice     (hPa)
  qsatw(T, p_mb)  — saturation mixing ratio over liquid     (kg/kg)
  qsati(T, p_mb)  — saturation mixing ratio over ice        (kg/kg)

Driver reads inputs.bin (n, T(n), p(n)) and writes fortran_out.bin.
Jsam side calls jsam.core.physics.slm.sat at the identical (T, p)
points.

Called from run.sh:
    python dump_inputs.py <mode>
where <mode> ∈ {all, esatw, esati, qsatw, qsati}.

Binary inputs.bin layout (no record markers, little-endian stream)
------------------------------------------------------------------
    i4 n
    f4 T(n)     # Kelvin
    f4 P(n)     # hPa  — gSAM/jsam sat convention
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
# Test points — cover the full (T, p) envelope that shows up in ERA5
# IRMA init at step 1 stage 0 (pre_step):
#
#   TABS_min ≈ 183 K   (tropopause / polar vortex)
#   TABS_max ≈ 318 K   (tropical SST skin)
#   pressure ≈  1 hPa @ model top, 1013 hPa @ surface
#
# We include:
#   - hot/wet tropics   (qsatw hits >30 g/kg, QC starts here)
#   - freezing level    (305→230 K, both qsatw and qsati matter)
#   - stratosphere      (very cold + very low p, qsati dominates)
#   - polar surface     (240 K, 1010 hPa)
#   - warm dry plateau  (300 K, 800 hPa)
#   - saturation edge   (T close to where e_sat → p, tests the max() guard)
# ---------------------------------------------------------------------------

def _test_points() -> tuple[np.ndarray, np.ndarray]:
    T = np.array([
        # tropical troposphere sweep
        300.0, 295.0, 288.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0,
        # stratosphere / tropopause
        220.0, 210.0, 200.0, 195.0, 190.0, 185.0, 183.0,
        # warm extreme (QC init hot zone)
        305.0, 310.0, 315.0, 318.0,
        # polar surface
        240.0, 245.0, 250.0,
        # saturation-near-p edge (warm, low pressure)
        305.0, 310.0,
    ], dtype=np.float32)

    P = np.array([
        # tropical troposphere sweep (descending pressure)
        1013.0, 900.0, 800.0, 700.0, 500.0, 400.0, 300.0, 200.0, 150.0,
        # stratosphere / tropopause
        100.0, 70.0, 40.0, 20.0, 10.0, 5.0, 1.5,
        # warm extreme
        1013.0, 1013.0, 1013.0, 1013.0,
        # polar surface
        1005.0, 1010.0, 1013.0,
        # saturation-near-p edge — deliberately low p
        100.0, 50.0,
    ], dtype=np.float32)

    assert T.shape == P.shape
    return T, P


# ---------------------------------------------------------------------------
# jsam side — same four primitives
# ---------------------------------------------------------------------------

def _jsam_compute(mode: str, T: np.ndarray, P: np.ndarray) -> np.ndarray:
    import jax
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp
    from jsam.core.physics.slm.sat import esatw, esati, qsatw, qsati

    T_j = jnp.asarray(T)
    P_j = jnp.asarray(P)

    def _as_f32(x):
        return np.asarray(x, dtype=np.float32).ravel(order="C")

    if mode == "esatw":
        return _as_f32(esatw(T_j))
    if mode == "esati":
        return _as_f32(esati(T_j))
    if mode == "qsatw":
        return _as_f32(qsatw(T_j, P_j))
    if mode == "qsati":
        return _as_f32(qsati(T_j, P_j))
    if mode == "all":
        return np.concatenate([
            _as_f32(esatw(T_j)),
            _as_f32(esati(T_j)),
            _as_f32(qsatw(T_j, P_j)),
            _as_f32(qsati(T_j, P_j)),
        ])
    raise SystemExit(f"unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in ("all", "esatw", "esati", "qsatw", "qsati"):
        raise SystemExit(f"unknown mode: {mode}")

    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    T, P = _test_points()
    n = T.size

    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("i", n))
        f.write(T.tobytes(order="C"))
        f.write(P.tobytes(order="C"))

    jsam_vals = _jsam_compute(mode, T, P)
    write_bin(workdir / "jsam_out.bin", jsam_vals)

    print(f"[dump_inputs] mode={mode}  n_points={n}  n_values={jsam_vals.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
