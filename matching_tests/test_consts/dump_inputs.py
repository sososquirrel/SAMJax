"""Dump jsam-side constant values in the same order as test_consts/driver.f90.

After microphysics.py was fixed (LS=2.834e6 literal; CPV/RAD_EARTH/
SIGMA_SB/EMIS_WATER/CPW now exported), this test should PASS at 4
decimals against the gSAM consts.f90 values.

If a constant is missing from the jsam module the script raises a clear
error noting the gap — it does NOT fall back to a hardcoded value
(the parallel microphysics fix is responsible for adding the missing
exports).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))

from common.bin_io import write_bin  # noqa: E402

from jsam.core.physics import microphysics as M  # noqa: E402

# Names that jsam should export, in the order driver.f90 writes them.
NAMES = [
    "CP", "CPV", "G_GRAV", "LV", "LF", "LS", "RV", "RGAS",
    "DIFFELQ", "THERCO", "MUELQ", "FAC_COND", "FAC_FUS", "FAC_SUB",
    "RAD_EARTH", "SIGMA_SB", "EMIS_WATER", "CPW",
]


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    missing = [n for n in NAMES if not hasattr(M, n)]
    if missing:
        sys.stderr.write(
            "test_consts: jsam.core.physics.microphysics is missing the "
            f"following exported constants: {missing}\n"
        )
        return 1

    out = np.array([float(getattr(M, n)) for n in NAMES], dtype=np.float32)
    write_bin(workdir / "jsam_out.bin", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
