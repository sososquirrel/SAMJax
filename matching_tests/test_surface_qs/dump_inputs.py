"""test_surface_qs — saturation specific humidity at the ocean surface.

Both sides compute  qs_sfc = salt_factor * qsatw(SST, p_sfc / 100).
The python side imports `qsatw` directly from
`jsam.core.physics.microphysics` (the same call that
`bulk_surface_fluxes` makes internally — see surface.py:236).
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

from common.bin_io import write_bin                          # noqa: E402
from jsam.core.physics.microphysics import qsatw             # noqa: E402
from jsam.core.physics.surface import BulkParams             # noqa: E402


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    ny, nx = 8, 16
    rng = np.random.default_rng(20260415)
    sst   = (290.0 + 10.0 * rng.random((ny, nx))).astype(np.float32)
    presi = (98000.0 + 4000.0 * rng.random((ny, nx))).astype(np.float32)

    params = BulkParams()
    salt_factor = float(params.salt_factor)

    qs = float(salt_factor) * np.asarray(
        qsatw(jnp.asarray(sst), jnp.asarray(presi) / 100.0),
        dtype=np.float32,
    )

    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("ii", ny, nx))
        f.write(sst.tobytes(order="C"))
        f.write(presi.tobytes(order="C"))
        f.write(struct.pack("f", salt_factor))

    write_bin(workdir / "jsam_out.bin", qs.ravel())
    return 0


if __name__ == "__main__":
    sys.exit(main())
