"""Dump inputs and jsam outputs for test_surface matching tests.

Called from run.sh:
    python dump_inputs.py qsat_w_at_300K
    python dump_inputs.py qsat_monotone_in_T
    python dump_inputs.py bulk_fluxes_warm_sst_shf
    python dump_inputs.py bulk_fluxes_tau_opposes
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
from jsam.core.physics.surface import _qsat_w, bulk_surface_fluxes, BulkParams  # noqa: E402
from jsam.core.state import ModelState  # noqa: E402


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]

    if mode == "qsat_w_at_300K":
        T = jnp.array(300.0)
        p = jnp.array(1.0e5)
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", 1))          # dummy mode_id
            f.write(struct.pack("ff", float(T), float(p)))
        qs = _qsat_w(T, p)
        write_bin(workdir / "jsam_out.bin", np.array([float(qs)], dtype=np.float32))
        return 0

    if mode == "qsat_monotone_in_T":
        T_arr = jnp.linspace(250.0, 310.0, 13)   # 13 values
        p     = jnp.array(1.0e5)
        nT    = len(T_arr)
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", nT))
            f.write(np.asarray(T_arr, dtype=np.float32).tobytes())
            f.write(struct.pack("f", float(p)))
        qs = _qsat_w(T_arr, p)
        write_bin(workdir / "jsam_out.bin", np.asarray(qs, dtype=np.float32))
        return 0

    if mode in ("bulk_fluxes_warm_sst_shf", "bulk_fluxes_tau_opposes"):
        # Build a minimal 1-column (nz=1, ny=1, nx=1) ModelState
        # to call bulk_surface_fluxes.
        if mode == "bulk_fluxes_warm_sst_shf":
            T_atm   = 295.0
            QV_atm  = 0.015
            u_atm   = 2.0
            v_atm   = 0.0
            SST_val = 302.0
        else:  # tau_opposes
            T_atm   = 295.0
            QV_atm  = 0.015
            u_atm   = 8.0
            v_atm   = 0.0
            SST_val = 295.0

        # Grid constants matching what bulk_surface_fluxes expects
        rho0  = 1.2
        pres0 = 1.0e5
        dz0   = 200.0
        z0    = dz0 / 2.0   # 100 m

        # Write inputs.bin
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", 3))
            f.write(struct.pack("ffff", T_atm, QV_atm, u_atm, v_atm))
            f.write(struct.pack("f", SST_val))
            f.write(struct.pack("ffff", rho0, pres0, dz0, z0))

        # Build a ModelState with shape (nz=1, ny=1, nx=1)
        # U: (1, 1, 2), V: (1, 2, 1), W: (2, 1, 1)
        U = jnp.full((1, 1, 2), u_atm)
        V = jnp.full((1, 2, 1), v_atm)
        W = jnp.zeros((2, 1, 1))
        TABS = jnp.full((1, 1, 1), T_atm)
        QV   = jnp.full((1, 1, 1), QV_atm)
        zero = jnp.zeros((1, 1, 1))

        state = ModelState(
            U=U, V=V, W=W,
            TABS=TABS, QV=QV,
            QC=zero, QI=zero, QR=zero, QS=zero, QG=zero,
            TKE=zero,
        )

        metric = {
            "rho":  jnp.array([rho0]),
            "pres": jnp.array([pres0]),
            "dz":   jnp.array([dz0]),
            "z":    jnp.array([z0]),
        }

        sst = jnp.full((1, 1), SST_val)
        params = BulkParams()
        fluxes = bulk_surface_fluxes(state, metric, sst, params)

        out = np.array([
            float(fluxes.shf.ravel()[0]),
            float(fluxes.lhf.ravel()[0]),
            float(fluxes.tau_x.ravel()[0]),
            float(fluxes.tau_y.ravel()[0]),
        ], dtype=np.float32)
        write_bin(workdir / "jsam_out.bin", out)
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
