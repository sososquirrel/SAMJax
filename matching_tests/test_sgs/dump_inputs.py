"""Dump inputs and jsam outputs for test_sgs matching tests.

Called from run.sh:
    python dump_inputs.py shear_prod_zero_velocity
    python dump_inputs.py shear_prod_uniform_u
    python dump_inputs.py smag_zero_def2
    python dump_inputs.py smag_positive_def2
    python dump_inputs.py diffuse_scalar_uniform

inputs.bin layout (all cases):
    int32  nz, ny, nx
    float32 U(nz, ny, nx+1)
    float32 V(nz, ny+1, nx)
    float32 W(nz+1, ny, nx)
    float32 def2(nz, ny, nx)   (zeros for shear_prod cases, actual for smag/diffuse)
    float32 phi(nz, ny, nx)    (zeros unless diffuse case)
    float32 tkh(nz, ny, nx)    (zeros unless diffuse case)
    float32 dx                 scalar
    float32 dy(ny)
    float32 dz(nz)
    float32 Cs, Pr
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
from jsam.core.physics.sgs import shear_prod, smag_viscosity, diffuse_scalar, SGSParams  # noqa: E402


def _make_metric(nz, ny, nx, dx=10_000.0, dy_val=10_000.0, dz_val=1_000.0):
    return {
        "dx_lon":  dx,
        "dy_lat":  jnp.full((ny,), dy_val),
        "dz":      jnp.full((nz,), dz_val),
        "cos_lat": jnp.ones((ny,)),
        "rho":     jnp.ones((nz,)),
        "rhow":    jnp.ones((nz + 1,)),
        "z":       jnp.arange(nz, dtype=float) * dz_val,
    }


def _write_inputs(workdir, U, V, W, def2, phi, tkh, metric, params):
    nz, ny, nx_p1 = np.asarray(U).shape
    nx = nx_p1 - 1
    dx  = float(metric["dx_lon"])
    dy  = np.asarray(metric["dy_lat"], dtype=np.float32)
    dz  = np.asarray(metric["dz"],     dtype=np.float32)
    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(np.asarray(U,    dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(V,    dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(W,    dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(def2, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(phi,  dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(tkh,  dtype=np.float32).tobytes(order="C"))
        f.write(struct.pack("f", dx))
        f.write(dy.tobytes())
        f.write(dz.tobytes())
        f.write(struct.pack("ff", params.Cs, params.Pr))


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode   = sys.argv[1]
    nz, ny, nx = 4, 8, 16
    params = SGSParams()
    metric = _make_metric(nz, ny, nx)

    if mode == "shear_prod_zero_velocity":
        U   = jnp.zeros((nz, ny, nx + 1))
        V   = jnp.zeros((nz, ny + 1, nx))
        W   = jnp.zeros((nz + 1, ny, nx))
        def2 = jnp.zeros((nz, ny, nx))
        phi  = jnp.zeros((nz, ny, nx))
        tkh  = jnp.zeros((nz, ny, nx))
        _write_inputs(workdir, U, V, W, def2, phi, tkh, metric, params)
        out = shear_prod(U, V, W, metric)
        write_bin(workdir / "jsam_out.bin", np.asarray(out, dtype=np.float32).ravel())
        return 0

    if mode == "shear_prod_uniform_u":
        U   = jnp.full((nz, ny, nx + 1), 5.0)
        V   = jnp.zeros((nz, ny + 1, nx))
        W   = jnp.zeros((nz + 1, ny, nx))
        def2 = jnp.zeros((nz, ny, nx))
        phi  = jnp.zeros((nz, ny, nx))
        tkh  = jnp.zeros((nz, ny, nx))
        _write_inputs(workdir, U, V, W, def2, phi, tkh, metric, params)
        out = shear_prod(U, V, W, metric)
        write_bin(workdir / "jsam_out.bin", np.asarray(out, dtype=np.float32).ravel())
        return 0

    if mode == "smag_zero_def2":
        U   = jnp.zeros((nz, ny, nx + 1))
        V   = jnp.zeros((nz, ny + 1, nx))
        W   = jnp.zeros((nz + 1, ny, nx))
        def2 = jnp.zeros((nz, ny, nx))
        phi  = jnp.zeros((nz, ny, nx))
        tkh  = jnp.zeros((nz, ny, nx))
        _write_inputs(workdir, U, V, W, def2, phi, tkh, metric, params)
        tk_out, tkh_out = smag_viscosity(def2, metric, params)
        combined = np.concatenate([
            np.asarray(tk_out,  dtype=np.float32).ravel(),
            np.asarray(tkh_out, dtype=np.float32).ravel(),
        ])
        write_bin(workdir / "jsam_out.bin", combined)
        return 0

    if mode == "smag_positive_def2":
        U   = jnp.zeros((nz, ny, nx + 1))
        V   = jnp.zeros((nz, ny + 1, nx))
        W   = jnp.zeros((nz + 1, ny, nx))
        def2 = jnp.full((nz, ny, nx), 1e-4)
        phi  = jnp.zeros((nz, ny, nx))
        tkh  = jnp.zeros((nz, ny, nx))
        _write_inputs(workdir, U, V, W, def2, phi, tkh, metric, params)
        tk_out, tkh_out = smag_viscosity(def2, metric, params)
        combined = np.concatenate([
            np.asarray(tk_out,  dtype=np.float32).ravel(),
            np.asarray(tkh_out, dtype=np.float32).ravel(),
        ])
        write_bin(workdir / "jsam_out.bin", combined)
        return 0

    if mode == "diffuse_scalar_uniform":
        U   = jnp.zeros((nz, ny, nx + 1))
        V   = jnp.zeros((nz, ny + 1, nx))
        W   = jnp.zeros((nz + 1, ny, nx))
        def2 = jnp.zeros((nz, ny, nx))
        phi  = jnp.ones((nz, ny, nx)) * 300.0
        tkh  = jnp.full((nz, ny, nx), 10.0)
        _write_inputs(workdir, U, V, W, def2, phi, tkh, metric, params)
        out = diffuse_scalar(phi, tkh, metric)
        write_bin(workdir / "jsam_out.bin", np.asarray(out, dtype=np.float32).ravel())
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
