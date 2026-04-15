"""
Restart I/O for jsam.

Writes a NetCDF checkpoint containing every prognostic field of ModelState
on its native staggered grid, plus the scalar step counter and simulation
time.  Round-tripping ``save_restart`` → ``load_restart`` yields a byte-
identical ModelState so a run can resume from exactly where it stopped.

This is intentionally distinct from ``jsam.io.writer.write_3d_atm`` —
``writer`` de-staggers velocities and converts units for gSAM-compatible
analysis output; ``restart`` preserves the native C-grid shapes.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import xarray as xr

from jsam.core.state import ModelState


_FIELDS_3D = ("U", "V", "W", "TABS", "QV", "QC", "QI", "QR", "QS", "QG", "TKE")


def save_restart(state: ModelState, path: str | Path) -> Path:
    """
    Write ``state`` to a NetCDF restart file at ``path``.

    Each prognostic field is stored on its native staggered shape with its
    own dimension names so xarray can round-trip the differing shapes
    (e.g. U has nx+1 while QV has nx).  ``nstep`` and ``time`` are stored
    as scalar variables.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data_vars = {}
    for name in _FIELDS_3D:
        arr = np.asarray(getattr(state, name))
        dims = (f"{name}_k", f"{name}_j", f"{name}_i")
        data_vars[name] = (dims, arr)

    if state.p_prev is not None:
        data_vars["p_prev"] = (("p_k", "p_j", "p_i"), np.asarray(state.p_prev))

    ds = xr.Dataset(
        data_vars=data_vars,
        attrs={
            "jsam_restart_version": "1",
            "nstep": int(state.nstep),
            "time":  float(state.time),
            "has_p_prev": int(state.p_prev is not None),
        },
    )
    encoding = {v: {"zlib": True, "complevel": 1} for v in data_vars}
    ds.to_netcdf(path, encoding=encoding)
    return path


def load_restart(path: str | Path) -> ModelState:
    """
    Load a restart file written by ``save_restart`` into a fresh ModelState.
    """
    path = Path(path)
    with xr.open_dataset(path) as ds:
        def _get(name):
            return jnp.asarray(np.asarray(ds[name].values))

        kwargs = {name: _get(name) for name in _FIELDS_3D}

        if int(ds.attrs.get("has_p_prev", 0)):
            kwargs["p_prev"] = _get("p_prev")
        else:
            kwargs["p_prev"] = None

        kwargs["nstep"] = jnp.int32(int(ds.attrs["nstep"]))
        kwargs["time"]  = jnp.float64(float(ds.attrs["time"]))

    return ModelState(**kwargs)
