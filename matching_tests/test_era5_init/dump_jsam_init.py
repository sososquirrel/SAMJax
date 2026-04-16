"""
Run jsam ERA5 initialization and dump the IRMA sub-region tensors.

Writes one binary per field to work/jsam_{field}.bin using the
common bin_io wire format.

This reproduces the same init pipeline as run_irma.py but only
runs the initialization (no time stepping).

Usage
-----
    python dump_jsam_init.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
SAMJAX_ROOT = MT_ROOT.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, str(SAMJAX_ROOT))

from common.bin_io import write_bin  # noqa: E402

# IRMA sub-region box (matches gSAM debug_dump.f90)
IRMA_LAT_MIN = 0.0
IRMA_LAT_MAX = 35.0
IRMA_LON_MIN = 260.0
IRMA_LON_MAX = 340.0

FIELDS = ["U", "V", "W", "TABS", "QC", "QV", "QI"]


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", False)

    from jsam.utils.IRMALoader import IRMALoader
    from jsam.io.era5 import era5_init

    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    # ── Load the gSAM reference grid (same as run_irma.py) ──
    loader = IRMALoader()
    g = loader.grid
    lat = np.asarray(g["lat"])      # (720,)
    lon = np.asarray(g["lon"])      # (1440,)
    z   = np.asarray(g["z"])        # (74,)
    zi  = np.asarray(g["zi"])       # (74,) — gSAM convention (bottom interface)

    # Build zi with nz+1 entries if needed (jsam expects nz+1 interfaces)
    if zi.shape[0] == z.shape[0]:
        # zi from gSAM netCDF has nz entries (bottom interfaces); add top
        dz = np.asarray(g["dz"])
        zi_full = np.zeros(len(z) + 1)
        zi_full[0] = 0.0
        for k in range(len(z)):
            zi_full[k + 1] = zi_full[k] + dz[k]
        zi = zi_full

    print(f"[dump_jsam_init] Grid: lat={lat.shape} lon={lon.shape} "
          f"z={z.shape} zi={zi.shape}")
    print(f"[dump_jsam_init] Running era5_init for 2017-09-05 00UTC ...")

    # ── Run ERA5 init (same settings as run_irma_debug500.pbs) ──
    out = era5_init(
        lat=lat, lon=lon, z=z, zi=zi,
        dt=datetime(2017, 9, 5, 0),
        polar_filter=False,  # --no-polar-filter in PBS script
    )
    state = out["state"]
    grid  = out["grid"]

    print(f"[dump_jsam_init] Init done. Extracting IRMA sub-region ...")

    # ── IRMA box indices (0-based) ──
    lon_arr = np.asarray(grid.lon)
    lat_arr = np.asarray(grid.lat)
    i_mask = (lon_arr >= IRMA_LON_MIN) & (lon_arr <= IRMA_LON_MAX)
    j_mask = (lat_arr >= IRMA_LAT_MIN) & (lat_arr <= IRMA_LAT_MAX)
    ii = np.where(i_mask)[0]
    jj = np.where(j_mask)[0]
    i_lo, i_hi = int(ii[0]), int(ii[-1]) + 1
    j_lo, j_hi = int(jj[0]), int(jj[-1]) + 1
    ni = i_hi - i_lo
    nj = j_hi - j_lo
    nz = len(grid.z)

    print(f"[dump_jsam_init] IRMA box: i=[{i_lo},{i_hi}) j=[{j_lo},{j_hi}) "
          f"=> {ni}x{nj}x{nz}")

    # ── Extract fields matching debug_dump convention ──
    # gSAM debug_dump writes: U[:nzm, :, :nx], V[:nzm, :ny, :],
    #   W[:nzm, :, :], TABS, QC, QV, QI (all on mass-grid slice)
    U_full    = np.asarray(state.U[:nz, :, :-1])     # drop stagger dim
    V_full    = np.asarray(state.V[:nz, :-1, :])
    W_full    = np.asarray(state.W[:nz, :, :])       # drop top half-level
    TABS_full = np.asarray(state.TABS)
    QC_full   = np.asarray(state.QC)
    QV_full   = np.asarray(state.QV)
    QI_full   = np.asarray(state.QI)

    field_arrays = {
        "U": U_full, "V": V_full, "W": W_full,
        "TABS": TABS_full, "QC": QC_full, "QV": QV_full, "QI": QI_full,
    }

    for name in FIELDS:
        arr = field_arrays[name]
        sub = arr[:, j_lo:j_hi, i_lo:i_hi].astype(np.float32)
        out_path = workdir / f"jsam_{name}.bin"
        write_bin(out_path, sub.ravel(order="C"))
        print(f"  {name}: shape={sub.shape}  min={sub.min():.6e}  max={sub.max():.6e}  "
              f"mean={sub.mean():.6e}  -> {out_path.name}")

    # ── Also dump a summary CSV for quick inspection ──
    csv_path = workdir / "init_comparison_stats.csv"
    with open(csv_path, "w") as f:
        f.write("field,jsam_min,jsam_max,jsam_mean\n")
        for name in FIELDS:
            arr = field_arrays[name]
            sub = arr[:, j_lo:j_hi, i_lo:i_hi]
            f.write(f"{name},{sub.min():.9e},{sub.max():.9e},{sub.mean():.9e}\n")
    print(f"\n  Stats written to {csv_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
