"""
Run jsam ERA5 initialization and dump the IRMA sub-region tensors.

Writes one binary per field to work/jsam_{field}.bin using the
common bin_io wire format.

This reproduces the same init pipeline as run_irma_debug500.pbs:
    --float32 --no-polar-filter --sponge-tau 0 --slm --rad rrtmg
    --nsteps 500 --nrad 90 --co2 400.0

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

# Paths matching run_irma_debug500.pbs
SLM_DATA_ROOT = "/glade/u/home/sabramian/gSAM1.8.7/GLOBAL_DATA/BIN_D"

FIELDS = ["U", "V", "W", "TABS", "QC", "QV", "QI"]
START_TIME = datetime(2017, 9, 5, 0)


def _build_zi(z, dz):
    zi = np.empty(len(z) + 1)
    zi[0]  = z[0] - 0.5 * dz[0]
    zi[1:] = zi[0] + np.cumsum(dz)
    return zi


def main() -> int:
    import jax
    import jax.numpy as jnp
    # float32 — matches --float32 in run_irma_debug500.pbs
    jax.config.update("jax_enable_x64", False)

    from jsam.utils.IRMALoader import IRMALoader
    from jsam.io.era5 import era5_init
    from jsam.core.state import ModelState

    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    # ── Load the gSAM reference grid (same as run_irma.py) ──
    loader = IRMALoader()
    g = loader.grid
    lat = np.asarray(g["lat"])      # (720,)
    lon = np.asarray(g["lon"])      # (1440,)
    z   = np.asarray(g["z"])        # (74,)
    dz  = np.asarray(g["dz"])
    zi_raw = np.asarray(g["zi"])

    # Reconstruct nz+1 interface heights (same logic as run_irma.py:_build_zi)
    if len(zi_raw) == len(z):
        zi = _build_zi(z, dz)
    else:
        zi = zi_raw
    assert len(zi) == len(z) + 1

    print(f"[dump_jsam_init] Grid: lat={lat.shape} lon={lon.shape} "
          f"z={z.shape} zi={zi.shape}")
    print(f"[dump_jsam_init] Running era5_init for 2017-09-05 00UTC ...")

    # ── Run ERA5 init — matches run_irma_debug500.pbs flags ──
    # --no-polar-filter  → polar_filter=False
    # --float32          → float64 init then downcast (era5_init always float64)
    out = era5_init(
        lat=lat, lon=lon, z=z, zi=zi,
        dt=START_TIME,
        polar_filter=False,
    )
    state = out["state"]
    grid  = out["grid"]
    metric = out["metric"]

    # Downcast to float32, matching run_irma.py --float32 logic
    state = ModelState(
        U    =state.U.astype(jnp.float32),
        V    =state.V.astype(jnp.float32),
        W    =state.W.astype(jnp.float32),
        TABS =state.TABS.astype(jnp.float32),
        QV   =state.QV.astype(jnp.float32),
        QC   =state.QC.astype(jnp.float32),
        QI   =state.QI.astype(jnp.float32),
        QR   =state.QR.astype(jnp.float32),
        QS   =state.QS.astype(jnp.float32),
        QG   =state.QG.astype(jnp.float32),
        TKE  =state.TKE.astype(jnp.float32),
        p_prev =state.p_prev.astype(jnp.float32),
        p_pprev=state.p_pprev.astype(jnp.float32),
        nstep=state.nstep,
        time =state.time,
    )
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
