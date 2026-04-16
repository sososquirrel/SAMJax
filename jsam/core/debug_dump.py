"""
jsam stage-dump — bit-compare hook for matching gSAM's debug_dump.f90 oracle.

Enables dumping the prognostic state (U, V, W, TABS, QC, QV, QI) at each of
the 19 gSAM stages of one step(), plus per-stage global min/max/mean stats.
Output matches the on-disk format documented in
``.claude/Oracle_Tensor_Structure.md`` so an offline script can diff jsam vs
the gSAM oracle tensors record-by-record.

Usage
-----

    from jsam.core import debug_dump as _dd

    _dd.DUMPER = _dd.DebugDumper(
        debug_dir="/path/to/jsam_debug",
        lat=lat_deg, lon=lon_deg, z_len=nz,
    )
    # run the time loop (step() will pick up DUMPER automatically)
    _dd.DUMPER.finalize()

Set ``DUMPER = None`` (default) to disable — step() will be a zero-cost no-op.

Single-rank only: jsam runs one process, so one ``rank_000000.bin`` is
written with the full IRMA sub-box (no inter-rank stitching needed).

Notes on stage IDs
------------------

jsam bundles several gSAM sub-operations into fused ops (e.g. buoyancy,
coriolis, sgs_mom and advect_mom all land in a single ``advance_momentum``
call).  The dumper uses gSAM's 19 stage IDs, but some of them will record
*identical* state because jsam never touches U/V/W/TABS between them.
That is expected and matches gSAM oracle behaviour: gSAM stages 3-9 also
only accumulate tendencies without changing state.

Stage labels and IDs (verbatim from gSAM main.f90):

    0  pre_step         7  coriolis         14  advect_scalars
    1  forcing          8  sgs_proc         15  sgs_scalars
    2  nudging          9  sgs_mom          16  upperbound
    3  buoyancy        10  adamsA           17  micro
    4  radiation       11  damping          18  diagnose
    5  surface         12  adamsB
    6  advect_mom      13  pressure
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Module-level singleton — set from the driver (run_irma*.py) to enable dumps.
DUMPER: "DebugDumper | None" = None

# Same constants as gSAM debug_dump.f90
IRMA_LAT_MIN = 0.0
IRMA_LAT_MAX = 35.0
IRMA_LON_MIN = 260.0
IRMA_LON_MAX = 340.0

NFIELDS = 7
FIELD_NAMES = ("U       ", "V       ", "W       ",
               "TABS    ", "QC      ", "QV      ", "QI      ")

STAGE_NAMES = {
     0: "pre_step",
     1: "forcing",
     2: "nudging",
     3: "buoyancy",
     4: "radiation",
     5: "surface",
     6: "advect_mom",
     7: "coriolis",
     8: "sgs_proc",
     9: "sgs_mom",
    10: "adamsA",
    11: "damping",
    12: "adamsB",
    13: "pressure",
    14: "advect_scalars",
    15: "sgs_scalars",
    16: "upperbound",
    17: "micro",
    18: "diagnose",
}


@dataclass
class DebugDumper:
    """
    Bit-compatible stage dumper matching gSAM's debug_dump.f90 on-disk format.

    Parameters
    ----------
    debug_dir : output directory (created if missing)
    lat       : (ny,) latitudes in degrees (matches gSAM lat_gl)
    lon       : (nx,) longitudes in degrees, 0..360 convention (matches lon_gl)
    z_len     : number of vertical levels (nzm)
    """

    debug_dir: str
    lat: np.ndarray
    lon: np.ndarray
    z_len: int

    def __post_init__(self):
        Path(self.debug_dir).mkdir(parents=True, exist_ok=True)
        self.lat = np.asarray(self.lat)
        self.lon = np.asarray(self.lon)
        self.nx_gl = int(self.lon.shape[0])
        self.ny_gl = int(self.lat.shape[0])
        self.nzm   = int(self.z_len)

        # --- IRMA box → 1-based inclusive global indices (gSAM convention) ---
        lon_in  = (self.lon >= IRMA_LON_MIN) & (self.lon <= IRMA_LON_MAX)
        lat_in  = (self.lat >= IRMA_LAT_MIN) & (self.lat <= IRMA_LAT_MAX)
        if not lon_in.any() or not lat_in.any():
            raise ValueError(
                f"DebugDumper: IRMA box does not intersect the grid — "
                f"lat range [{self.lat.min():.2f},{self.lat.max():.2f}], "
                f"lon range [{self.lon.min():.2f},{self.lon.max():.2f}]"
            )
        ii = np.where(lon_in)[0]
        jj = np.where(lat_in)[0]
        self.i_min_gl = int(ii[0])  + 1   # Fortran 1-based
        self.i_max_gl = int(ii[-1]) + 1
        self.j_min_gl = int(jj[0])  + 1
        self.j_max_gl = int(jj[-1]) + 1
        self._i_lo = int(ii[0])           # python 0-based slice
        self._i_hi = int(ii[-1]) + 1
        self._j_lo = int(jj[0])
        self._j_hi = int(jj[-1]) + 1
        self.ni = self._i_hi - self._i_lo
        self.nj = self._j_hi - self._j_lo

        # Single-rank layout: one "rank" covers the full domain.
        self.it_off = 0
        self.jt_off = 0
        self.i1_loc = self.i_min_gl
        self.i2_loc = self.i_max_gl
        self.j1_loc = self.j_min_gl
        self.j2_loc = self.j_max_gl

        self._rank_path = os.path.join(self.debug_dir, "rank_000000.bin")
        self._csv_path  = os.path.join(self.debug_dir, "globals.csv")

        self._fbin = open(self._rank_path, "wb")
        self._write_header()

        self._fcsv = open(self._csv_path, "w")
        self._fcsv.write(
            "nstep,stage_id,stage_name,dtn,"
            "U_min,U_max,U_mean,V_min,V_max,V_mean,W_min,W_max,W_mean,"
            "TABS_min,TABS_max,TABS_mean,QC_min,QC_max,QC_mean,"
            "QV_min,QV_max,QV_mean,QI_min,QI_max,QI_mean\n"
        )

        self._count = 0
        self._global_denom = float(self.nx_gl * self.ny_gl * self.nzm)
        print(f"[debug_dump] enabled → {self.debug_dir}")
        print(f"[debug_dump] IRMA box i=[{self.i_min_gl},{self.i_max_gl}] "
              f"j=[{self.j_min_gl},{self.j_max_gl}] ni={self.ni} nj={self.nj} "
              f"nzm={self.nzm}")

    # ------------------------------------------------------------------
    def _write_header(self):
        f = self._fbin
        np.array([self.nx_gl, self.ny_gl, self.nzm], dtype="<i4").tofile(f)
        np.array([IRMA_LAT_MIN, IRMA_LAT_MAX, IRMA_LON_MIN, IRMA_LON_MAX],
                 dtype="<f4").tofile(f)
        np.array([self.i_min_gl, self.i_max_gl,
                  self.j_min_gl, self.j_max_gl], dtype="<i4").tofile(f)
        np.array([self.it_off, self.jt_off], dtype="<i4").tofile(f)
        np.array([self.i1_loc, self.i2_loc,
                  self.j1_loc, self.j2_loc], dtype="<i4").tofile(f)
        np.array([NFIELDS], dtype="<i4").tofile(f)
        for name in FIELD_NAMES:
            f.write(name.encode("ascii"))
        f.flush()

    # ------------------------------------------------------------------
    def dump(self, state, stage_id: int, dtn: float, force_nstep: int | None = None):
        """
        Capture the current state and append one record to the binary + CSV.

        Parameters
        ----------
        state      : ModelState (values pulled to host via ``jax.device_get``)
        stage_id   : gSAM stage id (0..18)
        dtn        : current sub-step size (for CSV dtn column)
        force_nstep: if given, overrides state.nstep for the record number.
                     Used to keep a constant nstep across all 19 stages of a
                     single step (jsam's advance_scalars increments nstep
                     mid-step, which would otherwise jump the counter at
                     stage 14).
        """
        nz = self.nzm
        # C-grid staggering: U(nz,ny,nx+1), V(nz,ny+1,nx), W(nz+1,ny,nx), scalars(nz,ny,nx)
        # Slice to cell centers (nz, ny, nx) for all fields.
        # Important: U and V have staggered x/y dimensions; must slice them BEFORE
        # putting in the fields tuple to avoid shape mismatches in jnp.stack.
        U_full    = state.U[:nz, :, :-1]                # (nz, ny, nx)
        V_full    = state.V[:nz, :-1, :]                # (nz, ny, nx)
        W_full    = state.W[:nz, :, :]                  # (nz, ny, nx)
        TABS_full = state.TABS
        QC_full   = state.QC
        QV_full   = state.QV
        QI_full   = state.QI

        # Debug: check actual shapes before stacking
        shapes = [U_full.shape, V_full.shape, W_full.shape, TABS_full.shape,
                  QC_full.shape, QV_full.shape, QI_full.shape]
        names = ["U_full", "V_full", "W_full", "TABS", "QC", "QV", "QI"]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(
                f"Shape mismatch in debug dump: {list(zip(names, shapes))}"
            )

        fields = (U_full, V_full, W_full, TABS_full, QC_full, QV_full, QI_full)

        # Global reductions on device — 21 scalars in one go to minimise H2D sync.
        stats = jnp.stack([jnp.stack([jnp.min(F), jnp.max(F), jnp.sum(F)])
                           for F in fields])  # (7, 3)

        # Slice IRMA box on device, stack into (7, nz, nj, ni) for bit-compatible
        # memory order (i fastest, then j, then k, then field-slow — matches
        # Fortran (ni, nj, nzm, 7) in gSAM debug_dump.f90).
        # All fields now have consistent shape (nz, ny, nx) after the slicing above.
        boxes = jnp.stack([
            F[:, self._j_lo:self._j_hi, self._i_lo:self._i_hi] for F in fields
        ])

        # Host transfer — one jax.device_get batches both arrays.
        stats_np, boxes_np = jax.device_get((stats, boxes))
        nstep_val = int(force_nstep) if force_nstep is not None else int(state.nstep)

        # --- binary record ---
        np.array([nstep_val, stage_id], dtype="<i4").tofile(self._fbin)
        # float32, C-contiguous (7, nz, nj, ni) → memory order matches
        # Fortran-order (ni, nj, nzm, 7) that gSAM writes.
        boxes_np.astype("<f4", copy=False).tofile(self._fbin)
        self._fbin.flush()

        # --- CSV row ---
        fmin  = stats_np[:, 0]
        fmax  = stats_np[:, 1]
        fmean = stats_np[:, 2] / self._global_denom
        cols  = [f"{nstep_val}", f"{stage_id}",
                 STAGE_NAMES.get(stage_id, f"stage_{stage_id}"),
                 f"{float(dtn):.9e}"]
        for i in range(NFIELDS):
            cols.append(f"{float(fmin[i]):.9e}")
            cols.append(f"{float(fmax[i]):.9e}")
            cols.append(f"{float(fmean[i]):.9e}")
        self._fcsv.write(",".join(cols) + "\n")
        self._fcsv.flush()

        self._count += 1

    # ------------------------------------------------------------------
    def finalize(self):
        try:
            self._fbin.close()
        except Exception:
            pass
        try:
            self._fcsv.close()
        except Exception:
            pass
        print(f"[debug_dump] finalized: {self._count} records → "
              f"{self._rank_path}, {self._csv_path}")
