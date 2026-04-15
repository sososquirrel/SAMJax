"""Reassemble an IRMA-box tensor from gSAM debug_dump rank_*.bin files.

See /glade/u/home/sabramian/SAMJax/.claude/Oracle_Tensor_Structure.md
for the on-disk layout. Usage:

    from oracle_reassemble import reassemble_stage
    arr = reassemble_stage(oracle_dir, nstep=1, stage_id=0)
    # arr.shape == (NI_GL, NJ_GL, NZM, 7)  float32, (U,V,W,TABS,QC,QV,QI)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

NFIELDS = 7
NZM = 74
FIELD_NAMES = ("U", "V", "W", "TABS", "QC", "QV", "QI")


def _read_header(f):
    nx_gl, ny_gl, nzm = np.fromfile(f, "<i4", 3)
    lat_min, lat_max, lon_min, lon_max = np.fromfile(f, "<f4", 4)
    i_min_gl, i_max_gl, j_min_gl, j_max_gl = np.fromfile(f, "<i4", 4)
    it_off, jt_off = np.fromfile(f, "<i4", 2)
    i1, i2, j1, j2 = np.fromfile(f, "<i4", 4)
    nfields = int(np.fromfile(f, "<i4", 1)[0])
    _ = [f.read(8) for _ in range(nfields)]
    return dict(
        nx_gl=int(nx_gl), ny_gl=int(ny_gl), nzm=int(nzm),
        lat_min=float(lat_min), lat_max=float(lat_max),
        lon_min=float(lon_min), lon_max=float(lon_max),
        i_min_gl=int(i_min_gl), i_max_gl=int(i_max_gl),
        j_min_gl=int(j_min_gl), j_max_gl=int(j_max_gl),
        it_off=int(it_off), jt_off=int(jt_off),
        i1=int(i1), i2=int(i2), j1=int(j1), j2=int(j2),
        nfields=nfields,
        has_overlap=(int(i1) <= int(i2) and int(j1) <= int(j2)),
    )


def reassemble_stage(oracle_dir: str | Path, nstep: int, stage_id: int) -> np.ndarray:
    """Return float32 array of shape (NI_GL, NJ_GL, NZM, 7) for the IRMA box.

    Cells not covered by any rank stay NaN.
    """
    oracle_dir = Path(oracle_dir)
    rank_files = sorted(oracle_dir.glob("rank_*.bin"))
    if not rank_files:
        raise FileNotFoundError(f"no rank_*.bin files under {oracle_dir}")

    # Prime dimensions from rank 0.
    with open(rank_files[0], "rb") as f:
        h0 = _read_header(f)

    NI_GL = h0["i_max_gl"] - h0["i_min_gl"] + 1
    NJ_GL = h0["j_max_gl"] - h0["j_min_gl"] + 1
    assert h0["nzm"] == NZM, f"expected nzm=74, got {h0['nzm']}"

    out = np.full((NI_GL, NJ_GL, NZM, NFIELDS), np.nan, dtype=np.float32)

    for rf in rank_files:
        with open(rf, "rb") as f:
            h = _read_header(f)
            if not h["has_overlap"]:
                continue
            ni = h["i2"] - h["i1"] + 1
            nj = h["j2"] - h["j1"] + 1
            rec_bytes = 4 + 4 + ni * nj * NZM * NFIELDS * 4
            # Scan records until we find (nstep, stage_id).
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                ns = int(np.frombuffer(hdr[:4], "<i4")[0])
                sid = int(np.frombuffer(hdr[4:], "<i4")[0])
                blob = f.read(rec_bytes - 8)
                if ns == nstep and sid == stage_id:
                    rec = np.frombuffer(blob, dtype="<f4").reshape(
                        (ni, nj, NZM, NFIELDS), order="F"
                    )
                    # Global indices for this rank's overlap slice:
                    ig_lo = (h["it_off"] + h["i1"]) - h["i_min_gl"]
                    jg_lo = (h["jt_off"] + h["j1"]) - h["j_min_gl"]
                    out[ig_lo:ig_lo + ni, jg_lo:jg_lo + nj, :, :] = rec
                    break

    n_missing = int(np.isnan(out).any(axis=(-1, -2)).sum())
    if n_missing:
        print(f"[oracle] warning: {n_missing}/{NI_GL*NJ_GL} columns missing "
              f"(no rank covered them)")
    return out


def get_box_info(oracle_dir: str | Path) -> dict:
    """Return {i_min_gl, i_max_gl, j_min_gl, j_max_gl, nx_gl, ny_gl, nzm}."""
    oracle_dir = Path(oracle_dir)
    with open(sorted(oracle_dir.glob("rank_*.bin"))[0], "rb") as f:
        h = _read_header(f)
    return {k: h[k] for k in ("i_min_gl", "i_max_gl", "j_min_gl", "j_max_gl",
                              "nx_gl", "ny_gl", "nzm")}
