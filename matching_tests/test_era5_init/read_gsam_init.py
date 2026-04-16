"""
Read gSAM debug dump rank files and reconstruct the IRMA sub-region
at nstep=1, stage_id=0 (pre_step = initial state).

Writes one binary per field to work/gsam_{field}.bin using the
common bin_io wire format.

Usage
-----
    python read_gsam_init.py [debug_dir]

debug_dir defaults to the 500-step IRMA debug run on scratch.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))

from common.bin_io import write_bin  # noqa: E402

GSAM_DEBUG = "/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug"
FIELDS = ["U", "V", "W", "TABS", "QC", "QV", "QI"]
TARGET_NSTEP = 1
TARGET_STAGE = 0


def _read_rank_header(f):
    """Read the 128-byte + field-names header from a rank binary."""
    nx_gl, ny_gl, nzm = struct.unpack("iii", f.read(12))
    lat_min, lat_max, lon_min, lon_max = struct.unpack("ffff", f.read(16))
    i_min_gl, i_max_gl, j_min_gl, j_max_gl = struct.unpack("iiii", f.read(16))
    it_off, jt_off = struct.unpack("ii", f.read(8))
    i1_loc, i2_loc, j1_loc, j2_loc = struct.unpack("iiii", f.read(16))
    nfields = struct.unpack("i", f.read(4))[0]
    names = [f.read(8).decode("ascii").strip() for _ in range(nfields)]
    return {
        "nx_gl": nx_gl, "ny_gl": ny_gl, "nzm": nzm,
        "i_min_gl": i_min_gl, "i_max_gl": i_max_gl,
        "j_min_gl": j_min_gl, "j_max_gl": j_max_gl,
        "it_off": it_off, "jt_off": jt_off,
        "i1_loc": i1_loc, "i2_loc": i2_loc,
        "j1_loc": j1_loc, "j2_loc": j2_loc,
        "nfields": nfields, "names": names,
    }


def _has_overlap(hdr):
    return hdr["i1_loc"] <= hdr["i2_loc"] and hdr["j1_loc"] <= hdr["j2_loc"]


def reconstruct_irma_box(debug_dir: str) -> dict[str, np.ndarray]:
    """
    Read all rank files and stitch together the IRMA sub-region
    for nstep=TARGET_NSTEP, stage_id=TARGET_STAGE.

    Returns dict mapping field name -> (nzm, nj_irma, ni_irma) float32 array.
    """
    debug_path = Path(debug_dir)
    rank_files = sorted(debug_path.glob("rank_*.bin"))
    if not rank_files:
        raise FileNotFoundError(f"No rank files in {debug_dir}")

    # First pass: read headers to determine global IRMA box shape
    headers = []
    for rp in rank_files:
        with open(rp, "rb") as f:
            hdr = _read_rank_header(f)
            hdr["path"] = rp
            headers.append(hdr)

    h0 = headers[0]
    nzm = h0["nzm"]
    # IRMA box global indices (1-based inclusive)
    ig_min = h0["i_min_gl"]
    ig_max = h0["i_max_gl"]
    jg_min = h0["j_min_gl"]
    jg_max = h0["j_max_gl"]
    ni_irma = ig_max - ig_min + 1
    nj_irma = jg_max - jg_min + 1
    nfields = h0["nfields"]

    print(f"[read_gsam_init] IRMA box: i=[{ig_min},{ig_max}] j=[{jg_min},{jg_max}] "
          f"=> {ni_irma}x{nj_irma}x{nzm}")

    # Allocate output: (nfields, nzm, nj_irma, ni_irma)
    box = np.zeros((nfields, nzm, nj_irma, ni_irma), dtype=np.float32)
    filled = np.zeros((nj_irma, ni_irma), dtype=bool)

    # Second pass: read data from each rank that overlaps
    for hdr in headers:
        if not _has_overlap(hdr):
            continue

        ni_loc = hdr["i2_loc"] - hdr["i1_loc"] + 1
        nj_loc = hdr["j2_loc"] - hdr["j1_loc"] + 1

        with open(hdr["path"], "rb") as f:
            # Skip header
            header_size = 12 + 16 + 16 + 8 + 16 + 4 + nfields * 8
            f.seek(header_size)

            # Read records until we find the target
            record_data_size = ni_loc * nj_loc * nzm * nfields * 4
            found = False
            while True:
                rec_hdr = f.read(8)
                if len(rec_hdr) < 8:
                    break
                nstep, stage_id = struct.unpack("ii", rec_hdr)
                if nstep == TARGET_NSTEP and stage_id == TARGET_STAGE:
                    raw = np.frombuffer(f.read(record_data_size), dtype=np.float32)
                    # gSAM writes out_box(ni, nj, nzm, nfields) via stream I/O
                    # Fortran column-major: i fastest, j, k, field slowest
                    # = C-order (nfields, nzm, nj, ni) with ni fastest
                    data = raw.reshape((nfields, nzm, nj_loc, ni_loc))
                    found = True
                    break
                else:
                    f.seek(record_data_size, 1)  # skip this record's data

            if not found:
                print(f"  [WARN] rank {hdr['path'].name}: target record not found")
                continue

            # Map local IRMA overlap to global IRMA box coordinates
            # Local indices are 1-based Fortran; global offset is it_off, jt_off
            i_global_start = hdr["it_off"] + hdr["i1_loc"]  # 1-based global
            j_global_start = hdr["jt_off"] + hdr["j1_loc"]
            # Offset into IRMA box (0-based)
            i_box = i_global_start - ig_min
            j_box = j_global_start - jg_min

            box[:, :, j_box:j_box + nj_loc, i_box:i_box + ni_loc] = data
            filled[j_box:j_box + nj_loc, i_box:i_box + ni_loc] = True

    unfilled = (~filled).sum()
    if unfilled > 0:
        print(f"  [WARN] {unfilled} grid points not filled from any rank!")
    else:
        print(f"  All {ni_irma * nj_irma} grid points filled.")

    result = {}
    for fi, name in enumerate(FIELDS):
        result[name] = box[fi]  # (nzm, nj_irma, ni_irma)
    return result


def main() -> int:
    debug_dir = sys.argv[1] if len(sys.argv) > 1 else GSAM_DEBUG
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    fields = reconstruct_irma_box(debug_dir)

    for name, arr in fields.items():
        out_path = workdir / f"gsam_{name}.bin"
        write_bin(out_path, arr.ravel(order="C"))
        print(f"  {name}: shape={arr.shape}  min={arr.min():.6e}  max={arr.max():.6e}  "
              f"mean={arr.mean():.6e}  -> {out_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
