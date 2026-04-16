"""
Read jsam debug dump rank file and extract the IRMA sub-region
at nstep=1, stage_id=0 (pre_step = initial state).

jsam uses a single rank file covering the full domain, so no
inter-rank stitching is needed.

Writes one binary per field to work/jsam_{field}.bin using the
common bin_io wire format.

Usage
-----
    python read_jsam_init.py [debug_dir]

debug_dir defaults to the IRMA debug500 run on scratch.
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

JSAM_DEBUG = "/glade/derecho/scratch/sabramian/jsam_IRMA_debug500/debug"
FIELDS = ["U", "V", "W", "TABS", "QC", "QV", "QI"]
TARGET_NSTEP = 1
TARGET_STAGE = 0


def read_jsam_irma_box(debug_dir: str) -> dict[str, np.ndarray]:
    """
    Read the single jsam rank file and extract the IRMA sub-region
    at the target nstep/stage.

    Returns dict mapping field name -> (nzm, nj_irma, ni_irma) float32 array.
    """
    rank_path = Path(debug_dir) / "rank_000000.bin"
    if not rank_path.exists():
        raise FileNotFoundError(f"jsam rank file not found: {rank_path}")

    with open(rank_path, "rb") as f:
        # Header
        nx_gl, ny_gl, nzm = struct.unpack("iii", f.read(12))
        f.read(16)  # lat/lon box
        f.read(16)  # global idx
        f.read(8)   # it_off, jt_off
        i1, i2, j1, j2 = struct.unpack("iiii", f.read(16))
        nfields = struct.unpack("i", f.read(4))[0]
        names = [f.read(8).decode("ascii").strip() for _ in range(nfields)]

        ni = i2 - i1 + 1
        nj = j2 - j1 + 1

        print(f"[read_jsam_init] IRMA box: i=[{i1},{i2}] j=[{j1},{j2}] "
              f"=> {ni}x{nj}x{nzm}, {nfields} fields")

        # jsam writes data as C-contiguous (nfields, nzm, nj, ni) with
        # ni fastest — which is the same memory layout as Fortran (ni, nj, nzm, nfields).
        record_data_size = ni * nj * nzm * nfields * 4

        # Scan for target record
        while True:
            rec_hdr = f.read(8)
            if len(rec_hdr) < 8:
                raise ValueError(f"Target record (nstep={TARGET_NSTEP}, "
                                 f"stage={TARGET_STAGE}) not found")
            nstep, stage_id = struct.unpack("ii", rec_hdr)
            if nstep == TARGET_NSTEP and stage_id == TARGET_STAGE:
                raw = np.frombuffer(f.read(record_data_size), dtype=np.float32)
                data = raw.reshape((nfields, nzm, nj, ni))
                break
            else:
                f.seek(record_data_size, 1)

    result = {}
    for fi, name in enumerate(FIELDS):
        if fi < nfields:
            result[name] = data[fi].copy()  # (nzm, nj, ni)
    return result


def main() -> int:
    debug_dir = sys.argv[1] if len(sys.argv) > 1 else JSAM_DEBUG
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    fields = read_jsam_irma_box(debug_dir)

    for name, arr in fields.items():
        out_path = workdir / f"jsam_{name}.bin"
        write_bin(out_path, arr.ravel(order="C"))
        print(f"  {name}: shape={arr.shape}  min={arr.min():.6e}  max={arr.max():.6e}  "
              f"mean={arr.mean():.6e}  -> {out_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
