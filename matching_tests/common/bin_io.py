"""
Float32 binary I/O shared by all matching-test dump_inputs.py scripts.

Wire format (matches the Fortran stream-access drivers):
    int32  ndim   = 1      (always 1; present for compatibility)
    int32  size   = N      (number of float32 elements)
    float32[N]             (data in C row-major order)

The Fortran side writes:
    write(u_out) 1_4
    write(u_out) int(N, 4)
    write(u_out) arr        ! float32, stream access → raw bytes
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_bin(path: "str | Path", arr: "np.ndarray") -> None:
    """Write arr as a matching-test float32 binary."""
    arr = np.asarray(arr, dtype=np.float32).ravel()
    with open(path, "wb") as f:
        f.write(struct.pack("ii", 1, len(arr)))
        f.write(arr.tobytes())


def read_bin(path: "str | Path") -> np.ndarray:
    """Read a float32 binary written by write_bin or the Fortran driver."""
    with open(path, "rb") as f:
        ndim, size = struct.unpack("ii", f.read(8))
        return np.frombuffer(f.read(size * 4), dtype=np.float32).copy()
