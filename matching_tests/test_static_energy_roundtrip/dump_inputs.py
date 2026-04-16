"""Dump inputs and jsam outputs for test_static_energy_roundtrip.

Cases
-----
roundtrip_identity  — clear air: TABS -> s=TABS+gamaz -> TABS'=s-gamaz, verify identity
roundtrip_strat     — stratospheric T where gamaz >> TABS
roundtrip_cloudy    — non-zero QC/QI: compare jsam s vs gSAM t
gamaz_native        — verify jsam gamaz matches gSAM on 74-level grid
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
SAMJAX_ROOT = MT_ROOT.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, str(SAMJAX_ROOT))

from common.bin_io import write_bin  # noqa: E402

# gSAM physical constants
G  = 9.81
CP = 1004.0
FAC_COND = 2.5104e6 / CP   # LV / CP
FAC_SUB  = 2.8440e6 / CP   # LS / CP


def _build_state(case: str):
    if case == "roundtrip_identity":
        nz, ny, nx = 8, 4, 4
        z = np.linspace(0, 15000, nz, dtype=np.float32)
        gamaz = (G * z / CP).astype(np.float32)
        TABS = np.broadcast_to(
            np.linspace(295, 210, nz, dtype=np.float32)[:, None, None],
            (nz, ny, nx)).copy()
        QC = np.zeros((nz, ny, nx), dtype=np.float32)
        QI = np.zeros((nz, ny, nx), dtype=np.float32)
        QR = np.zeros((nz, ny, nx), dtype=np.float32)
        QS = np.zeros((nz, ny, nx), dtype=np.float32)
    elif case == "roundtrip_strat":
        nz, ny, nx = 8, 4, 4
        z = np.linspace(20000, 60000, nz, dtype=np.float32)
        gamaz = (G * z / CP).astype(np.float32)
        TABS = np.broadcast_to(
            np.linspace(210, 200, nz, dtype=np.float32)[:, None, None],
            (nz, ny, nx)).copy()
        QC = np.zeros((nz, ny, nx), dtype=np.float32)
        QI = np.zeros((nz, ny, nx), dtype=np.float32)
        QR = np.zeros((nz, ny, nx), dtype=np.float32)
        QS = np.zeros((nz, ny, nx), dtype=np.float32)
    elif case == "roundtrip_cloudy":
        nz, ny, nx = 8, 4, 4
        z = np.linspace(0, 15000, nz, dtype=np.float32)
        gamaz = (G * z / CP).astype(np.float32)
        TABS = np.broadcast_to(
            np.linspace(295, 230, nz, dtype=np.float32)[:, None, None],
            (nz, ny, nx)).copy()
        QC = np.full((nz, ny, nx), 0.5e-3, dtype=np.float32)
        QI = np.full((nz, ny, nx), 0.3e-3, dtype=np.float32)
        QR = np.full((nz, ny, nx), 0.1e-3, dtype=np.float32)
        QS = np.full((nz, ny, nx), 0.05e-3, dtype=np.float32)
    elif case == "gamaz_native":
        from jsam.utils.IRMALoader import IRMALoader
        g = IRMALoader().grid
        z = np.asarray(g["z"], dtype=np.float32)
        nz = len(z)
        ny, nx = 4, 4
        gamaz = (G * z / CP).astype(np.float32)
        TABS = np.broadcast_to(
            (295.0 - 6.5e-3 * z).clip(min=200.0).astype(np.float32)[:, None, None],
            (nz, ny, nx)).copy()
        QC = np.zeros((nz, ny, nx), dtype=np.float32)
        QI = np.zeros((nz, ny, nx), dtype=np.float32)
        QR = np.zeros((nz, ny, nx), dtype=np.float32)
        QS = np.zeros((nz, ny, nx), dtype=np.float32)
    else:
        raise ValueError(f"unknown case: {case}")

    return nz, ny, nx, gamaz, TABS, QC, QI, QR, QS


def _jsam_static_energy(nz, ny, nx, gamaz, TABS, QC, QI, QR, QS):
    """Compute jsam's s = TABS + gamaz and gSAM's t, plus recoveries."""
    import jax
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp

    gamaz_3d = jnp.asarray(gamaz)[:, None, None]
    tabs = jnp.asarray(TABS)
    qc = jnp.asarray(QC)
    qi = jnp.asarray(QI)
    qr = jnp.asarray(QR)
    qs = jnp.asarray(QS)

    # gSAM: t = TABS + gamaz - fac_cond*(qcl+qpl) - fac_sub*(qci+qpi)
    gsam_t = tabs + gamaz_3d - FAC_COND * (qc + qr) - FAC_SUB * (qi + qs)
    gsam_trec = gsam_t - gamaz_3d + FAC_COND * (qc + qr) + FAC_SUB * (qi + qs)

    # jsam: s = TABS + gamaz
    jsam_s = tabs + gamaz_3d
    jsam_srec = jsam_s - gamaz_3d

    return (np.asarray(gsam_t, dtype=np.float32),
            np.asarray(gsam_trec, dtype=np.float32),
            np.asarray(jsam_s, dtype=np.float32),
            np.asarray(jsam_srec, dtype=np.float32))


def main() -> int:
    case = sys.argv[1]

    nz, ny, nx, gamaz, TABS, QC, QI, QR, QS = _build_state(case)

    # Write inputs.bin
    with open("inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(gamaz.astype(np.float32).tobytes())
        f.write(struct.pack("ff", float(FAC_COND), float(FAC_SUB)))
        for a in (TABS, QC, QI, QR, QS):
            f.write(a.astype(np.float32).tobytes(order="C"))

    # jsam side
    gsam_t, gsam_trec, jsam_s, jsam_srec = _jsam_static_energy(
        nz, ny, nx, gamaz, TABS, QC, QI, QR, QS)

    out = np.concatenate([
        gsam_t.ravel(order="C"),
        gsam_trec.ravel(order="C"),
        jsam_s.ravel(order="C"),
        jsam_srec.ravel(order="C"),
    ])
    write_bin("jsam_out.bin", out)

    print(f"[static_energy] case={case}  nz={nz}  ny={ny}  nx={nx}")
    if case == "roundtrip_cloudy":
        condensate_term = FAC_COND * (QC + QR) + FAC_SUB * (QI + QS)
        print(f"  condensate term: max={condensate_term.max():.4f} K "
              f"(this is what jsam's s is missing vs gSAM's t)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
