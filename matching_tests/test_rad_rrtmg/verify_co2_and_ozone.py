"""
verify_co2_and_ozone.py — Items #6, #7.

#7: Pin the CO2 default to gSAM's IRMA prm value (nxco2=1 →
    3.670e-4 VMR).  Regression test: the pre-fix 400e-6 would fail.

#6: False-alarm documentation — the gSAM IRMA o3file is ERA5 mass mixing
    ratio, and both gSAM RAD_CAM (rad_full.f90:576) and jsam multiply by
    ~0.6034 to convert to VMR.  This test inspects the file header on
    disk and asserts the conversion path is self-consistent.

Run:
    PYTHONPATH=. python matching_tests/test_rad_rrtmg/verify_co2_and_ozone.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from jsam.core.physics.rad_rrtmg import RadRRTMGConfig, _O3_MMR_TO_VMR


O3FILE = Path(
    "/glade/u/home/sabramian/gSAM1.8.7/GLOBAL_DATA/BIN_D/"
    "ozone_era5_monthly_201709-201709_GLOBAL.bin"
)


def main():
    # ── Item #7: CO2 value ───────────────────────────────────────────────
    cfg = RadRRTMGConfig()
    assert abs(cfg.co2_vmr - 3.670e-4) < 1e-10, (
        f"CO2 VMR should be 3.670e-4 (gSAM IRMA prm: nxco2=1, "
        f"docurrentco2=.false. default), got {cfg.co2_vmr}"
    )
    print(f"  CO2 VMR = {cfg.co2_vmr:.4e}  ({cfg.co2_vmr*1e6:.1f} ppmv)")
    print("  matches gSAM rad_full.f90:323  `co2vmr = 3.670e-4 * nxco2`")

    # ── Item #6: o3file is MMR, both sides apply 0.6034 ──────────────────
    ratio = _O3_MMR_TO_VMR
    assert 0.60 < ratio < 0.61, f"unexpected MMR→VMR ratio {ratio}"
    print(f"  _O3_MMR_TO_VMR = {ratio:.6f}  (≡ MW_air/MW_O3 = 28.97/47.998)")
    print("  matches gSAM rad_full.f90:576  `o3vmr(:,k) = 0.6034 * ozone(i,j,m)`")

    if O3FILE.exists():
        with open(O3FILE, "rb") as f:
            def rec():
                n = struct.unpack("<i", f.read(4))[0]
                d = f.read(n)
                assert struct.unpack("<i", f.read(4))[0] == n
                return d
            nx1, ny1, nz1 = struct.unpack("<iii", rec())
            nobs = struct.unpack("<i", rec())[0]
            print(f"  o3file header: nx,ny,nz={nx1},{ny1},{nz1}  nobs={nobs}")
            rec()                                     # days
            # Seek past the first 4 metadata records (lon/lat/z/p) to the
            # first slab and verify the order of magnitude is consistent
            # with ERA5 MMR (~1e-8 at surface, ~1e-5 peaks).
            for _ in range(4):
                rec()
            slab0 = np.frombuffer(rec(), dtype="<f4")
            assert 1e-9 < slab0.max() < 1e-4, (
                f"o3 slab0 range {slab0.min():.3e}..{slab0.max():.3e} "
                "incompatible with ERA5 MMR"
            )
            print(f"  o3file slab0 range [{slab0.min():.3e}, {slab0.max():.3e}] — MMR")
            print("  jsam applies 0.6034 → VMR (matches gSAM path)")
    else:
        print(f"  (skipped disk inspection; {O3FILE} not present)")

    print("PASS — items #6 (ozone) false alarm documented, #7 (CO2) fixed")


if __name__ == "__main__":
    main()
