"""
verify_hydrostatic_rho.py — Item #3 fix.

Pins jsam's `_gsam_reference_column` helper to gSAM setdata.f90:427-466
(dolatlon branch).  Uses a synthetic tabs0 profile that has the
tropospheric lapse rate + stratospheric inversion typical of IRMA, and
reproduces the gSAM Fortran loop line by line in pure NumPy, then
compares to the helper.

Run:
    PYTHONPATH=. python matching_tests/test_era5/verify_hydrostatic_rho.py
"""

from __future__ import annotations

import numpy as np

from jsam.io.era5 import _gsam_reference_column


def _gsam_fortran_loop(z, zi, tabs0, pres0, pres_seed):
    """Line-for-line transcription of gSAM setdata.f90:427-466."""
    RGAS = 287.04
    CP   = 1004.64
    GGR  = 9.79764
    nz = len(z)
    pres = np.array(pres_seed, dtype=np.float64)
    presi = np.zeros(nz + 1, dtype=np.float64)
    presr = np.zeros(nz + 1, dtype=np.float64)
    t0    = np.zeros(nz,     dtype=np.float64)

    for _sweep in range(2):
        presr[0] = (pres0 / 1000.0) ** (RGAS / CP)
        presi[0] = pres0
        for k in range(nz):
            prespot_k = (1000.0 / pres[k]) ** (RGAS / CP)
            t0[k]     = tabs0[k] * prespot_k
            presr[k + 1] = presr[k] - GGR / CP / t0[k] * (zi[k + 1] - zi[k])
            presi[k + 1] = 1000.0 * presr[k + 1] ** (CP / RGAS)
            pres[k] = np.exp(
                np.log(presi[k])
                + np.log(presi[k + 1] / presi[k])
                * (z[k] - zi[k]) / (zi[k + 1] - zi[k])
            )
    rho = (presi[:-1] - presi[1:]) / (zi[1:] - zi[:-1]) / GGR * 100.0
    return rho, presi


def main():
    # Realistic IRMA-like vertical grid: 80 levels up to 30 km.
    nz = 80
    zi = np.linspace(0.0, 30_000.0, nz + 1)
    z  = 0.5 * (zi[:-1] + zi[1:])

    # Tropopause at 16 km, 288 K surface, -6.5 K/km troposphere,
    # +3 K/km stratosphere above 20 km.
    tabs0 = np.where(z < 16000.0,
                     288.0 - 6.5e-3 * z,
                     np.where(z < 20000.0, 288.0 - 6.5e-3 * 16000.0,
                              288.0 - 6.5e-3 * 16000.0 + 3.0e-3 * (z - 20000.0)))
    pres0 = 1013.25     # hPa
    pres_seed = pres0 * np.exp(-z / 8500.0)   # isothermal guess

    ref = _gsam_reference_column(z=z, zi=zi, tabs0=tabs0,
                                 pres0=pres0, pres_seed=pres_seed)
    rho_fort, presi_fort = _gsam_fortran_loop(z, zi, tabs0, pres0, pres_seed)

    max_rel = np.max(np.abs(ref["rho"] - rho_fort) / rho_fort)
    print(f"  max rel rho diff (helper vs gSAM loop) = {max_rel:.3e}")
    print(f"  rho range      = [{ref['rho'].min():.3f}, {ref['rho'].max():.3f}] kg/m³")
    print(f"  presi range    = [{ref['presi'].min():.3f}, {ref['presi'].max():.3f}] hPa")
    print(f"  rhow range     = [{ref['rhow'].min():.3f}, {ref['rhow'].max():.3f}]")

    assert max_rel < 1e-14, f"helper mismatch: rel={max_rel:.3e}"

    # Sanity: show how different the old ideal-gas approach would be.
    RD = 287.04
    pres_pa = ref["pres"] * 100.0
    rho_idealgas_direct = pres_pa / (RD * tabs0)
    rel_gap = np.max(np.abs(rho_idealgas_direct - ref["rho"]) / ref["rho"])
    print(f"  vs. ideal-gas p/(Rd T) (old approach) = {rel_gap:.3e}")
    print("PASS — item #3 hydrostatic reference column matches gSAM")


if __name__ == "__main__":
    main()
