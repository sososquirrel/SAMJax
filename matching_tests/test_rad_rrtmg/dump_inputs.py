"""
dump_inputs.py -- RRTMG SW+LW wrapper-vs-raw-f2py test.

Builds a single realistic tropical column (nz=74 layers), calls jsam
_rrtmg_sw_numpy and _rrtmg_lw_numpy wrappers, and also calls the f2py
extensions directly with the same prepared arrays.

Writes:
    work/jsam_out.bin      -- wrapper output
    work/fortran_out.bin   -- raw f2py output
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))

from common.bin_io import write_bin

from jsam.core.physics.rad_rrtmg import (
    RadRRTMGConfig,
    _ensure_sw_initialized,
    _ensure_initialized,
    _rrtmg_sw_numpy,
    _rrtmg_lw_numpy,
    _ice_re_from_T,
    _LIQ_RE_OCEAN,
    _CP_DAIR_DEFAULT,
    _NBNDSW,
    _NAERSW,
)


def _tropical_column(nlay=74):
    p_top, p_sfc = 0.4, 1013.25
    plev = np.exp(np.linspace(np.log(p_sfc), np.log(p_top), nlay + 1))
    play = 0.5 * (plev[:-1] + plev[1:])
    T_sfc = 300.0
    tlay = np.empty(nlay)
    for k in range(nlay):
        p = play[k]
        if p > 100.0:
            tlay[k] = T_sfc - 80.0 * (np.log(p_sfc/p) / np.log(p_sfc/100.0))
        elif p > 10.0:
            tlay[k] = 200.0
        else:
            tlay[k] = 200.0 + 30.0 * (np.log(10.0/p) / np.log(10.0/p_top))
    tlay = np.clip(tlay, 180.0, 310.0)
    tlev = np.empty(nlay + 1)
    tlev[0] = T_sfc
    tlev[1:-1] = 0.5 * (tlay[:-1] + tlay[1:])
    tlev[-1] = 2.0 * tlay[-1] - tlev[-2]
    qv = 0.020 * np.exp(-np.log(p_sfc/play) / np.log(p_sfc/200.0) * 6.0)
    qv = np.clip(qv, 1e-7, 0.025)
    h2ovmr = qv * (28.97 / 18.02)
    o3vmr = 8e-6 * np.exp(-0.5 * (np.log(play/10.0)/1.2)**2)
    dp = plev[:-1] - plev[1:]
    lm = 100.0 * dp / 9.79764
    cliqwp = np.zeros(nlay)
    cicewp = np.zeros(nlay)
    cldfr = np.zeros(nlay)
    for k in range(nlay):
        if 750.0 < play[k] < 850.0:
            cliqwp[k] = 3e-4 * 1e3 * lm[k]; cldfr[k] = 1.0
    for k in range(nlay):
        if 250.0 < play[k] < 350.0:
            cicewp[k] = 5e-5 * 1e3 * lm[k]; cldfr[k] = 1.0
    reliq = np.clip(np.where(cliqwp > 0, _LIQ_RE_OCEAN, 2.5), 2.5, 60.0)
    reice = np.clip(np.where(cicewp > 0, _ice_re_from_T(tlay), 5.0), 5.0, 131.0)
    def col(x): return x.reshape(1, -1).astype(np.float64)
    return dict(
        play_hpa=col(play), plev_hpa=plev.reshape(1,-1).astype(np.float64),
        tlay=col(tlay), tlev=tlev.reshape(1,-1).astype(np.float64),
        tsfc=np.array([T_sfc], dtype=np.float64),
        h2ovmr=col(h2ovmr), o3vmr=col(o3vmr),
        cldfr=col(cldfr), cicewp=col(cicewp), cliqwp=col(cliqwp),
        reice=col(reice), reliq=col(reliq),
        coszen=np.array([0.7], dtype=np.float64), eccf=1.0,
        asdir=np.array([0.03], dtype=np.float64),
        asdif=np.array([0.07], dtype=np.float64),
        aldir=np.array([0.03], dtype=np.float64),
        aldif=np.array([0.07], dtype=np.float64),
    )


def _call_raw_sw(inp, cfg):
    sw = _ensure_sw_initialized(cfg.cpdair)
    ncol, nlay = inp["play_hpa"].shape
    def _f(v): return np.full((ncol, nlay), float(v), dtype=np.float64)
    taucld = np.zeros((_NBNDSW, ncol, nlay), dtype=np.float64)
    ssacld = np.ones((_NBNDSW, ncol, nlay), dtype=np.float64)
    asmcld = np.zeros((_NBNDSW, ncol, nlay), dtype=np.float64)
    fsfcld = np.zeros((_NBNDSW, ncol, nlay), dtype=np.float64)
    tauaer = np.zeros((ncol, nlay, _NBNDSW), dtype=np.float64)
    ssaaer = np.zeros((ncol, nlay, _NBNDSW), dtype=np.float64)
    asmaer = np.zeros((ncol, nlay, _NBNDSW), dtype=np.float64)
    ecaer = np.zeros((ncol, nlay, _NAERSW), dtype=np.float64)
    icld = np.array(1, dtype=np.int32)
    C = np.ascontiguousarray
    r = sw.rrtmg_sw_rad_nomcica.rrtmg_sw(
        ncol, nlay, icld,
        C(inp["play_hpa"]), C(inp["plev_hpa"]),
        C(inp["tlay"]), C(inp["tlev"]), C(inp["tsfc"]),
        C(inp["h2ovmr"]), C(inp["o3vmr"]),
        _f(cfg.co2_vmr), _f(cfg.ch4_vmr), _f(cfg.n2o_vmr), _f(cfg.o2_vmr),
        C(inp["asdir"]), C(inp["asdif"]), C(inp["aldir"]), C(inp["aldif"]),
        C(inp["coszen"]), float(inp["eccf"]), 0, 1367.0,
        2, 3, 1, C(inp["cldfr"]),
        taucld, ssacld, asmcld, fsfcld,
        C(inp["cicewp"]), C(inp["cliqwp"]),
        C(inp["reice"]), C(inp["reliq"]),
        tauaer, ssaaer, asmaer, ecaer,
    )
    return np.asarray(r[2], dtype=np.float64), np.asarray(r[1], dtype=np.float64)


def _call_raw_lw(inp, cfg):
    _ensure_initialized(cfg.cpdair)
    from jsam.core.physics.rad_rrtmg import _LW
    ncol, nlay = inp["play_hpa"].shape
    def _f(v): return np.full((ncol, nlay), float(v), dtype=np.float64)
    emis = np.full((ncol, 16), float(cfg.emis), dtype=np.float64)
    taucld = np.zeros((16, ncol, nlay), dtype=np.float64)
    tauaer = np.zeros((ncol, nlay, 16), dtype=np.float64)
    C = np.ascontiguousarray
    r = _LW.rrtmg_lw_rad_nomcica.rrtmg_lw(
        ncol, nlay, 1, 0,
        C(inp["play_hpa"]), C(inp["plev_hpa"]),
        C(inp["tlay"]), C(inp["tlev"]), C(inp["tsfc"]),
        C(inp["h2ovmr"]), C(inp["o3vmr"]),
        _f(cfg.co2_vmr), _f(cfg.ch4_vmr), _f(cfg.n2o_vmr), _f(cfg.o2_vmr),
        _f(cfg.cfc11_vmr), _f(cfg.cfc12_vmr), _f(cfg.cfc22_vmr), _f(cfg.ccl4_vmr),
        emis, 2, 3, 1, C(inp["cldfr"]), taucld,
        C(inp["cicewp"]), C(inp["cliqwp"]),
        C(inp["reice"]), C(inp["reliq"]), tauaer,
    )
    return np.asarray(r[2], dtype=np.float64), np.asarray(r[1], dtype=np.float64)


def main():
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = RadRRTMGConfig()
    inp = _tropical_column()

    print("  calling jsam _rrtmg_sw_numpy wrapper ...")
    sw_hr_j, sw_df_j = _rrtmg_sw_numpy(
        play_hpa=inp["play_hpa"], plev_hpa=inp["plev_hpa"],
        tlay=inp["tlay"], tlev=inp["tlev"], tsfc=inp["tsfc"],
        h2ovmr=inp["h2ovmr"], o3vmr=inp["o3vmr"],
        asdir=inp["asdir"], asdif=inp["asdif"],
        aldir=inp["aldir"], aldif=inp["aldif"],
        coszen=inp["coszen"], eccf=inp["eccf"],
        cldfr=inp["cldfr"],
        cicewp=inp["cicewp"], cliqwp=inp["cliqwp"],
        reice=inp["reice"], reliq=inp["reliq"], cfg=cfg,
    )
    print("  calling jsam _rrtmg_lw_numpy wrapper ...")
    lw_hr_j, lw_df_j = _rrtmg_lw_numpy(
        play_hpa=inp["play_hpa"], plev_hpa=inp["plev_hpa"],
        tlay=inp["tlay"], tlev=inp["tlev"], tsfc=inp["tsfc"],
        h2ovmr=inp["h2ovmr"], o3vmr=inp["o3vmr"],
        cldfr=inp["cldfr"],
        cicewp=inp["cicewp"], cliqwp=inp["cliqwp"],
        reice=inp["reice"], reliq=inp["reliq"], cfg=cfg,
    )

    jsam_out = np.concatenate([
        sw_hr_j.ravel(), lw_hr_j.ravel(),
        sw_df_j[0, :1], lw_df_j[0, :1],
    ]).astype(np.float32)
    write_bin(workdir / "jsam_out.bin", jsam_out)
    print(f"  wrote jsam_out.bin  ({jsam_out.size} floats)")

    print("  calling raw f2py RRTMG_SW ...")
    sw_hr_r, sw_df_r = _call_raw_sw(inp, cfg)
    print("  calling raw f2py RRTMG_LW ...")
    lw_hr_r, lw_df_r = _call_raw_lw(inp, cfg)

    fort_out = np.concatenate([
        sw_hr_r.ravel(), lw_hr_r.ravel(),
        sw_df_r[0, :1], lw_df_r[0, :1],
    ]).astype(np.float32)
    write_bin(workdir / "fortran_out.bin", fort_out)
    print(f"  wrote fortran_out.bin  ({fort_out.size} floats)")

    print(f"  SW hr range: [{sw_hr_j.min():.3f}, {sw_hr_j.max():.3f}] K/day")
    print(f"  LW hr range: [{lw_hr_j.min():.3f}, {lw_hr_j.max():.3f}] K/day")
    print(f"  SW sfc down: {sw_df_j[0,0]:.2f} W/m^2")
    print(f"  LW sfc down: {lw_df_j[0,0]:.2f} W/m^2")


if __name__ == "__main__":
    main()
