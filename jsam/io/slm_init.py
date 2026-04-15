"""
Build an ``(SLMStatic, SLMState)`` pair matching the gSAM IRMA
``prm_debug500`` namelist.

Mirrors the Fortran pipeline in ``SRC/SLM/slm_vars.f90``:
    slm_init → init_soil_tw → init_soil_par → vege_root_init → init_slm_vars

Reads the same seven binary files that ``prm_debug500`` consumes from
``GLOBAL_DATA/BIN_D/``, regrids them from the gSAM native
``lat_720_dyvar`` grid (1440×720) to the jsam target grid, and assembles
all static Cosby 1984 soil hydraulics + rooting profile + IGBP-derived
land parameters.

The ``prm_debug500`` configuration is hard-wired here (we do not read the
namelist): readland=T, readsoil=T, readinitsoil=T, readsnow=T, readsnowt=T,
readseaice=F, readimperv=F, dosoilvolumetric=F, dorunoff=F.

gSAM uses 9 soil layers with thicknesses taken from the IRMA case file
``CASES/IRMA/soil``:
    [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.37, 0.5, 1.0] m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from jsam.core.state import ModelState
from jsam.core.physics.slm.params import SLMParams
from jsam.core.physics.slm.state import SLMStatic, SLMState, NSOIL
from jsam.core.physics.slm.landtypes import lookup_igbp_fields
from jsam.io.gsam_binary import (
    read_landtype, read_landmask, read_lai_monthly,
    read_soil_sand_clay, read_snow, read_snowt, read_soil_init,
    read_lat_dyvar, interp_horiz_dyvar,
)


# 9-layer thicknesses (m) from gSAM CASES/IRMA/soil
_SOIL_S_DEPTH = np.array(
    [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.37, 0.5, 1.0], dtype=np.float32
)


def _compute_layer_geometry(s_depth: np.ndarray):
    """Given a (nsoil,) thickness array, return node_z and interface_z.

    Mirrors slm_vars.f90 lines 1134–1148.
    node_z(k) = layer centre depth; interface_z(k) = bottom of layer k.
    """
    nsoil = s_depth.shape[0]
    node_z = np.zeros(nsoil, dtype=np.float32)
    node_z[0] = 0.5 * s_depth[0]
    for k in range(1, nsoil):
        node_z[k] = 0.5 * s_depth[k] + float(np.sum(s_depth[:k]))
    interface_z = node_z + 0.5 * s_depth
    return node_z, interface_z


def _cosby_1984(SAND: np.ndarray, CLAY: np.ndarray):
    """Cosby 1984 soil hydraulic parameters from SAND/CLAY percentages.

    Ports slm_vars.f90:1194-1261. SAND, CLAY are arrays with values in %.
    Returns a dict of numpy arrays matching input shape.
    """
    sand_frac = SAND / 100.0

    sst_cond_other = np.where(SAND > 20.0, 2.0, 3.0).astype(np.float32)
    sst_cond = (7.7 ** sand_frac) * (sst_cond_other ** (1.0 - sand_frac))

    # hydraulic conductivity at saturation (mm/s)
    ks = (10.0 ** (0.0153 * SAND - 0.884)) * (25.4 / 3600.0)
    # constant B
    Bconst = 0.159 * CLAY + 2.91
    # porosity
    poro_soil = -0.00126 * SAND + 0.489
    # moisture potential at saturation (mm, negative)
    m_pot_sat = np.minimum(
        -150.0,
        -10.0 * (10.0 ** (1.88 - 0.0131 * SAND)),
    )
    # soil heat capacity (J/m³/K) — de Vries 1963
    denom = np.maximum(SAND + CLAY, 1e-30)
    sst_capa = (2.128 * SAND + 2.385 * CLAY) / denom * 1.0e6
    # field-capacity / wilting-point volumetric contents
    # theta_FC = poro * (0.1/86400 / ks)**(1/(2*B+3))
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_FC = poro_soil * (
            (0.1 / 86400.0 / np.maximum(ks, 1e-30)) ** (1.0 / (2.0 * Bconst + 3.0))
        )
        theta_WP = poro_soil * (
            (-150000.0 / m_pot_sat) ** (-1.0 / Bconst)
        )
    # soil wetness (normalised) at FC / WP
    w_s_FC = theta_FC / np.maximum(poro_soil, 1e-30)
    w_s_WP = theta_WP / np.maximum(poro_soil, 1e-30)
    return dict(
        ks=ks.astype(np.float32),
        Bconst=Bconst.astype(np.float32),
        poro_soil=poro_soil.astype(np.float32),
        m_pot_sat=m_pot_sat.astype(np.float32),
        sst_capa=sst_capa.astype(np.float32),
        sst_cond=sst_cond.astype(np.float32),
        theta_FC=theta_FC.astype(np.float32),
        theta_WP=theta_WP.astype(np.float32),
        w_s_FC=w_s_FC.astype(np.float32),
        w_s_WP=w_s_WP.astype(np.float32),
    )


def _vege_root_init(rootL, root_a, root_b, interface_z):
    """Port of slm_vars.f90:1266-1306.

    rootL, root_a, root_b : (ny, nx)
    interface_z            : (nsoil, ny, nx)
    Returns rootF of shape (nsoil, ny, nx) summing to 1 over the nrootind
    layers that contain live roots.
    """
    nsoil, ny, nx = interface_z.shape
    rootF = np.zeros((nsoil, ny, nx), dtype=np.float32)

    # nrootind[i,j] = smallest k (1-indexed) such that interface_z[k] >= rootL
    # Fortran loops k=nsoil..2 and sets nrootind=k when interface_z[k]>=rootL
    # and interface_z[k-1]<rootL. Default is 1.
    # Equivalent: searchsorted.
    # For vectorisation, compute per-cell nrootind as
    # argmax along the layer axis of the condition.
    rootL_b = rootL[None]  # (1, ny, nx)
    ge_mask = interface_z >= rootL_b          # (nsoil, ny, nx)
    # first layer where ge_mask is True:
    first_ge = np.argmax(ge_mask, axis=0).astype(np.int32)  # 0 if never True
    any_ge = ge_mask.any(axis=0)
    nrootind = np.where(any_ge, first_ge, nsoil - 1).astype(np.int32)  # 0-indexed

    # Compute rootF per cell using a Python loop in k (9 layers) — cheap.
    for j in range(ny):
        for i in range(nx):
            k_ind = int(nrootind[j, i])
            if rootL[j, i] <= 0.0:
                continue  # non-vegetated
            a = float(root_a[j, i])
            b = float(root_b[j, i])
            # root fraction ABOVE each interface
            frac = np.zeros(nsoil, dtype=np.float64)
            # Deepest rooted layer: fraction up to rootL
            frac[k_ind] = 1.0 - 0.5 * (
                np.exp(-a * float(rootL[j, i])) + np.exp(-b * float(rootL[j, i]))
            )
            tot_root_density = frac[k_ind]
            for k in range(k_ind):
                z = float(interface_z[k, j, i])
                frac[k] = 1.0 - 0.5 * (np.exp(-a * z) + np.exp(-b * z))
            # Convert cumulative fractions to layer fractions (top-down)
            if k_ind > 0:
                for k in range(k_ind, 0, -1):
                    frac[k] = frac[k] - frac[k - 1]
                if tot_root_density > 0:
                    for k in range(k_ind + 1):
                        frac[k] = frac[k] / tot_root_density
            rootF[:, j, i] = frac.astype(np.float32)
    return rootF


def _vertical_interp_soil(src_z: np.ndarray, src_field: np.ndarray,
                          tgt_z: np.ndarray) -> np.ndarray:
    """Interpolate soil profile from src layers to tgt layers.

    Mirrors slm_vars.f90:871-888 / 898-922. Linear in depth. Layer 0 is
    extrapolated from layer 1 (flds(0) = flds(1)), then for each target
    depth we find the first source interface above it and linearly
    interpolate.

    src_z : (nsoil_src,)
    src_field : (nsoil_src, ny, nx)
    tgt_z : (nsoil_tgt,)
    Returns (nsoil_tgt, ny, nx).
    """
    nsoil_src = src_z.shape[0]
    # Prepend level 0 at z=0 with value = level 1
    z_ext = np.concatenate([[0.0], src_z]).astype(np.float32)          # (nsoil_src+1,)
    f_ext = np.concatenate([src_field[:1], src_field], axis=0)         # (nsoil_src+1, ny, nx)

    nsoil_tgt = tgt_z.shape[0]
    ny, nx = src_field.shape[1], src_field.shape[2]
    out = np.zeros((nsoil_tgt, ny, nx), dtype=np.float32)
    for k, z in enumerate(tgt_z):
        # find smallest i such that z_ext[i] >= z; Fortran does the same
        placed = False
        for iz in range(1, nsoil_src + 1):
            if z_ext[iz] >= z:
                w = (z - z_ext[iz - 1]) / max(z_ext[iz] - z_ext[iz - 1], 1e-30)
                out[k] = f_ext[iz - 1] + w * (f_ext[iz] - f_ext[iz - 1])
                placed = True
                break
        if not placed:
            out[k] = f_ext[-1]
    return out


def build_slm_static_and_state(
    grid,                  # LatLonGrid (jsam target)
    metric: dict,          # jsam metric dict
    state: ModelState,     # initial atmospheric state (for init_slm_vars)
    *,
    date_month: int = 9,   # 1..12 — month used to pick LAI climatology slice
    data_root: str = "/glade/u/home/sabramian/gSAM1.8.7/GLOBAL_DATA/BIN_D",
    params: SLMParams | None = None,
    landtypefile: str | None = None,
    soilfile:     str | None = None,
    LAIfile:      str | None = None,
    landmaskfile: str | None = None,
    soilinitfile: str | None = None,
    snowfile:     str | None = None,
    snowtfile:    str | None = None,
) -> tuple[SLMStatic, SLMState]:
    """Mirror of gSAM ``slm_init`` for the ``prm_debug500`` configuration.

    All seven binary files are read at the gSAM native 1440×720 dyvar grid,
    regridded to the jsam ``(grid.lat, grid.lon)`` target with
    ``interp_horiz_dyvar`` (nearest for int fields, bilinear for real),
    then fed through the 16-class IGBP lookup + Cosby 1984 soil hydraulics
    + rooting profile + ``init_slm_vars``.
    """
    if params is None:
        params = SLMParams()

    landtypefile = landtypefile or f"{data_root}/landtype_1440x720_dyvar.bin"
    soilfile     = soilfile     or f"{data_root}/soil_1440x720_dyvar.bin"
    LAIfile      = LAIfile      or f"{data_root}/lai_1440x720_dyvar.bin"
    landmaskfile = landmaskfile or f"{data_root}/landmask_1440x720_dyvar.bin"
    soilinitfile = soilinitfile or f"{data_root}/soil_init_2017090500_1440x720_dyvar_era5.bin"
    snowfile     = snowfile     or f"{data_root}/snow_2017090500_1440x720_dyvar_era5.bin"
    snowtfile    = snowtfile    or f"{data_root}/snowt_2017090500_1440x720_dyvar_era5.bin"

    # ── Read source files at gSAM native grid ────────────────────────────
    lt_src   = read_landtype(landtypefile)           # (720, 1440) int32
    lm_src   = read_landmask(landmaskfile)           # (720, 1440) int32
    lai_src  = read_lai_monthly(LAIfile)             # (12, 720, 1440) float32
    sand_src, clay_src = read_soil_sand_clay(soilfile)  # (720, 1440) each
    snow_src  = read_snow(snowfile)                  # (720, 1440) float32 [m snow depth]
    snowt_src = read_snowt(snowtfile)                # (720, 1440) float32
    zsoil_src, soilt_src, soilw_src = read_soil_init(soilinitfile)
    # zsoil_src:   (nsoil_src,)
    # soilt_src:   (nsoil_src, 720, 1440)
    # soilw_src:   (nsoil_src, 720, 1440)

    # ── Regrid to jsam target grid ────────────────────────────────────────
    src_lat = read_lat_dyvar().astype(np.float32)
    src_lon = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float32)
    tgt_lat = np.asarray(grid.lat, dtype=np.float32)
    tgt_lon = np.asarray(grid.lon, dtype=np.float32)
    # jsam lon may be in [-180, 180]; shift to [0, 360) so it matches gSAM
    tgt_lon_mod = np.mod(tgt_lon, 360.0)

    def rg(f, method="bilinear"):
        return interp_horiz_dyvar(f, src_lat, src_lon, tgt_lat, tgt_lon_mod, method=method)

    landtype  = rg(lt_src.astype(np.float32), method="nearest").astype(np.int32)
    landmask  = rg(lm_src.astype(np.float32), method="nearest").astype(np.int8)
    lai_12    = rg(lai_src)                                                     # (12, ny, nx)
    SAND_2d   = rg(sand_src)
    CLAY_2d   = rg(clay_src)
    snow_2d   = rg(snow_src)
    snowt_2d  = rg(snowt_src)
    soilt_src_tgt = rg(soilt_src)                                               # (nsoil_src, ny, nx)
    soilw_src_tgt = rg(soilw_src)

    # Pick the LAI month (1-indexed → 0-indexed).
    lai_month = np.clip(int(date_month) - 1, 0, 11)
    lai_raw = lai_12[lai_month]                                                # (ny, nx)

    # ── IGBP lookup → per-cell land parameter fields ─────────────────────
    ig = lookup_igbp_fields(landtype, lai_raw, params)

    # disp_hgt = min(0.4*z(1), 0.65*ztop)  [slm_vars.f90:446]
    z1 = float(np.asarray(metric["z"])[0])
    disp_hgt = np.minimum(0.4 * z1, 0.65 * ig["ztop"]).astype(np.float32)

    # ── Soil layer geometry (9 layers from CASES/IRMA/soil) ──────────────
    ny, nx = landtype.shape
    s_depth_1d = _SOIL_S_DEPTH                                                  # (nsoil,)
    node_z_1d, interface_z_1d = _compute_layer_geometry(s_depth_1d)             # (nsoil,)

    s_depth_3d  = np.broadcast_to(s_depth_1d[:, None, None], (NSOIL, ny, nx)).copy()
    node_z_3d   = np.broadcast_to(node_z_1d[:, None, None],  (NSOIL, ny, nx)).copy()
    interface_z_3d = np.broadcast_to(interface_z_1d[:, None, None], (NSOIL, ny, nx)).copy()

    # Seaice depth adjustment (slm_vars.f90:1113) — with readseaice=F all
    # seaicemask cells are zero, but apply the formula anyway for
    # completeness. For cells with landmask==0 the depth is scaled by
    # seaicedepth/sum(s_depth).
    scale = float(params.seaicedepth / s_depth_1d.sum())
    ocean_mask_3d = (landmask[None] == 0)
    s_depth_3d = np.where(ocean_mask_3d, s_depth_1d[:, None, None] * scale, s_depth_3d)

    # Broadcast SAND/CLAY to (nsoil, ny, nx)
    SAND_3d = np.broadcast_to(SAND_2d[None], (NSOIL, ny, nx)).astype(np.float32).copy()
    CLAY_3d = np.broadcast_to(CLAY_2d[None], (NSOIL, ny, nx)).astype(np.float32).copy()

    # ── Cosby 1984 soil hydraulics ───────────────────────────────────────
    cosby = _cosby_1984(SAND_3d, CLAY_3d)

    # ── Root fraction profile ────────────────────────────────────────────
    rootF = _vege_root_init(
        ig["rootL"], ig["root_a"], ig["root_b"], interface_z_3d
    )

    # ── Vertical interpolation of soilt/soilw from source layers ─────────
    soilt_new = _vertical_interp_soil(zsoil_src.astype(np.float32), soilt_src_tgt, node_z_1d)
    soilw_raw = _vertical_interp_soil(zsoil_src.astype(np.float32), soilw_src_tgt, node_z_1d)

    # Convert volumetric→wetness (slm_vars.f90:912-916): positive values are
    # volumetric content and get divided by poro_soil; negative values are
    # already wetness.
    soilw_new = np.where(
        soilw_raw >= 0.0,
        np.minimum(1.0, soilw_raw / np.maximum(cosby["poro_soil"], 1e-30)),
        np.maximum(0.0, np.minimum(1.0, -soilw_raw)),
    ).astype(np.float32)
    # Mask ocean/land-ice cells to 0 (slm_vars.f90:918-920)
    land_mask_3d = (landmask[None] == 1) & (ig["landicemask"][None] == 0)
    soilw_new = np.where(land_mask_3d, soilw_new, 0.0).astype(np.float32)

    # Wetland (landtype==11): force saturated (slm_vars.f90:938-944)
    is_wet = (landtype == 11)
    soilw_new = np.where(
        is_wet[None], cosby["w_s_FC"], soilw_new
    ).astype(np.float32)

    # Land-ice cells: soilt capped at tfriz (slm_vars.f90:884)
    is_landice = (ig["landicemask"] == 1)
    soilt_new = np.where(
        is_landice[None], np.minimum(soilt_new, params.tfriz), soilt_new
    ).astype(np.float32)

    # ── Masks ────────────────────────────────────────────────────────────
    # prm_debug500: readseaice=F ⇒ seaicemask = 0 everywhere
    seaicemask  = np.zeros_like(landmask, dtype=np.int8)
    landicemask = ig["landicemask"].astype(np.int8)
    icemask     = ((landicemask == 1) | (seaicemask == 1)).astype(np.int8)

    # ── Snow ─────────────────────────────────────────────────────────────
    # snow_file holds snow depth in metres → convert to snow mass (kg/m²).
    # Mask off ocean and land-ice cells (slm_vars.f90:954).
    snow_mass = snow_2d * float(params.rho_snow)
    snow_mass = np.where(
        (landmask == 1) & (icemask == 0), snow_mass, 0.0
    ).astype(np.float32)

    snowt_raw = snowt_2d.astype(np.float32)
    fallback_snowt = np.minimum(params.tfriz, soilt_new[0])
    snowt_init = np.where(
        (landmask == 1) & (icemask == 0), snowt_raw, fallback_snowt
    ).astype(np.float32)

    # ── init_slm_vars: canopy / CAS from the first atmospheric level ─────
    TABS0 = np.asarray(state.TABS[0])           # (ny, nx)
    QV0   = np.asarray(state.QV[0])             # (ny, nx)
    t_cas_init   = TABS0.astype(np.float32)
    t_canop_init = TABS0.astype(np.float32)
    q_cas_init   = QV0.astype(np.float32)
    mw_init      = np.zeros((ny, nx), dtype=np.float32)
    mws_init     = np.zeros((ny, nx), dtype=np.float32)
    t_skin_init  = soilt_new[0].copy()

    ustar_init = np.full((ny, nx), 0.1, dtype=np.float32)
    tstar_init = np.zeros((ny, nx), dtype=np.float32)

    # ── Assemble SLMStatic ───────────────────────────────────────────────
    def J(a, dtype=jnp.float32):
        return jnp.asarray(a, dtype=dtype)

    static = SLMStatic(
        landmask    = J(landmask,   jnp.int8),
        seaicemask  = J(seaicemask, jnp.int8),
        landicemask = J(landicemask, jnp.int8),
        icemask     = J(icemask,    jnp.int8),
        landtype    = J(landtype.astype(np.int8), jnp.int8),
        vegetated   = J(ig["vegetated"], jnp.bool_),
        vege_YES    = J(ig["vege_YES"]),

        z0_sfc      = J(ig["z0_sfc"]),
        ztop        = J(ig["ztop"]),
        disp_hgt    = J(disp_hgt),
        BAI         = J(ig["BAI"]),
        IMPERV      = J(ig["IMPERV"]),

        alb_vis_v   = J(ig["alb_vis_v"]),
        alb_nir_v   = J(ig["alb_nir_v"]),
        alb_vis_s   = J(ig["alb_vis_s"]),
        alb_nir_s   = J(ig["alb_nir_s"]),

        IR_emis_vege = J(ig["IR_emis_vege"]),
        IR_emis_grnd = J(ig["IR_emis_grnd"]),

        khai_L      = J(ig["khai_L"]),
        phi_1       = J(ig["phi_1"]),
        phi_2       = J(ig["phi_2"]),
        precip_extinc = J(ig["precip_extinc"]),

        Rc_min      = J(ig["Rc_min"]),
        Rgl         = J(ig["Rgl"]),
        hs_rc       = J(ig["hs_rc"]),

        rootL       = J(ig["rootL"]),
        root_a      = J(ig["root_a"]),
        root_b      = J(ig["root_b"]),
        rootF       = J(rootF),

        SAND        = J(SAND_3d),
        CLAY        = J(CLAY_3d),
        s_depth     = J(s_depth_3d),
        node_z      = J(node_z_3d),
        interface_z = J(interface_z_3d),

        Bconst      = J(cosby["Bconst"]),
        m_pot_sat   = J(cosby["m_pot_sat"]),
        ks          = J(cosby["ks"]),
        poro_soil   = J(cosby["poro_soil"]),
        theta_FC    = J(cosby["theta_FC"]),
        theta_WP    = J(cosby["theta_WP"]),
        w_s_FC      = J(cosby["w_s_FC"]),
        w_s_WP      = J(cosby["w_s_WP"]),
        sst_cond    = J(cosby["sst_cond"]),
        sst_capa    = J(cosby["sst_capa"]),

        mw_mx       = J(ig["mw_mx"]),
        mws_mx      = J(np.full((ny, nx), float(params.mws_mx0), dtype=np.float32)),

        LAI         = J(ig["LAI"]),
    )

    slm_state = SLMState(
        soilt   = J(soilt_new),
        soilw   = J(soilw_new),
        t_canop = J(t_canop_init),
        t_cas   = J(t_cas_init),
        q_cas   = J(q_cas_init),
        mw      = J(mw_init),
        mws     = J(mws_init),
        snow_mass = J(snow_mass),
        snowt     = J(snowt_init),
        t_skin    = J(t_skin_init),
        ustar     = J(ustar_init),
        tstar     = J(tstar_init),
    )

    return static, slm_state
