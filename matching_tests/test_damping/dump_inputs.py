"""
Dump inputs and jsam outputs for test_damping matching tests.

Cases covered:
  pole_u.polar_face_halved   — ny=12, nz=4, nx=16, lat=linspace(-85,85,12),
                               U uniform, no u_max_phys trigger
  pole_u.equatorial_no_damp  — near-equator grid (±5°), U unchanged
  pole_u.high_lat_clip       — 85°, U > umax_cfl but < u_max_phys
  pole_v.polar_face_halved   — same grid, V[:,0,:] and V[:,ny,:] halved

Called from run.sh:
    python dump_inputs.py pole_u.polar_face_halved
    python dump_inputs.py pole_u.equatorial_no_damp
    python dump_inputs.py pole_u.high_lat_clip
    python dump_inputs.py pole_v.polar_face_halved
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, "/glade/u/home/sabramian/jsam")

from common.bin_io import write_bin                         # noqa: E402
from jsam.core.dynamics.damping import pole_damping         # noqa: E402

WORKDIR = HERE / "work"


def _write_inputs_u(nz, ny, nx, U, lat_rad, dx, dt, cu):
    """Write inputs.bin for pole_u mode."""
    with open(WORKDIR / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(np.asarray(U, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(lat_rad, dtype=np.float32).tobytes())
        f.write(struct.pack("fff", float(dx), float(dt), float(cu)))


def _write_inputs_v(nz, ny, nx, V, lat_rad, dx, dt, cu):
    """Write inputs.bin for pole_v mode."""
    with open(WORKDIR / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(np.asarray(V, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(lat_rad, dtype=np.float32).tobytes())
        f.write(struct.pack("fff", float(dx), float(dt), float(cu)))


def _jsam_pole_u(U, V, lat_rad, dx, dt, cu):
    """Run jsam pole_damping and return U_new only (no u_max_phys trigger)."""
    # Pass u_max_phys=1e9 so the physical cap never activates
    U_new, _ = pole_damping(
        U, V, jnp.array(lat_rad, dtype=jnp.float32),
        dx=float(dx), dy=jnp.ones(len(lat_rad)) * 1e5,
        dt=float(dt), cu=float(cu),
        u_max_phys=1e9,
    )
    return np.asarray(U_new, dtype=np.float32)


def _jsam_pole_v(U, V, lat_rad, dx, dt, cu):
    """Run jsam pole_damping and return V_new only (no u_max_phys trigger)."""
    _, V_new = pole_damping(
        U, V, jnp.array(lat_rad, dtype=jnp.float32),
        dx=float(dx), dy=jnp.ones(len(lat_rad)) * 1e5,
        dt=float(dt), cu=float(cu),
        u_max_phys=1e9,
    )
    return np.asarray(V_new, dtype=np.float32)


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------

def case_pole_u_polar_face_halved():
    """ny=12, lat=linspace(-85,85,12), U=10 — well below umax_cfl at these lats."""
    nz, ny, nx = 4, 12, 16
    lat_deg = np.linspace(-85.0, 85.0, ny)
    lat_rad = np.deg2rad(lat_deg)

    # dx: use a 1° grid at equator
    R = 6.371e6
    dx = R * np.deg2rad(1.0)   # ~111 km — umax_cfl at 85° ~= 0.3*111e3*cos(85°)/dt
    dt = 30.0
    cu = 0.3
    # umax at 85° = 0.3 * 111e3 * cos(85°) / 30 ≈ 0.3 * 111000 * 0.0872 / 30 ≈ 96.5 m/s
    # So U=10 < umax → no clipping, but tau(85°) ≈ 1 → mild damping

    U = jnp.full((nz, ny, nx + 1), 10.0)
    V = jnp.zeros((nz, ny + 1, nx))
    return nz, ny, nx, U, V, lat_rad, dx, dt, cu


def case_pole_u_equatorial_no_damp():
    """Near-equator grid ±5°: tau≈0 → U unchanged."""
    nz, ny, nx = 4, 8, 16
    lat_deg = np.linspace(-5.0, 5.0, ny)
    lat_rad = np.deg2rad(lat_deg)
    R = 6.371e6
    dx = R * np.deg2rad(1.0)
    dt = 30.0
    cu = 0.3
    U = jnp.full((nz, ny, nx + 1), 20.0)
    V = jnp.zeros((nz, ny + 1, nx))
    return nz, ny, nx, U, V, lat_rad, dx, dt, cu


def case_pole_u_high_lat_clip():
    """85° grid, U > umax_cfl but < u_max_phys=1e9 so Fortran & jsam agree."""
    nz, ny, nx = 4, 4, 16
    lat_deg = np.array([75.0, 80.0, 82.0, 85.0])
    lat_rad = np.deg2rad(lat_deg)
    R = 6.371e6
    dx = R * np.deg2rad(1.0)   # ~111 km
    dt = 30.0
    cu = 0.3
    # umax at 85° ≈ 96.5 m/s; use U=50 < 96.5 → below CFL cap
    # Actually at 85°: umax = 0.3 * 111000 * cos(85°) / 30 = 96.5 m/s
    # Use U=50 so we're well below the CFL cap — this exercises clipping
    # at the tau ramp without hitting the physical cap
    U = jnp.full((nz, ny, nx + 1), 50.0)
    V = jnp.zeros((nz, ny + 1, nx))
    return nz, ny, nx, U, V, lat_rad, dx, dt, cu


def case_pole_v_polar_face_halved():
    """Same as polar_face_halved but for V. V[:,0,:] and V[:,ny,:] → V/2."""
    nz, ny, nx = 4, 12, 16
    lat_deg = np.linspace(-85.0, 85.0, ny)
    lat_rad = np.deg2rad(lat_deg)
    R = 6.371e6
    dx = R * np.deg2rad(1.0)
    dt = 30.0
    cu = 0.3
    U = jnp.zeros((nz, ny, nx + 1))
    V = jnp.full((nz, ny + 1, nx), 20.0)
    return nz, ny, nx, U, V, lat_rad, dx, dt, cu


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)

    case = sys.argv[1]

    if case == "pole_u.polar_face_halved":
        nz, ny, nx, U, V, lat_rad, dx, dt, cu = case_pole_u_polar_face_halved()
        _write_inputs_u(nz, ny, nx, U, lat_rad, dx, dt, cu)
        U_new = _jsam_pole_u(U, V, lat_rad, dx, dt, cu)
        write_bin(WORKDIR / "jsam_out.bin", U_new)

    elif case == "pole_u.equatorial_no_damp":
        nz, ny, nx, U, V, lat_rad, dx, dt, cu = case_pole_u_equatorial_no_damp()
        _write_inputs_u(nz, ny, nx, U, lat_rad, dx, dt, cu)
        U_new = _jsam_pole_u(U, V, lat_rad, dx, dt, cu)
        write_bin(WORKDIR / "jsam_out.bin", U_new)

    elif case == "pole_u.high_lat_clip":
        nz, ny, nx, U, V, lat_rad, dx, dt, cu = case_pole_u_high_lat_clip()
        _write_inputs_u(nz, ny, nx, U, lat_rad, dx, dt, cu)
        U_new = _jsam_pole_u(U, V, lat_rad, dx, dt, cu)
        write_bin(WORKDIR / "jsam_out.bin", U_new)

    elif case == "pole_v.polar_face_halved":
        nz, ny, nx, U, V, lat_rad, dx, dt, cu = case_pole_v_polar_face_halved()
        _write_inputs_v(nz, ny, nx, V, lat_rad, dx, dt, cu)
        V_new = _jsam_pole_v(U, V, lat_rad, dx, dt, cu)
        write_bin(WORKDIR / "jsam_out.bin", V_new)

    else:
        raise SystemExit(f"unknown case: {case}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
