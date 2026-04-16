"""ERA5 unit-conversion tests - dump inputs and jsam outputs."""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.bin_io import write_bin
from jsam.io.era5 import (_G, _RD, omega_to_w, interp_pressure,
    stagger_u, stagger_v, stagger_w, _gsam_reference_column)

def _write_raw_f64(path, *arrays):
    with open(path, "wb") as f:
        for a in arrays:
            f.write(np.asarray(a, dtype=np.float64).ravel().tobytes())

def run_z_from_Z(wd):
    n = 200
    Z = np.linspace(0.0, 300000.0, n, dtype=np.float32)
    with open(wd/"inputs.bin","wb") as f:
        f.write(struct.pack("i",n)); f.write(Z.tobytes())
    write_bin(wd/"jsam_out.bin", Z / np.float32(_G))

def run_omega_to_w(wd):
    nlev = 37
    p = np.array([100,200,300,500,700,1000,2000,3000,5000,7000,10000,12500,
        15000,17500,20000,22500,25000,30000,35000,40000,45000,50000,55000,
        60000,65000,70000,75000,77500,80000,82500,85000,87500,90000,92500,
        95000,97500,100000], dtype=np.float32)
    T = np.linspace(200.0,300.0,nlev,dtype=np.float32)
    om = np.linspace(-0.5,0.5,nlev,dtype=np.float32)
    w = omega_to_w(om.astype(np.float64),T.astype(np.float64),
                   p.astype(np.float64)).astype(np.float32)
    with open(wd/"inputs.bin","wb") as f:
        f.write(struct.pack("i",nlev)); f.write(p.tobytes())
        f.write(T.tobytes()); f.write(om.tobytes())
    write_bin(wd/"jsam_out.bin", w)

def run_interp_pres(wd):
    ns,nt = 10,15
    ps = np.linspace(100.0,100000.0,ns,dtype=np.float32)
    pt = np.linspace(50.0,110000.0,nt,dtype=np.float32)
    fld = np.linspace(200.0,300.0,ns,dtype=np.float32)
    out = interp_pressure(fld.astype(np.float64),ps.astype(np.float64),
                          pt.astype(np.float64)).astype(np.float32)
    with open(wd/"inputs.bin","wb") as f:
        f.write(struct.pack("iii",ns,nt,0))
        f.write(ps.tobytes()); f.write(pt.tobytes()); f.write(fld.tobytes())
    write_bin(wd/"jsam_out.bin", out)

def run_stagger_u(wd):
    nz,ny,nx = 3,4,6; np.random.seed(7)
    u = np.random.randn(nz,ny,nx).astype(np.float32)
    with open(wd/"inputs.bin","wb") as f:
        f.write(struct.pack("iii",nz,ny,nx)); f.write(u.tobytes())
    write_bin(wd/"jsam_out.bin", stagger_u(u.astype(np.float64)).astype(np.float32))

def run_stagger_v(wd):
    nz,ny,nx = 3,4,6; np.random.seed(8)
    v = np.random.randn(nz,ny,nx).astype(np.float32)
    with open(wd/"inputs.bin","wb") as f:
        f.write(struct.pack("iii",nz,ny,nx)); f.write(v.tobytes())
    write_bin(wd/"jsam_out.bin", stagger_v(v.astype(np.float64)).astype(np.float32))

def run_stagger_w(wd):
    nz,ny,nx = 3,4,6; np.random.seed(9)
    w = np.random.randn(nz,ny,nx).astype(np.float32)
    with open(wd/"inputs.bin","wb") as f:
        f.write(struct.pack("iii",nz,ny,nx)); f.write(w.tobytes())
    write_bin(wd/"jsam_out.bin", stagger_w(w.astype(np.float64)).astype(np.float32))

def run_ref_column(wd):
    nz = 40
    zi = np.linspace(0.0,25000.0,nz+1); z = 0.5*(zi[:-1]+zi[1:])
    tabs0 = np.where(z<12000.0, 288.0-6.5e-3*z, 288.0-6.5e-3*12000.0)
    pres0 = 1013.25; ps = pres0*np.exp(-z/8500.0)
    ref = _gsam_reference_column(z=z,zi=zi,tabs0=tabs0,pres0=pres0,pres_seed=ps)
    _write_raw_f64(wd/"inputs.bin", np.array([nz],dtype=np.float64),
                   z, zi, tabs0, np.array([pres0],dtype=np.float64), ps)
    write_bin(wd/"jsam_out.bin",
              np.concatenate([ref["rho"].astype(np.float32),
                              ref["presi"].astype(np.float32)]))

def main():
    wd = HERE/"work"; wd.mkdir(parents=True, exist_ok=True)
    d = {"z_from_Z":run_z_from_Z, "omega_to_w":run_omega_to_w,
         "interp_pres":run_interp_pres, "stagger_u":run_stagger_u,
         "stagger_v":run_stagger_v, "stagger_w":run_stagger_w,
         "ref_column":run_ref_column}
    mode = sys.argv[1]
    if mode not in d: raise SystemExit(f"unknown mode: {mode}")
    d[mode](wd); return 0

if __name__ == "__main__":
    sys.exit(main())
