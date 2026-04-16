! test_era5 -- Fortran shadow of jsam.io.era5 pure conversion functions.
! Modes: z_from_Z, omega_to_w, interp_pres, stagger_u, stagger_v, stagger_w, ref_column
program era5_driver
  implicit none
  character(len=64) :: mode
  call get_command_argument(1, mode)
  select case (trim(mode))
  case ('z_from_Z');    call run_z_from_Z()
  case ('omega_to_w');  call run_omega_to_w()
  case ('interp_pres'); call run_interp_pres()
  case ('stagger_u');   call run_stagger_u()
  case ('stagger_v');   call run_stagger_v()
  case ('stagger_w');   call run_stagger_w()
  case ('ref_column');  call run_ref_column()
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select
contains

  subroutine run_z_from_Z()
    real, parameter :: ggr = 9.79764
    integer :: n, k, fin, fout
    real, allocatable :: Z_geopot(:), z_out(:)
    open(newunit=fin, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(fin) n
    allocate(Z_geopot(n), z_out(n))
    read(fin) Z_geopot
    close(fin)
    do k = 1, n
      z_out(k) = Z_geopot(k) / ggr
    end do
    open(newunit=fout, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(fout) 1_4
    write(fout) int(n, 4)
    write(fout) z_out
    close(fout)
  end subroutine

  subroutine run_omega_to_w()
    real, parameter :: ggr = 9.79764, Rd = 287.04
    integer :: nlev, k, fin, fout
    real, allocatable :: p_pa(:), T(:), omega(:), w_out(:)
    open(newunit=fin, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(fin) nlev
    allocate(p_pa(nlev), T(nlev), omega(nlev), w_out(nlev))
    read(fin) p_pa; read(fin) T; read(fin) omega
    close(fin)
    do k = 1, nlev
      w_out(k) = -omega(k) * Rd * T(k) / (p_pa(k) * ggr)
    end do
    open(newunit=fout, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(fout) 1_4; write(fout) int(nlev, 4); write(fout) w_out
    close(fout)
  end subroutine

  subroutine run_interp_pres()
    integer :: nz_src, nz_tgt, dummy, k, fin, fout, idx
    real, allocatable :: p_src(:), p_tgt(:), field_src(:), field_out(:)
    real :: p_clamped, frac
    open(newunit=fin, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(fin) nz_src, nz_tgt, dummy
    allocate(p_src(nz_src), p_tgt(nz_tgt), field_src(nz_src), field_out(nz_tgt))
    read(fin) p_src; read(fin) p_tgt; read(fin) field_src
    close(fin)
    do k = 1, nz_tgt
      p_clamped = min(max(p_tgt(k), p_src(1)), p_src(nz_src))
      idx = 1
      do while (idx < nz_src .and. p_src(idx+1) < p_clamped)
        idx = idx + 1
      end do
      if (idx >= nz_src) then
        field_out(k) = field_src(nz_src)
      else
        frac = (p_clamped - p_src(idx)) / (p_src(idx+1) - p_src(idx))
        field_out(k) = field_src(idx) + frac * (field_src(idx+1) - field_src(idx))
      end if
    end do
    open(newunit=fout, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(fout) 1_4; write(fout) int(nz_tgt, 4); write(fout) field_out
    close(fout)
  end subroutine

  subroutine run_stagger_u()
    integer :: nz, ny, nx, iz, iy, ix, fin, fout, ntot
    real, allocatable :: u(:,:,:), Usg(:,:,:)
    open(newunit=fin, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(fin) nz, ny, nx
    allocate(u(nx, ny, nz), Usg(nx+1, ny, nz))
    read(fin) u
    close(fin)
    do iz = 1, nz
      do iy = 1, ny
        do ix = 1, nx
          if (ix < nx) then
            Usg(ix, iy, iz) = 0.5 * (u(ix, iy, iz) + u(ix+1, iy, iz))
          else
            Usg(ix, iy, iz) = 0.5 * (u(ix, iy, iz) + u(1, iy, iz))
          end if
        end do
        Usg(nx+1, iy, iz) = Usg(1, iy, iz)
      end do
    end do
    ntot = (nx+1) * ny * nz
    open(newunit=fout, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(fout) 1_4; write(fout) int(ntot, 4); write(fout) Usg
    close(fout)
  end subroutine

  subroutine run_stagger_v()
    integer :: nz, ny, nx, iz, iy, ix, fin, fout, ntot
    real, allocatable :: v(:,:,:), Vsg(:,:,:)
    open(newunit=fin, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(fin) nz, ny, nx
    allocate(v(nx, ny, nz), Vsg(nx, ny+1, nz))
    read(fin) v
    close(fin)
    Vsg = 0.0
    do iz = 1, nz
      do iy = 2, ny
        do ix = 1, nx
          Vsg(ix, iy, iz) = 0.5 * (v(ix, iy-1, iz) + v(ix, iy, iz))
        end do
      end do
    end do
    ntot = nx * (ny+1) * nz
    open(newunit=fout, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(fout) 1_4; write(fout) int(ntot, 4); write(fout) Vsg
    close(fout)
  end subroutine

  subroutine run_stagger_w()
    integer :: nz, ny, nx, iz, iy, ix, fin, fout, ntot
    real, allocatable :: w(:,:,:), Wsg(:,:,:)
    open(newunit=fin, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(fin) nz, ny, nx
    allocate(w(nx, ny, nz), Wsg(nx, ny, nz+1))
    read(fin) w
    close(fin)
    Wsg = 0.0
    do iz = 2, nz
      do iy = 1, ny
        do ix = 1, nx
          Wsg(ix, iy, iz) = 0.5 * (w(ix, iy, iz-1) + w(ix, iy, iz))
        end do
      end do
    end do
    ntot = nx * ny * (nz+1)
    open(newunit=fout, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(fout) 1_4; write(fout) int(ntot, 4); write(fout) Wsg
    close(fout)
  end subroutine

  subroutine run_ref_column()
    double precision, parameter :: RGAS=287.04d0, CP=1004.64d0, GGR=9.79764d0
    integer :: nz, k, sweep, fin, fout, ntot
    double precision :: nz_dbl, pres0, prespot_k
    double precision, allocatable :: z(:), zi(:), tabs0(:), pres_seed(:)
    double precision, allocatable :: pres(:), presi(:), presr(:), t0(:), rho(:)
    real, allocatable :: out(:)
    open(newunit=fin, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(fin) nz_dbl
    nz = int(nz_dbl)
    allocate(z(nz), zi(nz+1), tabs0(nz), pres_seed(nz))
    read(fin) z; read(fin) zi; read(fin) tabs0; read(fin) pres0; read(fin) pres_seed
    close(fin)
    allocate(pres(nz), presi(nz+1), presr(nz+1), t0(nz), rho(nz))
    pres = pres_seed
    do sweep = 1, 2
      presr(1) = (pres0 / 1000.0d0) ** (RGAS / CP)
      presi(1) = pres0
      do k = 1, nz
        prespot_k = (1000.0d0 / pres(k)) ** (RGAS / CP)
        t0(k) = tabs0(k) * prespot_k
        presr(k+1) = presr(k) - GGR / CP / t0(k) * (zi(k+1) - zi(k))
        presi(k+1) = 1000.0d0 * presr(k+1) ** (CP / RGAS)
        pres(k) = dexp(dlog(presi(k)) + dlog(presi(k+1)/presi(k)) * (z(k)-zi(k))/(zi(k+1)-zi(k)))
      end do
    end do
    do k = 1, nz
      rho(k) = (presi(k) - presi(k+1)) / (zi(k+1) - zi(k)) / GGR * 100.0d0
    end do
    ntot = nz + (nz + 1)
    allocate(out(ntot))
    do k = 1, nz
      out(k) = real(rho(k))
    end do
    do k = 1, nz+1
      out(nz + k) = real(presi(k))
    end do
    open(newunit=fout, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(fout) 1_4; write(fout) int(ntot, 4); write(fout) out
    close(fout)
  end subroutine

end program era5_driver
