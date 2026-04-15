! test_surface_qs — surface saturation specific humidity for the bulk
! flux scheme.  Verbatim copy of the gSAM esatw / qsatw snippets from
! SRC/sat.f90:8-41 plus the salinity reduction factor 0.98 used in
! oceflx.f90 (qs_sfc = 0.98 * qsatw(SST, presi_mb)).
!
! Inputs:
!   int32  ny, nx
!   float32 sst(ny, nx)         (K)
!   float32 presi(ny, nx)       (Pa)
!   float32 salt_factor          (-)  default 0.98
!
! Output:
!   float32 qs_sfc(ny, nx) flattened C-order
program surface_qs_driver
  implicit none
  integer :: ny, nx, j, i, u_in, u_out, n
  real, allocatable :: sst(:,:), presi(:,:), out(:)
  real :: salt_factor

  open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
  read(u_in) ny, nx
  allocate(sst(ny, nx), presi(ny, nx), out(ny*nx))
  read(u_in) sst
  read(u_in) presi
  read(u_in) salt_factor
  close(u_in)

  n = 0
  do j = 1, ny
    do i = 1, nx
      n = n + 1
      out(n) = salt_factor * qsatw_local(sst(j, i), presi(j, i) / 100.0)
    end do
  end do

  open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) int(ny*nx, 4)
  write(u_out) out
  close(u_out)

contains

  pure real function esatw_local(t)
    real, intent(in) :: t
    real, parameter :: e0 = 6.1121, a1 = 17.502, T0 = 273.16, T1 = 32.19
    esatw_local = e0 * exp( a1 * (t - T0) / (t - T1) )
  end function esatw_local

  pure real function qsatw_local(t, p)
    real, intent(in) :: t, p
    real :: esat
    esat = esatw_local(t)
    qsatw_local = 0.622 * esat / max(esat, p - esat)
  end function qsatw_local

end program surface_qs_driver
