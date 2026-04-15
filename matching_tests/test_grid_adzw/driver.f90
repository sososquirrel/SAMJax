! test_grid_adzw — verify jsam build_metric() yields the same adz / adzw /
! dz_ref as gSAM SRC/setgrid.f90 lines 99-108 on a synthetic stretched
! vertical grid.
!
! Snippet ported verbatim from setgrid.f90 (new-style "grd specifies zi"
! branch), nzm = nz - 1:
!
!     dz = zi(2) - zi(1)
!     do k = 1, nzm
!       adz(k) = (zi(k+1) - zi(k)) / dz
!       z(k)   = 0.5*(zi(k+1) + zi(k))
!     end do
!     do k = 2, nzm
!       adzw(k) = (z(k) - z(k-1)) / dz
!     end do
!     adzw(1)  = 1.0
!     adzw(nz) = adzw(nzm)
!
! Inputs (inputs.bin):
!   int32  nz                         (= number of full-cell levels nzm; the
!                                       interface array has nz+1 entries)
!   float32 zi(nz+1)                  (interface heights, m)
!
! Output (fortran_out.bin):
!   single contiguous float32 vector of length 2*nz + (nz+1) + 1:
!     dz_ref           (1)
!     adz(1..nz)       (nz)
!     adzw(1..nz+1)    (nz+1)        — gSAM convention with adzw(nz+1)=adzw(nz)
program adzw_driver
  implicit none
  integer :: nz, nzm, k, u_in, u_out
  real, allocatable :: zi(:), z(:), adz(:), adzw(:), out(:)
  real :: dz

  open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
  read(u_in) nz                              ! number of full cells (nzm in gSAM)
  allocate(zi(nz+1), z(nz), adz(nz), adzw(nz+1))
  read(u_in) zi
  close(u_in)
  nzm = nz

  dz = zi(2) - zi(1)
  do k = 1, nzm
    adz(k) = (zi(k+1) - zi(k)) / dz
    z(k)   = 0.5 * (zi(k+1) + zi(k))
  end do
  do k = 2, nzm
    adzw(k) = (z(k) - z(k-1)) / dz
  end do
  adzw(1)    = 1.0
  adzw(nzm+1)= adzw(nzm)

  allocate(out(1 + nzm + (nzm + 1)))
  out(1) = dz
  out(2:1+nzm) = adz(1:nzm)
  out(2+nzm:1+nzm+nzm+1) = adzw(1:nzm+1)

  open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) int(size(out), 4)
  write(u_out) out
  close(u_out)
end program adzw_driver
