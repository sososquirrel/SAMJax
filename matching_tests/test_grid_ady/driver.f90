! test_grid_ady — verify LatLonGrid.ady / LatLonGrid.dy_ref against the
! gSAM setgrid.f90 adyy formula on a synthetic non-uniform latv array.
!
! gSAM SRC/setgrid.f90 lines 238-241 (dolatlon branch):
!
!   dy = y_gl(ny_gl/2 + 1) - y_gl(ny_gl/2)         ! mid-latitude scalar
!   do j = 1, ny_gl
!     adyy(j) = (yv_gl(j+1) - yv_gl(j)) / dy
!   end do
!
! where y_gl(j) = 0.5*(yv_gl(j) + yv_gl(j+1)).  Equivalently
!   dy_ref = 0.5*(dy_per_row(ny/2-1) + dy_per_row(ny/2))     (Python 0-index)
! with dy_per_row(j) = yv_gl(j+1) - yv_gl(j).  The test feeds a synthetic
! latv (8 mass rows → 9 v-faces, mimicking the middle of lat_720_dyvar)
! and requires jsam to reproduce this exactly.
!
! Inputs (inputs.bin):
!   int32  ny                                ! number of mass rows
!   float32 yv_meters(ny+1)                  ! v-face positions in metres
!
! Output (fortran_out.bin):
!   float32 vector of length 1 + ny:
!     dy_ref           (1)
!     ady(1..ny)       (ny)
program ady_driver
  implicit none
  integer :: ny, j, u_in, u_out
  real, allocatable :: yv(:), y(:), dy_per_row(:), ady(:), out(:)
  real :: dy_ref

  open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
  read(u_in) ny
  allocate(yv(ny+1), y(ny), dy_per_row(ny), ady(ny))
  read(u_in) yv
  close(u_in)

  do j = 1, ny
    y(j)          = 0.5 * (yv(j) + yv(j+1))
    dy_per_row(j) = yv(j+1) - yv(j)
  end do

  ! gSAM dy_ref = y(ny/2+1) - y(ny/2)  (Fortran 1-based; ny even)
  dy_ref = y(ny/2 + 1) - y(ny/2)

  do j = 1, ny
    ady(j) = dy_per_row(j) / dy_ref
  end do

  allocate(out(1 + ny))
  out(1)        = dy_ref
  out(2:1+ny)   = ady(1:ny)

  open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) int(size(out), 4)
  write(u_out) out
  close(u_out)
end program ady_driver
