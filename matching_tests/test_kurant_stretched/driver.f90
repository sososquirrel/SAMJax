! test_kurant_stretched — kurant CFL on a non-uniform vertical grid.
!
! Verbatim port of gSAM SRC/kurant.f90 lines 19-47, restricted to the
! advective Courant computation (no SGS, no MPI):
!
!     do k = 1, nzm
!       idz = dtn / (dz * adzw(k))                   ! gSAM line 20
!       do j = 1, ny
!         idx = imu(j) * dtn / dx                    ! line 22  (here cos(lat) form)
!         idy = YES3D * dtn / (dy * ady(j))          ! line 23
!         do i = 1, nx
!           cflz1 = abs(w(i,j,k)) * idz
!           cflh1 = sqrt((u*idx)**2 + (v*idy)**2)    ! line 31
!           cfll  = sqrt(cflh1**2 + cflz1**2)        ! line 37
!           cfl   = max(cfl, cfll)
!         end do
!       end do
!     end do
!
! Inputs (cell-centered absolute velocity magnitudes — the python
! harness already does the staggered→mass max reduction):
!   int32  nz, ny, nx
!   float32 U_abs(nz,ny,nx)
!   float32 V_abs(nz,ny,nx)
!   float32 W_abs(nz,ny,nx)
!   float32 dx_ref                              (= equatorial dx, m)
!   float32 dy_ref                              (= scalar reference dy, m)
!   float32 dz_ref                              (= scalar reference dz, m)
!   float32 cos_lat(ny)
!   float32 ady(ny)                             (per-row meridional stretch)
!   float32 adzw(nz+1)                          (per-level vertical stretch)
!   float32 dt
!
! Output:
!   single float32 = global maximum cfl
program kurant_stretched_driver
  implicit none
  integer :: nz, ny, nx, i, j, k, u_in, u_out
  real, allocatable :: U_abs(:,:,:), V_abs(:,:,:), W_abs(:,:,:)
  real, allocatable :: cos_lat(:), ady(:), adzw(:)
  real :: dx, dy, dz, dt
  real :: idx, idy, idz, cflz1, cflh1, cfll, cfl_max

  open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
  read(u_in) nz, ny, nx
  allocate(U_abs(nz, ny, nx), V_abs(nz, ny, nx), W_abs(nz, ny, nx))
  allocate(cos_lat(ny), ady(ny), adzw(nz+1))
  read(u_in) U_abs
  read(u_in) V_abs
  read(u_in) W_abs
  read(u_in) dx, dy, dz
  read(u_in) cos_lat
  read(u_in) ady
  read(u_in) adzw
  read(u_in) dt
  close(u_in)

  cfl_max = 0.0
  do k = 1, nz
    idz = dt / (dz * adzw(k))                  ! gSAM kurant.f90:20
    do j = 1, ny
      idx = dt / (dx * max(cos_lat(j), 1e-6))
      idy = dt / (dy * ady(j))                 ! gSAM kurant.f90:23
      do i = 1, nx
        cflz1 = abs(W_abs(k, j, i)) * idz
        cflh1 = sqrt((U_abs(k, j, i)*idx)**2 + (V_abs(k, j, i)*idy)**2)
        cfll  = sqrt(cflh1**2 + cflz1**2)
        if (cfll > cfl_max) cfl_max = cfll
      end do
    end do
  end do

  open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) 1_4
  write(u_out) cfl_max
  close(u_out)
end program kurant_stretched_driver
