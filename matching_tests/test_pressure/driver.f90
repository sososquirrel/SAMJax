! Fortran shadow of jsam/core/dynamics/pressure.py — press_rhs
!
! Cases (residual tests — compare RHS, not pressure solution):
!   press_rhs_zero_velocity    — U=V=W=0 → RHS=0
!   press_rhs_constant_U       — uniform U → RHS=0 (zero divergence)
!   press_gradient_zero_p      — p=0 → U,V,W unchanged → RHS=0 applied to zero vel
!   press_gradient_constant_p  — p=const → no gradient → U,V,W unchanged
!
! inputs.bin layout:
!   int32  nz, ny, nx
!   float32 U(nz, ny, nx+1)
!   float32 V(nz, ny+1, nx)
!   float32 W(nz+1, ny, nx)
!   float32 rho(nz)
!   float32 rhow(nz+1)
!   float32 dz(nz)
!   float32 dx                  scalar (dx_lon, equatorial)
!   float32 dy(ny)              per-row
!   float32 imu(ny)             1/cos(lat_j)
!   float32 cos_v(ny+1)         cos at v-faces
!   float32 cos_lat(ny)
!   float32 dt                  scalar
!
! Output: fortran_out.bin  =  RHS(nz,ny,nx)  as float32
program pressure_driver
  implicit none
  character(len=64) :: case_name
  call get_command_argument(1, case_name)

  select case (trim(case_name))
  case ('press_rhs_zero_velocity');   call run_press_rhs()
  case ('press_rhs_constant_U');      call run_press_rhs()
  case ('press_gradient_zero_p');     call run_press_rhs()
  case ('press_gradient_constant_p'); call run_press_rhs()
  case default
    write(*,*) 'unknown case: ', trim(case_name); stop 2
  end select

contains

  subroutine write_result(result, n)
    integer, intent(in) :: n
    real, intent(in)    :: result(n)
    integer :: u_out
    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(n, 4)
    write(u_out) result
    close(u_out)
  end subroutine write_result

  ! -----------------------------------------------------------------------
  ! press_rhs: anelastic spherical divergence / dt
  !
  !   div_u(i,j,k) = imu(j) * (U(k,j,i+1) - U(k,j,i)) / dx
  !   div_v(i,j,k) = (cos_v(j+1)*V(k,j+1,i) - cos_v(j)*V(k,j,i)) / (dy(j)*cos_lat(j))
  !   div_w(i,j,k) = (rhow(k+1)*W(k+1,j,i) - rhow(k)*W(k,j,i)) / (rho(k)*dz(k))
  !   RHS = (div_u + div_v + div_w) / dt
  ! -----------------------------------------------------------------------
  subroutine run_press_rhs()
    integer :: nz, ny, nx, k, j, i
    real, allocatable :: U(:,:,:), V(:,:,:), W(:,:,:)
    real, allocatable :: rho(:), rhow(:), dz(:), dy(:)
    real, allocatable :: imu(:), cos_v(:), cos_lat(:)
    real, allocatable :: rhs(:,:,:)
    real :: dx, dt
    real :: div_u, div_v, div_w
    integer :: u_in

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    allocate(U(nz, ny, nx+1), V(nz, ny+1, nx), W(nz+1, ny, nx))
    allocate(rho(nz), rhow(nz+1), dz(nz), dy(ny))
    allocate(imu(ny), cos_v(ny+1), cos_lat(ny))
    read(u_in) U
    read(u_in) V
    read(u_in) W
    read(u_in) rho
    read(u_in) rhow
    read(u_in) dz
    read(u_in) dx
    read(u_in) dy
    read(u_in) imu
    read(u_in) cos_v
    read(u_in) cos_lat
    read(u_in) dt
    close(u_in)

    allocate(rhs(nz, ny, nx))

    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          ! Spherical zonal divergence
          div_u = imu(j) * (U(k, j, i+1) - U(k, j, i)) / dx

          ! Spherical meridional divergence
          div_v = (cos_v(j+1) * V(k, j+1, i) - cos_v(j) * V(k, j, i)) &
                / (dy(j) * cos_lat(j))

          ! Anelastic vertical divergence
          div_w = (rhow(k+1) * W(k+1, j, i) - rhow(k) * W(k, j, i)) &
                / (rho(k) * dz(k))

          rhs(k, j, i) = (div_u + div_v + div_w) / dt
        end do
      end do
    end do

    call write_result(reshape(rhs, [nz*ny*nx]), nz*ny*nx)
    deallocate(U, V, W, rho, rhow, dz, dy, imu, cos_v, cos_lat, rhs)
  end subroutine run_press_rhs

end program pressure_driver
