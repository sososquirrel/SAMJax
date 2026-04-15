! Fortran shadow of jsam/tests/unit/test_operators.py
!
! Array layout: Python writes C row-major.
!   Python (nz, ny, nx+1) → Fortran U(nx+1, ny, nz)
!   Python (nz, ny+1, nx) → Fortran V(nx, ny+1, nz)
!   Python (nz, ny, nx)   → Fortran phi(nx, ny, nz)
! U(i,j,k) in Fortran == U[k-1,j-1,i-1] in Python.
!
! Three cases:
!   1. divergence_zero_v_const — U=0, V=const → div = 0
!   2. divergence_linear_u    — U[i]=i, V=0  → div = 1/dx per row
!   3. laplacian_const         — phi=42       → lap = 0

program operators_driver
  implicit none
  character(len=64) :: case_name
  call get_command_argument(1, case_name)

  select case (trim(case_name))
  case ('divergence_zero_v_const');  call run_divergence()
  case ('divergence_linear_u');      call run_divergence()
  case ('laplacian_const');          call run_laplacian()
  case default
    write(*,*) 'unknown case: ', trim(case_name); stop 2
  end select

contains

  ! -----------------------------------------------------------------------
  ! Read shared inputs (reversed axes)
  ! -----------------------------------------------------------------------
  subroutine read_inputs(nz, ny, nx, U, V, phi, dx, dy)
    integer, intent(out) :: nz, ny, nx
    real, allocatable, intent(out) :: U(:,:,:), V(:,:,:), phi(:,:,:)
    real, allocatable, intent(out) :: dx(:), dy(:)
    integer :: u_in
    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    ! Reversed axes: U(nx+1, ny, nz), V(nx, ny+1, nz), phi(nx, ny, nz)
    allocate(U(nx+1, ny, nz), V(nx, ny+1, nz), phi(nx, ny, nz))
    allocate(dx(ny), dy(ny))
    read(u_in) U
    read(u_in) V
    read(u_in) phi
    read(u_in) dx
    read(u_in) dy
    close(u_in)
  end subroutine read_inputs

  ! -----------------------------------------------------------------------
  ! Write result as flat float32 in standard format
  ! -----------------------------------------------------------------------
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
  ! Divergence: div = (U(i+1,j,k)-U(i,j,k))/dx(j) + (V(i,j+1,k)-V(i,j,k))/dy(j)
  ! Result shape: div(nx, ny, nz)  → matches Python (nz, ny, nx) C-order
  ! -----------------------------------------------------------------------
  subroutine run_divergence()
    integer :: nz, ny, nx, i, j, k
    real, allocatable :: U(:,:,:), V(:,:,:), phi(:,:,:), dx(:), dy(:)
    real, allocatable :: div(:,:,:), out(:)

    call read_inputs(nz, ny, nx, U, V, phi, dx, dy)
    allocate(div(nx, ny, nz))

    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          div(i, j, k) = (U(i+1, j, k) - U(i, j, k)) / dx(j) &
                       + (V(i, j+1, k) - V(i, j, k)) / dy(j)
        end do
      end do
    end do

    allocate(out(nx*ny*nz))
    out = reshape(div, [nx*ny*nz])
    call write_result(out, nx*ny*nz)
    deallocate(U, V, phi, dx, dy, div, out)
  end subroutine run_divergence

  ! -----------------------------------------------------------------------
  ! Laplacian via gradient applied twice (centred differences, periodic x)
  ! -----------------------------------------------------------------------
  subroutine compute_gradient(phi, dx, dy, nx, ny, nz, dphi_dx, dphi_dy)
    integer, intent(in) :: nx, ny, nz
    real, intent(in)  :: phi(nx, ny, nz), dx(ny), dy(ny)
    real, intent(out) :: dphi_dx(nx, ny, nz), dphi_dy(nx, ny, nz)
    integer :: i, j, k, ip1, im1
    real :: dy_v_int(ny-1), dy_v_north(ny), dy_v_south(ny), denom_y
    integer :: jj

    do jj = 1, ny-1
      dy_v_int(jj) = 0.5 * (dy(jj) + dy(jj+1))
    end do
    ! edge-pad: north uses interior value at boundary, south same
    do jj = 1, ny-1
      dy_v_north(jj) = dy_v_int(jj)
    end do
    dy_v_north(ny) = dy_v_int(ny-1)
    dy_v_south(1)  = dy_v_int(1)
    do jj = 2, ny
      dy_v_south(jj) = dy_v_int(jj-1)
    end do

    do k = 1, nz
      do j = 1, ny
        denom_y = dy_v_north(j) + dy_v_south(j)
        do i = 1, nx
          ! periodic x
          ip1 = mod(i, nx) + 1
          im1 = mod(i - 2 + nx, nx) + 1
          dphi_dx(i, j, k) = (phi(ip1, j, k) - phi(im1, j, k)) / (2.0 * dx(j))
          ! y: edge-padded at boundaries
          if (j == 1) then
            dphi_dy(i, j, k) = (phi(i, j+1, k) - phi(i, j, k)) / denom_y
          else if (j == ny) then
            dphi_dy(i, j, k) = (phi(i, j, k) - phi(i, j-1, k)) / denom_y
          else
            dphi_dy(i, j, k) = (phi(i, j+1, k) - phi(i, j-1, k)) / denom_y
          end if
        end do
      end do
    end do
  end subroutine compute_gradient

  subroutine run_laplacian()
    integer :: nz, ny, nx
    real, allocatable :: U(:,:,:), V(:,:,:), phi(:,:,:), dx(:), dy(:)
    real, allocatable :: dphi_dx(:,:,:), dphi_dy(:,:,:)
    real, allocatable :: d2dx2(:,:,:), tmp(:,:,:), d2dy2(:,:,:)
    real, allocatable :: lap(:,:,:), out(:)

    call read_inputs(nz, ny, nx, U, V, phi, dx, dy)
    allocate(dphi_dx(nx,ny,nz), dphi_dy(nx,ny,nz))
    allocate(d2dx2(nx,ny,nz), tmp(nx,ny,nz), d2dy2(nx,ny,nz))
    allocate(lap(nx,ny,nz), out(nx*ny*nz))

    call compute_gradient(phi,     dx, dy, nx, ny, nz, dphi_dx, dphi_dy)
    call compute_gradient(dphi_dx, dx, dy, nx, ny, nz, d2dx2,   tmp)
    call compute_gradient(dphi_dy, dx, dy, nx, ny, nz, tmp,     d2dy2)
    lap = d2dx2 + d2dy2

    out = reshape(lap, [nx*ny*nz])
    call write_result(out, nx*ny*nz)
    deallocate(U, V, phi, dx, dy, dphi_dx, dphi_dy, d2dx2, tmp, d2dy2, lap, out)
  end subroutine run_laplacian

end program operators_driver
