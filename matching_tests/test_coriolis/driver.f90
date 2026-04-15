! Fortran shadow of jsam/tests/unit/test_coriolis.py
!
! Implements the gSAM coriolis.f90 dolatlon=.true. branch.
!
! Array layout note: Python writes C row-major.  For shape (nz, ny, nx+1)
! the bytes are [k=0,j=0,i=0],[k=0,j=0,i=1],...  Fortran reads column-major
! (first index varies fastest).  Arrays are declared with REVERSED axes:
!   Python (nz, ny, nx+1) → Fortran U(nx+1, ny, nz)
!   Python (nz, ny+1, nx) → Fortran V(nx, ny+1, nz)
! so U(i,j,k) in Fortran == U[k-1,j-1,i-1] in Python.
!
! Coriolis formulae (gSAM dolatlon=.true.):
!
!   dU(i,j,k) = (f(j) + U(i,j,k)*tanr(j)) * v_bar(i,j,k)
!     v_bar = (adyv(j)*(V(i,j,k)+V(i-1,j,k)) + adyv(j+1)*(V(i,j+1,k)+V(i-1,j+1,k)))
!             / (4 * ady(j))
!
!   dV(i,jv,k) = -0.25*imuv(jv)*(q(i,j_n,k)+q(i+1,j_n,k)+q(i,j_s,k)+q(i+1,j_s,k))
!     where q(i,j,k) = (f(j)+U(i,j,k)*tanr(j))*mu(j)*U(i,j,k)
!     j_n = jv, j_s = jv-1   (1-based)
!     imuv(jv) = 1/max(cos_v(jv), 1e-6)
!   dV(i,1,k) = 0, dV(i,ny+1,k) = 0  (pole BCs)
!
! Mode: coriolis_tend
!   Output: dU(nz,ny,nx) concat dV(nz,ny+1,nx) as flat float32 vector
!
! inputs.bin:
!   int32 nz, int32 ny, int32 nx
!   float32 U(nz,ny,nx+1)  [C order]
!   float32 V(nz,ny+1,nx)  [C order]
!   float32 fcory(ny), tanr(ny), mu(ny), cos_v(ny+1), ady(ny), adyv(ny+1)

program coriolis_driver
  implicit none
  character(len=64) :: mode

  call get_command_argument(1, mode)

  select case (trim(mode))
  case ('coriolis_tend');  call run_coriolis()
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select

contains

  subroutine run_coriolis()
    integer :: nz, ny, nx, u_in, u_out
    integer :: N_dU, N_dV, N_total, k, j, jv, i, ip1, i_west
    ! Reversed-axis arrays: U(nx+1,ny,nz), V(nx,ny+1,nz)
    real, allocatable :: U(:,:,:), V(:,:,:)
    real, allocatable :: fcory(:), tanr(:), mu(:), cos_v(:), ady(:), adyv(:)
    ! dU and dV also reversed: dU(nx,ny,nz), dV(nx,ny+1,nz)
    real, allocatable :: dU(:,:,:), dV(:,:,:), out(:)
    real :: imuv_j, q_ij, q_ip1j, q_ijm1, q_ip1jm1, v_bar

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    ! Reversed axes relative to Python shapes
    allocate(U(nx+1, ny, nz), V(nx, ny+1, nz))
    allocate(fcory(ny), tanr(ny), mu(ny), cos_v(ny+1), ady(ny), adyv(ny+1))
    read(u_in) U
    read(u_in) V
    read(u_in) fcory
    read(u_in) tanr
    read(u_in) mu
    read(u_in) cos_v
    read(u_in) ady
    read(u_in) adyv
    close(u_in)

    allocate(dU(nx, ny, nz), dV(nx, ny+1, nz))
    dU = 0.0
    dV = 0.0

    ! Fortran mapping: U(i,j,k) = Python U[k-1, j-1, i-1]
    !                  V(i,j,k) = Python V[k-1, j-1, i-1]
    ! Periodic x: i-1 = mod(i-2+nx,nx)+1; i+1 = mod(i,nx)+1

    do k = 1, nz
      ! --- dU ---
      do j = 1, ny
        do i = 1, nx
          ! U at west face of mass cell i: Python U_left[k-1,j-1,i-1] = U(i,j,k)
          ! V corners (periodic in x):
          i_west = mod(i - 2 + nx, nx) + 1   ! i-1 with wrap
          v_bar = (adyv(j)   * (V(i, j,   k) + V(i_west, j,   k)) &
                +  adyv(j+1) * (V(i, j+1, k) + V(i_west, j+1, k))) &
                / (4.0 * ady(j))
          dU(i, j, k) = (fcory(j) + U(i, j, k) * tanr(j)) * v_bar
        end do
      end do

      ! --- dV interior (jv=2..ny) ---
      do jv = 2, ny
        imuv_j = 1.0 / max(cos_v(jv), 1.0e-6)
        do i = 1, nx
          ip1 = mod(i, nx) + 1   ! i+1 with wrap
          ! q(i,j) = (f(j)+U(i,j)*tanr(j))*mu(j)*U(i,j)
          ! north mass row j_n = jv; south mass row j_s = jv-1
          q_ij      = (fcory(jv)   + U(i,   jv,   k) * tanr(jv))   * mu(jv)   * U(i,   jv,   k)
          q_ip1j    = (fcory(jv)   + U(ip1, jv,   k) * tanr(jv))   * mu(jv)   * U(ip1, jv,   k)
          q_ijm1    = (fcory(jv-1) + U(i,   jv-1, k) * tanr(jv-1)) * mu(jv-1) * U(i,   jv-1, k)
          q_ip1jm1  = (fcory(jv-1) + U(ip1, jv-1, k) * tanr(jv-1)) * mu(jv-1) * U(ip1, jv-1, k)
          dV(i, jv, k) = -0.25 * imuv_j * (q_ij + q_ip1j + q_ijm1 + q_ip1jm1)
        end do
      end do
      ! dV(i,1,k) = 0, dV(i,ny+1,k) = 0  (already zero from allocation)
    end do

    ! Flatten dU then dV; both are in reversed-axis order, so the bytes
    ! already match Python's C-order output of shape (nz,ny,nx) and (nz,ny+1,nx).
    N_dU    = nx * ny * nz
    N_dV    = nx * (ny+1) * nz
    N_total = N_dU + N_dV
    allocate(out(N_total))
    out(1:N_dU)         = reshape(dU, [N_dU])
    out(N_dU+1:N_total) = reshape(dV, [N_dV])

    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(N_total, 4)
    write(u_out) out
    close(u_out)
  end subroutine run_coriolis

end program coriolis_driver
