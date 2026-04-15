! Fortran shadow of jsam/tests/unit/test_lsforcing.py
!
! Implements two kernels:
!
! 1. subsidence — first-order upwind −w·∂φ/∂z:
!      wsub > 0 (upward): backward diff (phi[k] - phi[k-1]) / dz[k]
!      wsub < 0 (downward): forward diff (phi[k+1] - phi[k]) / dz[k]
!      Edge pad: phi[k=0] = phi[k=1], phi[k=nz+1] = phi[k=nz] → tend=0 at boundaries
!
! 2. ls_proc_direct:
!      TABS += dtls * dt
!      QV   = max(0, QV + dqls * dt)
!      TABS += dt * subsidence(TABS, wsub, dz)
!      QV   = max(0, QV + dt * subsidence(QV, wsub, dz))
!      Output: TABS_new concat QV_new
!
! Array layout: Python C row-major shape (nz,ny,nx) → Fortran (nx,ny,nz)
! so column-major Fortran storage matches Python bytes.
!
! inputs.bin for subsidence:
!   int32 nz, int32 ny, int32 nx
!   float32 phi(nz,ny,nx) [C order]
!   float32 wsub(nz), dz(nz)
!
! inputs.bin for ls_proc_direct:
!   int32 nz, int32 ny, int32 nx
!   float32 TABS(nz,ny,nx), QV(nz,ny,nx) [C order]
!   float32 dtls(nz), dqls(nz), wsub(nz), dz(nz)
!   float32 dt

program lsforcing_driver
  implicit none
  character(len=64) :: mode

  call get_command_argument(1, mode)

  select case (trim(mode))
  case ('subsidence');       call run_subsidence()
  case ('ls_proc_direct');   call run_ls_proc()
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select

contains

  ! -----------------------------------------------------------------------
  ! subsidence_tend: first-order upwind −w·∂φ/∂z
  ! phi(nx,ny,nz) in Fortran (reversed), wsub(nz), dz(nz)
  ! tend same shape
  ! -----------------------------------------------------------------------
  subroutine subsidence_tend_3d(nx, ny, nz, phi, wsub, dz, tend)
    integer, intent(in)  :: nx, ny, nz
    real,    intent(in)  :: phi(nx, ny, nz), wsub(nz), dz(nz)
    real,    intent(out) :: tend(nx, ny, nz)
    integer :: k, j, i
    real :: dphi, w_k, inv_dz

    ! Fortran phi(i,j,k) = Python phi[k-1,j-1,i-1]  (0-indexed Python)
    do k = 1, nz
      w_k    = wsub(k)
      inv_dz = 1.0 / dz(k)
      do j = 1, ny
        do i = 1, nx
          if (w_k >= 0.0) then
            ! backward difference (upward advection)
            if (k == 1) then
              dphi = 0.0   ! edge pad: phi[k-1] = phi[k]
            else
              dphi = phi(i,j,k) - phi(i,j,k-1)
            end if
          else
            ! forward difference (downward advection)
            if (k == nz) then
              dphi = 0.0   ! edge pad: phi[k+1] = phi[k]
            else
              dphi = phi(i,j,k+1) - phi(i,j,k)
            end if
          end if
          tend(i,j,k) = -w_k * dphi * inv_dz
        end do
      end do
    end do
  end subroutine subsidence_tend_3d

  ! -----------------------------------------------------------------------
  subroutine run_subsidence()
    integer :: nz, ny, nx, u_in, u_out, N
    real, allocatable :: phi(:,:,:), wsub(:), dz(:), tend(:,:,:)

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    ! Reversed axes: Python (nz,ny,nx) → Fortran (nx,ny,nz)
    allocate(phi(nx,ny,nz), wsub(nz), dz(nz), tend(nx,ny,nz))
    read(u_in) phi
    read(u_in) wsub
    read(u_in) dz
    close(u_in)

    call subsidence_tend_3d(nx, ny, nz, phi, wsub, dz, tend)

    N = nx * ny * nz
    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(N, 4)
    write(u_out) tend
    close(u_out)
  end subroutine run_subsidence

  ! -----------------------------------------------------------------------
  subroutine run_ls_proc()
    integer :: nz, ny, nx, u_in, u_out, N_half, N_total
    real, allocatable :: TABS(:,:,:), QV(:,:,:), dtls(:), dqls(:), wsub(:), dz(:)
    real, allocatable :: tend(:,:,:), out(:)
    real :: dt
    integer :: k, j, i

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    ! Reversed axes: Python (nz,ny,nx) → Fortran (nx,ny,nz)
    allocate(TABS(nx,ny,nz), QV(nx,ny,nz), dtls(nz), dqls(nz), wsub(nz), dz(nz))
    allocate(tend(nx,ny,nz))
    read(u_in) TABS
    read(u_in) QV
    read(u_in) dtls
    read(u_in) dqls
    read(u_in) wsub
    read(u_in) dz
    read(u_in) dt
    close(u_in)

    ! 1. Horizontal advective tendencies
    ! TABS(i,j,k) and dtls(k) — need to broadcast over i,j
    do k = 1, nz
      TABS(:,:,k) = TABS(:,:,k) + dtls(k) * dt
      QV(:,:,k)   = max(0.0, QV(:,:,k) + dqls(k) * dt)
    end do

    ! 2. Subsidence on TABS
    call subsidence_tend_3d(nx, ny, nz, TABS, wsub, dz, tend)
    TABS = TABS + dt * tend

    ! 3. Subsidence on QV + clamp
    call subsidence_tend_3d(nx, ny, nz, QV, wsub, dz, tend)
    QV = max(0.0, QV + dt * tend)

    N_half  = nx * ny * nz
    N_total = 2 * N_half
    allocate(out(N_total))
    out(1:N_half)          = reshape(TABS, [N_half])
    out(N_half+1:N_total)  = reshape(QV,   [N_half])

    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(N_total, 4)
    write(u_out) out
    close(u_out)
  end subroutine run_ls_proc

end program lsforcing_driver
