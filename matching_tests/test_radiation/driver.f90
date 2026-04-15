! Fortran shadow of jsam/tests/unit/test_radiation.py
!
! Two modes:
!   interp1d    — reads (n, xp(n), fp(n), x); writes interpolated scalar
!
!                 Linear interp with edge clamping matching jsam _interp1d:
!                   idx = searchsorted_right(xp, x) clamped to [1, n-1]
!                   w   = clip((x - xp(idx-1)) / (xp(idx) - xp(idx-1)), 0, 1)
!                   out = fp(idx-1) + w * (fp(idx) - fp(idx-1))
!
!   rad_proc    — reads (nz, ny, nx, TABS(nz,ny,nx), q_profile(nz), z_model(nz), dt)
!                 Applies TABS_new = TABS + q_profile * dt (profile already on model grid)
!                 writes TABS_new(nz,ny,nx) flattened C-order
!
! Binary output format (matching common/bin_io.py):
!   write(u_out) 1_4          ! ndim
!   write(u_out) int(N, 4)    ! size
!   write(u_out) arr          ! float32 data
!
program radiation_driver
  implicit none
  character(len=64) :: mode

  call get_command_argument(1, mode)

  select case (trim(mode))
  case ('interp1d_midpoint', 'interp1d_clamp_below', 'interp1d_clamp_above')
    call run_interp1d()
  case ('rad_proc_magnitude')
    call run_rad_proc()
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select

contains

  ! -------------------------------------------------------------------
  ! Linear interpolation with boundary clamping
  ! -------------------------------------------------------------------
  real function interp1d(x, xp, fp, n)
    integer, intent(in) :: n
    real,    intent(in) :: x, xp(n), fp(n)
    integer :: idx, k
    real    :: x0, x1, f0, f1, dx, w

    ! searchsorted right: find first index where xp(idx) > x
    ! then clamp to [2, n] so that idx-1 is always valid
    idx = n   ! default: beyond all xp
    do k = 1, n
      if (xp(k) > x) then
        idx = k
        exit
      end if
    end do
    ! clamp to [2, n]  (i.e. idx-1 in [1, n-1])
    if (idx < 2) idx = 2
    if (idx > n) idx = n

    x0 = xp(idx - 1);  x1 = xp(idx)
    f0 = fp(idx - 1);  f1 = fp(idx)
    dx = x1 - x0
    if (dx > 0.0) then
      w = (x - x0) / dx
    else
      w = 0.0
    end if
    w = min(max(w, 0.0), 1.0)
    interp1d = f0 + w * (f1 - f0)
  end function interp1d

  ! -------------------------------------------------------------------
  subroutine run_interp1d()
    integer :: n, u_in, u_out
    real, allocatable :: xp(:), fp(:)
    real :: x, result

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    ! Layout: int32 n; float32 xp(n); float32 fp(n); float32 x
    read(u_in) n
    allocate(xp(n), fp(n))
    read(u_in) xp
    read(u_in) fp
    read(u_in) x
    close(u_in)

    result = interp1d(x, xp, fp, n)

    open(newunit=u_out, file='fortran_out.bin', access='stream', &
         form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) 1_4
    write(u_out) result
    close(u_out)
  end subroutine run_interp1d

  ! -------------------------------------------------------------------
  ! rad_proc: TABS_new(i,j,k) = TABS(i,j,k) + q_profile(k) * dt
  ! -------------------------------------------------------------------
  subroutine run_rad_proc()
    integer :: nz, ny, nx, k, j, i, u_in, u_out
    real, allocatable :: TABS(:,:,:), q_profile(:), z_model(:), TABS_new(:,:,:)
    real :: dt

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    ! Layout: int32 nz, ny, nx; float32 TABS(nz,ny,nx) C-order; float32 q_profile(nz);
    !         float32 z_model(nz); float32 dt
    read(u_in) nz, ny, nx
    allocate(TABS(nx, ny, nz), q_profile(nz), z_model(nz), TABS_new(nx, ny, nz))
    ! C-order from Python: i varies fastest in memory -> Fortran(nx,ny,nz)
    read(u_in) TABS
    read(u_in) q_profile
    read(u_in) z_model
    read(u_in) dt
    close(u_in)

    ! Apply heating: TABS_new(i,j,k) = TABS(i,j,k) + q_profile(k) * dt
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          TABS_new(i, j, k) = TABS(i, j, k) + q_profile(k) * dt
        end do
      end do
    end do

    open(newunit=u_out, file='fortran_out.bin', access='stream', &
         form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(nz * ny * nx, 4)
    write(u_out) TABS_new
    close(u_out)
  end subroutine run_rad_proc

end program radiation_driver
