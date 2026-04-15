! Fortran shadow of jsam/tests/unit/test_timestepping.py
!
! Two modes:
!   ab2_coefs  — reads nrec cases of (nstep, dt_curr, dt_prev)
!                writes (at, bt) pairs
!
!                Matching gSAM abcoefs.f90 (nadams=2 / AB2 branch):
!                  nstep == 0            -> at=1, bt=0       (Euler)
!                  else (AB2, any nstep) -> alpha = dt_prev/dt_curr
!                                          at = (1+2*alpha)/(2*alpha)
!                                          bt = -1/(2*alpha)
!
!   ab2_step   — reads nrec cases of (phi, tend_n, tend_nm1, dt, dt_prev, nstep)
!                writes phi_new = phi + dt*(at*tend_n + bt*tend_nm1)
!
! Binary output format (matching common/bin_io.py):
!   write(u_out) 1_4          ! ndim
!   write(u_out) int(N, 4)    ! size
!   write(u_out) arr          ! float32 data
!
program timestepping_driver
  implicit none
  character(len=64) :: mode

  call get_command_argument(1, mode)

  select case (trim(mode))
  case ('ab2_coefs');  call run_ab2_coefs()
  case ('ab2_step');   call run_ab2_step()
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select

contains

  ! -------------------------------------------------------------------
  subroutine run_ab2_coefs()
    integer :: nrec, k, nstep, u_in, u_out
    real :: dt_curr, dt_prev, at, bt, alpha
    real, allocatable :: out(:)

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    ! Layout: int32 nrec; then nrec records of (int32 nstep, float32 dt_curr, float32 dt_prev)
    read(u_in) nrec
    allocate(out(2 * nrec))
    do k = 1, nrec
      read(u_in) nstep, dt_curr, dt_prev
      if (nstep == 0) then
        at = 1.0;  bt = 0.0
      else
        alpha = dt_prev / dt_curr
        at    = (1.0 + 2.0 * alpha) / (2.0 * alpha)
        bt    = -1.0 / (2.0 * alpha)
      end if
      out(2 * k - 1) = at
      out(2 * k    ) = bt
    end do
    close(u_in)

    open(newunit=u_out, file='fortran_out.bin', access='stream', &
         form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(2 * nrec, 4)
    write(u_out) out
    close(u_out)
  end subroutine run_ab2_coefs

  ! -------------------------------------------------------------------
  subroutine run_ab2_step()
    integer :: nrec, k, nstep, u_in, u_out
    real :: phi, tend_n, tend_nm1, dt, dt_prev, at, bt, alpha, phi_new
    real, allocatable :: out(:)

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    ! Layout: int32 nrec; then nrec records of
    !   (float32 phi, float32 tend_n, float32 tend_nm1, float32 dt, float32 dt_prev, int32 nstep)
    read(u_in) nrec
    allocate(out(nrec))
    do k = 1, nrec
      read(u_in) phi, tend_n, tend_nm1, dt, dt_prev
      read(u_in) nstep
      if (nstep == 0) then
        at = 1.0;  bt = 0.0
      else
        alpha = dt_prev / dt
        at    = (1.0 + 2.0 * alpha) / (2.0 * alpha)
        bt    = -1.0 / (2.0 * alpha)
      end if
      phi_new = phi + dt * (at * tend_n + bt * tend_nm1)
      out(k) = phi_new
    end do
    close(u_in)

    open(newunit=u_out, file='fortran_out.bin', access='stream', &
         form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(nrec, 4)
    write(u_out) out
    close(u_out)
  end subroutine run_ab2_step

end program timestepping_driver
