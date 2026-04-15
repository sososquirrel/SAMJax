! Fortran shadow of jsam/tests/unit/test_timestepping.py
!
! Two modes — both ported verbatim from gSAM SRC/abcoefs.f90 (lines 15-30):
!
!   ab2_coefs  — reads nrec records of (int32 nstep, f32 dt_curr, f32 dt_prev,
!                                       f32 dt_pprev) and writes (at,bt,ct)
!                triples (length 3*nrec).
!
!                Branches (gSAM abcoefs.f90):
!                  nstep == 0                 → Euler  (1, 0, 0)
!                  nstep == 1                 → AB2    bootstrap
!                                                  alpha = dt_prev/dt_curr
!                                                  at = (1+2α)/(2α)
!                                                  bt = -1/(2α)
!                                                  ct = 0
!                  nstep >= 2                 → AB3    (gSAM lines 15-20)
!                                                  alpha = dt_prev /dt_curr
!                                                  beta  = dt_pprev/dt_curr
!                                                  ct = (2+3α)/(6(α+β)β)
!                                                  bt = -(1+2(α+β)ct)/(2α)
!                                                  at = 1 - bt - ct
!
!   ab2_step   — reads nrec records of
!                  (f32 phi, f32 tend_n, f32 tend_nm1, f32 tend_nm2,
!                   f32 dt_curr, f32 dt_prev, f32 dt_pprev, int32 nstep)
!                and writes
!                  phi_new = phi + dt*(at*tend_n + bt*tend_nm1 + ct*tend_nm2).
!
! Binary output format (matching common/bin_io.py):
!   write(u_out) 1_4              ! ndim
!   write(u_out) int(N, 4)        ! size
!   write(u_out) arr              ! float32 data
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

  subroutine ab3_branch(nstep, dt_curr, dt_prev, dt_pprev, at, bt, ct)
    ! Verbatim port of gSAM SRC/abcoefs.f90:15-30 (nadams==3 branch).
    integer, intent(in) :: nstep
    real, intent(in)    :: dt_curr, dt_prev, dt_pprev
    real, intent(out)   :: at, bt, ct
    real :: alpha, beta

    if (nstep == 0) then
      at = 1.0;  bt = 0.0;  ct = 0.0
    else if (nstep == 1) then
      alpha = dt_prev / dt_curr
      at = (1.0 + 2.0*alpha) / (2.0*alpha)
      bt = -1.0 / (2.0*alpha)
      ct = 0.0
    else
      alpha = dt_prev  / dt_curr
      beta  = dt_pprev / dt_curr
      ct    = (2.0 + 3.0*alpha) / (6.0*(alpha + beta)*beta)
      bt    = -(1.0 + 2.0*(alpha + beta)*ct) / (2.0*alpha)
      at    = 1.0 - bt - ct
    end if
  end subroutine ab3_branch

  ! -------------------------------------------------------------------
  subroutine run_ab2_coefs()
    integer :: nrec, k, nstep, u_in, u_out
    real :: dt_curr, dt_prev, dt_pprev, at, bt, ct
    real, allocatable :: out(:)

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    ! Layout: int32 nrec; then nrec records of
    !   (int32 nstep, float32 dt_curr, float32 dt_prev, float32 dt_pprev)
    read(u_in) nrec
    allocate(out(3 * nrec))
    do k = 1, nrec
      read(u_in) nstep, dt_curr, dt_prev, dt_pprev
      call ab3_branch(nstep, dt_curr, dt_prev, dt_pprev, at, bt, ct)
      out(3*k - 2) = at
      out(3*k - 1) = bt
      out(3*k    ) = ct
    end do
    close(u_in)

    open(newunit=u_out, file='fortran_out.bin', access='stream', &
         form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(3 * nrec, 4)
    write(u_out) out
    close(u_out)
  end subroutine run_ab2_coefs

  ! -------------------------------------------------------------------
  subroutine run_ab2_step()
    integer :: nrec, k, nstep, u_in, u_out
    real :: phi, tend_n, tend_nm1, tend_nm2, dt_curr, dt_prev, dt_pprev
    real :: at, bt, ct, phi_new
    real, allocatable :: out(:)

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    ! Layout: int32 nrec; then nrec records of
    !   (float32 phi, float32 tend_n, float32 tend_nm1, float32 tend_nm2,
    !    float32 dt_curr, float32 dt_prev, float32 dt_pprev, int32 nstep)
    read(u_in) nrec
    allocate(out(nrec))
    do k = 1, nrec
      read(u_in) phi, tend_n, tend_nm1, tend_nm2, dt_curr, dt_prev, dt_pprev
      read(u_in) nstep
      call ab3_branch(nstep, dt_curr, dt_prev, dt_pprev, at, bt, ct)
      phi_new = phi + dt_curr * (at*tend_n + bt*tend_nm1 + ct*tend_nm2)
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
