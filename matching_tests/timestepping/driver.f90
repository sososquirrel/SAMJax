! Fortran shadow of jsam AB3 timestepping (ab_coefs / ab_step).
!
! Modes:
!   ab3_coefs  — reads nrec of (int32 nstep, float32 dc, float32 dp, float32 dpp)
!                writes (at, bt, ct) per case (3*nrec floats)
!
!   ab3_step   — reads nzm, phi(nzm), tend_n, tend_nm1, tend_nm2, dt, nstep
!                writes phi_new(nzm)
!
! Coefficient formulas (matching jsam ab_coefs / gSAM abcoefs.f90 nadams=3):
!   nstep == 0   ->  (1, 0, 0)   Euler
!   nstep == 1   ->  alpha = dp/dc
!                    at = (1+2a)/(2a),  bt = -1/(2a),  ct = 0
!   nstep >= 2   ->  alpha = dp/dc,  beta = dpp/dc
!                    ct = (2+3a)/(6*(a+b)*b)
!                    bt = -(1+2*(a+b)*ct)/(2*a)
!                    at = 1 - bt - ct

program timestepping_ab3_driver
  implicit none
  character(len=64) :: mode

  call get_command_argument(1, mode)

  select case (trim(mode))
  case ('ab3_coefs');  call run_ab3_coefs()
  case ('ab3_step');   call run_ab3_step()
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select

contains

  ! -------------------------------------------------------------------
  subroutine ab3_coef_calc(nstep, dc, dp, dpp, at, bt, ct)
    integer, intent(in)  :: nstep
    real,    intent(in)  :: dc, dp, dpp
    real,    intent(out) :: at, bt, ct
    real :: alpha, beta

    if (nstep == 0) then
      at = 1.0;  bt = 0.0;  ct = 0.0
    else if (nstep == 1) then
      alpha = dp / dc
      at = (1.0 + 2.0 * alpha) / (2.0 * alpha)
      bt = -1.0 / (2.0 * alpha)
      ct = 0.0
    else
      alpha = dp / dc
      beta  = dpp / dc
      ct = (2.0 + 3.0 * alpha) / (6.0 * (alpha + beta) * beta)
      bt = -(1.0 + 2.0 * (alpha + beta) * ct) / (2.0 * alpha)
      at = 1.0 - bt - ct
    end if
  end subroutine ab3_coef_calc

  ! -------------------------------------------------------------------
  subroutine run_ab3_coefs()
    integer :: nrec, k, nstep, u_in, u_out
    real :: dc, dp, dpp, at, bt, ct
    real, allocatable :: out(:)

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    read(u_in) nrec
    allocate(out(3 * nrec))
    do k = 1, nrec
      read(u_in) nstep, dc, dp, dpp
      call ab3_coef_calc(nstep, dc, dp, dpp, at, bt, ct)
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
  end subroutine run_ab3_coefs

  ! -------------------------------------------------------------------
  subroutine run_ab3_step()
    integer :: nzm, nstep, k, u_in, u_out
    real :: dt, at, bt, ct, dc
    real, allocatable :: phi(:), tn(:), tnm1(:), tnm2(:), phi_new(:)

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    read(u_in) nzm
    allocate(phi(nzm), tn(nzm), tnm1(nzm), tnm2(nzm), phi_new(nzm))
    read(u_in) phi
    read(u_in) tn
    read(u_in) tnm1
    read(u_in) tnm2
    read(u_in) dt, nstep
    close(u_in)

    ! For ab3_step test: dt_curr=dt_prev=dt_pprev=dt, nstep=2 -> AB3
    dc = dt
    call ab3_coef_calc(nstep, dc, dc, dc, at, bt, ct)

    do k = 1, nzm
      phi_new(k) = phi(k) + dt * (at * tn(k) + bt * tnm1(k) + ct * tnm2(k))
    end do

    open(newunit=u_out, file='fortran_out.bin', access='stream', &
         form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(nzm, 4)
    write(u_out) phi_new
    close(u_out)
  end subroutine run_ab3_step

end program timestepping_ab3_driver
