! Fortran shadow of jsam/core/physics/microphysics.py — saturation functions
!
! Cases:
!   qsatw_at_20C_1000mb   — T=293.16K, p=1000mb → check range 0.013..0.016
!   qsati_below_freezing  — T=263.16K, p=500mb  → qsati < qsatw
!   qsatw_monotone        — T=270..300K, p=1000mb → monotone increasing
!
! gSAM / Buck (1981) formulas matching jsam microphysics.py:
!
!   esatw(T) = 6.1121 * exp(17.502*(T-273.16)/(T-32.18))   mb
!   esati(T) = 6.1121 * exp(22.587*(T-273.16)/(T+0.7))     mb
!   EPS = 0.6220  (RGAS/RV = 287.04/461.5)
!   qsatw = EPS * esatw / max(p - esatw, 1e-3)   (p in mb)
!   qsati = EPS * esati / max(p - esati, 1e-3)
!
! Note: jsam uses EPS=RGAS/RV = 287.04/461.5 ≈ 0.6220
!
! inputs.bin:
!   int32  nT         (number of temperature values)
!   float32 T(nT)     (K)
!   float32 p         (mb, scalar)
!
! Output: fortran_out.bin
!   For qsatw* cases: qsatw(nT)
!   For qsati* cases: [qsatw(nT), qsati(nT)] concatenated (2*nT floats)
program microphysics_driver
  implicit none
  character(len=64) :: case_name
  call get_command_argument(1, case_name)

  select case (trim(case_name))
  case ('qsatw_at_20C_1000mb');   call run_qsatw()
  case ('qsatw_monotone');        call run_qsatw()
  case ('qsati_below_freezing');  call run_qsati_compare()
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

  ! Buck (1981) saturation vapour pressure — liquid water (mb)
  function esatw(T) result(es)
    real, intent(in) :: T   ! K
    real :: es
    es = 6.1121 * exp(17.502 * (T - 273.16) / (T - 32.18))
  end function esatw

  ! Buck (1981) saturation vapour pressure — ice (mb)
  function esati(T) result(es)
    real, intent(in) :: T   ! K
    real :: es
    es = 6.1121 * exp(22.587 * (T - 273.16) / (T + 0.7))
  end function esati

  real function qsatw_fn(T, p)
    real, intent(in) :: T, p   ! T in K, p in mb
    real :: es, eps
    eps = 287.04 / 461.5   ! RGAS/RV matching jsam consts
    es = esatw(T)
    qsatw_fn = eps * es / max(p - es, 1.0e-3)
  end function qsatw_fn

  real function qsati_fn(T, p)
    real, intent(in) :: T, p   ! T in K, p in mb
    real :: es, eps
    eps = 287.04 / 461.5
    es = esati(T)
    qsati_fn = eps * es / max(p - es, 1.0e-3)
  end function qsati_fn

  ! -----------------------------------------------------------------------
  ! Run qsatw for all T values; output qsatw(nT)
  ! -----------------------------------------------------------------------
  subroutine run_qsatw()
    integer :: nT, k, u_in
    real :: p
    real, allocatable :: T(:), qs(:)
    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nT
    allocate(T(nT), qs(nT))
    read(u_in) T
    read(u_in) p
    close(u_in)
    do k = 1, nT
      qs(k) = qsatw_fn(T(k), p)
    end do
    call write_result(qs, nT)
    deallocate(T, qs)
  end subroutine run_qsatw

  ! -----------------------------------------------------------------------
  ! Run both qsatw and qsati; output [qsatw(nT), qsati(nT)]
  ! -----------------------------------------------------------------------
  subroutine run_qsati_compare()
    integer :: nT, k, u_in
    real :: p
    real, allocatable :: T(:), qw(:), qi(:), out(:)
    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nT
    allocate(T(nT), qw(nT), qi(nT), out(2*nT))
    read(u_in) T
    read(u_in) p
    close(u_in)
    do k = 1, nT
      qw(k) = qsatw_fn(T(k), p)
      qi(k) = qsati_fn(T(k), p)
    end do
    out(1:nT)       = qw
    out(nT+1:2*nT)  = qi
    call write_result(out, 2*nT)
    deallocate(T, qw, qi, out)
  end subroutine run_qsati_compare

end program microphysics_driver
