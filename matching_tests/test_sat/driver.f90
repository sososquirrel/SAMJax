! test_sat — Fortran shadow of jsam.core.physics.microphysics saturation
! routines.  Verbatim copy of the relevant snippets in
! /glade/u/home/sabramian/gSAM1.8.7/SRC/sat.f90 (esatw, qsatw, dtqsatw)
! so the driver builds with bare gfortran — no link against the gSAM
! library required.
!
! Modes (selected via argv[1]):
!   esatw    — reads nrec T(K) values, writes nrec esatw(T) values
!   qsatw    — reads nrec (T,p) pairs, writes nrec qsatw(T,p) values
!              The python harness includes >=50 (T,p) points at low p
!              where es > p-es so the gSAM `max(es, p-es)` clamp matters.
!   dtqsatw  — reads nrec (T,p) pairs, writes nrec dqsatw/dT values
!
! Binary I/O: matches matching_tests/common/bin_io.py
!   write(u_out) 1_4 ; write(u_out) int(N,4) ; write(u_out) arr
program sat_driver
  implicit none
  character(len=64) :: mode

  call get_command_argument(1, mode)

  select case (trim(mode))
  case ('esatw');    call run_esatw()
  case ('qsatw');    call run_qsatw()
  case ('dtqsatw');  call run_dtqsatw()
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select

contains

  ! gSAM SRC/sat.f90:8-17 — esatw, returns hPa
  pure real function esatw_local(t)
    real, intent(in) :: t
    real, parameter :: e0 = 6.1121
    real, parameter :: a1 = 17.502
    real, parameter :: T0 = 273.16
    real, parameter :: T1 =  32.19
    esatw_local = e0 * exp( a1 * (t - T0) / (t - T1) )
  end function esatw_local

  ! gSAM SRC/sat.f90:33-41 — qsatw, with max(es, p-es) clamp
  pure real function qsatw_local(t, p)
    real, intent(in) :: t, p
    real :: esat
    esat = esatw_local(t)
    qsatw_local = 0.622 * esat / max(esat, p - esat)
  end function qsatw_local

  ! gSAM SRC/sat.f90:124-137 — dtqsatw
  pure real function dtqsatw_local(t, p)
    real, intent(in) :: t, p
    real, parameter :: a1 = 17.502
    real, parameter :: T0 = 273.16
    real, parameter :: T1 =  32.19
    real :: esat, dtesatw
    esat = esatw_local(t)
    dtesatw = esat * a1 * (T0 - T1) / ((t - T1) * (t - T1))
    dtqsatw_local = 0.622 * dtesatw / (p - esat) * (1.0 + esat / (p - esat))
  end function dtqsatw_local

  ! ------------------------------------------------------------------
  subroutine run_esatw()
    integer :: nrec, k, u_in, u_out
    real, allocatable :: t(:), out(:)
    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nrec
    allocate(t(nrec), out(nrec))
    read(u_in) t
    close(u_in)
    do k = 1, nrec
      out(k) = esatw_local(t(k))
    end do
    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(nrec, 4)
    write(u_out) out
    close(u_out)
  end subroutine run_esatw

  subroutine run_qsatw()
    integer :: nrec, k, u_in, u_out
    real, allocatable :: t(:), p(:), out(:)
    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nrec
    allocate(t(nrec), p(nrec), out(nrec))
    read(u_in) t
    read(u_in) p
    close(u_in)
    do k = 1, nrec
      out(k) = qsatw_local(t(k), p(k))
    end do
    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(nrec, 4)
    write(u_out) out
    close(u_out)
  end subroutine run_qsatw

  subroutine run_dtqsatw()
    integer :: nrec, k, u_in, u_out
    real, allocatable :: t(:), p(:), out(:)
    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nrec
    allocate(t(nrec), p(nrec), out(nrec))
    read(u_in) t
    read(u_in) p
    close(u_in)
    do k = 1, nrec
      out(k) = dtqsatw_local(t(k), p(k))
    end do
    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(nrec, 4)
    write(u_out) out
    close(u_out)
  end subroutine run_dtqsatw

end program sat_driver
