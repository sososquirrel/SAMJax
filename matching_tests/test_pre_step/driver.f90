! test_pre_step/driver.f90
!
! Fortran shadow of jsam/core/physics/slm/sat.py — the bit-level saturation
! functions that feed every init-time QC/QI computation. This is the
! smallest isolated brick that can explain the QC/QI mismatch observed in
! the IRMA debug500 run at step 1 stage 0 (see
! scripts/compare_debug_globals.py — QC max rel ~3.8e-1, QI mean rel ~3.1e-1).
!
! What it tests
! -------------
!   esatw(T)       -> hPa        (gSAM SRC/sat.f90:8)
!   esati(T)       -> hPa        (gSAM SRC/sat.f90:19)
!   qsatw(T, p_mb) -> kg/kg      (gSAM SRC/sat.f90:33)
!   qsati(T, p_mb) -> kg/kg      (gSAM SRC/sat.f90:43)
!
! The four functions are copy-pasted in-file from gSAM's sat.f90 so this
! driver stays standalone — no linking against the SAM tree. If gSAM
! sat.f90 changes upstream, sync here and re-run.
!
! Binary I/O (matches dump_inputs.py + common/bin_io.py)
! ------------------------------------------------------
! inputs.bin  (little-endian stream, no record markers):
!   i4 n_points
!   f4 T(n_points)      ! Kelvin
!   f4 P(n_points)      ! hPa (millibars) — jsam/sat.py convention
! fortran_out.bin  (common/bin_io.py format):
!   i4 1                ! ndim
!   i4 N                ! length (= 4 * n_points for mode=all)
!   f4 result(N)        ! [esatw, esati, qsatw, qsati] concatenated
!
! Modes (command-line argument):
!   all     — dump all four functions (length = 4*n_points)
!   esatw   — just esatw   (length = n_points)
!   esati   — just esati
!   qsatw   — just qsatw
!   qsati   — just qsati

program sat_driver
  implicit none
  character(len=16) :: mode
  integer :: u_in, u_out
  integer(4) :: n
  real, allocatable :: T(:), P(:), out(:)
  integer :: i
  real :: esatw_r, esati_r, qsatw_r, qsati_r
  real, external :: esatw, esati, qsatw, qsati

  call get_command_argument(1, mode)
  if (len_trim(mode) == 0) mode = 'all'

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  read(u_in) n
  allocate(T(n), P(n))
  read(u_in) T
  read(u_in) P
  close(u_in)

  select case (trim(mode))
  case ('all')
    allocate(out(4 * n))
    do i = 1, n
      out(          i) = esatw(T(i))
      out(    n +   i) = esati(T(i))
      out(2 * n + i) = qsatw(T(i), P(i))
      out(3 * n + i) = qsati(T(i), P(i))
    end do
  case ('esatw')
    allocate(out(n))
    do i = 1, n;  out(i) = esatw(T(i));         end do
  case ('esati')
    allocate(out(n))
    do i = 1, n;  out(i) = esati(T(i));         end do
  case ('qsatw')
    allocate(out(n))
    do i = 1, n;  out(i) = qsatw(T(i), P(i));   end do
  case ('qsati')
    allocate(out(n))
    do i = 1, n;  out(i) = qsati(T(i), P(i));   end do
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select

  open(newunit=u_out, file='fortran_out.bin', access='stream', &
       form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) int(size(out), 4)
  write(u_out) out
  close(u_out)

end program sat_driver


! ===========================================================================
! gSAM SRC/sat.f90 — copy-pasted so the driver is standalone. Keep in sync.
! ===========================================================================

real function esatw(t)
  implicit none
  real t  ! temperature (K)
  real, parameter :: e0 = 6.1121 ! esat(T0) in hPa
  real, parameter :: a1 = 17.502
  real, parameter :: T0 = 273.16
  real, parameter :: T1 =  32.19
  esatw = e0 * exp( a1 * (T - T0) / (T - T1) )
end function esatw

real function esati(t)
  implicit none
  real t
  real, parameter :: e0 = 6.1121
  real, parameter :: a1 = 22.587
  real, parameter :: T0 = 273.16
  real, parameter :: T1 =   -0.7
  esati = e0 * exp( a1 * (T - T0) / (T - T1) )
end function esati

real function qsatw(t, p)
  implicit none
  real t  ! K
  real p  ! hPa
  real esat
  real, external :: esatw
  esat  = esatw(t)
  qsatw = 0.622 * esat / max(esat, p - esat)
end function qsatw

real function qsati(t, p)
  implicit none
  real t
  real p
  real esat
  real, external :: esati
  esat  = esati(t)
  qsati = 0.622 * esat / max(esat, p - esat)
end function qsati
