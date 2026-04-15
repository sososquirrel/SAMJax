! test_consts — Fortran shadow of the constants jsam exports from
! jsam.core.physics.microphysics.  Numbers are verbatim from
! /glade/u/home/sabramian/gSAM1.8.7/SRC/consts.f90 (gSAM declares
! these as real(8) parameters; we down-cast to real(4) for binary
! comparison via common/bin_io.py).
!
! Output is a single (18,)-vector, in the same order the
! python harness writes its jsam side:
!
!   1  cp          (J/kg/K)
!   2  cpv         (J/kg/K)
!   3  ggr         (m/s²)
!   4  lcond       (J/kg)
!   5  lfus        (J/kg)
!   6  lsub        (J/kg)
!   7  rv          (J/kg/K)
!   8  rgas        (J/kg/K)
!   9  diffelq     (m²/s)
!  10  therco      (J/m/s/K)
!  11  muelq       (Pa·s)
!  12  fac_cond    (= lcond/cp)
!  13  fac_fus     (= lfus/cp)
!  14  fac_sub     (= lsub/cp)
!  15  rad_earth   (m)
!  16  sigmaSB     (W/m²/K⁴)
!  17  emis_water  (-)
!  18  cpw         (J/kg/K, seawater)
program consts_driver
  implicit none

  real(8), parameter :: cp        = 1004.64d0
  real(8), parameter :: cpv       = 1870.0d0
  real(8), parameter :: ggr       = 9.79764d0
  real(8), parameter :: lcond     = 2.501d6
  real(8), parameter :: lfus      = 0.337d6
  real(8), parameter :: lsub      = 2.834d6
  real(8), parameter :: rv        = 461.5d0
  real(8), parameter :: rgas      = 287.04d0
  real(8), parameter :: diffelq   = 2.21d-5
  real(8), parameter :: therco    = 2.40d-2
  real(8), parameter :: muelq     = 1.717d-5
  real(8), parameter :: fac_cond  = lcond / cp
  real(8), parameter :: fac_fus   = lfus  / cp
  real(8), parameter :: fac_sub   = lsub  / cp
  real(8), parameter :: rad_earth = 6371229.d0
  real(8), parameter :: sigmaSB   = 5.670373d-8
  real(8), parameter :: emis_water= 0.98d0
  real(8), parameter :: cpw       = 3991.86795711963d0

  real, dimension(18) :: out
  integer :: u_out

  out( 1) = real(cp)
  out( 2) = real(cpv)
  out( 3) = real(ggr)
  out( 4) = real(lcond)
  out( 5) = real(lfus)
  out( 6) = real(lsub)
  out( 7) = real(rv)
  out( 8) = real(rgas)
  out( 9) = real(diffelq)
  out(10) = real(therco)
  out(11) = real(muelq)
  out(12) = real(fac_cond)
  out(13) = real(fac_fus)
  out(14) = real(fac_sub)
  out(15) = real(rad_earth)
  out(16) = real(sigmaSB)
  out(17) = real(emis_water)
  out(18) = real(cpw)

  open(newunit=u_out, file='fortran_out.bin', access='stream', &
       form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) 18_4
  write(u_out) out
  close(u_out)
end program consts_driver
