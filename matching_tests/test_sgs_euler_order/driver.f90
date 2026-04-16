! test_sgs_euler_order/driver.f90
!
! Tests the order of SGS momentum application in gSAM's adamsA.f90.
!
! gSAM applies SGS diffusion (dudtd) INSIDE adamsA, but NOT in the
! AB3 history buffer:
!
!   u(i,j,k) = u(i,j,k) + dt * ( at*dudt(na) + bt*dudt(nb) + ct*dudt(nc)
!                                + dudtd(i,j,k) )
!
! The dudtd term is fresh (single-step), while dudt(na/nb/nc) are
! AB3-weighted. This is equivalent to:
!   1. AB3 step: u_ab3 = u + dt*(at*F_n + bt*F_{n-1} + ct*F_{n-2})
!   2. Euler add: u_final = u_ab3 + dt*dudtd
!
! This test verifies both approaches produce the same result, and
! that putting dudtd into the AB3 buffer (wrong order) diverges.
!
! Binary inputs.bin layout:
!   i4 n
!   f4 dt
!   f4 at, bt, ct
!   f4 phi(n)         [initial field]
!   f4 tend_na(n)     [AB3 tendency slot na]
!   f4 tend_nb(n)     [AB3 tendency slot nb]
!   f4 tend_nc(n)     [AB3 tendency slot nc]
!   f4 dudtd(n)       [SGS diffusion tendency]
!   i4 mode           [0=correct, 1=wrong_order]
!
! Output fortran_out.bin:
!   f4 phi_final(n)

program sgsorder_driver
  implicit none
  integer(4) :: n, mode
  real :: dt_s, at, bt, ct
  integer :: i, u_in, u_out
  real, allocatable :: phi(:), tend_na(:), tend_nb(:), tend_nc(:), dudtd(:)

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  read(u_in) n
  read(u_in) dt_s
  read(u_in) at, bt, ct

  allocate(phi(n), tend_na(n), tend_nb(n), tend_nc(n), dudtd(n))
  read(u_in) phi
  read(u_in) tend_na
  read(u_in) tend_nb
  read(u_in) tend_nc
  read(u_in) dudtd
  read(u_in) mode
  close(u_in)

  if (mode == 0) then
    ! Correct: gSAM adamsA style — dudtd added as fresh tendency
    do i = 1, n
      phi(i) = phi(i) + dt_s * ( &
               at * tend_na(i) + bt * tend_nb(i) + ct * tend_nc(i) &
             + dudtd(i) )
    end do
  else
    ! Wrong: put dudtd INTO the AB3 buffer (pollutes history)
    ! This simulates what would happen if SGS were included in
    ! the tend_na slot:  tend_na_wrong = tend_na + dudtd
    do i = 1, n
      phi(i) = phi(i) + dt_s * ( &
               at * (tend_na(i) + dudtd(i)) &
             + bt * tend_nb(i) + ct * tend_nc(i) )
    end do
  end if

  open(newunit=u_out, file='fortran_out.bin', access='stream', &
       form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) int(n, 4)
  write(u_out) phi
  close(u_out)

end program sgsorder_driver
