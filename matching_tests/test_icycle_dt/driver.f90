! test_icycle_dt/driver.f90
!
! Minimal reproduction of gSAM's icycle subcycling loop for AB3
! tendency rotation. Demonstrates how the 3-slot circular buffer
! (na, nb, nc) rotates after each icycle.
!
! gSAM main.f90 pattern:
!   dt3(na) = dtn
!   [all dynamics: compute dudt(:,:,:,na)]
!   nn=na; na=nc; nc=nb; nb=nn    ! rotate
!   if(icycle.gt.1) cycle
!
! This driver simulates 3 icycles of a simple tendency accumulation
! to verify the buffer rotation matches jsam's Python-level rotation.
!
! Binary inputs.bin layout:
!   i4 n               [vector size]
!   i4 n_icycles       [1 or 3]
!   f4 phi(n)          [initial scalar field]
!   f4 tend_vals(n_icycles, n)  [pre-computed tendency per icycle]
!   f4 dt_vals(n_icycles)       [dt per icycle]
!
! Output fortran_out.bin:
!   f4 phi_final(n) + tend_na(n) + tend_nb(n) + tend_nc(n)  [4*n values]

program icycle_driver
  implicit none
  integer(4) :: n, n_icycles
  integer :: ic, i, nn, u_in, u_out, n_out
  integer :: na, nb, nc
  real, allocatable :: phi(:), tend_vals(:,:), dt_vals(:)
  real, allocatable :: tend(:,:)   ! tend(n, 3) — 3-slot circular buffer
  real, allocatable :: buf(:)
  real :: dt_ic, at, bt, ct

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  read(u_in) n, n_icycles

  allocate(phi(n), tend_vals(n, n_icycles), dt_vals(n_icycles))
  read(u_in) phi
  read(u_in) tend_vals
  read(u_in) dt_vals
  close(u_in)

  ! Initialize 3-slot tendency buffer
  allocate(tend(n, 3))
  tend = 0.0

  ! gSAM initial slot indices (1-based)
  na = 1
  nb = 2
  nc = 3

  do ic = 1, n_icycles
    dt_ic = dt_vals(ic)

    ! AB coefficients (Euler for ic=1, AB2 for ic=2, AB3 for ic>=3)
    if (ic == 1) then
      at = 1.0; bt = 0.0; ct = 0.0
    else if (ic == 2) then
      at = 1.5; bt = -0.5; ct = 0.0
    else
      at = 23.0/12.0; bt = -16.0/12.0; ct = 5.0/12.0
    end if

    ! Store current tendency in slot na
    tend(:, na) = tend_vals(:, ic)

    ! AB step: phi += dt * (at*tend_na + bt*tend_nb + ct*tend_nc)
    do i = 1, n
      phi(i) = phi(i) + dt_ic * ( &
               at * tend(i, na) + bt * tend(i, nb) + ct * tend(i, nc) )
    end do

    ! Rotate: nn=na; na=nc; nc=nb; nb=nn (gSAM main.f90:353-356)
    nn = na
    na = nc
    nc = nb
    nb = nn
  end do

  ! Write output: phi_final + tend(na) + tend(nb) + tend(nc)
  n_out = 4 * n
  allocate(buf(n_out))
  buf(1:n)        = phi
  buf(n+1:2*n)    = tend(:, na)
  buf(2*n+1:3*n)  = tend(:, nb)
  buf(3*n+1:4*n)  = tend(:, nc)

  open(newunit=u_out, file='fortran_out.bin', access='stream', &
       form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) int(n_out, 4)
  write(u_out) buf
  close(u_out)

end program icycle_driver
