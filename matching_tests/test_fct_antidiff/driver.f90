! test_fct_antidiff/driver.f90
!
! Standalone 1D FCT flux limiter test.
!
! Implements the Zalesak (1979) FCT algorithm in 1D:
!   1. Compute upwind (1st-order) flux and high-order (5th-order) flux
!   2. Antidiffusive flux = F_high - F_low
!   3. Limit antidiffusive flux to maintain monotonicity
!   4. phi_new = phi_upwind + limited_correction
!
! Simplified 1D version for isolated testing of the limiter logic.
!
! Binary inputs.bin layout:
!   i4 n              [number of cells]
!   f4 phi(n)         [scalar field]
!   f4 u(n+1)         [face velocities]
!   f4 dt, dx
!
! Output fortran_out.bin:
!   f4 phi_new(n)     [after 1 advection step with FCT]

program fct_driver
  implicit none
  integer(4) :: n
  real :: dt_s, dx_s
  integer :: i, ip, im, u_in, u_out
  real, allocatable :: phi(:), u_f(:), phi_new(:)
  real, allocatable :: f_low(:), f_high(:), f_anti(:)
  real, allocatable :: phi_up(:), p_plus(:), p_minus(:)
  real, allocatable :: q_plus(:), q_minus(:), r_plus(:), r_minus(:), scale(:)
  real :: cr, phi_face_up, phi_face_ho
  real :: local_max, local_min
  real, parameter :: eps = 1.0e-30

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  read(u_in) n
  allocate(phi(n), u_f(n+1))
  read(u_in) phi
  read(u_in) u_f
  read(u_in) dt_s, dx_s
  close(u_in)

  allocate(f_low(n+1), f_high(n+1), f_anti(n+1))
  allocate(phi_up(n), phi_new(n))
  allocate(p_plus(n), p_minus(n), q_plus(n), q_minus(n))
  allocate(r_plus(n), r_minus(n), scale(n+1))

  ! --- Step 1: Compute low-order (upwind) and high-order fluxes ---
  do i = 1, n+1
    im = mod(i-2+n, n) + 1  ! periodic
    ip = mod(i, n) + 1       ! periodic (cell to the right of face i)
    cr = u_f(i) * dt_s / dx_s

    ! Upwind flux
    if (u_f(i) >= 0.0) then
      phi_face_up = phi(im)
    else
      phi_face_up = phi(ip)
    end if
    f_low(i) = u_f(i) * phi_face_up

    ! High-order: simple centered (2nd-order for this test)
    phi_face_ho = 0.5 * (phi(im) + phi(ip))
    f_high(i) = u_f(i) * phi_face_ho

    f_anti(i) = f_high(i) - f_low(i)
  end do

  ! --- Step 2: Upwind update ---
  do i = 1, n
    phi_up(i) = phi(i) - dt_s/dx_s * (f_low(i+1) - f_low(i))
  end do

  ! --- Step 3: FCT limiting (Zalesak 1979) ---
  ! P+/P-: sum of incoming/outgoing antidiffusive fluxes
  do i = 1, n
    ip = mod(i, n) + 1
    p_plus(i)  = max(0.0, f_anti(i)) - min(0.0, f_anti(i+1))
    p_minus(i) = max(0.0, f_anti(i+1)) - min(0.0, f_anti(i))

    ! Q+/Q-: headroom
    im = mod(i-2+n, n) + 1
    local_max = max(phi(im), phi(i), phi(ip))
    local_min = min(phi(im), phi(i), phi(ip))
    q_plus(i)  = (local_max - phi_up(i)) * dx_s / dt_s
    q_minus(i) = (phi_up(i) - local_min) * dx_s / dt_s
  end do

  do i = 1, n
    if (p_plus(i) > eps) then
      r_plus(i) = min(1.0, q_plus(i) / (p_plus(i) + eps))
    else
      r_plus(i) = 1.0
    end if
    if (p_minus(i) > eps) then
      r_minus(i) = min(1.0, q_minus(i) / (p_minus(i) + eps))
    else
      r_minus(i) = 1.0
    end if
  end do

  ! Scale each antidiffusive flux
  do i = 1, n+1
    im = mod(i-2+n, n) + 1
    ip = mod(i, n) + 1
    if (f_anti(i) >= 0.0) then
      scale(i) = min(r_plus(ip), r_minus(im))
    else
      scale(i) = min(r_plus(im), r_minus(ip))
    end if
  end do

  ! --- Step 4: Final update ---
  do i = 1, n
    phi_new(i) = phi_up(i) - dt_s/dx_s * &
                 (scale(i+1)*f_anti(i+1) - scale(i)*f_anti(i))
  end do

  ! Write output
  open(newunit=u_out, file='fortran_out.bin', access='stream', &
       form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) int(n, 4)
  write(u_out) phi_new
  close(u_out)

end program fct_driver
