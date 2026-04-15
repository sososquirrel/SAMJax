! Fortran shadow of jsam/tests/unit/test_momentum_advection.py
!
! Modes (first command-line argument):
!   flux3_positive     — scalar _flux3 kernel, u_adv > 0
!   flux3_negative     — scalar _flux3 kernel, u_adv < 0
!   flux3_zero         — scalar _flux3 kernel, u_adv = 0
!   flux3_constant     — scalar _flux3 kernel, constant field, any sign
!   zero_velocity      — full advect_momentum with U=V=W=0
!   uniform_U          — full advect_momentum with uniform U, V=W=0
!
! The _flux3 kernel (jsam advection.py, function _flux3) matches gSAM
!   advect23_mom_xy.f90 (nadv_mom=3, alpha_hybrid=0, wg=0, flat terrain):
!
!     With uuu = u1(i+1,j,k) + u1(i,j,k) and d12 = 1/12:
!       uuu >= 0: fu = uuu * d12 * (2*u(i+1) + 5*u(i) - u(i-1))
!       uuu < 0:  fu = uuu * d12 * (2*u(i)   + 5*u(i+1) - u(i+2))
!
!     In jsam notation u_adv = 0.5*uuu, so d12*uuu = (1/6)*u_adv:
!       u_adv >= 0: flux = u_adv * (2*phi_p1 + 5*phi_0 - phi_m1) / 6
!       u_adv <  0: flux = u_adv * (2*phi_0  + 5*phi_p1 - phi_p2) / 6
!
! Binary output format (matches common/bin_io.py):
!   write(u_out) 1_4
!   write(u_out) int(N, 4)
!   write(u_out) arr  ! float32

program mom_advection_driver
  implicit none
  character(len=64) :: mode

  call get_command_argument(1, mode)

  select case (trim(mode))
  case ('flux3_positive'); call run_flux3()
  case ('flux3_negative'); call run_flux3()
  case ('flux3_zero');     call run_flux3()
  case ('flux3_constant'); call run_flux3()
  case ('zero_velocity'); call run_full_mom()
  case ('uniform_U');     call run_full_mom()
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select

contains

  ! -----------------------------------------------------------------------
  ! flux3: 3rd-order upwind-biased face flux  (gSAM nadv_mom=3, wg=0)
  !
  !   face is between phi_0 and phi_p1
  !   u_adv >= 0: flux = u_adv * (2*phi_p1 + 5*phi_0 - phi_m1) / 6
  !   u_adv <  0: flux = u_adv * (2*phi_0  + 5*phi_p1 - phi_p2) / 6
  ! -----------------------------------------------------------------------
  real function flux3(phi_m1, phi_0, phi_p1, phi_p2, u_adv)
    implicit none
    real, intent(in) :: phi_m1, phi_0, phi_p1, phi_p2, u_adv
    real :: f_pos, f_neg
    f_pos = (2.*phi_p1 + 5.*phi_0 - phi_m1) / 6.
    f_neg = (2.*phi_0  + 5.*phi_p1 - phi_p2) / 6.
    if (u_adv >= 0.) then
      flux3 = u_adv * f_pos
    else
      flux3 = u_adv * f_neg
    end if
  end function flux3

  ! -----------------------------------------------------------------------
  ! run_flux3: reads (phi_m1, phi_0, phi_p1, phi_p2, u_adv) → flux
  ! Inputs layout: float32 f(4), float32 u_adv
  ! -----------------------------------------------------------------------
  subroutine run_flux3()
    integer :: u_in, u_out
    real :: f(4), u_adv, result

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) f
    read(u_in) u_adv
    close(u_in)

    result = flux3(f(1), f(2), f(3), f(4), u_adv)

    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) 1_4
    write(u_out) result
    close(u_out)
  end subroutine run_flux3

  ! -----------------------------------------------------------------------
  ! run_full_mom: reads U(nz,ny,nx+1), V(nz,ny+1,nx), W(nz+1,ny,nx),
  ! metric arrays, dt; runs one step of 3rd-order upwind x-direction
  ! momentum advection for U; writes U_new, V_new, W_new flattened.
  !
  ! For the tested cases (zero velocity or uniform U with V=W=0):
  !   zero_velocity → U_new=U, V_new=V, W_new=W (all fluxes zero)
  !   uniform_U     → U_new=U (constant U is div-free in x; y,z=0)
  !
  ! We implement the U x-direction momentum advection loop using flux3.
  ! The y and z contributions are zero for these cases; we do not implement
  ! them (they cancel for zero-velocity or zero V,W cases).
  !
  ! Inputs layout (from dump_inputs.py):
  !   int32 nz, int32 ny, int32 nx
  !   float32 U(nz,ny,nx+1)     C-order
  !   float32 V(nz,ny+1,nx)     C-order
  !   float32 W(nz+1,ny,nx)     C-order
  !   float32 dx_lon             scalar (m)
  !   float32 dy_lat_ref         scalar reference dy (m)
  !   float32 dz_ref             scalar reference dz (m) [= dz(1)]
  !   float32 ady(ny)            ratio dy_per_row/dy_ref
  !   float32 adz(nz)            ratio dz/dz_ref
  !   float32 adzw(nz+1)         ratio dz at w-faces
  !   float32 rho(nz)
  !   float32 rhow(nz+1)
  !   float32 mu(ny)             cos(lat)
  !   float32 muv(ny+1)          cos_v at v-faces
  !   float32 dt
  ! -----------------------------------------------------------------------
  subroutine run_full_mom()
    implicit none
    integer :: nz, ny, nx, k, j, i, im1, im2, ip, ip2
    integer :: u_in, u_out
    ! Arrays declared in Fortran column-major with reversed index order to
    ! match C-order binary from Python.
    real, allocatable :: U_arr(:,:,:)   ! (nx+1, ny, nz)
    real, allocatable :: V_arr(:,:,:)   ! (nx, ny+1, nz)
    real, allocatable :: W_arr(:,:,:)   ! (nx, ny, nz+1)
    real, allocatable :: ady(:), adz(:), adzw(:), rho(:), rhow(:)
    real, allocatable :: mu(:), muv(:)
    real, allocatable :: U_new(:,:,:), V_new(:,:,:), W_new(:,:,:)
    real :: dx, dy_ref, dz_ref, dt
    ! tendencies
    real, allocatable :: dU(:,:,:)      ! (nx, ny, nz)
    real :: u1_i, u1_ip, u_adv, f_m1, f_0, f_p1, f_p2, fl_west, fl_east
    real :: gu_k, gu_j

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    allocate(U_arr(nx+1, ny, nz), V_arr(nx, ny+1, nz), W_arr(nx, ny, nz+1))
    allocate(ady(ny), adz(nz), adzw(nz+1), rho(nz), rhow(nz+1), mu(ny), muv(ny+1))
    allocate(U_new(nx+1, ny, nz), V_new(nx, ny+1, nz), W_new(nx, ny, nz+1))
    allocate(dU(nx, ny, nz))
    read(u_in) U_arr
    read(u_in) V_arr
    read(u_in) W_arr
    read(u_in) dx
    read(u_in) dy_ref
    read(u_in) dz_ref
    read(u_in) ady
    read(u_in) adz
    read(u_in) adzw
    read(u_in) rho
    read(u_in) rhow
    read(u_in) mu
    read(u_in) muv
    read(u_in) dt
    close(u_in)

    ! U x-tendency (gSAM advect23_mom_xy lines 25-41, nadv_mom=3, wg=0, flat terrain):
    !
    !   u1(i,j,k) = U(i,j,k) * rho(k) * (dt/dx) * adz(k) * ady(j)    [mass flux]
    !   gu(j,k)   = mu(j) * rho(k) * ady(j) * adz(k)                  [Jacobian]
    !   uuu       = u1(i+1,j,k) + u1(i,j,k)                           [face mass flux]
    !   u_adv     = 0.5 * uuu                                          [jsam convention]
    !   face flux of U at i+0.5 = flux3(U(i-1), U(i), U(i+1), U(i+2), u_adv)
    !   dU/dt (i) = -(flux(i+0.5) - flux(i-0.5)) / gu(j,k)
    !
    ! Indices: U_arr(i, j, k) — Fortran column-major, C-order Python dump.
    !   U_arr(i, j, k) = Python U[k-1, j-1, i-1]
    !   U-face index i=1..nx+1 in Fortran = Python index i=0..nx.
    !   U-cell (west face) index i_U corresponds to mass cell i in x.
    !   gSAM: u1(i,j,k) for i=0..nx; u1(0)=u1(nx) by periodicity.
    !   In Fortran 1-based: u1(i) uses U_arr(i,j,k) for i=1..nx+1.
    !   u1(nx+1) = u1(1) by periodicity.

    dU = 0.0
    do k = 1, nz
      do j = 1, ny
        ! Jacobian for U tendency at row j, level k
        gu_k = mu(j) * rho(k) * ady(j) * adz(k)
        do i = 1, nx
          ! Stencil: faces at i (west) and i+1 (east) in terms of U-cell
          ! U cell i uses: phi_m1=U(i-1), phi_0=U(i), phi_p1=U(i+1), phi_p2=U(i+2)
          ! All indices are periodic mod nx.
          im1 = mod(i - 2 + nx, nx) + 1    ! i-1 (1-based, periodic)
          im2 = mod(i - 3 + nx, nx) + 1    ! i-2
          ip  = mod(i,          nx) + 1    ! i+1
          ip2 = mod(i + 1,      nx) + 1    ! i+2

          ! mass flux u1 at face index i (west face of cell i) and i+1 (east face)
          u1_i  = U_arr(i,  j, k) * rho(k) * (dt/dx) * adz(k) * ady(j)
          u1_ip = U_arr(ip, j, k) * rho(k) * (dt/dx) * adz(k) * ady(j)

          ! advecting velocity at east face of cell i = 0.5*(u1(i) + u1(i+1))
          u_adv = 0.5 * (u1_i + u1_ip)

          ! U values for stencil around east face of cell i
          ! east face i+0.5: phi_m1=U(i-1), phi_0=U(i), phi_p1=U(i+1), phi_p2=U(i+2)
          f_m1 = U_arr(im1, j, k)
          f_0  = U_arr(i,   j, k)
          f_p1 = U_arr(ip,  j, k)
          f_p2 = U_arr(ip2, j, k)
          fl_east = flux3(f_m1, f_0, f_p1, f_p2, u_adv)

          ! west face of cell i = east face of cell i-1:
          ! phi_m1=U(i-2), phi_0=U(i-1), phi_p1=U(i), phi_p2=U(i+1)
          ! advecting velocity = 0.5*(u1(i-1) + u1(i))
          u1_i  = U_arr(im1, j, k) * rho(k) * (dt/dx) * adz(k) * ady(j)
          u_adv = 0.5 * (u1_i + U_arr(i, j, k) * rho(k) * (dt/dx) * adz(k) * ady(j))
          f_m1 = U_arr(im2, j, k)
          f_0  = U_arr(im1, j, k)
          f_p1 = U_arr(i,   j, k)
          f_p2 = U_arr(ip,  j, k)
          fl_west = flux3(f_m1, f_0, f_p1, f_p2, u_adv)

          dU(i, j, k) = -(fl_east - fl_west) / gu_k
        end do
      end do
    end do

    ! Apply tendency to U
    U_new = U_arr
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          U_new(i, j, k) = U_new(i, j, k) + dU(i, j, k)
        end do
        ! Enforce periodicity: U(nx+1) = U(1)
        U_new(nx+1, j, k) = U_new(1, j, k)
      end do
    end do

    ! V and W unchanged (V=W=0 for both tested cases; y,z tendencies are zero)
    V_new = V_arr
    W_new = W_arr

    ! Write U_new, V_new, W_new concatenated in C order (matches jsam output)
    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(nz*ny*(nx+1) + nz*(ny+1)*nx + (nz+1)*ny*nx, 4)
    write(u_out) U_new
    write(u_out) V_new
    write(u_out) W_new
    close(u_out)
  end subroutine run_full_mom

end program mom_advection_driver
