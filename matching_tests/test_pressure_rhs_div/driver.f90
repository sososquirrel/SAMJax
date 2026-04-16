! test_pressure_rhs_div/driver.f90
!
! Standalone port of gSAM SRC/press_rhs.f90 (the 3D branch),
! computing the RHS of the pressure Poisson equation:
!
!   ppp(i,j,k) = ( rdx*(u(ic,j,k)-u(i,j,k))
!                + rdy*(muv(jc)*v(i,jc,k)-muv(j)*v(i,j,k))
!                + (w(i,j,kc)*rup - w(i,j,k)*rdn) ) * dta
!
! where:
!   rdx  = imu(j)/dx
!   rdy  = imu(j)/(dy*ady(j))
!   rup  = rhow(kc)/rho(k) / (adz(k)*dz)
!   rdn  = rhow(k) /rho(k) / (adz(k)*dz)
!   dta  = 1/(dt*at)
!
! Binary inputs.bin layout:
!   i4 nz, ny, nx
!   f4 dx, dy, dz, dt_at     [dt_at = dt * at]
!   f4 imu(ny), ady(ny), muv(ny+1), adz(nz)
!   f4 rho(nz), rhow(nz+1)
!   f4 U(nz,ny,nx+1), V(nz,ny+1,nx), W(nz+1,ny,nx)    [C order]
!
! Output fortran_out.bin:
!   f4 ppp(nz,ny,nx)    [C order]

program prhs_driver
  implicit none
  integer(4) :: nz, ny, nx
  real :: dx_s, dy_s, dz_s, dt_at
  integer :: i, j, k, ic, jc, kc, n_out, idx, u_in, u_out
  real :: rdx, rdy, rdz, rup, rdn, dta
  real, allocatable :: imu(:), ady(:), muv(:), adz(:), rho(:), rhow(:)
  real, allocatable :: U(:,:,:), V(:,:,:), W(:,:,:), ppp(:,:,:)
  real, allocatable :: buf(:)

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  read(u_in) nz, ny, nx
  read(u_in) dx_s, dy_s, dz_s, dt_at

  allocate(imu(ny), ady(ny), muv(ny+1), adz(nz), rho(nz), rhow(nz+1))
  read(u_in) imu
  read(u_in) ady
  read(u_in) muv
  read(u_in) adz
  read(u_in) rho
  read(u_in) rhow
  close(u_in)

  ! Read staggered velocity fields
  allocate(U(nx+1, ny, nz))
  call read_carray_stag(10, 'inputs.bin', U, nz, ny, nx+1, &
       4*(3 + 4 + ny + ny + (ny+1) + nz + nz + (nz+1)))

  ! Reopen for V and W with correct offsets
  ! ... Actually, let's just read sequentially from a second pass.
  ! Simpler: read all velocities from a separate velocity file.
  ! For simplicity, read from inputs.bin with explicit positioning.

  ! Re-read the file fully
  block
    integer :: u2
    integer :: hdr_bytes
    real, allocatable :: u_buf(:), v_buf(:), w_buf(:)

    hdr_bytes = 4*3 + 4*4 + 4*ny + 4*ny + 4*(ny+1) + 4*nz + 4*nz + 4*(nz+1)

    open(newunit=u2, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    ! Skip header
    read(u2, pos=hdr_bytes+1)  ! position past header

    ! Read U: (nz, ny, nx+1) in C order → Fortran (nx+1, ny, nz)
    allocate(u_buf(nz*ny*(nx+1)))
    read(u2) u_buf
    call c_to_f(u_buf, U, nz, ny, nx+1)

    ! Read V: (nz, ny+1, nx) in C order → Fortran (nx, ny+1, nz)
    allocate(V(nx, ny+1, nz))
    allocate(v_buf(nz*(ny+1)*nx))
    read(u2) v_buf
    call c_to_f_v(v_buf, V, nz, ny+1, nx)

    ! Read W: (nz+1, ny, nx) in C order → Fortran (nx, ny, nz+1)
    allocate(W(nx, ny, nz+1))
    allocate(w_buf((nz+1)*ny*nx))
    read(u2) w_buf
    call c_to_f_w(w_buf, W, nz+1, ny, nx)

    close(u2)
  end block

  allocate(ppp(nx, ny, nz))
  dta = 1.0 / dt_at

  do k = 1, nz
    kc = k + 1
    rdz = 1.0 / (adz(k) * dz_s)
    rup = rhow(kc) / rho(k) * rdz
    rdn = rhow(k)  / rho(k) * rdz
    do j = 1, ny
      jc = j + 1
      rdx = imu(j) / dx_s
      rdy = imu(j) / (dy_s * ady(j))
      do i = 1, nx
        ic = i + 1
        ppp(i,j,k) = ( rdx * (U(ic,j,k) - U(i,j,k)) &
                      + rdy * (muv(jc)*V(i,jc,k) - muv(j)*V(i,j,k)) &
                      + (W(i,j,kc)*rup - W(i,j,k)*rdn) ) * dta
      end do
    end do
  end do

  ! Write output
  n_out = nz * ny * nx
  allocate(buf(n_out))
  idx = 0
  do k = 1, nz
    do j = 1, ny
      do i = 1, nx
        idx = idx + 1
        buf(idx) = ppp(i, j, k)
      end do
    end do
  end do

  open(newunit=u_out, file='fortran_out.bin', access='stream', &
       form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) int(n_out, 4)
  write(u_out) buf
  close(u_out)

contains

  subroutine c_to_f(cbuf, a, d1, d2, d3)
    ! C order (d1,d2,d3) i3 fastest → Fortran (d3,d2,d1)
    real, intent(in) :: cbuf(*)
    real, intent(out) :: a(d3, d2, d1)
    integer, intent(in) :: d1, d2, d3
    integer :: i1, i2, i3, pos
    pos = 0
    do i1 = 1, d1
      do i2 = 1, d2
        do i3 = 1, d3
          pos = pos + 1
          a(i3, i2, i1) = cbuf(pos)
        end do
      end do
    end do
  end subroutine c_to_f

  subroutine c_to_f_v(cbuf, a, d1, d2, d3)
    real, intent(in) :: cbuf(*)
    real, intent(out) :: a(d3, d2, d1)
    integer, intent(in) :: d1, d2, d3
    integer :: i1, i2, i3, pos
    pos = 0
    do i1 = 1, d1
      do i2 = 1, d2
        do i3 = 1, d3
          pos = pos + 1
          a(i3, i2, i1) = cbuf(pos)
        end do
      end do
    end do
  end subroutine c_to_f_v

  subroutine c_to_f_w(cbuf, a, d1, d2, d3)
    real, intent(in) :: cbuf(*)
    real, intent(out) :: a(d3, d2, d1)
    integer, intent(in) :: d1, d2, d3
    integer :: i1, i2, i3, pos
    pos = 0
    do i1 = 1, d1
      do i2 = 1, d2
        do i3 = 1, d3
          pos = pos + 1
          a(i3, i2, i1) = cbuf(pos)
        end do
      end do
    end do
  end subroutine c_to_f_w

  subroutine read_carray_stag(unit_num, fname, a, d1, d2, d3, skip_bytes)
    integer, intent(in) :: unit_num, d1, d2, d3, skip_bytes
    character(*), intent(in) :: fname
    real, intent(out) :: a(d3, d2, d1)
    ! placeholder — actual reading done in main block
  end subroutine read_carray_stag

end program prhs_driver
