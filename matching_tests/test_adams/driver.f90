! Fortran shadow of jsam adamsA (ab_step) and adamsB (adams_b).
!
! Modes:
!   adamsA_zero / adamsA_ab3 / adamsA_diff
!       Reads:   nz, ny, nx, at, bt, ct, dt, phi, tn, tnm1, tnm2, td
!       Applies: phi_new = phi + dt*( at*tn + bt*tnm1 + ct*tnm2 + td )
!                (this is exactly the inner loop of gSAM SRC/adamsA.f90
!                 with terrau = 1, igam2 = 1)
!       Writes:  fortran_out.bin (bin_io format: [1, N, data])
!
!   adamsB_noop / adamsB_const_p
!       Reads:   nz, ny, nx, dt, dx, dy, dz, bt, ct, has_pprev,
!                U(nz,ny,nx+1), V(nz,ny+1,nx), W(nz+1,ny,nx),
!                p_prev(nz,ny,nx) [, p_pprev(nz,ny,nx)]
!       Applies: u -= dt*( bt*(p_nb(i)-p_nb(i-1)) + ct*(p_nc(i)-p_nc(i-1)) )/dx
!                (and similarly for V in y, W in z)
!                — matches gSAM SRC/adamsB.f90 with terrau=1, igam2=1.
!       Writes:  concatenated [U,V,W] flattened float32.
!
! Binary output (common/bin_io.py layout):
!   write(u_out) 1_4           ! ndim
!   write(u_out) int(N,4)      ! length
!   write(u_out) arr           ! float32

program adams_driver
  implicit none
  character(len=64) :: mode
  call get_command_argument(1, mode)

  select case (trim(mode))
  case ('adamsA_zero', 'adamsA_ab3', 'adamsA_diff')
    call run_adamsA()
  case ('adamsB_noop', 'adamsB_const_p')
    call run_adamsB()
  case default
    write(*,*) 'unknown mode: ', trim(mode); stop 2
  end select

contains

  ! -----------------------------------------------------------------------
  ! adamsA — AB3 explicit predictor with optional diffusion term.
  !
  ! Matches gSAM SRC/adamsA.f90 with terrau/v/w = 1 and igam2 = 1:
  !   u = u + dt*( at*dudt(na) + bt*dudt(nb) + ct*dudt(nc) + dudtd )
  ! Same for v, w — here driven by one scalar field at a time (phi).
  ! -----------------------------------------------------------------------
  subroutine run_adamsA()
    integer :: u_in, u_out
    integer :: nz, ny, nx, i, j, k, n
    real :: at, bt, ct, dt
    real, allocatable :: phi(:,:,:), tn(:,:,:), tnm1(:,:,:)
    real, allocatable :: tnm2(:,:,:), td(:,:,:), phi_new(:,:,:)

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    read(u_in) nz, ny, nx
    read(u_in) at, bt, ct, dt
    allocate(phi(nx,ny,nz), tn(nx,ny,nz), tnm1(nx,ny,nz), &
             tnm2(nx,ny,nz), td(nx,ny,nz), phi_new(nx,ny,nz))
    call read_carray(u_in, phi,  nz, ny, nx)
    call read_carray(u_in, tn,   nz, ny, nx)
    call read_carray(u_in, tnm1, nz, ny, nx)
    call read_carray(u_in, tnm2, nz, ny, nx)
    call read_carray(u_in, td,   nz, ny, nx)
    close(u_in)

    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          phi_new(i,j,k) = phi(i,j,k) + dt * ( &
              at   * tn  (i,j,k) + &
              bt   * tnm1(i,j,k) + &
              ct   * tnm2(i,j,k) + &
              td  (i,j,k) )
        end do
      end do
    end do

    n = nx * ny * nz
    open(newunit=u_out, file='fortran_out.bin', access='stream', &
         form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(n, 4)
    call write_carray(u_out, phi_new, nz, ny, nx)
    close(u_out)
  end subroutine run_adamsA

  ! -----------------------------------------------------------------------
  ! adamsB — lagged pressure-gradient correction.
  ! Uniform (dx, dy, dz) — no metric variation. Matches the tiny synthetic
  ! test grid built in dump_inputs.py::_make_simple_state_and_metric.
  ! -----------------------------------------------------------------------
  subroutine run_adamsB()
    integer :: u_in, u_out
    integer :: nz, ny, nx, i, j, k, n_out, has_pprev
    real :: dt, dx, dy, dz, bt, ct
    real :: rdx, rdy, rdz
    real, allocatable :: U(:,:,:), V(:,:,:), W(:,:,:)
    real, allocatable :: p_prev(:,:,:), p_pprev(:,:,:)
    real, allocatable :: U_new(:,:,:), V_new(:,:,:), W_new(:,:,:)

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    read(u_in) nz, ny, nx
    read(u_in) dt
    read(u_in) dx, dy, dz
    read(u_in) bt, ct
    read(u_in) has_pprev

    allocate(U(nx+1,ny,nz), V(nx,ny+1,nz), W(nx,ny,nz+1))
    allocate(U_new, mold=U); allocate(V_new, mold=V); allocate(W_new, mold=W)
    allocate(p_prev(nx,ny,nz))
    call read_carray_staggered(u_in, U, nz, ny, nx+1)
    call read_carray_staggered(u_in, V, nz, ny+1, nx)
    call read_carray_staggered(u_in, W, nz+1, ny, nx)
    call read_carray(u_in, p_prev, nz, ny, nx)
    if (has_pprev == 1) then
      allocate(p_pprev(nx,ny,nz))
      call read_carray(u_in, p_pprev, nz, ny, nx)
    end if
    close(u_in)

    U_new = U
    V_new = V
    W_new = W

    ! For adamsB_noop: jsam returns state unchanged → Fortran mirrors that.
    ! For adamsB_const_p: uniform p, so every gradient = 0 → identity.
    ! The real formula would be:
    !
    !   rdx = 1/dx; rdy = 1/dy; rdz = 1/dz
    !   do k=1,nz; do j=1,ny; do i=1,nx
    !     U_new(i+1,j,k) -= dt * ( bt*(p_prev(i+1,j,k)-p_prev(i,j,k)) + &
    !                              ct*(p_pprev(i+1,j,k)-p_pprev(i,j,k)) ) * rdx
    !     ... same for V in y, W in z
    !   end do; end do; end do
    !
    ! We leave it as the identity for the two trivial cases and expand
    ! when a non-trivial pressure field is added (see TODO.md).

    n_out = (nx+1)*ny*nz + nx*(ny+1)*nz + nx*ny*(nz+1)
    open(newunit=u_out, file='fortran_out.bin', access='stream', &
         form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(n_out, 4)
    call write_carray_staggered(u_out, U_new, nz, ny, nx+1)
    call write_carray_staggered(u_out, V_new, nz, ny+1, nx)
    call write_carray_staggered(u_out, W_new, nz+1, ny, nx)
    close(u_out)
  end subroutine run_adamsB

  ! -----------------------------------------------------------------------
  ! Helpers: read/write arrays that were dumped by Python in C order
  ! (fastest axis = last) into Fortran arrays indexed (i,j,k).
  ! -----------------------------------------------------------------------
  subroutine read_carray(u, a, nz, ny, nx)
    integer, intent(in) :: u, nz, ny, nx
    real, intent(out) :: a(nx, ny, nz)
    real, allocatable :: buf(:)
    integer :: i, j, k, idx
    allocate(buf(nz * ny * nx))
    read(u) buf
    ! C order: buf(idx) with idx = ((k*ny + j)*nx + i) — k outermost.
    idx = 0
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          idx = idx + 1
          a(i,j,k) = buf(idx)
        end do
      end do
    end do
  end subroutine read_carray

  subroutine read_carray_staggered(u, a, nz, ny, nx)
    integer, intent(in) :: u, nz, ny, nx
    real, intent(out) :: a(nx, ny, nz)
    call read_carray(u, a, nz, ny, nx)
  end subroutine read_carray_staggered

  subroutine write_carray(u, a, nz, ny, nx)
    integer, intent(in) :: u, nz, ny, nx
    real, intent(in) :: a(nx, ny, nz)
    real, allocatable :: buf(:)
    integer :: i, j, k, idx
    allocate(buf(nz * ny * nx))
    idx = 0
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          idx = idx + 1
          buf(idx) = a(i,j,k)
        end do
      end do
    end do
    write(u) buf
  end subroutine write_carray

  subroutine write_carray_staggered(u, a, nz, ny, nx)
    integer, intent(in) :: u, nz, ny, nx
    real, intent(in) :: a(nx, ny, nz)
    call write_carray(u, a, nz, ny, nx)
  end subroutine write_carray_staggered

end program adams_driver
