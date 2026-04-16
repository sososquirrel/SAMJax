! test_adamsb_p_buffer/driver.f90
!
! Standalone port of gSAM adamsB.f90 — lagged pressure gradient
! correction applied to U, V, W.
!
! gSAM formula:
!   u(i,j,k) -= dt * ( bt*(p(i,j,k,nb)-p(ib,j,k,nb))
!                     + ct*(p(i,j,k,nc)-p(ib,j,k,nc)) ) * rdx
!   v(i,j,k) -= dt * ( bt*(p(i,j,k,nb)-p(i,jb,k,nb))
!                     + ct*(p(i,j,k,nc)-p(i,jb,k,nc)) ) * rdy
!   w(i,j,k) -= dt * ( bt*(p(i,j,k,nb)-p(i,j,kb,nb))
!                     + ct*(p(i,j,k,nc)-p(i,j,kb,k,nc)) ) * rdz
!
! Binary inputs.bin layout:
!   i4 nz, ny, nx
!   f4 dt, bt, ct
!   f4 rdx, rdy(ny), rdz(nz)
!   f4 U(nz,ny,nx+1)      [C order, staggered]
!   f4 V(nz,ny+1,nx)
!   f4 W(nz+1,ny,nx)
!   f4 p_prev(nz,ny,nx)   [pressure at n-1]
!   f4 p_pprev(nz,ny,nx)  [pressure at n-2, may be all zeros]
!
! Output fortran_out.bin:
!   f4 U_new(nz,ny,nx+1) + V_new(nz,ny+1,nx) + W_new(nz+1,ny,nx)

program pb_driver
  implicit none
  integer(4) :: nz, ny, nx
  real :: dt_s, bt, ct, rdx_s
  integer :: i, j, k, ib, jb, kb, u_in, u_out, n_out, idx
  real, allocatable :: rdy(:), rdz(:)
  real, allocatable :: U(:,:,:), V(:,:,:), W(:,:,:)
  real, allocatable :: pp(:,:,:), ppp(:,:,:)  ! p_prev, p_pprev
  real, allocatable :: buf(:)
  real :: dpx, dpy, dpz

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  read(u_in) nz, ny, nx
  read(u_in) dt_s, bt, ct, rdx_s

  allocate(rdy(ny), rdz(nz))
  read(u_in) rdy
  read(u_in) rdz

  ! Read staggered velocities and pressure fields
  allocate(U(nx+1, ny, nz), V(nx, ny+1, nz), W(nx, ny, nz+1))
  allocate(pp(nx, ny, nz), ppp(nx, ny, nz))

  block
    real, allocatable :: tmp(:)
    integer :: pos, ii, jj, kk

    ! U: (nz, ny, nx+1) C-order
    allocate(tmp(nz*ny*(nx+1)))
    read(u_in) tmp
    pos = 0
    do kk = 1, nz; do jj = 1, ny; do ii = 1, nx+1
      pos = pos + 1; U(ii,jj,kk) = tmp(pos)
    end do; end do; end do
    deallocate(tmp)

    ! V: (nz, ny+1, nx) C-order
    allocate(tmp(nz*(ny+1)*nx))
    read(u_in) tmp
    pos = 0
    do kk = 1, nz; do jj = 1, ny+1; do ii = 1, nx
      pos = pos + 1; V(ii,jj,kk) = tmp(pos)
    end do; end do; end do
    deallocate(tmp)

    ! W: (nz+1, ny, nx) C-order
    allocate(tmp((nz+1)*ny*nx))
    read(u_in) tmp
    pos = 0
    do kk = 1, nz+1; do jj = 1, ny; do ii = 1, nx
      pos = pos + 1; W(ii,jj,kk) = tmp(pos)
    end do; end do; end do
    deallocate(tmp)

    ! p_prev, p_pprev: (nz, ny, nx) C-order
    allocate(tmp(nz*ny*nx))
    read(u_in) tmp
    pos = 0
    do kk = 1, nz; do jj = 1, ny; do ii = 1, nx
      pos = pos + 1; pp(ii,jj,kk) = tmp(pos)
    end do; end do; end do

    read(u_in) tmp
    pos = 0
    do kk = 1, nz; do jj = 1, ny; do ii = 1, nx
      pos = pos + 1; ppp(ii,jj,kk) = tmp(pos)
    end do; end do; end do
    deallocate(tmp)
  end block
  close(u_in)

  ! --- Apply adamsB correction ---
  ! U correction: dp/dx at east faces
  do k = 1, nz
    do j = 1, ny
      do i = 2, nx  ! interior U-faces (skip periodic boundaries)
        ib = i - 1
        dpx = bt * (pp(i,j,k) - pp(ib,j,k)) + ct * (ppp(i,j,k) - ppp(ib,j,k))
        U(i,j,k) = U(i,j,k) - dt_s * dpx * rdx_s
      end do
      ! Periodic: i=1 wraps to ib=nx, i=nx+1 wraps to ib=1
      dpx = bt * (pp(1,j,k) - pp(nx,j,k)) + ct * (ppp(1,j,k) - ppp(nx,j,k))
      U(1,j,k) = U(1,j,k) - dt_s * dpx * rdx_s
      U(nx+1,j,k) = U(1,j,k)  ! periodic wrap
    end do
  end do

  ! V correction: dp/dy at north faces
  do k = 1, nz
    do j = 2, ny  ! interior V-faces; j=1 and j=ny+1 are walls
      jb = j - 1
      do i = 1, nx
        dpy = bt * (pp(i,j,k) - pp(i,jb,k)) + ct * (ppp(i,j,k) - ppp(i,jb,k))
        V(i,j,k) = V(i,j,k) - dt_s * dpy * rdy(j)
      end do
    end do
  end do

  ! W correction: dp/dz at top faces
  do k = 2, nz  ! interior W-faces; k=1 (ground) and k=nz+1 (lid) fixed
    kb = k - 1
    do j = 1, ny
      do i = 1, nx
        dpz = bt * (pp(i,j,k) - pp(i,j,kb)) + ct * (ppp(i,j,k) - ppp(i,j,kb))
        W(i,j,k) = W(i,j,k) - dt_s * dpz * rdz(k)
      end do
    end do
  end do

  ! --- Write output ---
  n_out = nz*ny*(nx+1) + nz*(ny+1)*nx + (nz+1)*ny*nx
  allocate(buf(n_out))
  idx = 0

  do k=1,nz; do j=1,ny; do i=1,nx+1
    idx=idx+1; buf(idx)=U(i,j,k)
  end do; end do; end do

  do k=1,nz; do j=1,ny+1; do i=1,nx
    idx=idx+1; buf(idx)=V(i,j,k)
  end do; end do; end do

  do k=1,nz+1; do j=1,ny; do i=1,nx
    idx=idx+1; buf(idx)=W(i,j,k)
  end do; end do; end do

  open(newunit=u_out, file='fortran_out.bin', access='stream', &
       form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) int(n_out, 4)
  write(u_out) buf
  close(u_out)

end program pb_driver
