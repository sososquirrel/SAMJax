! test_w_courant_clip/driver.f90
!
! Standalone port of gSAM damping.f90 §1 (dodamping: W sponge) and
! §2 (dodamping_w: W CFL limiter below sponge).
!
! Algorithm:
!   1. Compute taudamp(k) from model height profile:
!      nu = (zi(k) - zi(1)) / (zi(nzm) - zi(1))
!      if nu > nub:  zzz = 100*((nu-nub)/(1-nub))^2
!                    taudamp = 0.333 * tau_max * zzz / (1+zzz)
!   2. Sponge: where taudamp > 0:
!      W(k) = W(k) / (1 + taudamp(k))
!   3. CFL limiter: where taudamp == 0:
!      wmax = damping_w_cu * dz * adzw(k) / dtn
!      W(k) = (W(k) + min(wmax, max(-wmax, W(k))) * tau_max) / (1 + tau_max)
!
! Binary inputs.bin layout:
!   i4 nz, ny, nx
!   f4 dtn, dt_base, dz_s
!   f4 nub, damping_w_cu
!   f4 zi(nz+1)        [interface heights]
!   f4 adzw(nz)        [dz at W-face / dz_ref]
!   f4 W(nz+1,ny,nx)   [C order]
!
! Output fortran_out.bin:
!   f4 W_new(nz+1,ny,nx) + taudamp(nz)    [for diagnostics]

program wclip_driver
  implicit none
  integer(4) :: nz, ny, nx
  real :: dtn, dt_base, dz_s, nub_val, damping_w_cu
  integer :: i, j, k, n_out, idx, u_in, u_out
  real :: tau_max, nu, zzz, wmax1, wmax
  real, allocatable :: zi(:), adzw(:), W(:,:,:), taudamp(:)
  real, allocatable :: buf(:)

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  read(u_in) nz, ny, nx
  read(u_in) dtn, dt_base, dz_s
  read(u_in) nub_val, damping_w_cu

  allocate(zi(nz+1), adzw(nz), taudamp(nz))
  read(u_in) zi
  read(u_in) adzw

  ! Read W: (nz+1, ny, nx) C-order
  allocate(W(nx, ny, nz+1))
  block
    real, allocatable :: tmp(:)
    integer :: ii, jj, kk, pos
    allocate(tmp((nz+1)*ny*nx))
    read(u_in) tmp
    pos = 0
    do kk = 1, nz+1
      do jj = 1, ny
        do ii = 1, nx
          pos = pos + 1
          W(ii, jj, kk) = tmp(pos)
        end do
      end do
    end do
  end block
  close(u_in)

  tau_max = dtn / dt_base

  ! --- Step 1: Compute taudamp ---
  taudamp = 0.0
  do k = 1, nz
    nu = (zi(k) - zi(1)) / (zi(nz+1) - zi(1))
    if (nu > nub_val) then
      zzz = 100.0 * ((nu - nub_val) / (1.0 - nub_val))**2
      taudamp(k) = 0.333 * tau_max * zzz / (1.0 + zzz)
    end if
  end do

  ! --- Step 2: Sponge damping (where taudamp > 0) ---
  do k = 1, nz
    if (taudamp(k) > 0.0) then
      do j = 1, ny
        do i = 1, nx
          W(i, j, k) = W(i, j, k) / (1.0 + taudamp(k))
        end do
      end do
    end if
  end do

  ! --- Step 3: CFL limiter (where taudamp == 0) ---
  do k = 1, nz
    if (taudamp(k) == 0.0) then
      wmax1 = damping_w_cu * dz_s * adzw(k) / dtn
      do j = 1, ny
        wmax = wmax1
        do i = 1, nx
          W(i,j,k) = (W(i,j,k) + min(wmax, max(-wmax, W(i,j,k))) * tau_max) &
                    / (1.0 + tau_max)
        end do
      end do
    end if
  end do

  ! --- Write output: W_new + taudamp ---
  n_out = (nz+1)*ny*nx + nz
  allocate(buf(n_out))
  idx = 0
  do k = 1, nz+1
    do j = 1, ny
      do i = 1, nx
        idx = idx + 1
        buf(idx) = W(i, j, k)
      end do
    end do
  end do
  do k = 1, nz
    idx = idx + 1
    buf(idx) = taudamp(k)
  end do

  open(newunit=u_out, file='fortran_out.bin', access='stream', &
       form='unformatted', status='replace')
  write(u_out) 1_4
  write(u_out) int(n_out, 4)
  write(u_out) buf
  close(u_out)

end program wclip_driver
