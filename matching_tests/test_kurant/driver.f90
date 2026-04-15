! Fortran shadow of jsam/tests/unit/test_kurant.py
!
! Reads inputs that the .sh harness has dumped (U, V, W, dx, dy, dz, cos_lat,
! dt) and a small directive file that says which case to run, then writes the
! Fortran-side answer to fortran_out.bin.
!
! Two functions are exercised:
!   1. compute_cfl   — gSAM SRC/kurant.f90 cell-Courant kernel:
!                        cfll = sqrt((u*idx)^2 + (v*idy)^2 + (w*idz)^2)
!                        idx = dt/(dx*cos_lat),  idy = dt/dy,  idz = dt/dz
!   2. ab2_coefs     — variable-dt Adams-Bashforth coefficients (now AB3,
!                       matching gSAM SRC/abcoefs.f90 lines 15-30):
!                         nstep == 0  → Euler  (1, 0, 0)
!                         nstep == 1  → AB2    (at, bt, 0) bootstrap
!                         nstep >= 2  → AB3    (at, bt, ct)
!                       Output is (at, bt, ct) triples → length 3*nrec.
!
! For (1) we read the staggered (U, V, W) arrays directly. The python harness
! takes max over the two adjacent faces on its side and dumps a (nz, ny, nx)
! mass-centered absolute-velocity field, so this driver works on cell-centered
! magnitudes — identical to what jsam.compute_cfl does internally.
program kurant_driver
  implicit none
  character(len=64) :: case
  integer :: u_in, u_out

  call get_command_argument(1, case)

  select case (trim(case))
  case ('compute_cfl');  call run_cfl()
  case ('ab2_coefs');    call run_ab2()
  case default
    write(*,*) 'unknown case: ', trim(case);  stop 2
  end select

contains

  subroutine run_cfl()
    integer :: ndim, nz, ny, nx, k, j, i
    integer :: nz_v, ny_v, nx_v
    real, allocatable :: U_abs(:, :, :), V_abs(:, :, :), W_abs(:, :, :)
    real, allocatable :: dy(:), dz(:), cos_lat(:)
    real :: dx, dt, cfll, cflz1, cflh1_sq, idx, idy, idz, cfl_max
    integer :: u_in, u_out

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')

    ! Layout (matching dump_inputs.py for compute_cfl):
    !   int32 nz, int32 ny, int32 nx
    !   float32 U_abs(nz, ny, nx)        ! mass-centered |u|
    !   float32 V_abs(nz, ny, nx)
    !   float32 W_abs(nz, ny, nx)
    !   float32 dx                       ! scalar
    !   float32 dy(ny), dz(nz), cos_lat(ny)
    !   float32 dt                       ! scalar
    read(u_in) nz, ny, nx
    allocate(U_abs(nz, ny, nx), V_abs(nz, ny, nx), W_abs(nz, ny, nx))
    allocate(dy(ny), dz(nz), cos_lat(ny))
    read(u_in) U_abs
    read(u_in) V_abs
    read(u_in) W_abs
    read(u_in) dx
    read(u_in) dy, dz, cos_lat
    read(u_in) dt
    close(u_in)

    cfl_max = 0.0
    do k = 1, nz
      idz = dt / dz(k)
      do j = 1, ny
        idx = dt / (dx * max(cos_lat(j), 1e-6))
        idy = dt / dy(j)
        do i = 1, nx
          cflz1   = abs(W_abs(k, j, i)) * idz
          cflh1_sq = (U_abs(k, j, i) * idx) ** 2 + (V_abs(k, j, i) * idy) ** 2
          cfll    = sqrt(cflh1_sq + cflz1 ** 2)
          if (cfll > cfl_max) cfl_max = cfll
        end do
      end do
    end do

    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) 1_4
    write(u_out) cfl_max
    close(u_out)
  end subroutine run_cfl

  subroutine run_ab2()
    ! gSAM SRC/abcoefs.f90 lines 15-30 — full AB3 state machine.
    integer :: nstep, nrec, k
    real :: dt_curr, dt_prev, dt_pprev, at, bt, ct, alpha, beta
    real, allocatable :: out(:)
    integer :: u_in, u_out

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    ! Layout: int32 nrec; then nrec records of
    !   (int32 nstep, float32 dt_curr, float32 dt_prev, float32 dt_pprev)
    read(u_in) nrec
    allocate(out(3 * nrec))
    do k = 1, nrec
      read(u_in) nstep, dt_curr, dt_prev, dt_pprev
      if (nstep == 0) then                       ! Euler bootstrap
        at = 1.0;  bt = 0.0;  ct = 0.0
      else if (nstep == 1) then                  ! AB2 bootstrap
        alpha = dt_prev / dt_curr
        at = (1.0 + 2.0*alpha) / (2.0*alpha)
        bt = -1.0 / (2.0*alpha)
        ct = 0.0
      else                                        ! AB3 (gSAM abcoefs.f90:15-20)
        alpha = dt_prev  / dt_curr
        beta  = dt_pprev / dt_curr
        ct    = (2.0 + 3.0*alpha) / (6.0*(alpha + beta)*beta)
        bt    = -(1.0 + 2.0*(alpha + beta)*ct) / (2.0*alpha)
        at    = 1.0 - bt - ct
      end if
      out(3*k - 2) = at
      out(3*k - 1) = bt
      out(3*k    ) = ct
    end do
    close(u_in)

    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(3 * nrec, 4)
    write(u_out) out
    close(u_out)
  end subroutine run_ab2

end program kurant_driver
