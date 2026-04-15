! Fortran shadow of jsam/tests/unit/test_nudging.py
!
! Two modes:
!   band_mask     — reads (nz, z(nz), z1, z2)
!                   writes mask(nz) where mask(k) = 1 if z1 <= z(k) <= z2, else 0
!
!   nudge_scalar  — reads (nz, ny, nx, phi(nz,ny,nx), ref(nz), z(nz), dt, z1, z2, tau)
!                   writes phi_new(nz,ny,nx) flattened (C order)
!
! Binary output format (matching common/bin_io.py):
!   write(u_out) 1_4          ! ndim
!   write(u_out) int(N, 4)    ! size
!   write(u_out) arr          ! float32 data
!
program nudging_driver
  implicit none
  character(len=64) :: mode

  call get_command_argument(1, mode)

  select case (trim(mode))
  case ('band_mask');     call run_band_mask()
  case ('band_mask_full');   call run_band_mask()
  case ('band_mask_partial'); call run_band_mask()
  case ('nudge_decay');   call run_nudge_scalar()
  case ('nudge_zero_outside'); call run_nudge_scalar()
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select

contains

  ! -------------------------------------------------------------------
  subroutine run_band_mask()
    integer :: nz, k, u_in, u_out
    real, allocatable :: z(:), mask(:)
    real :: z1, z2

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    ! Layout: int32 nz, float32 z(nz), float32 z1, float32 z2
    read(u_in) nz
    allocate(z(nz), mask(nz))
    read(u_in) z
    read(u_in) z1, z2
    close(u_in)

    do k = 1, nz
      if (z(k) >= z1 .and. z(k) <= z2) then
        mask(k) = 1.0
      else
        mask(k) = 0.0
      end if
    end do

    open(newunit=u_out, file='fortran_out.bin', access='stream', &
         form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(nz, 4)
    write(u_out) mask
    close(u_out)
  end subroutine run_band_mask

  ! -------------------------------------------------------------------
  subroutine run_nudge_scalar()
    integer :: nz, ny, nx, k, j, i, u_in, u_out
    real, allocatable :: phi(:,:,:), ref(:), z(:), phi_new(:,:,:), mask(:)
    real :: dt, z1, z2, tau, coef, coef_k

    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    ! Layout: int32 nz, ny, nx; float32 phi(nz,ny,nx) C-order; float32 ref(nz);
    !         float32 z(nz); float32 dt, z1, z2, tau
    read(u_in) nz, ny, nx
    allocate(phi(nx, ny, nz))   ! read in C-major (z-slowest) order as F col-major
    allocate(ref(nz), z(nz), mask(nz))
    allocate(phi_new(nx, ny, nz))

    ! Python writes row-major (C order): slowest index is k (nz), then j (ny), then i (nx).
    ! In Fortran stream-access the bytes come in as: i varies fastest, then j, then k.
    ! That matches Fortran array phi(nx, ny, nz) where the first index is fastest.
    read(u_in) phi
    read(u_in) ref
    read(u_in) z
    read(u_in) dt, z1, z2, tau
    close(u_in)

    ! band mask
    do k = 1, nz
      if (z(k) >= z1 .and. z(k) <= z2) then
        mask(k) = 1.0
      else
        mask(k) = 0.0
      end if
    end do

    ! nudge: phi_new = phi + clip(dt/tau,0,1) * mask(k) * (ref(k) - phi)
    coef = min(max(dt / tau, 0.0), 1.0)
    do k = 1, nz
      coef_k = coef * mask(k)
      do j = 1, ny
        do i = 1, nx
          phi_new(i, j, k) = phi(i, j, k) + coef_k * (ref(k) - phi(i, j, k))
        end do
      end do
    end do

    open(newunit=u_out, file='fortran_out.bin', access='stream', &
         form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(nz * ny * nx, 4)
    write(u_out) phi_new
    close(u_out)
  end subroutine run_nudge_scalar

end program nudging_driver
