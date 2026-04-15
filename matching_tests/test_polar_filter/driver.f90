! Fortran shadow of jsam/tests/unit/test_polar_filter.py
!
! Implements the SAME FFT-based Fourier polar filter as jsam's
! polar_fourier_filter (NOT the gSAM box smoother — see TODO.md).
! This is a self-consistency test: Fortran DFT == jsam rfft/irfft.
!
! Algorithm:
!   For each row j:
!     m_max(j) = max(1, floor(nx/2 * |cos(lat(j))|))
!     Zero spectral modes m > m_max via brute-force DFT
!
! Array layout: Python C row-major shape (nz,ny,nx) → Fortran (nx,ny,nz)
! so column-major Fortran storage matches Python bytes.
!
! inputs.bin:
!   int32 nz, int32 ny, int32 nx
!   float32 lat_rad(ny)
!   float32 field(nz,ny,nx) [C order]
!
! fortran_out.bin: float32 field_filtered(nz,ny,nx) [same layout]

program polar_filter_driver
  implicit none
  character(len=64) :: mode

  call get_command_argument(1, mode)

  select case (trim(mode))
  case ('polar_fourier_filter');  call run_filter()
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select

contains

  subroutine run_filter()
    integer  :: nz, ny, nx, u_in, u_out, N
    ! Reversed axes: Python (nz,ny,nx) → Fortran (nx,ny,nz)
    real, allocatable :: lat_rad(:), field(:,:,:), out_field(:,:,:)
    integer :: k, j, i, m, m_max_j
    real    :: cos_lat_j, pi, half_nx
    double precision :: re, angle
    double precision, allocatable :: Re_hat(:), Im_hat(:)

    pi = 4.0 * atan(1.0)

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    allocate(lat_rad(ny), field(nx, ny, nz), out_field(nx, ny, nz))
    read(u_in) lat_rad
    read(u_in) field
    close(u_in)

    allocate(Re_hat(0:nx/2), Im_hat(0:nx/2))

    half_nx = real(nx) / 2.0

    ! Loop: Fortran field(i,j,k) = Python field[k-1,j-1,i-1]
    ! The filter acts on each (j,k) row of x-points, i=1..nx
    do k = 1, nz
      do j = 1, ny
        cos_lat_j = abs(cos(lat_rad(j)))
        m_max_j   = max(1, int(half_nx * cos_lat_j))   ! floor

        ! Forward DFT (real input row i=1..nx → complex spectrum m=0..nx/2)
        do m = 0, nx/2
          Re_hat(m) = 0.0d0
          Im_hat(m) = 0.0d0
          do i = 1, nx
            angle = 2.0d0 * pi * m * (i - 1) / nx
            Re_hat(m) = Re_hat(m) + field(i, j, k) * cos(angle)
            Im_hat(m) = Im_hat(m) - field(i, j, k) * sin(angle)
          end do
          ! Zero modes above m_max
          if (m > m_max_j) then
            Re_hat(m) = 0.0d0
            Im_hat(m) = 0.0d0
          end if
        end do

        ! Inverse DFT: reconstruct real signal from truncated Hermitian spectrum
        do i = 1, nx
          re = Re_hat(0)   ! m=0 term (DC)
          do m = 1, nx/2 - 1
            angle = 2.0d0 * pi * m * (i - 1) / nx
            re = re + 2.0d0 * (Re_hat(m) * cos(angle) - Im_hat(m) * sin(angle))
          end do
          ! Nyquist term m=nx/2 (real-only for even nx)
          if (mod(nx, 2) == 0) then
            angle = 2.0d0 * pi * (nx/2) * (i - 1) / nx
            re = re + Re_hat(nx/2) * cos(angle) - Im_hat(nx/2) * sin(angle)
          end if
          out_field(i, j, k) = real(re / nx)
        end do
      end do
    end do

    N = nx * ny * nz
    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(N, 4)
    write(u_out) out_field
    close(u_out)
  end subroutine run_filter

end program polar_filter_driver
