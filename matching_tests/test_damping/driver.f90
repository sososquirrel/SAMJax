! Fortran shadow of jsam/tests/unit/test_damping.py
!
! Implements the gSAM dodamping_poles kernel for U and V:
!   tau(j) = (1 - cos²(lat(j)))^200
!   umax(j) = cu * dx * cos(lat(j)) / dt
!   phi_new = (phi + clip(phi, -umax, umax) * tau) / (1 + tau)
!
! Two modes:
!   pole_u : apply to U (nz, ny, nx+1)
!   pole_v : apply to V (nz, ny+1, nx)
!
! Array layout note: Python writes arrays in C row-major order.  For a
! Python array of shape (nz, ny, nx) the bytes are [k=0,j=0,i=0],
! [k=0,j=0,i=1], ..., [k=0,j=1,i=0], etc.  Fortran reads stream data in
! column-major order (first index varies fastest).  To match, Fortran arrays
! are declared with axes REVERSED relative to the Python shape:
!   Python (nz, ny, nx+1) → Fortran U(nx+1, ny, nz)
! so that element U[k,j,i] in Python == U(i+1, j+1, k+1) in Fortran.
!
! Binary format (stream, float32):
!   inputs.bin  : int32 nz, int32 ny, int32 nx, float32 field(...),
!                 float32 lat_rad(ny), float32 dx, float32 dt, float32 cu
!   fortran_out.bin : [int32 1, int32 N, float32 field_new(N)]

program damping_driver
  implicit none
  character(len=64) :: mode

  call get_command_argument(1, mode)

  select case (trim(mode))
  case ('pole_u');  call run_pole_u()
  case ('pole_v');  call run_pole_v()
  case default
    write(*,*) 'unknown mode: ', trim(mode);  stop 2
  end select

contains

  ! -----------------------------------------------------------------------
  ! Helper: compute tau and umax for mass-cell latitudes
  ! -----------------------------------------------------------------------
  subroutine compute_tau_umax(ny, lat_rad, dx, dt, cu, tau, umax)
    integer,    intent(in)  :: ny
    real,       intent(in)  :: lat_rad(ny), dx, dt, cu
    real,       intent(out) :: tau(ny), umax(ny)
    integer :: j
    real    :: c, s2
    do j = 1, ny
      c      = cos(lat_rad(j))
      s2     = 1.0 - c * c              ! sin²(lat)
      tau(j) = s2 ** 200               ! (sin²)^200
      umax(j) = cu * dx * c / dt       ! CFL-based umax (no physical cap — matches gSAM)
    end do
  end subroutine

  ! -----------------------------------------------------------------------
  ! pole_u: damp U of Python shape (nz, ny, nx+1)
  ! Fortran layout: U(nx+1, ny, nz) so column-major matches C row-major
  ! -----------------------------------------------------------------------
  subroutine run_pole_u()
    integer :: nz, ny, nx, u_in, u_out, N
    real, allocatable :: U(:,:,:), lat_rad(:), tau(:), umax(:)
    real :: dx, dt, cu, tau_j, umax_j, u_clipped
    integer :: k, j, i

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    ! Reverse axes: Python (nz,ny,nx+1) → Fortran (nx+1,ny,nz)
    allocate(U(nx+1, ny, nz), lat_rad(ny), tau(ny), umax(ny))
    read(u_in) U
    read(u_in) lat_rad
    read(u_in) dx, dt, cu
    close(u_in)

    call compute_tau_umax(ny, lat_rad, dx, dt, cu, tau, umax)

    ! Loop: Fortran U(i, j, k) = Python U[k, j, i]
    do k = 1, nz
      do j = 1, ny
        tau_j  = tau(j)
        umax_j = umax(j)
        do i = 1, nx+1
          u_clipped = min(umax_j, max(-umax_j, U(i,j,k)))
          U(i,j,k) = (U(i,j,k) + u_clipped * tau_j) / (1.0 + tau_j)
        end do
      end do
    end do

    N = (nx+1) * ny * nz
    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(N, 4)
    write(u_out) U
    close(u_out)
  end subroutine run_pole_u

  ! -----------------------------------------------------------------------
  ! pole_v: damp V of Python shape (nz, ny+1, nx)
  ! Fortran layout: V(nx, ny+1, nz)
  !
  ! Interior v-faces (j=2..ny in 1-based): lat_v = 0.5*(lat[j-1]+lat[j-2])
  ! Pole rows (j=1, j=ny+1): tau=1, umax=0 → V_new = V/2
  ! -----------------------------------------------------------------------
  subroutine run_pole_v()
    integer :: nz, ny, nx, u_in, u_out, N, nyv
    real, allocatable :: V(:,:,:), lat_rad(:)
    real, allocatable :: tau_v(:), umax_v(:)
    real :: dx, dt, cu
    real :: lat_v, c_v, s2_v, tau_j, umax_j, v_clipped
    integer :: k, j, i

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    nyv = ny + 1
    ! Reverse axes: Python (nz,ny+1,nx) → Fortran (nx,ny+1,nz)
    allocate(V(nx, nyv, nz), lat_rad(ny), tau_v(nyv), umax_v(nyv))
    read(u_in) V
    read(u_in) lat_rad
    read(u_in) dx, dt, cu
    close(u_in)

    ! Pole rows: tau=1, umax=0
    tau_v(1)   = 1.0;  umax_v(1)   = 0.0
    tau_v(nyv) = 1.0;  umax_v(nyv) = 0.0

    ! Interior v-faces (j=2..ny in 1-based): lat_v = 0.5*(lat[j-2]+lat[j-1])
    ! i.e. for Fortran jv=2..ny: lat_rad(jv-1) and lat_rad(jv) in 1-based
    do j = 2, ny
      lat_v = 0.5 * (lat_rad(j-1) + lat_rad(j))
      c_v   = cos(lat_v)
      s2_v  = 1.0 - c_v * c_v
      tau_v(j)  = s2_v ** 200
      umax_v(j) = cu * dx * c_v / dt
    end do

    ! Loop: Fortran V(i, j, k) = Python V[k, j-1, i-1] (0-indexed)
    do k = 1, nz
      do j = 1, nyv
        tau_j  = tau_v(j)
        umax_j = umax_v(j)
        do i = 1, nx
          v_clipped = min(umax_j, max(-umax_j, V(i,j,k)))
          V(i,j,k) = (V(i,j,k) + v_clipped * tau_j) / (1.0 + tau_j)
        end do
      end do
    end do

    N = nx * nyv * nz
    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(N, 4)
    write(u_out) V
    close(u_out)
  end subroutine run_pole_v

end program damping_driver
