! test_buoyancy/driver.f90
!
! Standalone Fortran port of the inner loop of gSAM SRC/buoyancy.f90
! (the terrain-free branch), reading inputs.bin produced by
! dump_inputs.py and writing fortran_out.bin in bin_io format.
!
! Formula (per cell k, for each (i,j)):
!
!   buo_cell(k) = (g/tabs0(k)) * (
!        tabs0(k)*(epsv*(qv(k)-qv0(k)) - (qcl(k)+qci(k)-qn0(k)
!                                         + qpl(k)+qpi(k)+qpg(k)-qp0(k)))
!      + (tabs(k)-tabs0(k)) * (1 + epsv*qv0(k) - qn0(k) - qp0(k))
!   )
!
! then interpolated to W-faces:
!
!   betu(k) = adz(kb) / (adz(k) + adz(kb))
!   betd(k) = adz(k)  / (adz(k) + adz(kb))
!   buo_face(k) = betu(k)*buo_cell(k) + betd(k)*buo_cell(kb)
!
! with rigid-lid BCs buo_face(1) = buo_face(nz+1) = 0.
!
! gSAM uses adz = dz/dz_ref (dimensionless), but since betu/betd only
! involve ratios of adz the scaling drops out — we can use dz directly.
!
! Binary inputs.bin layout (matches dump_inputs.py)
! --------------------------------------------------
!   i4 nz, ny, nx
!   f4 g, epsv
!   f4 tabs0(nz), qv0(nz), qn0(nz), qp0(nz)
!   f4 dz(nz)
!   f4 TABS(nz,ny,nx)  [C order: index i fastest]
!   f4 QV  (nz,ny,nx)
!   f4 QC  (nz,ny,nx)
!   f4 QI  (nz,ny,nx)
!   f4 QR  (nz,ny,nx)
!   f4 QS  (nz,ny,nx)
!   f4 QG  (nz,ny,nx)
!
! Output (common/bin_io.py format)
!   i4 1                  ! ndim
!   i4 (nz+1)*ny*nx       ! length
!   f4 buoy_face((nz+1)*ny*nx)    [C order: (k,j,i), i fastest]

program buoy_driver
  implicit none
  integer(4) :: nz, ny, nx
  real :: g, epsv
  integer :: i, j, k, kb, n_out, idx
  real, allocatable :: tabs0(:), qv0(:), qn0(:), qp0(:), dz(:)
  real, allocatable :: TABS(:,:,:), QV(:,:,:), QC(:,:,:), QI(:,:,:)
  real, allocatable :: QR(:,:,:), QS(:,:,:), QG(:,:,:)
  real, allocatable :: buo_cell(:,:,:), buo_face(:,:,:), buf(:)
  real :: qn, qp, qv_anom, qn_anom, qp_anom, thermal, buo_val
  real :: betu, betd
  integer :: u_in, u_out

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  read(u_in) nz, ny, nx
  read(u_in) g, epsv

  allocate(tabs0(nz), qv0(nz), qn0(nz), qp0(nz), dz(nz))
  read(u_in) tabs0
  read(u_in) qv0
  read(u_in) qn0
  read(u_in) qp0
  read(u_in) dz

  allocate(TABS(nx,ny,nz), QV(nx,ny,nz), QC(nx,ny,nz), QI(nx,ny,nz))
  allocate(QR(nx,ny,nz), QS(nx,ny,nz), QG(nx,ny,nz))
  call read_carray(u_in, TABS, nz, ny, nx)
  call read_carray(u_in, QV,   nz, ny, nx)
  call read_carray(u_in, QC,   nz, ny, nx)
  call read_carray(u_in, QI,   nz, ny, nx)
  call read_carray(u_in, QR,   nz, ny, nx)
  call read_carray(u_in, QS,   nz, ny, nx)
  call read_carray(u_in, QG,   nz, ny, nx)
  close(u_in)

  allocate(buo_cell(nx,ny,nz), buo_face(nx,ny,nz+1))

  ! ---- Cell-centre buoyancy ----
  do k = 1, nz
    do j = 1, ny
      do i = 1, nx
        qn      = QC(i,j,k) + QI(i,j,k)                    ! qcl + qci
        qp      = QR(i,j,k) + QS(i,j,k) + QG(i,j,k)        ! qpl + qpi (SAM1MOM lumps)
        qv_anom = epsv * (QV(i,j,k) - qv0(k))
        qn_anom = qn - qn0(k)
        qp_anom = qp - qp0(k)
        thermal = (TABS(i,j,k) - tabs0(k)) &
                * (1.0 + epsv * qv0(k) - qn0(k) - qp0(k))
        buo_val = (g / tabs0(k)) * ( &
                    tabs0(k) * (qv_anom - qn_anom - qp_anom) + thermal )
        buo_cell(i,j,k) = buo_val
      end do
    end do
  end do

  ! ---- W-face interpolation ----
  buo_face = 0.0
  do k = 2, nz
    kb   = k - 1
    betu = dz(kb) / (dz(k) + dz(kb))
    betd = dz(k)  / (dz(k) + dz(kb))
    do j = 1, ny
      do i = 1, nx
        buo_face(i,j,k) = betu * buo_cell(i,j,k) + betd * buo_cell(i,j,kb)
      end do
    end do
  end do
  ! Rigid lid / ground BCs are already 0 (array was zero-initialised).

  ! ---- Write fortran_out.bin ----
  n_out = (nz + 1) * ny * nx
  allocate(buf(n_out))
  idx = 0
  do k = 1, nz + 1
    do j = 1, ny
      do i = 1, nx
        idx = idx + 1
        buf(idx) = buo_face(i,j,k)
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

  subroutine read_carray(u, a, nz_l, ny_l, nx_l)
    integer, intent(in) :: u, nz_l, ny_l, nx_l
    real, intent(out) :: a(nx_l, ny_l, nz_l)
    real, allocatable :: buf(:)
    integer :: ii, jj, kk, idx_l
    allocate(buf(nz_l * ny_l * nx_l))
    read(u) buf
    idx_l = 0
    do kk = 1, nz_l
      do jj = 1, ny_l
        do ii = 1, nx_l
          idx_l = idx_l + 1
          a(ii, jj, kk) = buf(idx_l)
        end do
      end do
    end do
  end subroutine read_carray

end program buoy_driver
