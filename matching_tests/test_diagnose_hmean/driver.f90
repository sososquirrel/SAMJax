! test_diagnose_hmean/driver.f90
!
! Standalone port of gSAM diagnose.f90 horizontal mean computation.
!
! gSAM formula (flat terrain, single-rank):
!   tabs0d(k) = sum_j sum_i [ tabs(i,j,k) * wgt(j,k) ]
!   where wgt(j,k) = mu(j) * ady(j) * (nx*ny) / sums(k)
!   and sums(k) = sum_j [ mu(j) * ady(j) * nx ]     (terra=1 everywhere)
!
! Simplifies to:
!   wgt(j) = mu(j) * ady(j) / sum_j [ mu(j) * ady(j) ]
!   tabs0(k) = sum_j [ mean_x(tabs(k,j,:)) * wgt(j) ]
!
! Binary inputs.bin layout:
!   i4 nz, ny, nx
!   f4 mu(ny)           [= cos(lat)]
!   f4 ady(ny)          [dy stretching factor]
!   f4 TABS(nz,ny,nx)   [C order]
!   f4 QV(nz,ny,nx)
!   f4 QC(nz,ny,nx)
!   f4 QI(nz,ny,nx)
!   f4 QR(nz,ny,nx)
!   f4 QS(nz,ny,nx)
!   f4 QG(nz,ny,nx)
!
! Output fortran_out.bin:
!   f4 tabs0(nz) + qv0(nz) + qn0(nz) + qp0(nz)   [4*nz values]

program hmean_driver
  implicit none
  integer(4) :: nz, ny, nx
  integer :: i, j, k, n_out, u_in, u_out
  real, allocatable :: mu(:), ady(:), wgt(:)
  real, allocatable :: TABS(:,:,:), QV(:,:,:), QC(:,:,:), QI(:,:,:)
  real, allocatable :: QR(:,:,:), QS(:,:,:), QG(:,:,:)
  real(8), allocatable :: tabs0d(:), q0d(:), qn0d(:), qp0d(:)
  real, allocatable :: tabs0(:), qv0(:), qn0(:), qp0(:)
  real, allocatable :: buf(:)
  real(8) :: coefd, wsum

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  read(u_in) nz, ny, nx

  allocate(mu(ny), ady(ny), wgt(ny))
  read(u_in) mu
  read(u_in) ady
  close(u_in)

  ! Compute gSAM-style weights
  wsum = 0.0d0
  do j = 1, ny
    wsum = wsum + dble(mu(j)) * dble(ady(j))
  end do
  do j = 1, ny
    wgt(j) = real(dble(mu(j)) * dble(ady(j)) / wsum)
  end do

  ! Re-read fields
  allocate(TABS(nx,ny,nz), QV(nx,ny,nz), QC(nx,ny,nz), QI(nx,ny,nz))
  allocate(QR(nx,ny,nz), QS(nx,ny,nz), QG(nx,ny,nz))

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  ! Skip header: 3 ints + 2*ny floats = 12 + 8*ny bytes
  read(u_in, pos=13+8*ny)
  call read_carray(u_in, TABS, nz, ny, nx)
  call read_carray(u_in, QV,   nz, ny, nx)
  call read_carray(u_in, QC,   nz, ny, nx)
  call read_carray(u_in, QI,   nz, ny, nx)
  call read_carray(u_in, QR,   nz, ny, nx)
  call read_carray(u_in, QS,   nz, ny, nx)
  call read_carray(u_in, QG,   nz, ny, nx)
  close(u_in)

  ! Compute horizontal means (gSAM diagnose.f90:27-86 pattern)
  allocate(tabs0d(nz), q0d(nz), qn0d(nz), qp0d(nz))
  allocate(tabs0(nz), qv0(nz), qn0(nz), qp0(nz))

  do k = 1, nz
    tabs0d(k) = 0.0d0
    q0d(k)    = 0.0d0
    qn0d(k)   = 0.0d0
    qp0d(k)   = 0.0d0
    do j = 1, ny
      coefd = dble(wgt(j))
      do i = 1, nx
        tabs0d(k) = tabs0d(k) + dble(TABS(i,j,k)) * coefd
        q0d(k)    = q0d(k)    + dble(QV(i,j,k) + QC(i,j,k) + QI(i,j,k)) * coefd
        qn0d(k)   = qn0d(k)   + dble(QC(i,j,k) + QI(i,j,k)) * coefd
        qp0d(k)   = qp0d(k)   + dble(QR(i,j,k) + QS(i,j,k) + QG(i,j,k)) * coefd
      end do
    end do
    ! Divide by nx (the mean_x part)
    coefd = 1.0d0 / dble(nx)
    tabs0(k) = real(tabs0d(k) * coefd)
    qn0(k)   = real(qn0d(k)   * coefd)
    qp0(k)   = real(qp0d(k)   * coefd)
    qv0(k)   = real(q0d(k) * coefd) - qn0(k)   ! diagnose.f90:86
  end do

  ! Write output: tabs0 + qv0 + qn0 + qp0
  n_out = 4 * nz
  allocate(buf(n_out))
  buf(1:nz)        = tabs0
  buf(nz+1:2*nz)   = qv0
  buf(2*nz+1:3*nz) = qn0
  buf(3*nz+1:4*nz) = qp0

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
    real, allocatable :: tmp(:)
    integer :: ii, jj, kk, pos
    allocate(tmp(nz_l * ny_l * nx_l))
    read(u) tmp
    pos = 0
    do kk = 1, nz_l
      do jj = 1, ny_l
        do ii = 1, nx_l
          pos = pos + 1
          a(ii, jj, kk) = tmp(pos)
        end do
      end do
    end do
  end subroutine read_carray

end program hmean_driver
