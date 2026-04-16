! test_static_energy_roundtrip/driver.f90
!
! Computes gSAM's moist static energy variable t and jsam's simplified
! static energy s, then recovers TABS from each:
!
!   gSAM: t = TABS + gamaz - fac_cond*(qcl+qpl) - fac_sub*(qci+qpi)
!          TABS_recovered = t - gamaz + fac_cond*(qcl+qpl) + fac_sub*(qci+qpi)
!
!   jsam: s = TABS + gamaz
!          TABS_recovered = s - gamaz
!
! Output: TABS round-trip error for both formulations.
!
! Binary inputs.bin layout:
!   i4 nz, ny, nx
!   f4 gamaz(nz)
!   f4 fac_cond, fac_sub
!   f4 TABS(nz,ny,nx), QC(nz,ny,nx), QI(nz,ny,nx), QR(nz,ny,nx), QS(nz,ny,nx)
!
! Output fortran_out.bin:
!   f4 concat: gsam_t(nz,ny,nx) + gsam_tabs_recovered(nz,ny,nx)
!              + jsam_s(nz,ny,nx) + jsam_tabs_recovered(nz,ny,nx)

program se_driver
  implicit none
  integer(4) :: nz, ny, nx
  real :: fac_cond, fac_sub
  integer :: i, j, k, n_out, idx, u_in, u_out
  real, allocatable :: gamaz(:)
  real, allocatable :: TABS(:,:,:), QC(:,:,:), QI(:,:,:), QR(:,:,:), QS(:,:,:)
  real, allocatable :: gsam_t(:,:,:), gsam_trec(:,:,:)
  real, allocatable :: jsam_s(:,:,:), jsam_srec(:,:,:)
  real, allocatable :: buf(:)

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  read(u_in) nz, ny, nx

  allocate(gamaz(nz))
  read(u_in) gamaz
  read(u_in) fac_cond, fac_sub

  allocate(TABS(nx,ny,nz), QC(nx,ny,nz), QI(nx,ny,nz), QR(nx,ny,nz), QS(nx,ny,nz))
  call read_carray(u_in, TABS, nz, ny, nx)
  call read_carray(u_in, QC,   nz, ny, nx)
  call read_carray(u_in, QI,   nz, ny, nx)
  call read_carray(u_in, QR,   nz, ny, nx)
  call read_carray(u_in, QS,   nz, ny, nx)
  close(u_in)

  allocate(gsam_t(nx,ny,nz), gsam_trec(nx,ny,nz))
  allocate(jsam_s(nx,ny,nz), jsam_srec(nx,ny,nz))

  do k = 1, nz
    do j = 1, ny
      do i = 1, nx
        ! gSAM moist static energy
        gsam_t(i,j,k) = TABS(i,j,k) + gamaz(k) &
                       - fac_cond * (QC(i,j,k) + QR(i,j,k)) &
                       - fac_sub  * (QI(i,j,k) + QS(i,j,k))
        ! gSAM recovery
        gsam_trec(i,j,k) = gsam_t(i,j,k) - gamaz(k) &
                          + fac_cond * (QC(i,j,k) + QR(i,j,k)) &
                          + fac_sub  * (QI(i,j,k) + QS(i,j,k))
        ! jsam simplified static energy
        jsam_s(i,j,k) = TABS(i,j,k) + gamaz(k)
        ! jsam recovery
        jsam_srec(i,j,k) = jsam_s(i,j,k) - gamaz(k)
      end do
    end do
  end do

  ! Write output
  n_out = 4 * nz * ny * nx
  allocate(buf(n_out))
  idx = 0
  call write_carray(buf, idx, gsam_t,    nz, ny, nx)
  call write_carray(buf, idx, gsam_trec, nz, ny, nx)
  call write_carray(buf, idx, jsam_s,    nz, ny, nx)
  call write_carray(buf, idx, jsam_srec, nz, ny, nx)

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

  subroutine write_carray(buf_out, offset, a, nz_l, ny_l, nx_l)
    real, intent(inout) :: buf_out(*)
    integer, intent(inout) :: offset
    real, intent(in) :: a(nx_l, ny_l, nz_l)
    integer, intent(in) :: nz_l, ny_l, nx_l
    integer :: ii, jj, kk
    do kk = 1, nz_l
      do jj = 1, ny_l
        do ii = 1, nx_l
          offset = offset + 1
          buf_out(offset) = a(ii, jj, kk)
        end do
      end do
    end do
  end subroutine write_carray

end program se_driver
