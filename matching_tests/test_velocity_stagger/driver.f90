! test_velocity_stagger/driver.f90
!
! Applies gSAM C-grid boundary conditions to mass-grid velocity fields:
!   U(nx+1, ny, nz): periodic wrap U(nx+1,:,:) = U(1,:,:)
!   V(nx, ny+1, nz): pole walls V(:,1,:)=0, V(:,ny+1,:)=0
!   W(nx, ny, nz+1): rigid lid W(:,:,1)=0, W(:,:,nz+1)=0
!
! Binary inputs.bin layout:
!   i4 nz, ny, nx
!   f4 U_mass(nz, ny, nx)   [C order]
!   f4 V_mass(nz, ny, nx)
!   f4 W_mass(nz, ny, nx)
!
! Output fortran_out.bin:
!   Concatenated: U_stag(nz,ny,nx+1) + V_stag(nz,ny+1,nx) + W_stag(nz+1,ny,nx)

program stagger_driver
  implicit none
  integer(4) :: nz, ny, nx
  integer :: i, j, k, n_out, idx, u_in, u_out
  real, allocatable :: U_mass(:,:,:), V_mass(:,:,:), W_mass(:,:,:)
  real, allocatable :: U_stag(:,:,:), V_stag(:,:,:), W_stag(:,:,:)
  real, allocatable :: buf(:)

  open(newunit=u_in, file='inputs.bin', access='stream', &
       form='unformatted', status='old')
  read(u_in) nz, ny, nx

  allocate(U_mass(nx,ny,nz), V_mass(nx,ny,nz), W_mass(nx,ny,nz))
  call read_carray(u_in, U_mass, nz, ny, nx)
  call read_carray(u_in, V_mass, nz, ny, nx)
  call read_carray(u_in, W_mass, nz, ny, nx)
  close(u_in)

  ! --- Stagger U: (nx+1, ny, nz) with periodic wrap ---
  allocate(U_stag(nx+1, ny, nz))
  U_stag = 0.0
  do k = 1, nz
    do j = 1, ny
      do i = 1, nx
        U_stag(i, j, k) = U_mass(i, j, k)
      end do
      U_stag(nx+1, j, k) = U_mass(1, j, k)  ! periodic BC
    end do
  end do

  ! --- Stagger V: (nx, ny+1, nz) with pole walls ---
  allocate(V_stag(nx, ny+1, nz))
  V_stag = 0.0
  ! V(:, 1, :) = 0 (south pole wall)
  ! V(:, ny+1, :) = 0 (north pole wall)
  do k = 1, nz
    do j = 2, ny  ! interior V-faces
      do i = 1, nx
        V_stag(i, j, k) = 0.5 * (V_mass(i, j-1, k) + V_mass(i, j, k))
      end do
    end do
  end do

  ! --- Stagger W: (nx, ny, nz+1) with rigid lid ---
  allocate(W_stag(nx, ny, nz+1))
  W_stag = 0.0
  ! W(:, :, 1) = 0 (ground)
  ! W(:, :, nz+1) = 0 (rigid lid)
  do k = 2, nz  ! interior W-faces
    do j = 1, ny
      do i = 1, nx
        W_stag(i, j, k) = 0.5 * (W_mass(i, j, k-1) + W_mass(i, j, k))
      end do
    end do
  end do

  ! --- Write fortran_out.bin: concat U_stag + V_stag + W_stag ---
  n_out = nz*ny*(nx+1) + nz*(ny+1)*nx + (nz+1)*ny*nx
  allocate(buf(n_out))
  idx = 0

  ! U_stag in C order: (nz, ny, nx+1)
  do k = 1, nz
    do j = 1, ny
      do i = 1, nx+1
        idx = idx + 1
        buf(idx) = U_stag(i, j, k)
      end do
    end do
  end do

  ! V_stag in C order: (nz, ny+1, nx)
  do k = 1, nz
    do j = 1, ny+1
      do i = 1, nx
        idx = idx + 1
        buf(idx) = V_stag(i, j, k)
      end do
    end do
  end do

  ! W_stag in C order: (nz+1, ny, nx)
  do k = 1, nz+1
    do j = 1, ny
      do i = 1, nx
        idx = idx + 1
        buf(idx) = W_stag(i, j, k)
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

end program stagger_driver
