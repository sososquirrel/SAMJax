! Fortran shadow of jsam/core/physics/sgs.py
!
! Cases:
!   shear_prod_zero_velocity   — U=V=W=0 → def2=0
!   shear_prod_uniform_u       — U=5, V=W=0 → def2=0 (uniform = no shear)
!   smag_zero_def2             — def2=0 → tk=tkh=0
!   smag_positive_def2         — def2=1e-4 → tk>0, tkh=Pr*tk
!   diffuse_scalar_uniform     — phi=const, tkh=any → dfdt=0
!
! inputs.bin layout (written by dump_inputs.py):
!   int32  nz, ny, nx
!   float32 U(nz, ny, nx+1)
!   float32 V(nz, ny+1, nx)
!   float32 W(nz+1, ny, nx)
!   float32 def2(nz, ny, nx)       (only read for smag/diffuse cases)
!   float32 phi(nz, ny, nx)        (only read for diffuse case)
!   float32 tkh(nz, ny, nx)        (only read for diffuse case)
!   float32 dx                     scalar
!   float32 dy(ny)
!   float32 dz(nz)
!   float32 Cs, Pr                 SGS params
program sgs_driver
  implicit none
  character(len=64) :: case_name
  call get_command_argument(1, case_name)

  select case (trim(case_name))
  case ('shear_prod_zero_velocity'); call run_shear_prod()
  case ('shear_prod_uniform_u');     call run_shear_prod()
  case ('smag_zero_def2');           call run_smag()
  case ('smag_positive_def2');       call run_smag()
  case ('diffuse_scalar_uniform');   call run_diffuse_scalar()
  case default
    write(*,*) 'unknown case: ', trim(case_name); stop 2
  end select

contains

  ! -----------------------------------------------------------------------
  ! Write result
  ! -----------------------------------------------------------------------
  subroutine write_result(result, n)
    integer, intent(in) :: n
    real, intent(in)    :: result(n)
    integer :: u_out
    open(newunit=u_out, file='fortran_out.bin', access='stream', form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(n, 4)
    write(u_out) result
    close(u_out)
  end subroutine write_result

  ! -----------------------------------------------------------------------
  ! dzw: distance between cell centres at w-faces, shape (nz+1,)
  !   dzw(0)   = dz(0)/2
  !   dzw(k)   = (dz(k-1)+dz(k))/2  for k=1..nz-1
  !   dzw(nz)  = dz(nz-1)/2
  ! (1-based indexing in Fortran, so dzw(1)=dz(1)/2, dzw(nz+1)=dz(nz)/2)
  ! -----------------------------------------------------------------------
  subroutine compute_dzw(dz, nz, dzw)
    integer, intent(in) :: nz
    real, intent(in)  :: dz(nz)
    real, intent(out) :: dzw(nz+1)
    integer :: k
    dzw(1) = 0.5 * dz(1)
    do k = 2, nz
      dzw(k) = 0.5 * (dz(k-1) + dz(k))
    end do
    dzw(nz+1) = 0.5 * dz(nz)
  end subroutine compute_dzw

  ! -----------------------------------------------------------------------
  ! shear_prod: def2 = 2 * S_ij * S_ij
  ! Mirrors jsam sgs.shear_prod (Cartesian approx for y).
  ! -----------------------------------------------------------------------
  subroutine run_shear_prod()
    integer :: nz, ny, nx, k, j, i, ip1, im1, jp1, jm1
    real, allocatable :: U(:,:,:), V(:,:,:), W(:,:,:)
    real, allocatable :: def2_dummy(:,:,:), phi(:,:,:), tkh(:,:,:)
    real, allocatable :: dx_arr(:), dy(:), dz(:)
    real :: dx, Cs, Pr
    real, allocatable :: def2(:,:,:)
    real, allocatable :: dzw(:)
    integer :: u_in
    ! Strain components
    real :: S11, S22, S33, rdx, rdy, diag_val
    real :: dudy_NE, dudy_NW, dudy_SE, dudy_SW
    real :: dvdx_NE, dvdx_NW, dvdx_SE, dvdx_SW
    real :: dudz_ab_ip1, dudz_ab_i, dudz_bel_ip1, dudz_bel_i
    real :: dwdx_above, dwdx_below
    real :: dvdz_ab_jp1, dvdz_ab_j, dvdz_bel_jp1, dvdz_bel_j
    real :: dwdy_above, dwdy_below
    real :: cross_uv, cross_uw, cross_vw
    real :: dzw_above, dzw_below

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    allocate(U(nz, ny, nx+1), V(nz, ny+1, nx), W(nz+1, ny, nx))
    allocate(def2_dummy(nz, ny, nx), phi(nz, ny, nx), tkh(nz, ny, nx))
    allocate(dx_arr(ny), dy(ny), dz(nz))
    read(u_in) U
    read(u_in) V
    read(u_in) W
    read(u_in) def2_dummy
    read(u_in) phi
    read(u_in) tkh
    read(u_in) dx
    read(u_in) dy
    read(u_in) dz
    read(u_in) Cs, Pr
    close(u_in)

    allocate(def2(nz, ny, nx), dzw(nz+1))
    call compute_dzw(dz, nz, dzw)

    rdx = 1.0 / dx

    do k = 1, nz
      dzw_above = dzw(k+1)
      dzw_below = dzw(k)
      do j = 1, ny
        rdy = 1.0 / dy(j)
        do i = 1, nx
          ! Periodic x
          ip1 = mod(i, nx) + 1
          im1 = mod(i - 2 + nx, nx) + 1
          ! y neighbours (Neumann/edge)
          jp1 = min(j+1, ny)
          jm1 = max(j-1, 1)

          ! Diagonal terms
          S11 = (U(k, j, i+1) - U(k, j, i)) * rdx
          S22 = (V(k, j+1, i) - V(k, j, i)) / dy(j)
          S33 = (W(k+1, j, i) - W(k, j, i)) / dz(k)
          diag_val = 2.0 * (S11**2 + S22**2 + S33**2)

          ! U-V cross terms: corners at (i+1/2, j+1/2), (i-1/2, j+1/2), etc.
          ! NE corner (i+1/2, j+1/2)
          dudy_NE = (U(k, min(j+1,ny), i+1) - U(k, j, i+1)) / dy(j)
          dvdx_NE = (V(k, min(j+1,ny), ip1) - V(k, min(j+1,ny), i)) * rdx
          ! NW corner (i-1/2, j+1/2)
          dudy_NW = (U(k, min(j+1,ny), i) - U(k, j, i)) / dy(j)
          dvdx_NW = (V(k, min(j+1,ny), i) - V(k, min(j+1,ny), im1)) * rdx
          ! SE corner (i+1/2, j-1/2)
          dudy_SE = (U(k, j, i+1) - U(k, max(j-1,1), i+1)) / dy(j)
          dvdx_SE = (V(k, j, ip1) - V(k, j, i)) * rdx
          ! SW corner (i-1/2, j-1/2)
          dudy_SW = (U(k, j, i) - U(k, max(j-1,1), i)) / dy(j)
          dvdx_SW = (V(k, j, i) - V(k, j, im1)) * rdx

          cross_uv = 0.25 * ((dudy_NE + dvdx_NE)**2 + (dudy_NW + dvdx_NW)**2 &
                            + (dudy_SE + dvdx_SE)**2 + (dudy_SW + dvdx_SW)**2)

          ! U-W cross terms: w-face above k+1, below k
          ! Uz padded edge in z: below k=1 → U(1,...), above k=nz → U(nz,...)
          dudz_ab_ip1 = (U(min(k+1,nz), j, i+1) - U(k, j, i+1)) / dzw_above
          dudz_ab_i   = (U(min(k+1,nz), j, i)   - U(k, j, i))   / dzw_above
          dudz_bel_ip1= (U(k, j, i+1) - U(max(k-1,1), j, i+1)) / dzw_below
          dudz_bel_i  = (U(k, j, i)   - U(max(k-1,1), j, i))   / dzw_below
          ! dw/dx at w-faces: W(k+1) and W(k)
          dwdx_above = (W(k+1, j, ip1) - W(k+1, j, i)) * rdx
          dwdx_below = (W(k,   j, ip1) - W(k,   j, i)) * rdx

          cross_uw = 0.25 * ((dudz_ab_ip1 + dwdx_above)**2 + (dudz_ab_i  + dwdx_above)**2 &
                            + (dudz_bel_ip1+ dwdx_below)**2 + (dudz_bel_i + dwdx_below)**2)

          ! V-W cross terms
          dvdz_ab_jp1 = (V(min(k+1,nz), min(j+1,ny+1), i) - V(k, min(j+1,ny+1), i)) / dzw_above
          dvdz_ab_j   = (V(min(k+1,nz), j, i)              - V(k, j, i))             / dzw_above
          dvdz_bel_jp1= (V(k, min(j+1,ny+1), i) - V(max(k-1,1), min(j+1,ny+1), i)) / dzw_below
          dvdz_bel_j  = (V(k, j, i)              - V(max(k-1,1), j, i))             / dzw_below
          ! dw/dy at w-faces: Wy padded edge in y
          dwdy_above = (W(k+1, min(j+1,ny), i) - W(k+1, j, i)) / dy(j)
          dwdy_below = (W(k,   min(j+1,ny), i) - W(k,   j, i)) / dy(j)

          cross_vw = 0.25 * ((dvdz_ab_jp1 + dwdy_above)**2 + (dvdz_ab_j  + dwdy_above)**2 &
                            + (dvdz_bel_jp1+ dwdy_below)**2 + (dvdz_bel_j + dwdy_below)**2)

          def2(k, j, i) = diag_val + cross_uv + cross_uw + cross_vw
        end do
      end do
    end do

    call write_result(reshape(def2, [nz*ny*nx]), nz*ny*nx)
    deallocate(U, V, W, def2_dummy, phi, tkh, dx_arr, dy, dz, def2, dzw)
  end subroutine run_shear_prod

  ! -----------------------------------------------------------------------
  ! smag_viscosity (pure Smagorinsky, no buoyancy)
  ! tk = Cs^2 * grd^2 * sqrt(max(0,def2))
  ! tkh = Pr * tk
  ! Returns tk and tkh concatenated: [tk(nz*ny*nx), tkh(nz*ny*nx)]
  ! -----------------------------------------------------------------------
  subroutine run_smag()
    integer :: nz, ny, nx, k, j
    real, allocatable :: U(:,:,:), V(:,:,:), W(:,:,:)
    real, allocatable :: def2(:,:,:), phi(:,:,:), tkh_in(:,:,:)
    real, allocatable :: dx_arr(:), dy(:), dz(:)
    real :: dx, Cs, Pr
    real, allocatable :: tk(:,:,:), tkh(:,:,:), out(:)
    real :: dx_eff, dy_eff, grd, delta_max
    integer :: u_in

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    allocate(U(nz, ny, nx+1), V(nz, ny+1, nx), W(nz+1, ny, nx))
    allocate(def2(nz, ny, nx), phi(nz, ny, nx), tkh_in(nz, ny, nx))
    allocate(dx_arr(ny), dy(ny), dz(nz))
    read(u_in) U
    read(u_in) V
    read(u_in) W
    read(u_in) def2
    read(u_in) phi
    read(u_in) tkh_in
    read(u_in) dx
    read(u_in) dy
    read(u_in) dz
    read(u_in) Cs, Pr
    close(u_in)

    allocate(tk(nz, ny, nx), tkh(nz, ny, nx))
    allocate(out(2 * nz * ny * nx))

    delta_max = 1000.0
    dx_eff = min(delta_max, dx)

    do k = 1, nz
      do j = 1, ny
        dy_eff = min(delta_max, dy(j))
        grd = (dz(k) * dx_eff * dy_eff) ** (1.0 / 3.0)
        tk(k, j, :)  = Cs**2 * grd**2 * sqrt(max(0.0, def2(k, j, :)))
        tkh(k, j, :) = Pr * tk(k, j, :)
      end do
    end do

    ! Output: [tk_flat, tkh_flat]
    out(1 : nz*ny*nx)              = reshape(tk,  [nz*ny*nx])
    out(nz*ny*nx+1 : 2*nz*ny*nx)  = reshape(tkh, [nz*ny*nx])
    call write_result(out, 2*nz*ny*nx)
    deallocate(U, V, W, def2, phi, tkh_in, dx_arr, dy, dz, tk, tkh, out)
  end subroutine run_smag

  ! -----------------------------------------------------------------------
  ! diffuse_scalar (forward Euler explicit, horizontal + vertical)
  ! For the uniform-field case: dfdt = 0.
  ! We compute dfdt = div(tkh * grad(phi)) and return it.
  !
  ! Horizontal: second-order centred (Cartesian)
  !   dfdt_x = (tkh[i+1]*dphi_x_east - tkh[i-1]*dphi_x_west) / dx^2
  ! For simplicity: flux_x at face i+1/2 = tkh_avg * (phi[i+1]-phi[i])/dx
  !   dfdt[k,j,i] += (F_x[i+1/2] - F_x[i-1/2]) / dx
  !                + (F_y[j+1/2] - F_y[j-1/2]) / dy
  !                + (F_z[k+1/2] - F_z[k-1/2]) / dz  (anelastic)
  ! -----------------------------------------------------------------------
  subroutine run_diffuse_scalar()
    integer :: nz, ny, nx, k, j, i, ip1, im1, jp1, jm1
    real, allocatable :: U(:,:,:), V(:,:,:), W(:,:,:)
    real, allocatable :: def2(:,:,:), phi(:,:,:), tkh(:,:,:)
    real, allocatable :: dx_arr(:), dy(:), dz(:)
    real :: dx, Cs, Pr
    real, allocatable :: dfdt(:,:,:)
    real :: tkh_east, tkh_west, flux_xe, flux_xw
    real :: tkh_north, tkh_south, flux_yn, flux_ys
    real :: tkh_top, tkh_bot, flux_zt, flux_zb
    integer :: u_in

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nz, ny, nx
    allocate(U(nz, ny, nx+1), V(nz, ny+1, nx), W(nz+1, ny, nx))
    allocate(def2(nz, ny, nx), phi(nz, ny, nx), tkh(nz, ny, nx))
    allocate(dx_arr(ny), dy(ny), dz(nz))
    read(u_in) U
    read(u_in) V
    read(u_in) W
    read(u_in) def2
    read(u_in) phi
    read(u_in) tkh
    read(u_in) dx
    read(u_in) dy
    read(u_in) dz
    read(u_in) Cs, Pr
    close(u_in)

    allocate(dfdt(nz, ny, nx))
    dfdt = 0.0

    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          ip1 = mod(i, nx) + 1
          im1 = mod(i - 2 + nx, nx) + 1
          jp1 = min(j + 1, ny)
          jm1 = max(j - 1, 1)

          ! Horizontal x flux (periodic)
          tkh_east  = 0.5 * (tkh(k,j,i)   + tkh(k,j,ip1))
          tkh_west  = 0.5 * (tkh(k,j,im1) + tkh(k,j,i))
          flux_xe = tkh_east  * (phi(k,j,ip1) - phi(k,j,i))   / dx
          flux_xw = tkh_west  * (phi(k,j,i)   - phi(k,j,im1)) / dx

          ! Horizontal y flux (zero-flux at boundary: clamp to same value)
          tkh_north = 0.5 * (tkh(k,j,i) + tkh(k,jp1,i))
          tkh_south = 0.5 * (tkh(k,jm1,i) + tkh(k,j,i))
          flux_yn = tkh_north * (phi(k,jp1,i) - phi(k,j,i))   / dy(j)
          flux_ys = tkh_south * (phi(k,j,i)   - phi(k,jm1,i)) / dy(j)
          ! Zero flux at j=1 and j=ny (Neumann)
          if (j == 1)  flux_ys = 0.0
          if (j == ny) flux_yn = 0.0

          ! Vertical flux (zero-flux at top/bottom)
          if (k == 1) then
            tkh_bot = tkh(k, j, i)
          else
            tkh_bot = 0.5 * (tkh(k-1,j,i) + tkh(k,j,i))
          end if
          if (k == nz) then
            tkh_top = tkh(k, j, i)
          else
            tkh_top = 0.5 * (tkh(k,j,i) + tkh(k+1,j,i))
          end if
          if (k == 1) then
            flux_zt = tkh_top * (phi(k+1,j,i) - phi(k,j,i)) / dz(k)
            flux_zb = 0.0
          else if (k == nz) then
            flux_zt = 0.0
            flux_zb = tkh_bot * (phi(k,j,i) - phi(k-1,j,i)) / dz(k)
          else
            flux_zt = tkh_top * (phi(k+1,j,i) - phi(k,j,i)) / dz(k)
            flux_zb = tkh_bot * (phi(k,j,i)   - phi(k-1,j,i)) / dz(k)
          end if

          dfdt(k,j,i) = (flux_xe - flux_xw) / dx &
                      + (flux_yn - flux_ys) / dy(j) &
                      + (flux_zt - flux_zb) / dz(k)
        end do
      end do
    end do

    call write_result(reshape(dfdt, [nz*ny*nx]), nz*ny*nx)
    deallocate(U, V, W, def2, phi, tkh, dx_arr, dy, dz, dfdt)
  end subroutine run_diffuse_scalar

end program sgs_driver
