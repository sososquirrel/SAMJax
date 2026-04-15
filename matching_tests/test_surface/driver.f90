! Fortran shadow of jsam/core/physics/surface.py
!
! Cases:
!   qsat_w_at_300K             — T=300K, p=1e5 Pa → qsat (Bolton formula)
!   qsat_monotone_in_T         — T=250..310K, p=1e5 Pa → array of qsat
!   bulk_fluxes_warm_sst_shf   — T_atm=295, SST=302 → SHF>0
!   bulk_fluxes_tau_opposes    — u=8, v=0 → tau_x < 0
!
! inputs.bin layout:
!   int32 mode_id   (1=qsat_single, 2=qsat_array, 3=bulk)
!   then mode-specific fields (see cases below)
!
! outputs: fortran_out.bin with the result
!
! Bolton (1980) qsat_w (matching jsam surface.py):
!   es = 611.2 * exp(17.67*(T-273.15)/(T-29.65))   Pa
!   eps = 0.6219
!   qsat = eps * es / (p - (1-eps)*es)
!
! For bulk fluxes: we implement the same Large & Pond / M-O iteration as jsam.
program surface_driver
  implicit none
  character(len=64) :: case_name
  call get_command_argument(1, case_name)

  select case (trim(case_name))
  case ('qsat_w_at_300K');           call run_qsat_single()
  case ('qsat_monotone_in_T');       call run_qsat_array()
  case ('bulk_fluxes_warm_sst_shf'); call run_bulk_fluxes()
  case ('bulk_fluxes_tau_opposes');  call run_bulk_fluxes()
  case default
    write(*,*) 'unknown case: ', trim(case_name); stop 2
  end select

contains

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

  ! Bolton (1980) saturation specific humidity
  !   T in K, p in Pa → qsat in kg/kg
  function qsatw_bolton(T, p) result(qsat)
    real, intent(in) :: T, p
    real :: qsat, es, eps
    eps = 0.6219
    es  = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65))   ! Pa
    qsat = eps * es / (p - (1.0 - eps) * es)
  end function qsatw_bolton

  ! -----------------------------------------------------------------------
  ! Case 1: single qsat at T=300K, p=1e5 Pa
  ! inputs.bin: int32 dummy=1; float32 T, p
  ! -----------------------------------------------------------------------
  subroutine run_qsat_single()
    real :: T, p, qs
    integer :: u_in, dummy
    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) dummy
    read(u_in) T, p
    close(u_in)
    qs = qsatw_bolton(T, p)
    call write_result([qs], 1)
  end subroutine run_qsat_single

  ! -----------------------------------------------------------------------
  ! Case 2: qsat array over T range
  ! inputs.bin: int32 nT; float32 T(nT), p
  ! -----------------------------------------------------------------------
  subroutine run_qsat_array()
    integer :: nT, k, u_in
    real :: p
    real, allocatable :: T(:), qs(:)
    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) nT
    allocate(T(nT), qs(nT))
    read(u_in) T
    read(u_in) p
    close(u_in)
    do k = 1, nT
      qs(k) = qsatw_bolton(T(k), p)
    end do
    call write_result(qs, nT)
    deallocate(T, qs)
  end subroutine run_qsat_array

  ! -----------------------------------------------------------------------
  ! Case 3 & 4: bulk surface fluxes
  ! inputs.bin:
  !   int32 dummy=3
  !   float32 T_atm, QV_atm, u_atm, v_atm  (atmosphere k=0)
  !   float32 SST
  !   float32 rho0, pres0, dz0, z0          (grid at k=0)
  !
  ! We implement the same 2-iteration M-O loop as jsam surface.py.
  ! Output: [shf, lhf, tau_x, tau_y]  (4 scalars)
  ! -----------------------------------------------------------------------
  subroutine run_bulk_fluxes()
    integer :: u_in, dummy
    real :: T_atm, QV_atm, u_atm, v_atm, SST
    real :: rho0, pres0, dz0, z0
    ! BulkParams (gSAM defaults matching surface.py)
    real, parameter :: umin        = 1.0
    real, parameter :: karman      = 0.4
    real, parameter :: epsv        = 0.61
    real, parameter :: salt_factor = 0.98
    real, parameter :: p00         = 1.0e5
    real, parameter :: Rd          = 287.0
    real, parameter :: Rv          = 461.5
    real, parameter :: cp          = 1004.0
    real, parameter :: g           = 9.81
    real, parameter :: eps         = 0.6219   ! Rd/Rv
    real, parameter :: zref        = 10.0
    real :: pres_sfc, exner0, exner_s, thbot, ts, vmag, delt, qs_sfc, delq
    real :: rdn, rhn, ren, ustar, tstar, qstar
    real :: shf, lhf, tau_x, tau_y
    real :: alz, hol, stable, xsq, xqq, psimh, psixh
    real :: rd_stab, u10n_val, delt_stab
    real :: rdn_new, rhn_new, ren_new
    real :: rd_new, rh_new, re_new
    real :: cdn_val
    integer :: iter
    real :: out(4)

    open(newunit=u_in, file='inputs.bin', access='stream', form='unformatted', status='old')
    read(u_in) dummy
    read(u_in) T_atm, QV_atm, u_atm, v_atm
    read(u_in) SST
    read(u_in) rho0, pres0, dz0, z0
    close(u_in)

    ! Surface pressure: hydrostatic half-step below cell centre
    pres_sfc = pres0 + rho0 * g * dz0 * 0.5

    ! Exner functions
    exner0  = (pres0    / p00) ** (Rd / cp)
    exner_s = (pres_sfc / p00) ** (Rd / cp)

    ! Potential temperatures
    thbot = T_atm * exner0
    ts    = SST   * exner_s

    delt  = thbot - ts

    ! Wind speed (min = umin)
    vmag = max(umin, sqrt(u_atm**2 + v_atm**2))

    ! Surface saturation humidity (salt factor)
    qs_sfc = salt_factor * qsatw_bolton(SST, pres_sfc)
    delq   = QV_atm - qs_sfc

    ! Neutral coefficients (first guess)
    cdn_val = 0.0027 / vmag + 0.000142 + 0.0000764 * vmag
    rdn = sqrt(cdn_val)
    if (delt > 0.0) then   ! stable
      rhn = 0.018
    else
      rhn = 0.0327
    end if
    ren = 0.0346

    ! 2 M-O iterations
    alz = log(z0 / zref)
    do iter = 1, 2
      ustar = rdn * vmag
      tstar = rhn * delt
      qstar = ren * delq

      hol = karman * g * z0 * (tstar / thbot + qstar / (1.0 / epsv + QV_atm)) &
          / (ustar**2 + 1.0e-20)
      hol = max(-10.0, min(10.0, hol))
      if (hol >= 0.0) then
        stable = 1.0
      else
        stable = 0.0
      end if

      xsq = max(sqrt(abs(1.0 - 16.0 * hol)), 1.0)
      xqq = sqrt(xsq)
      psimh = -5.0 * hol * stable + (1.0 - stable) * psimhu(xqq)
      psixh = -5.0 * hol * stable + (1.0 - stable) * psixhu(xqq)

      rd_stab  = rdn / (1.0 + rdn / karman * (alz - psimh))
      u10n_val = vmag * rd_stab / rdn

      ! Updated neutral coefficients
      if (delt > 0.0) then
        delt_stab = 1.0
      else
        delt_stab = 0.0
      end if
      cdn_val = 0.0027 / u10n_val + 0.000142 + 0.0000764 * u10n_val
      rdn_new = sqrt(cdn_val)
      rhn_new = (1.0 - delt_stab) * 0.0327 + delt_stab * 0.018
      ren_new = 0.0346

      rd_new = rdn_new / (1.0 + rdn_new / karman * (alz - psimh))
      rh_new = rhn_new / (1.0 + rhn_new / karman * (alz - psixh))
      re_new = ren_new / (1.0 + ren_new / karman * (alz - psixh))

      rdn = rdn_new
      rhn = rhn_new
      ren = ren_new
      ustar = rd_new * vmag
      tstar = rh_new * delt
      qstar = re_new * delq
    end do

    ! Fluxes
    shf   = -ustar * tstar
    lhf   = -ustar * qstar
    tau_x = -(ustar**2) * u_atm / vmag
    tau_y = -(ustar**2) * v_atm / vmag

    out = [shf, lhf, tau_x, tau_y]
    call write_result(out, 4)
  end subroutine run_bulk_fluxes

  ! Paulson (1970) unstable momentum stability function
  function psimhu(xd) result(r)
    real, intent(in) :: xd
    real :: r
    r = log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0) &
        - 2.0 * atan(xd) + 1.5707963
  end function psimhu

  ! Unstable heat/moisture stability function
  function psixhu(xd) result(r)
    real, intent(in) :: xd
    real :: r
    r = 2.0 * log((1.0 + xd * xd) / 2.0)
  end function psixhu

end program surface_driver
