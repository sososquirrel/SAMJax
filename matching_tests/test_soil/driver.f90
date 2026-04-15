! Fortran shadow of jsam/core/physics/slm/{sat,vapor_fluxes}.py and
! jsam/io/slm_init.py :: _cosby_1984 — the scalar arithmetic core of the
! SLM soil / saturation code. Used for Fortran-vs-jsam matching tests:
!
!   sat_qsatw       — saturation specific humidity over liquid water
!                     (Magnus / IFS Cy47r3, port of gSAM sat.f90)
!   sat_qsati       — saturation specific humidity over ice
!   cosby_1984      — Cosby 1984 hydraulic parameters (ks, Bconst,
!                     poro_soil, m_pot_sat, sst_capa, sst_cond,
!                     theta_FC, theta_WP, w_s_FC, w_s_WP) from
!                     SAND/CLAY percentages. Port of slm_vars.f90:1194-1261.
!   fh_calc         — fractional humidity at soil surface from Clapp &
!                     Hornberger moisture potential. Port of
!                     slm_vars.f90:1659-1666.
!
! All four cases operate on 1D arrays of length N. The Fortran formulas
! are bit-for-bit identical to the gSAM source they shadow and to the
! jsam implementation, so float32 matching should be within ~1 ULP.
!
! inputs.bin layout (per case):
!
!   sat_qsatw / sat_qsati:
!     int32  N
!     float32 T(N)          temperature (K)
!     float32 P(N)          pressure (hPa / mbar)
!
!   cosby_1984:
!     int32  N
!     float32 SAND(N)       sand fraction (%)
!     float32 CLAY(N)       clay fraction (%)
!
!   fh_calc:
!     int32  N
!     float32 T(N)          soil temperature (K)
!     float32 mps(N)        saturated moisture potential (mm, negative)
!     float32 sw(N)         soil wetness (0..1)
!     float32 B(N)          Cosby exponent
!
! Output: fortran_out.bin  — float32 stream (matches common/bin_io)
!   sat_qsatw / sat_qsati  : N values
!   cosby_1984             : 10*N values — ks, Bconst, poro_soil, m_pot_sat,
!                            sst_capa, sst_cond, theta_FC, theta_WP,
!                            w_s_FC, w_s_WP   (each concatenated)
!   fh_calc                : N values
program soil_driver
  implicit none
  character(len=64) :: case_name
  call get_command_argument(1, case_name)

  select case (trim(case_name))
  case ('sat_qsatw');  call run_sat_qsatw()
  case ('sat_qsati');  call run_sat_qsati()
  case ('cosby_1984'); call run_cosby()
  case ('fh_calc');    call run_fh_calc()
  case default
    write(*,*) 'unknown case: ', trim(case_name); stop 2
  end select

contains

  subroutine write_result(result, n)
    integer, intent(in) :: n
    real, intent(in)    :: result(n)
    integer :: u_out
    open(newunit=u_out, file='fortran_out.bin', access='stream', &
         form='unformatted', status='replace')
    write(u_out) 1_4
    write(u_out) int(n, 4)
    write(u_out) result
    close(u_out)
  end subroutine write_result

  ! -----------------------------------------------------------------
  ! Magnus / IFS Cy47r3 saturation vapour pressure (hPa)
  ! Port of gSAM SRC/sat.f90 lines 20-56 (verbatim).
  ! -----------------------------------------------------------------
  real function esatw_scalar(t)
    real, intent(in) :: t
    real, parameter :: e0 = 6.1121, t0 = 273.16, aw = 17.502, tw = 32.19
    esatw_scalar = e0 * exp(aw * (t - t0) / (t - tw))
  end function esatw_scalar

  real function esati_scalar(t)
    real, intent(in) :: t
    real, parameter :: e0 = 6.1121, t0 = 273.16, ai = 22.587, ti = -0.7
    esati_scalar = e0 * exp(ai * (t - t0) / (t - ti))
  end function esati_scalar

  real function qsatw_scalar(t, p)
    real, intent(in) :: t, p
    real :: es
    es = esatw_scalar(t)
    qsatw_scalar = 0.622 * es / max(es, p - es)
  end function qsatw_scalar

  real function qsati_scalar(t, p)
    real, intent(in) :: t, p
    real :: es
    es = esati_scalar(t)
    qsati_scalar = 0.622 * es / max(es, p - es)
  end function qsati_scalar

  ! -----------------------------------------------------------------
  ! sat_qsatw: qsatw(T,P)  — N samples
  ! -----------------------------------------------------------------
  subroutine run_sat_qsatw()
    integer :: n, i, u_in
    real, allocatable :: T(:), P(:), q(:)
    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    read(u_in) n
    allocate(T(n), P(n), q(n))
    read(u_in) T
    read(u_in) P
    close(u_in)
    do i = 1, n
      q(i) = qsatw_scalar(T(i), P(i))
    end do
    call write_result(q, n)
    deallocate(T, P, q)
  end subroutine run_sat_qsatw

  subroutine run_sat_qsati()
    integer :: n, i, u_in
    real, allocatable :: T(:), P(:), q(:)
    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    read(u_in) n
    allocate(T(n), P(n), q(n))
    read(u_in) T
    read(u_in) P
    close(u_in)
    do i = 1, n
      q(i) = qsati_scalar(T(i), P(i))
    end do
    call write_result(q, n)
    deallocate(T, P, q)
  end subroutine run_sat_qsati

  ! -----------------------------------------------------------------
  ! cosby_1984: compute 10 soil hydraulic fields from SAND/CLAY (%).
  ! Port of slm_vars.f90:1194-1261 and jsam/io/slm_init.py::_cosby_1984.
  ! -----------------------------------------------------------------
  subroutine run_cosby()
    integer :: n, i, u_in
    real, allocatable :: SAND(:), CLAY(:)
    real, allocatable :: ks(:), Bconst(:), poro_soil(:), m_pot_sat(:)
    real, allocatable :: sst_capa(:), sst_cond(:)
    real, allocatable :: theta_FC(:), theta_WP(:), w_s_FC(:), w_s_WP(:)
    real, allocatable :: out(:)
    real :: sand_frac, sst_cond_other, denom
    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    read(u_in) n
    allocate(SAND(n), CLAY(n))
    read(u_in) SAND
    read(u_in) CLAY
    close(u_in)

    allocate(ks(n), Bconst(n), poro_soil(n), m_pot_sat(n))
    allocate(sst_capa(n), sst_cond(n))
    allocate(theta_FC(n), theta_WP(n), w_s_FC(n), w_s_WP(n))

    do i = 1, n
      sand_frac = SAND(i) / 100.0
      if (SAND(i) > 20.0) then
        sst_cond_other = 2.0
      else
        sst_cond_other = 3.0
      end if
      sst_cond(i) = (7.7 ** sand_frac) * (sst_cond_other ** (1.0 - sand_frac))

      ks(i)        = (10.0 ** (0.0153 * SAND(i) - 0.884)) * (25.4 / 3600.0)
      Bconst(i)    = 0.159 * CLAY(i) + 2.91
      poro_soil(i) = -0.00126 * SAND(i) + 0.489
      m_pot_sat(i) = min(-150.0, -10.0 * (10.0 ** (1.88 - 0.0131 * SAND(i))))

      denom = max(SAND(i) + CLAY(i), 1.0e-30)
      sst_capa(i) = (2.128 * SAND(i) + 2.385 * CLAY(i)) / denom * 1.0e6

      theta_FC(i) = poro_soil(i) * &
                    ((0.1 / 86400.0 / max(ks(i), 1.0e-30)) &
                      ** (1.0 / (2.0 * Bconst(i) + 3.0)))
      theta_WP(i) = poro_soil(i) * &
                    ((-150000.0 / m_pot_sat(i)) ** (-1.0 / Bconst(i)))

      w_s_FC(i) = theta_FC(i) / max(poro_soil(i), 1.0e-30)
      w_s_WP(i) = theta_WP(i) / max(poro_soil(i), 1.0e-30)
    end do

    allocate(out(10 * n))
    out(        1:   n) = ks
    out(  n + 1: 2*n)   = Bconst
    out(2*n + 1: 3*n)   = poro_soil
    out(3*n + 1: 4*n)   = m_pot_sat
    out(4*n + 1: 5*n)   = sst_capa
    out(5*n + 1: 6*n)   = sst_cond
    out(6*n + 1: 7*n)   = theta_FC
    out(7*n + 1: 8*n)   = theta_WP
    out(8*n + 1: 9*n)   = w_s_FC
    out(9*n + 1:10*n)   = w_s_WP

    call write_result(out, 10 * n)

    deallocate(SAND, CLAY, ks, Bconst, poro_soil, m_pot_sat)
    deallocate(sst_capa, sst_cond, theta_FC, theta_WP, w_s_FC, w_s_WP, out)
  end subroutine run_cosby

  ! -----------------------------------------------------------------
  ! fh_calc: Clapp & Hornberger fractional humidity at soil surface
  ! Port of slm_vars.f90:1659-1666 (fh_calc) and
  ! jsam/core/physics/slm/vapor_fluxes.py::fh_calc.
  ! -----------------------------------------------------------------
  subroutine run_fh_calc()
    integer :: n, i, u_in
    real, allocatable :: T(:), mps(:), sw(:), B(:), fh(:)
    real, parameter :: g_grav = 9.81, rv = 461.5
    real :: denom, moist_pot1
    open(newunit=u_in, file='inputs.bin', access='stream', &
         form='unformatted', status='old')
    read(u_in) n
    allocate(T(n), mps(n), sw(n), B(n), fh(n))
    read(u_in) T
    read(u_in) mps
    read(u_in) sw
    read(u_in) B
    close(u_in)

    do i = 1, n
      denom = max(1.0e-10, sw(i) ** B(i))
      moist_pot1 = mps(i) / denom / 1000.0
      moist_pot1 = max(-1.0e8, moist_pot1)
      fh(i) = min(1.0, exp(moist_pot1 * g_grav / (rv * T(i))))
    end do

    call write_result(fh, n)
    deallocate(T, mps, sw, B, fh)
  end subroutine run_fh_calc

end program soil_driver
